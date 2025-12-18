import argparse
import os
import sys
import pathlib

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch

import numpy as np

# Make sure we can import stereo code from the project root (../core, ../utils)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "core") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "core"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from igev_stereo import IGEVStereo  # noqa: E402
from utils.utils import InputPadder  # noqa: E402

# Stereo parameters (must match the stereo node)
BASELINE_M = 0.4
FOCAL_LENGTH_PX = 831.38
# Drop detections whose bounding-box area is below this threshold (pixels^2)
MIN_BOX_AREA_PX = 500

# Simple obstacle avoidance tuning
CAM_OFFSET_X = 1.5
CAM_OFFSET_Y = 0.2
AVOID_RADIUS_M = 1.25          # required clearance from path center to rock
DETROUR_EXTRA_M = 0.75         # extra margin beyond rock clearance
DETROUR_COMPLETE_THRESH = 0.8  # how close we must get to detour waypoint before resuming goal
REPLAN_MIN_ADVANCE_M = 0.75    # ignore rocks behind the rover


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 rock/shadow inference on ROS left image.")
    parser.add_argument(
        "--weights",
        type=str,
        default="rock_shadow_polar_bestepoch200real.pt",
        help="Path to YOLOv5 weights (.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda device id (e.g. 0) or cpu. Empty = auto",
    )
    parser.add_argument(
        "--view-img",
        action="store_true",
        help="Show results window",
    )
    parser.add_argument(
        "--stereo-ckpt",
        type=str,
        default=None,
        help="Path to IGEV++ stereo checkpoint (.pth). If omitted, uses ../pretrained_models/middlebury.pth relative to this file.",
    )
    parser.add_argument(
        "--stereo-device",
        type=str,
        default="",
        help="Device for stereo model (e.g., cuda:0 or cpu). Empty = auto (cuda if available).",
    )
    return parser.parse_args()


def load_model(weights_path: str, device: str):
    """Load YOLOv5 without clashing with local `utils` package."""
    device = device.strip() if device else ""
    core_path = str(PROJECT_ROOT / "core")
    root_path = str(PROJECT_ROOT)
    yolo_repo = pathlib.Path(torch.hub.get_dir()) / "ultralytics_yolov5_master"
    repo_path = str(yolo_repo)

    # Remove conflicting paths and injected modules to avoid shadowing YOLO's utils
    removed = []
    for p in (core_path, root_path):
        if p in sys.path:
            sys.path.remove(p)
            removed.append(p)
    if repo_path not in sys.path and yolo_repo.exists():
        sys.path.insert(0, repo_path)

    saved_utils = sys.modules.pop("utils", None)
    try:
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=weights_path,
            trust_repo=True,
            device=device if device else None,
        )
    finally:
        if saved_utils is not None:
            sys.modules["utils"] = saved_utils
        for p in reversed(removed):
            sys.path.insert(0, p)
    return model


def load_stereo_model(ckpt_path: str, device: str):
    """Load IGEV++ stereo model for disparity estimation."""
    device = device.strip() if device else ("cuda" if torch.cuda.is_available() else "cpu")
    default_args = argparse.Namespace(
        restore_ckpt=ckpt_path,
        mixed_precision=False,
        precision_dtype="float32",
        valid_iters=16,
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=768,
        s_disp_range=48,
        m_disp_range=96,
        l_disp_range=192,
        s_disp_interval=1,
        m_disp_interval=2,
        l_disp_interval=4,
    )
    model = torch.nn.DataParallel(IGEVStereo(default_args))
    model.load_state_dict(torch.load(default_args.restore_ckpt, map_location=device))
    model = model.module.to(device).eval()
    return model, device


class YoloRockNode(Node):
    def __init__(self, model, bridge, imgsz: int, view_img: bool, stereo_model, stereo_device: str):
        super().__init__("yolov5_rock_detector")
        self.model = model
        self.bridge = bridge
        self.imgsz = imgsz
        self.view_img = view_img
        self.frame_count = 0
        self.disparity = None
        self.disparity_count = 0
        self.stereo_model = stereo_model
        self.stereo_device = stereo_device
        self.left_image = None
        self.right_image = None

        # State for avoidance and localization
        self.current_path = []          # list of np.array([x, y])
        self.final_goal = None          # np.array([x, y])
        self.rover_xy = None            # np.array([x, y])
        self.rover_yaw = 0.0
        self.active_detour = False
        self.detour_target = None       # np.array([x, y])
        self.pending_goal = None        # goal to restore after detour

        self.sub_left = self.create_subscription(
            Image,
            "/chrono_ros_node/stereo/left",
            self.left_callback,
            10,
        )
        self.sub_right = self.create_subscription(
            Image,
            "/chrono_ros_node/stereo/right",
            self.right_callback,
            10,
        )
        self.path_sub = self.create_subscription(
            Path,
            "/chrono_ros_node/output/rover/waypoint_path",
            self.path_callback,
            10,
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/chrono_ros_node/output/rover/state/pose",
            self.pose_callback,
            10,
        )
        self.waypoint_pub = self.create_publisher(
            Vector3,
            "/chrono_ros_node/input/driver_waypoint_update",
            10,
        )
        self.pub_annotated = self.create_publisher(
            Image,
            "/chrono_ros_node/yolov5/annotated_left",
            10,
        )
        self.get_logger().info("YOLOv5 rock detector node initialized.")

    # --- ROS callbacks for localization and path state ---
    def pose_callback(self, msg: PoseStamped):
        self.rover_xy = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float64)
        # yaw from quaternion (ROS uses x,y,z,w)
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.rover_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    def path_callback(self, msg: Path):
        if not msg.poses:
            self.current_path = []
            self.final_goal = None
            return
        pts = []
        for pose in msg.poses:
            pts.append(np.array([pose.pose.position.x, pose.pose.position.y], dtype=np.float64))
        self.current_path = pts
        self.final_goal = pts[-1]
        # If we just replanned to a detour, remember where to return
        if self.pending_goal is None:
            self.pending_goal = self.final_goal

    def right_callback(self, msg: Image):
        """Receive right image and cache it."""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed (right): {e}")
            return
        # Flip vertically to match left handling
        self.right_image = cv2.flip(bgr, 0)

    def compute_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        """Run IGEV++ stereo to compute disparity map."""
        if left_bgr is None or right_bgr is None:
            return None
        image1 = torch.from_numpy(left_bgr).permute(2, 0, 1).float().unsqueeze(0).to(self.stereo_device)
        image2 = torch.from_numpy(right_bgr).permute(2, 0, 1).float().unsqueeze(0).to(self.stereo_device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            disp = self.stereo_model(image1, image2, iters=16, test_mode=True)
        disp = padder.unpad(disp)
        disparity = disp.cpu().numpy().squeeze()
        self.disparity_count += 1
        if self.disparity_count % 30 == 0:
            self.get_logger().info(f"Computed disparities: {self.disparity_count}")
        return disparity

    def estimate_depth_m(self, box_xyxy):
        """
        Estimate depth (meters) at the box center using the latest disparity.
        Returns None if disparity is unavailable or invalid.
        """
        if self.disparity is None:
            return None

        x1, y1, x2, y2 = box_xyxy[:4]
        xc = int((x1 + x2) / 2)
        yc = int((y1 + y2) / 2)

        h_disp, w_disp = self.disparity.shape
        if not (0 <= xc < w_disp and 0 <= yc < h_disp):
            return None

        # Use a small patch to be more robust to sparse/zero disparity at a single pixel.
        x0 = max(0, xc - 2)
        x1p = min(w_disp, xc + 3)
        y0 = max(0, yc - 2)
        y1p = min(h_disp, yc + 3)
        patch = self.disparity[y0:y1p, x0:x1p]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        d = float(np.median(valid))
        return (BASELINE_M * FOCAL_LENGTH_PX) / d

    # --- Geometry helpers for avoidance ---
    def _project_on_path(self, point_xy: np.ndarray):
        """Return (s_along_path, distance, closest_point, tangent_dir)."""
        if len(self.current_path) < 2:
            return 0.0, float("inf"), point_xy, np.array([1.0, 0.0])
        best_dist = float("inf")
        best_s = 0.0
        best_pt = self.current_path[0]
        best_dir = np.array([1.0, 0.0])
        s_accum = 0.0
        for i in range(len(self.current_path) - 1):
            p0 = self.current_path[i]
            p1 = self.current_path[i + 1]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            t = float(np.clip(np.dot(point_xy - p0, seg) / (seg_len ** 2), 0.0, 1.0))
            proj = p0 + t * seg
            dist = float(np.linalg.norm(point_xy - proj))
            if dist < best_dist:
                best_dist = dist
                best_s = s_accum + t * seg_len
                best_pt = proj
                best_dir = seg / seg_len
            s_accum += seg_len
        return best_s, best_dist, best_pt, best_dir

    def _pixel_to_body_xy(self, det):
        """Approximate rock position in rover body frame using depth + pixel offset."""
        if self.disparity is None:
            return None
        depth_m = self.estimate_depth_m(det)
        if depth_m is None:
            return None
        x1, y1, x2, y2, *_ = det
        xc = float((x1 + x2) / 2.0)
        yc = float((y1 + y2) / 2.0)
        cx = self.left_image.shape[1] / 2.0
        cy = self.left_image.shape[0] / 2.0
        lateral = (xc - cx) * depth_m / FOCAL_LENGTH_PX
        forward = depth_m
        # Camera points + mount offset (assume camera faces +X, y left)
        return np.array([forward + CAM_OFFSET_X, CAM_OFFSET_Y - lateral], dtype=np.float64)

    def _body_to_world(self, body_xy: np.ndarray):
        if self.rover_xy is None:
            return None
        c = np.cos(self.rover_yaw)
        s = np.sin(self.rover_yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return self.rover_xy + rot @ body_xy

    def publish_waypoint(self, xy_world: np.ndarray):
        msg = Vector3()
        msg.x = float(xy_world[0])
        msg.y = float(xy_world[1])
        msg.z = 0.0
        self.waypoint_pub.publish(msg)
        self.get_logger().info(
            f"Published avoidance waypoint ({msg.x:.2f}, {msg.y:.2f})"
        )

    def handle_obstacle_avoidance(self, dets_np: np.ndarray):
        """Check detections against current path and request a detour if needed."""
        if (
            self.rover_xy is None
            or len(self.current_path) < 2
            or self.final_goal is None
        ):
            return

        rover_s, _, _, _ = self._project_on_path(self.rover_xy)
        blocking_candidate = None
        blocking_proj = None
        blocking_dir = None
        min_forward_s = float("inf")

        for det in dets_np:
            body_xy = self._pixel_to_body_xy(det)
            if body_xy is None:
                continue
            world_xy = self._body_to_world(body_xy)
            if world_xy is None:
                continue

            s, dist, proj, seg_dir = self._project_on_path(world_xy)
            if s <= rover_s + REPLAN_MIN_ADVANCE_M:
                continue  # behind or too close to ignore
            if dist > AVOID_RADIUS_M:
                continue  # not in our corridor
            if s < min_forward_s:
                min_forward_s = s
                blocking_candidate = world_xy
                blocking_proj = proj
                blocking_dir = seg_dir

        if blocking_candidate is None or self.active_detour:
            return

        # Build a simple lateral detour around the blocking rock
        delta = blocking_candidate - blocking_proj
        cross = blocking_dir[0] * delta[1] - blocking_dir[1] * delta[0]
        perp = np.array([-blocking_dir[1], blocking_dir[0]], dtype=np.float64)
        side = -1.0 if cross >= 0.0 else 1.0  # go opposite side of the rock
        offset_mag = AVOID_RADIUS_M + DETROUR_EXTRA_M
        detour_xy = blocking_candidate + side * offset_mag * perp

        self.active_detour = True
        self.detour_target = detour_xy
        self.pending_goal = self.final_goal
        self.publish_waypoint(detour_xy)

    def maybe_complete_detour(self):
        if not self.active_detour or self.rover_xy is None or self.detour_target is None:
            return
        dist = float(np.linalg.norm(self.rover_xy - self.detour_target))
        if dist > DETROUR_COMPLETE_THRESH:
            return
        if self.pending_goal is None:
            self.active_detour = False
            self.detour_target = None
            return
        self.publish_waypoint(self.pending_goal)
        self.active_detour = False
        self.detour_target = None

    def left_callback(self, msg: Image):
        self.frame_count += 1
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Flip vertically to match the stereo node behavior
        bgr = cv2.flip(bgr, 0)
        self.left_image = bgr

        # Compute disparity on-demand using the latest right image
        if self.right_image is None:
            if self.frame_count % 30 == 0:
                self.get_logger().warn("No right image received yet; disparity unavailable")
            self.disparity = None
        else:
            try:
                self.disparity = self.compute_disparity(self.left_image, self.right_image)
            except Exception as e:
                self.get_logger().error(f"Stereo disparity computation failed: {e}")
                self.disparity = None

        rgb = bgr[..., ::-1]
        try:
            with torch.no_grad():
                results = self.model(rgb, size=self.imgsz)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        # Filter out tiny detections before rendering/publishing
        dets = results.xyxy[0]
        if len(dets):
            widths = dets[:, 2] - dets[:, 0]
            heights = dets[:, 3] - dets[:, 1]
            areas = widths * heights
            keep = areas >= MIN_BOX_AREA_PX
            if keep.sum() != len(dets):
                filtered = dets[keep]
                results.xyxy[0] = filtered
                results.pred[0] = filtered

        # Throttle logging to avoid spamming the console
        if self.frame_count % 30 == 0:
            det_count = len(results.xyxy[0])
            self.get_logger().info(f"Frame {self.frame_count}: detections={det_count}")

        # Render returns a readonly array; make a writable contiguous copy for drawing.
        annotated_bgr = results.render()[0].copy()
        annotated_bgr = np.ascontiguousarray(annotated_bgr)
        annotated_bgr.setflags(write=1)

        # Add depth estimates if disparity is available
        if self.disparity is None and self.frame_count % 30 == 0:
            self.get_logger().warn("No disparity available; depth unavailable")

        if self.disparity is not None:
            h_disp, w_disp = self.disparity.shape
            dets = results.xyxy[0].cpu().numpy()
            for det in dets:
                x1, y1, x2, y2, conf, cls = det
                depth_m = self.estimate_depth_m(det)
                if depth_m is not None:
                    xc = int((x1 + x2) / 2)
                    yc = int((y1 + y2) / 2)
                    label = f"{depth_m:.2f}m"
                    cv2.putText(
                        annotated_bgr,
                        label,
                        (xc, max(0, yc - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                # If disparity missing/invalid, skip depth overlay

            # Run obstacle avoidance once per frame with current detections
            self.handle_obstacle_avoidance(dets)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
            self.pub_annotated.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Publishing annotated image failed: {e}")

        if self.view_img:
            cv2.imshow("YOLOv5 Rocks (left)", annotated_bgr)
            # Optional disparity visualization to confirm we're using the stereo output.
            # if self.disparity is not None:
            #     disp_vis = cv2.normalize(self.disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #     disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            #     cv2.imshow("Disparity (raw)", disp_color)
            cv2.waitKey(1)

        # If we are on a detour, check whether we can resume the original target
        self.maybe_complete_detour()


def main():
    args = parse_args()
    weights = pathlib.Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    # Resolve stereo checkpoint: CLI wins; otherwise default to ../pretrained_models/middlebury.pth
    if args.stereo_ckpt:
        stereo_ckpt = pathlib.Path(args.stereo_ckpt).expanduser().resolve()
    else:
        stereo_ckpt = pathlib.Path(__file__).resolve().parent.parent / "pretrained_models" / "middlebury.pth"
    if not stereo_ckpt.exists():
        raise FileNotFoundError(
            f"Stereo checkpoint not found: {stereo_ckpt}. "
            "Pass --stereo-ckpt or place middlebury.pth in ../pretrained_models relative to this file."
        )

    model = load_model(str(weights), args.device)
    model.conf = args.conf
    model.iou = args.iou
    model.max_det = args.max_det
    model.classes = None  # detect both rock (0) and shadow (1)

    stereo_model, stereo_device = load_stereo_model(str(stereo_ckpt), args.stereo_device)

    rclpy.init()
    bridge = CvBridge()
    node = YoloRockNode(
        model,
        bridge,
        imgsz=args.imgsz,
        view_img=args.view_img,
        stereo_model=stereo_model,
        stereo_device=stereo_device,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if args.view_img:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

