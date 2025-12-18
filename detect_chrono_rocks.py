import sys
sys.path.append('core')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
import os
import argparse

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from sklearn.cluster import DBSCAN
# import argparse

# Minimum object size requirements
MIN_WIDTH_PX = 20
MIN_HEIGHT_PX = 20
MIN_WIDTH_M = 0.10
MIN_HEIGHT_M = 0.10

def detect_obstacles(disparity):

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    disparity_blur = cv2.bilateralFilter(cleaned, 5, 50, 50)
    edge = cv2.Canny(disparity_blur, 15, 40)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            h > 0
            and w > 0
            and w * h > 50
            and w / h < 5
            and w >= MIN_WIDTH_PX
            and h >= MIN_HEIGHT_PX
        ):
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes, disparity_blur, edge

class StereoInferenceNode(Node):
    def __init__(self):
        super().__init__('stereo_inference_node')

        self.sub_left = self.create_subscription(Image, '/chrono_ros_node/stereo/left', self.left_callback, 10)
        self.sub_right = self.create_subscription(Image, '/chrono_ros_node/stereo/right', self.right_callback, 10)
        self.pub_disparity = self.create_publisher(Image, '/chrono_ros_node/stereo/disparity', 10)
        self.pub_disparity_raw = self.create_publisher(Image, '/chrono_ros_node/stereo/disparity_raw', 10)
        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None

        default_args = argparse.Namespace(
            restore_ckpt='./pretrained_models/middlebury.pth',
            mixed_precision=False,
            precision_dtype='float32',
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

        self.model = torch.nn.DataParallel(IGEVStereo(default_args))
        self.model.load_state_dict(torch.load(default_args.restore_ckpt))
        self.model = self.model.module.to(DEVICE).eval()
        self.get_logger().info("Stereo Inference Node Initialized")

    def left_callback(self, msg):
        self.left_image = cv2.flip(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), 0)
        self.process_images()

    def right_callback(self, msg):
        self.right_image = cv2.flip(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), 0) 
        self.process_images()

    def process_images(self):

        if self.left_image is None or self.right_image is None:
            return

        self.get_logger().info("Processing stereo images...")

        image1 = torch.from_numpy(self.left_image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        image2 = torch.from_numpy(self.right_image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            disparity = self.model(image1, image2, iters=16, test_mode=True)

        disparity = padder.unpad(disparity)
        disparity = disparity.cpu().numpy().squeeze()

        # Work on the full left image; pre-clean with morphology to drop stars/noise.
        img_gray = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)

        bounding_boxes, img_blur, img_edge = detect_obstacles(img_gray)

        B = 0.4  # Baseline in meters
        f = 831.38  # Focal length in pixels

        detections_overlay = self.left_image.copy()

        for x, y, w, h in bounding_boxes:
            # Get disparity value at the center pixel
            x_center = int(x + w / 2)
            y_center = int(y + h / 2)

            if y_center < 0 or y_center >= disparity.shape[0] or x_center < 0 or x_center >= disparity.shape[1]:
                continue

            d = disparity[y_center, x_center]

            # Compute depth (avoid division by zero)
            if d > 0:
                Z = (B * f) / d
                approx_width_m = (w / f) * Z
                approx_height_m = (h / f) * Z
                if (
                    approx_width_m > 3.0
                    or approx_width_m < MIN_WIDTH_M
                    or approx_height_m < MIN_HEIGHT_M
                ):
                    continue
                self.get_logger().info(f"Rock detected at ({x_center}, {y_center}) with depth: {Z:.2f} meters")
            else:
                Z = float('inf')  # Infinite depth if disparity is zero
                self.get_logger().warn(f"Rock at ({x_center}, {y_center}) has invalid disparity (depth unknown)")

            cv2.rectangle(detections_overlay, (x-5, y-5), (x + w+5, y + h+5), (0, 255, 0), 2)
            cv2.putText(detections_overlay, f"{Z:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Blurred", img_blur)
        cv2.imshow("Canny Edges", img_edge)
        cv2.imshow("Detections", detections_overlay)
        cv2.waitKey(1)

        # Still publish disparity for downstream consumers, even though detection uses ROI edges.
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        disparity_msg = self.bridge.cv2_to_imgmsg(disp_colored, encoding="bgr8")
        self.pub_disparity.publish(disparity_msg)
        disparity_raw_msg = self.bridge.cv2_to_imgmsg(disparity.astype(np.float32), encoding="32FC1")
        self.pub_disparity_raw.publish(disparity_raw_msg)
        self.left_image = None
        self.right_image = None

def main(args=None):
    rclpy.init(args=args)
    node = StereoInferenceNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                print("Exiting...")
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


