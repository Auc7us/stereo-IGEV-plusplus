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


# Set CUDA device
DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn.cluster import DBSCAN
import argparse

def detect_obstacles(disparity):

    disparity = cv2.bilateralFilter(disparity, 7, 50, 50)
    edge = cv2.Canny(disparity, 15, 40) 
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0 and w > 0 and w * h > 50 and w/h < 10:
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

class StereoInferenceNode(Node):
    def __init__(self):
        super().__init__('stereo_inference_node')

        # ROS 2 subscribers
        self.sub_left = self.create_subscription(Image, '/chrono_ros_node/stereo/left', self.left_callback, 10)
        self.sub_right = self.create_subscription(Image, '/chrono_ros_node/stereo/right', self.right_callback, 10)

        # ROS 2 publisher for the disparity map
        self.pub_disparity = self.create_publisher(Image, '/chrono_ros_node/stereo/disparity', 10)

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Store images until both left & right arrive
        self.left_image = None
        self.right_image = None

        default_args = argparse.Namespace(
            restore_ckpt='./pretrained_models/middlebury.pth',  # Add this line
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

        # Load the IGEV Stereo model
        self.model = torch.nn.DataParallel(IGEVStereo(default_args))
        self.model.load_state_dict(torch.load(default_args.restore_ckpt))
        self.model = self.model.module.to(DEVICE).eval()

        self.get_logger().info("Stereo Inference Node Initialized")

    def left_callback(self, msg):
        """ Callback for left stereo image """
        self.left_image = cv2.flip(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), 0)
        # cv2.imshow("Left Image", self.left_image)
        # cv2.waitKey(1) 
        self.process_images()

    def right_callback(self, msg):
        """ Callback for right stereo image """
        self.right_image = cv2.flip(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), 0) 
        # cv2.imshow("Right Image", self.right_image)
        # cv2.waitKey(1)
        self.process_images()

    def process_images(self):
        """ Runs inference on the stereo images and publishes the disparity map """
        if self.left_image is None or self.right_image is None:
            return

        self.get_logger().info("Processing stereo images...")

        # Convert to PyTorch tensors
        image1 = torch.from_numpy(self.left_image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        image2 = torch.from_numpy(self.right_image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        # Ensure dimensions compatibility
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            disparity = self.model(image1, image2, iters=16, test_mode=True)

        disparity = padder.unpad(disparity)
        disparity = disparity.cpu().numpy().squeeze()

        # Normalize disparity for visualization
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_vis3c = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
        disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        combined = np.concatenate([disp_vis3c, disp_colored], axis=1)

        bounding_boxes = detect_obstacles(disp_vis)

        for x, y, w, h in bounding_boxes:
            cv2.rectangle(combined, (x-5, y-5), (x + w+5, y + h+5), (0, 255, 0), 2)
        
        cv2.imshow("Disparity Map", combined)
        cv2.waitKey(1)

        # Publish disparity as a ROS 2 topic
        disparity_msg = self.bridge.cv2_to_imgmsg(disp_colored, encoding="bgr8")
        self.pub_disparity.publish(disparity_msg)

        # Reset images after processing
        self.left_image = None
        self.right_image = None

def main(args=None):
    rclpy.init(args=args)
    node = StereoInferenceNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)  # Process ROS messages
            key = cv2.waitKey(1) & 0xFF  # Wait for key press
            
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


