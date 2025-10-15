#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import struct

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ros2_camera_lidar_fusion.read_yaml import extract_configuration


def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'extrinsic_matrix' not in data:
        raise KeyError(f"YAML {yaml_path} has no 'extrinsic_matrix' key.")

    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Extrinsic matrix is not 4x4.")
    return T

def _reshape_camera_matrix(mat_any) -> np.ndarray:
    try:
        arr = np.array(mat_any, dtype=np.float64)
        if arr.size == 9:
            return arr.reshape(3, 3)
        if arr.shape == (3, 3):
            return arr
    except Exception:
        pass
    raise ValueError("Invalid camera_matrix format; expected 3x3 or flat 9 list.")


def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray, str):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")

    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)

    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = _reshape_camera_matrix(cam_mat_data)

    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))

    distortion_model = calib_data.get('distortion_model', 'plumb_bob')

    return camera_matrix, dist_coeffs, distortion_model


def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> np.ndarray:
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)

    field_names = [f.name for f in cloud_msg.fields]
    if not all(k in field_names for k in ('x','y','z')):
        return np.zeros((0,3), dtype=np.float32)

    x_field = next(f for f in cloud_msg.fields if f.name=='x')
    y_field = next(f for f in cloud_msg.fields if f.name=='y')
    z_field = next(f for f in cloud_msg.fields if f.name=='z')

    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('_', 'V{}'.format(cloud_msg.point_step - 12))
    ])

    raw_data = np.frombuffer(cloud_msg.data, dtype=dtype)
    points = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points[:,0] = raw_data['x']
    points[:,1] = raw_data['y']
    points[:,2] = raw_data['z']

    if skip_rate > 1:
        points = points[::skip_rate]

    return points

class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')
        
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = config_file['general']['camera_extrinsic_calibration']
        extrinsic_yaml = os.path.join(config_folder, extrinsic_yaml)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = config_file['general']['camera_intrinsic_calibration']
        camera_yaml = os.path.join(config_folder, camera_yaml)
        self.camera_matrix, self.dist_coeffs, self.distortion_model = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion model: {}".format(self.distortion_model))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")

        self.image_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=5,
            slop=0.07
        )
        self.ts.registerCallback(self.sync_callback)

        projected_topic = config_file['camera']['projected_topic']
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()

        self.skip_rate = 1

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
        n_points = xyz_lidar.shape[0]
        if n_points == 0:
            self.get_logger().warn("Empty cloud. Nothing to project.")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return

        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))

        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]

        mask_in_front = (xyz_cam[:, 2] > 0.0)
        xyz_cam_front = xyz_cam[mask_in_front]
        n_front = xyz_cam_front.shape[0]
        if n_front == 0:
            self.get_logger().info("No points in front of camera (z>0).")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return
        
        rvec = np.zeros((3,1), dtype=np.float64)
        tvec = np.zeros((3,1), dtype=np.float64)
        if str(self.distortion_model).lower() == 'equidistant':
            # Use OpenCV fisheye projection for equidistant model
            objp = xyz_cam_front.reshape(-1, 1, 3)
            image_points, _ = cv2.fisheye.projectPoints(
                objp,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
        else:
            image_points, _ = cv2.projectPoints(
                xyz_cam_front,
                rvec, tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
        image_points = image_points.reshape(-1, 2)

        h, w = cv_image.shape[:2]
        for (u, v) in image_points:
            u_int = int(u + 0.5)
            v_int = int(v + 0.5)
            if 0 <= u_int < w and 0 <= v_int < h:
                cv2.circle(cv_image, (u_int, v_int), 2, (0, 255, 0), -1)

        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
