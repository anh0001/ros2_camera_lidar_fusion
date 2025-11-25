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

        slop = float(config_file['general'].get('slop', 0.1))
        self.get_logger().info(f"Using time synchronization slop: {slop} seconds")

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=5,
            slop=slop
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
        xyz_lidar_front = xyz_lidar_f64[mask_in_front]
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

        distances = np.linalg.norm(xyz_cam_front, axis=1)
        min_dist = float(np.min(distances))
        max_dist = float(np.max(distances))
        distance_span = max_dist - min_dist
        if distance_span < 1e-6:
            normalized = np.zeros_like(distances, dtype=np.float64)
        else:
            normalized = (distances - min_dist) / distance_span

        colormap_input = (normalized * 255.0).astype(np.uint8).reshape(-1, 1)
        colors_bgr = cv2.applyColorMap(colormap_input, cv2.COLORMAP_TURBO).reshape(-1, 3)

        roi_width = max(int(0.14 * w), 60)
        roi_height = max(int(0.12 * h), 40)
        center_x = w / 2.0
        center_y = min(h - roi_height // 2 - 1, int(h * 0.6))
        roi_x1 = max(0, int(round(center_x - roi_width / 2.0)))
        roi_y1 = max(0, center_y - roi_height // 2)
        roi_x2 = min(w, roi_x1 + roi_width)
        roi_y2 = min(h, roi_y1 + roi_height)

        roi_distances = []
        roi_points_lidar = []

        for (u, v), color, distance, lidar_point in zip(image_points, colors_bgr, distances, xyz_lidar_front):
            u_int = int(u + 0.5)
            v_int = int(v + 0.5)
            if 0 <= u_int < w and 0 <= v_int < h:
                color_tuple = tuple(int(c) for c in color.tolist())
                cv2.circle(cv_image, (u_int, v_int), 2, color_tuple, -1)
                if roi_x1 <= u_int < roi_x2 and roi_y1 <= v_int < roi_y2:
                    roi_distances.append(distance)
                    roi_points_lidar.append(lidar_point)
        roi_width_px = roi_x2 - roi_x1
        roi_height_px = roi_y2 - roi_y1
        if roi_width_px > 1 and roi_height_px > 1:
            cv2.rectangle(cv_image, (roi_x1, roi_y1), (roi_x2 - 1, roi_y2 - 1), (0, 255, 0), 2)

        if roi_distances:
            roi_array = np.asarray(roi_distances, dtype=np.float64)
            precision_step = 0.02  # 2 cm buckets for higher resolution
            rounded = np.round(roi_array / precision_step) * precision_step
            unique_vals, counts = np.unique(rounded, return_counts=True)
            dominant_idx = int(np.argmax(counts))
            dominant_distance = float(unique_vals[dominant_idx])
            distance_text = f"{dominant_distance:.2f} m"
            text_color = (0, 255, 0)
        else:
            distance_text = "N/A"
            text_color = (0, 255, 255)

        if roi_points_lidar:
            roi_points_array = np.asarray(roi_points_lidar, dtype=np.float64)
            mode_resolution = 0.05  # bucketize to 5 cm to find a stable mode
            mode_buckets = np.round(roi_points_array / mode_resolution) * mode_resolution
            unique_points, point_counts = np.unique(mode_buckets, axis=0, return_counts=True)
            dominant_idx = int(np.argmax(point_counts))
            dominant_xyz = unique_points[dominant_idx]
            xyz_text = f"x:{dominant_xyz[0]:.2f} y:{dominant_xyz[1]:.2f} z:{dominant_xyz[2]:.2f} m"
            xyz_text_color = text_color
        else:
            xyz_text = "x:N/A y:N/A z:N/A"
            xyz_text_color = (0, 255, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (distance_width, distance_height), _ = cv2.getTextSize(distance_text, font, font_scale, thickness)
        (xyz_width, xyz_height), _ = cv2.getTextSize(xyz_text, font, font_scale, thickness)

        distance_x = max(0, int(round((roi_x1 + roi_x2) * 0.5 - distance_width * 0.5)))
        distance_y = min(h - 6, roi_y2 - 8)
        distance_origin = (distance_x, max(roi_y1 + distance_height + 4, distance_y))

        xyz_x = max(0, int(round((roi_x1 + roi_x2) * 0.5 - xyz_width * 0.5)))
        xyz_target_y = distance_origin[1] - distance_height - 6
        xyz_y = max(roi_y1 + xyz_height + 4, xyz_target_y)
        xyz_origin = (xyz_x, xyz_y)

        cv2.putText(cv_image, xyz_text, xyz_origin, font, font_scale, xyz_text_color, thickness, cv2.LINE_AA)
        cv2.putText(cv_image, distance_text, distance_origin, font, font_scale, text_color, thickness, cv2.LINE_AA)

        legend_margin = 10
        available_width = w - 2 * legend_margin
        available_height = h - 2 * legend_margin
        legend_width = min(22, available_width) if available_width > 0 else 0
        legend_height = min(max(int(0.3 * h), 80), available_height) if available_height > 0 else 0
        if legend_height > 0 and legend_width > 0:
            legend_x = w - legend_margin - legend_width
            legend_y = legend_margin
            legend_gradient = np.linspace(255, 0, legend_height, dtype=np.uint8).reshape(legend_height, 1)
            legend_colors = cv2.applyColorMap(legend_gradient, cv2.COLORMAP_TURBO)
            legend_block = np.repeat(legend_colors, legend_width, axis=1)

            roi = cv_image[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width]
            if roi.shape[:2] == legend_block.shape[:2]:
                cv_image[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = cv2.addWeighted(
                    legend_block, 0.85, roi, 0.15, 0.0
                )

            cv2.rectangle(
                cv_image,
                (legend_x - 1, legend_y - 1),
                (legend_x + legend_width + 1, legend_y + legend_height + 1),
                (30, 30, 30),
                1
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_color = (255, 255, 255)

            max_text = f"{max_dist:.1f} m"
            min_text = f"{min_dist:.1f} m"
            mid_dist = min_dist + (distance_span * 0.5 if distance_span > 1e-6 else 0.0)
            mid_text = f"{mid_dist:.1f} m"

            def _put_text(text: str, y_pos: int):
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = max(2, legend_x - 12 - text_width)
                text_y = y_pos
                cv2.putText(cv_image, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            _put_text(max_text, legend_y + 12)
            _put_text(min_text, legend_y + legend_height - 2)
            _put_text(mid_text, legend_y + legend_height // 2 + 4)

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
