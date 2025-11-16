#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2
from rclpy.node import Node
import rclpy

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class CameraLidarExtrinsicNode(Node):
    def __init__(self):
        super().__init__('camera_lidar_extrinsic_node')
        
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return

        # Use the configured data and config folders instead of hardcoded paths
        data_folder = config_file['general']['data_folder']
        config_folder = config_file['general']['config_folder']

        self.corr_file = os.path.join(data_folder, config_file['general']['correspondence_file'])
        self.camera_yaml = os.path.join(config_folder, config_file['general']['camera_intrinsic_calibration'])
        self.output_dir = config_folder
        self.file = config_file['general']['camera_extrinsic_calibration']

        self.get_logger().info('Starting extrinsic calibration...')
        self.solve_extrinsic_with_pnp()

    def _reshape_camera_matrix(self, mat_any) -> np.ndarray:
        arr = np.array(mat_any, dtype=np.float64)
        if arr.size == 9:
            return arr.reshape(3, 3)
        if arr.shape == (3, 3):
            return arr
        raise ValueError("Invalid camera_matrix format; expected 3x3 or flat 9 list.")

    def load_camera_calibration(self, yaml_path: str):
        """Loads camera calibration parameters from a YAML file."""
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        mat_data = config['camera_matrix']['data']
        camera_matrix = self._reshape_camera_matrix(mat_data)
        dist_data = config['distortion_coefficients']['data']
        dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
        distortion_model = config.get('distortion_model', 'plumb_bob')

        return camera_matrix, dist_coeffs, distortion_model

    def solve_extrinsic_with_pnp(self):
        """Solves for extrinsic parameters using 2D-3D correspondences and camera calibration."""
        camera_matrix, dist_coeffs, distortion_model = self.load_camera_calibration(self.camera_yaml)
        self.get_logger().info(f"Camera matrix:\n{camera_matrix}")
        self.get_logger().info(f"Distortion model: {distortion_model}")
        self.get_logger().info(f"Distortion coefficients: {dist_coeffs}")

        if not os.path.isfile(self.corr_file):
            raise FileNotFoundError(f"Correspondence file not found: {self.corr_file}")

        pts_2d = []
        pts_3d = []
        with open(self.corr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                splitted = line.split(',')
                if len(splitted) != 5:
                    continue
                u, v, X, Y, Z = [float(val) for val in splitted]
                pts_2d.append([u, v])
                pts_3d.append([X, Y, Z])

        pts_2d = np.array(pts_2d, dtype=np.float64)
        pts_3d = np.array(pts_3d, dtype=np.float64)

        num_points = len(pts_2d)
        self.get_logger().info(f"Loaded {num_points} correspondences from {self.corr_file}")

        if num_points < 4:
            raise ValueError("At least 4 correspondences are required for solvePnP")

        # Handle fisheye (equidistant) image points properly by undistorting first.
        if str(distortion_model).lower() == 'equidistant':
            # Undistort to rectified pixel coordinates using P=K, then solve with zero distortion
            pts_2d_cv = pts_2d.reshape(-1, 1, 2)
            undistorted = cv2.fisheye.undistortPoints(pts_2d_cv, camera_matrix, dist_coeffs, P=camera_matrix)
            undistorted = undistorted.reshape(-1, 2)
            success, rvec, tvec = cv2.solvePnP(
                pts_3d,
                undistorted,
                camera_matrix,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                pts_3d,
                pts_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            raise RuntimeError("solvePnP failed to find a solution.")

        self.get_logger().info("solvePnP succeeded.")
        self.get_logger().info(f"rvec: {rvec.ravel()}")
        self.get_logger().info(f"tvec: {tvec.ravel()}")

        R, _ = cv2.Rodrigues(rvec)

        T_lidar_to_cam = np.eye(4, dtype=np.float64)
        T_lidar_to_cam[0:3, 0:3] = R
        T_lidar_to_cam[0:3, 3] = tvec[:, 0]

        self.get_logger().info(f"Transformation matrix (LiDAR -> Camera):\n{T_lidar_to_cam}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        out_yaml = os.path.join(self.output_dir, self.file)
        data_out = {
            "extrinsic_matrix": T_lidar_to_cam.tolist()
        }

        with open(out_yaml, 'w') as f:
            yaml.dump(data_out, f, sort_keys=False)

        self.get_logger().info(f"Extrinsic matrix saved to: {out_yaml}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarExtrinsicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
