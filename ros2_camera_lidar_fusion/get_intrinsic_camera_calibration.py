#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
from datetime import datetime
import os
import sys
import threading
import signal
import select
import tty
import termios
import re

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.chessboard_rows = config_file['chessboard']['pattern_size']['rows']
        self.chessboard_cols = config_file['chessboard']['pattern_size']['columns']
        square_size_raw = config_file['chessboard']['square_size_meters']
        self.square_size = self._extract_numeric(square_size_raw, "chessboard.square_size_meters", 0.03)

        self.image_topic = config_file['camera']['image_topic']
        self.image_width = config_file['camera']['image_size']['width']
        self.image_height = config_file['camera']['image_size']['height']

        self.output_path = config_file['general']['config_folder']
        self.file = config_file['general']['camera_intrinsic_calibration']
        general_cfg = config_file.get('general', {})
        # Limit the number of samples to keep calibration fast and stable.
        self.max_samples = int(general_cfg.get('max_intrinsic_samples', 50))

        # Shutdown guard/state tracking
        self.stopping = False
        self._calibration_thread = None
        self._keyboard_thread = None
        self._keyboard_stop = threading.Event()
        self.keyboard_listener_enabled = bool(general_cfg.get('keyboard_listener', True))

        # Try to create the OpenCV window up front so key events work reliably.
        self.gui_available = True
        try:
            cv2.namedWindow("Calibration Image", cv2.WINDOW_AUTOSIZE)
        except Exception as exc:
            self.gui_available = False
            self.get_logger().warn(f"Unable to create OpenCV window (GUI disabled): {exc}")

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.bridge = CvBridge()

        self.obj_points = []
        self.img_points = []

        self.objp = np.zeros((self.chessboard_rows * self.chessboard_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_cols, 0:self.chessboard_rows].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.get_logger().info("Camera calibration node initialized. Waiting for images...")
        self.get_logger().info(f"Will save calibration to {self.output_path}/{self.file}")

        if self.keyboard_listener_enabled:
            self._start_keyboard_listener()

    def image_callback(self, msg):
        try:
            if self.stopping:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_cols, self.chessboard_rows), None)

            if ret:
                self.obj_points.append(self.objp.copy())
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                self.img_points.append(refined_corners)

                cv2.drawChessboardCorners(cv_image, (self.chessboard_cols, self.chessboard_rows), refined_corners, ret)
                current_samples = len(self.img_points)
                self.get_logger().info(f"Chessboard detected and points added. Samples: {current_samples}")
            else:
                # Reduce noise during shutdown or missing context
                if not self.stopping:
                    self.get_logger().warn("Chessboard not detected in image.")

            if not self.stopping and self.gui_available:
                try:
                    cv2.imshow("Calibration Image", cv_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('c'), ord('C')):
                        self.get_logger().info("'c' key pressed. Running calibration and shutting down...")
                        self.request_shutdown_and_calibrate("key_c")
                        return
                except Exception:
                    # Ignore GUI errors when display is not available
                    pass

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def save_calibration(self):
        # Resample to at most max_samples from the saved points (random without replacement).
        total = len(self.obj_points)
        if total > self.max_samples:
            try:
                idx = np.random.choice(total, size=self.max_samples, replace=False)
                # Keep list order consistent with chosen indices
                idx_sorted = sorted(int(i) for i in idx)
                self.obj_points = [self.obj_points[i] for i in idx_sorted]
                self.img_points = [self.img_points[i] for i in idx_sorted]
                self.get_logger().info(
                    f"Subsampled {total} -> {len(self.obj_points)} samples for calibration."
                )
            except Exception as e:
                self.get_logger().warn(f"Resampling failed ({e}); proceeding with first {self.max_samples} samples.")
                self.obj_points = self.obj_points[: self.max_samples]
                self.img_points = self.img_points[: self.max_samples]

        if len(self.obj_points) < 10:
            self.get_logger().error(
                f"Not enough images for calibration. Collected {len(self.obj_points)} / 10 minimum."
            )
            return

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, (self.image_width, self.image_height), None, None
        )

        calibration_data = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'camera_matrix': {
                'rows': 3,
                'columns': 3,
                'data': camera_matrix.tolist()
            },
            'distortion_coefficients': {
                'rows': 1,
                'columns': len(dist_coeffs[0]),
                'data': dist_coeffs[0].tolist()
            },
            'chessboard': {
                'pattern_size': {
                    'rows': self.chessboard_rows,
                    'columns': self.chessboard_cols
                },
                'square_size_meters': self.square_size
            },
            'image_size': {
                'width': int(self.image_width),
                'height': int(self.image_height)
            },
            'rms_reprojection_error': ret
        }

        output_file = f"{self.output_path}/{self.file}"
        try:
            os.makedirs(self.output_path, exist_ok=True)
            with open(output_file, 'w') as file:
                yaml.dump(calibration_data, file)
            self.get_logger().info(f"Calibration saved to {output_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration: {e}")

    def stop_subscriptions(self):
        try:
            if hasattr(self, 'image_sub') and self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
        except Exception:
            pass

    def _extract_numeric(self, value, param_name: str, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass

        self.get_logger().warn(
            f"Invalid numeric value '{value}' for {param_name}; falling back to {default}."
        )
        return default

    def _close_windows(self):
        if not self.gui_available:
            return
        try:
            cv2.waitKey(1)
        except Exception:
            pass
        try:
            cv2.destroyWindow("Calibration Image")
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _start_keyboard_listener(self):
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            self.get_logger().warn("Keyboard listener enabled but stdin has no valid file descriptor; only GUI input will work.")
            return

        if not os.isatty(fd):
            self.get_logger().warn("Keyboard listener enabled but stdin is not a TTY; only GUI input will work.")
            return

        def worker():
            old_settings = None
            try:
                old_settings = termios.tcgetattr(fd)
            except termios.error as exc:
                self.get_logger().warn(f"Keyboard listener could not read terminal settings: {exc}")
                return

            try:
                tty.setcbreak(fd)
                while not self.stopping and not self._keyboard_stop.is_set():
                    try:
                        rlist, _, _ = select.select([fd], [], [], 0.1)
                    except Exception as exc:
                        self.get_logger().warn(f"Keyboard listener select failed: {exc}")
                        break

                    if not rlist:
                        continue

                    try:
                        ch = sys.stdin.read(1)
                    except Exception as exc:
                        self.get_logger().warn(f"Keyboard listener read failed: {exc}")
                        break

                    if not ch:
                        continue

                    if ch.lower() == 'c':
                        self.get_logger().info("Received 'c' from terminal. Running calibration and shutting down...")
                        self.request_shutdown_and_calibrate("stdin_c")
                        break
                    elif ch.lower() == 'q':
                        self.get_logger().info("Received 'q' from terminal. Aborting calibration and shutting down...")
                        self.request_shutdown_and_calibrate("stdin_q_abort", skip_calibration=True)
                        break
            finally:
                if old_settings is not None:
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    except Exception:
                        pass

        self._keyboard_thread = threading.Thread(
            target=worker,
            name="intrinsic_keyboard_listener",
            daemon=True,
        )
        self._keyboard_thread.start()

    def wait_for_calibration(self):
        if self._calibration_thread and self._calibration_thread.is_alive():
            self._calibration_thread.join()
            if not self._calibration_thread.is_alive():
                self._calibration_thread = None
        if (
            self._keyboard_thread
            and self._keyboard_thread.is_alive()
            and threading.current_thread() is not self._keyboard_thread
        ):
            self._keyboard_thread.join(timeout=1.0)
            if not self._keyboard_thread.is_alive():
                self._keyboard_thread = None

    def request_shutdown_and_calibrate(self, reason: str = "shutdown", skip_calibration: bool = False):
        if self.stopping:
            return

        self.stopping = True
        try:
            self.get_logger().info(
                f"Shutdown requested ({reason}); skip_calibration={skip_calibration}"
            )
        except Exception:
            pass
        self._keyboard_stop.set()
        self.stop_subscriptions()
        self._close_windows()

        def worker():
            try:
                if skip_calibration:
                    try:
                        self.get_logger().info("Calibration skipped; shutting down per request.")
                    except Exception:
                        pass
                else:
                    self.save_calibration()
            finally:
                try:
                    self.get_logger().info("Calibration process completed.")
                except Exception:
                    print("Calibration process completed.", file=sys.stderr)
                try:
                    rclpy.shutdown()
                except Exception:
                    pass

        self._calibration_thread = threading.Thread(
            target=worker,
            name="intrinsic_calibration_worker",
            daemon=False,
        )
        self._calibration_thread.start()

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    
    def handle_sigterm(signum, frame):
        node.request_shutdown_and_calibrate("sigterm")

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown_and_calibrate("keyboard_interrupt")
        node.wait_for_calibration()
    except ExternalShutdownException:
        node.wait_for_calibration()
    finally:
        node.wait_for_calibration()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
