import cv2
import numpy as np
import argparse
import csv
from pathlib import Path
import sys
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time

class RobotTracker:
    def __init__(self, video_path, output_path=None, use_convex_hull=True, show_path=True, 
                 mat_width_mm=1100, mat_height_mm=1700, num_threads=4, frame_skip=1):
        self.video_path = video_path
        self.output_path = output_path
        self.use_convex_hull = use_convex_hull
        self.show_path = show_path
        self.num_threads = num_threads
        self.frame_skip = frame_skip  # Process every Nth frame
        
        # Real-world mat dimensions in millimeters
        self.mat_width_mm = mat_width_mm
        self.mat_height_mm = mat_height_mm
        
        # Calibration variables for pixel-to-millimeter conversion
        self.pixels_per_mm_x = None
        self.pixels_per_mm_y = None
        self.mat_origin_pixels = None
        self.calibrated = False
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        opencv_version = cv2.__version__.split('.')
        if int(opencv_version[0]) >= 4 and int(opencv_version[1]) >= 7:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
        
        self.tracking_data = []
        self.mat_corners = {}
        self.last_known_position = None
        self.smoothed_direction = None
        self.direction_history = []
        
        # Threading
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        
        self.cap = None
        self.out = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
    
    def calculate_speed_data(self):
        """Calculate speed at each timestamp"""
        if len(self.tracking_data) < 2:
            return None
        
        speed_data = []
        
        for i in range(1, len(self.tracking_data)):
            curr = self.tracking_data[i]
            prev = self.tracking_data[i-1]
            
            # Calculate distance
            dx = curr['position'][0] - prev['position'][0]
            dy = curr['position'][1] - prev['position'][1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate time difference
            time_diff = curr['timestamp'] - prev['timestamp']
            
            # Calculate speed
            if time_diff > 0:
                speed = distance / time_diff
            else:
                speed = 0
            
            speed_data.append({
                'frame': curr['frame'],
                'timestamp': curr['timestamp'],
                'speed': speed,
                'distance': distance,
                'unit': self.get_speed_unit()
            })
        
        return speed_data
    
    def get_speed_unit(self):
        """Get appropriate speed unit"""
        if self.calibrated:
            return 'mm/s'
        else:
            return 'px/s'
    
    def save_speed_data(self, csv_path):
        """Save speed data to separate CSV file"""
        speed_data = self.calculate_speed_data()
        
        if not speed_data:
            print("No speed data to save")
            return
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'timestamp_sec', f'speed_{self.get_speed_unit()}', f'distance_{self.get_speed_unit()}'])
            
            for data in speed_data:
                writer.writerow([
                    data['frame'],
                    f"{data['timestamp']:.4f}",
                    f"{data['speed']:.3f}",
                    f"{data['distance']:.3f}"
                ])
        
        print(f"Speed data saved to: {csv_path}")
    
    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Set buffer size for faster frame reading
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {self.width}x{self.height} @ {self.fps:.2f} fps, {self.frame_count} frames")
        print(f"Mat dimensions: {self.mat_width_mm}mm x {self.mat_height_mm}mm")
        print(f"Multi-threading: {self.num_threads} threads, Frame skip: {self.frame_skip}")
        
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, 
                                      (self.width, self.height))
            if not self.out.isOpened():
                print(f"Warning: Could not open output video file: {self.output_path}")
                self.out = None
    
    def detect_markers(self, frame):
        """Detect ArUco markers - optimized"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)
        
        return corners, ids
    
    def calculate_center(self, corners):
        if len(corners) == 0:
            return None
        corner_points = np.array(corners, dtype=np.float32).reshape(-1, 2)
        center_x = np.mean(corner_points[:, 0])
        center_y = np.mean(corner_points[:, 1])
        return (float(center_x), float(center_y))
    
    def calculate_orientation(self, corners):
        if len(corners) == 0:
            return 0.0
        corner_points = np.array(corners, dtype=np.float32).reshape(-1, 2)
        
        center_x = np.mean(corner_points[:, 0])
        center_y = np.mean(corner_points[:, 1])
        
        dx = float(corner_points[0][0]) - float(center_x)
        dy = float(corner_points[0][1]) - float(center_y)
        angle = np.arctan2(dy, dx)
        return float(np.degrees(angle))
    
    def calibrate_mat(self, corners, ids):
        """Calibrate the coordinate system based on mat corner markers (IDs 0-3)"""
        if ids is None or len(ids) == 0:
            return
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id <= 3:
                center = self.calculate_center(corners[i])
                if center:
                    if marker_id not in self.mat_corners:
                        self.mat_corners[marker_id] = center
        
        if len(self.mat_corners) >= 4 and not self.calibrated:
            corners_list = [self.mat_corners[i] for i in range(4) if i in self.mat_corners]
            
            if len(corners_list) == 4:
                x_coords = [c[0] for c in corners_list]
                y_coords = [c[1] for c in corners_list]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                mat_width_pixels = max_x - min_x
                mat_height_pixels = max_y - min_y
                
                self.pixels_per_mm_x = mat_width_pixels / self.mat_width_mm
                self.pixels_per_mm_y = mat_height_pixels / self.mat_height_mm
                self.mat_origin_pixels = (min_x, min_y)
                
                self.calibrated = True
                
                print(f"\n{'='*50}")
                print(f"CALIBRATION COMPLETE")
                print(f"{'='*50}")
                print(f"Mat detected at pixels: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
                print(f"Mat size in pixels: {mat_width_pixels:.1f} x {mat_height_pixels:.1f}")
                print(f"Pixels per mm: X={self.pixels_per_mm_x:.3f}, Y={self.pixels_per_mm_y:.3f}")
                print(f"{'='*50}\n")
    
    def pixels_to_mm(self, pixel_pos):
        """Convert pixel coordinates to real-world millimeters"""
        if not self.calibrated:
            return pixel_pos
        
        px, py = pixel_pos
        rel_x_pixels = px - self.mat_origin_pixels[0]
        rel_y_pixels = py - self.mat_origin_pixels[1]
        
        x_mm = rel_x_pixels / self.pixels_per_mm_x
        y_mm = rel_y_pixels / self.pixels_per_mm_y
        
        return (float(x_mm), float(y_mm))
    
    def mm_to_pixels(self, mm_pos):
        """Convert real-world millimeters to pixel coordinates"""
        if not self.calibrated:
            return mm_pos
        
        mx, my = mm_pos
        px = mx * self.pixels_per_mm_x + self.mat_origin_pixels[0]
        py = my * self.pixels_per_mm_y + self.mat_origin_pixels[1]
        
        return (float(px), float(py))
    
    def get_robot_position(self, corners, ids, robot_marker_ids=None, center_marker_id=None):
        if ids is None or len(ids) == 0:
            return None
        
        robot_markers = []
        detected_ids = ids.flatten()
        
        if center_marker_id is not None:
            for i, marker_id in enumerate(detected_ids):
                if marker_id == center_marker_id:
                    try:
                        center = self.calculate_center(corners[i])
                        orientation = self.calculate_orientation(corners[i])
                        
                        if center:
                            center_mm = self.pixels_to_mm(center)
                            
                            return {
                                'position_pixels': center,
                                'position': center_mm,
                                'orientation': float(orientation),
                                'markers': [{'id': int(marker_id), 'center': center, 'orientation': orientation}],
                                'num_markers': 1
                            }
                    except Exception as e:
                        pass
            return None
        
        for i, marker_id in enumerate(detected_ids):
            if robot_marker_ids is None:
                is_robot_marker = marker_id > 3
            else:
                is_robot_marker = marker_id in robot_marker_ids
            
            if is_robot_marker:
                try:
                    center = self.calculate_center(corners[i])
                    orientation = self.calculate_orientation(corners[i])
                    
                    if center:
                        robot_markers.append({
                            'id': int(marker_id),
                            'center': center,
                            'orientation': orientation,
                            'corners': corners[i]
                        })
                except Exception as e:
                    continue
        
        if robot_markers:
            all_corner_points = []
            for marker in robot_markers:
                corner_points = np.array(marker['corners'], dtype=np.float32).reshape(-1, 2)
                all_corner_points.extend(corner_points.tolist())
            
            all_corner_points = np.array(all_corner_points, dtype=np.float32)
            
            if self.use_convex_hull and len(robot_markers) >= 3:
                hull = cv2.convexHull(all_corner_points)
                M = cv2.moments(hull)
                if M['m00'] != 0:
                    avg_x = M['m10'] / M['m00']
                    avg_y = M['m01'] / M['m00']
                else:
                    avg_x = np.mean(all_corner_points[:, 0])
                    avg_y = np.mean(all_corner_points[:, 1])
            else:
                avg_x = np.mean(all_corner_points[:, 0])
                avg_y = np.mean(all_corner_points[:, 1])
            
            avg_orientation = np.mean([m['orientation'] for m in robot_markers])
            
            position_pixels = (float(avg_x), float(avg_y))
            position_mm = self.pixels_to_mm(position_pixels)
            
            position_data = {
                'position_pixels': position_pixels,
                'position': position_mm,
                'orientation': float(avg_orientation),
                'markers': robot_markers,
                'num_markers': len(robot_markers)
            }
            
            self.last_known_position = position_data
            return position_data
        
        return None
    
    def draw_tracking_info(self, frame, corners, ids, robot_data):
        display_frame = frame.copy()
        
        if self.show_path and len(self.tracking_data) > 1:
            points = [d['position_pixels'] for d in self.tracking_data]
            
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                
                ratio = i / max(len(points) - 1, 1)
                color = (int(255 * (1 - ratio)), int(255 * ratio), int(200 * ratio))
                thickness = max(2, int(4 * (ratio + 0.5)))
                
                cv2.line(display_frame, pt1, pt2, color, thickness)
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                center = self.calculate_center(corners[i])
                if center:
                    cv2.putText(display_frame, f"ID:{marker_id}", 
                              (int(center[0]) - 20, int(center[1]) - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if robot_data:
            pos_pixels = robot_data['position_pixels']
            pos_mm = robot_data['position']
            
            cv2.circle(display_frame, (int(pos_pixels[0]), int(pos_pixels[1])), 12, (0, 0, 255), -1)
            cv2.circle(display_frame, (int(pos_pixels[0]), int(pos_pixels[1])), 15, (255, 255, 255), 2)
            
            if len(self.tracking_data) >= 2:
                window_size = min(30, len(self.tracking_data))
                recent_positions = [d['position_pixels'] for d in self.tracking_data[-window_size:]]
                
                if len(recent_positions) >= 2:
                    curr_pos = recent_positions[-1]
                    start_pos = recent_positions[0]
                    
                    dx = curr_pos[0] - start_pos[0]
                    dy = curr_pos[1] - start_pos[1]
                    
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance > 10:
                        movement_angle = np.degrees(np.arctan2(dy, dx))
                        
                        self.direction_history.append(movement_angle)
                        if len(self.direction_history) > 15:
                            self.direction_history.pop(0)
                        
                        angles_rad = [np.radians(a) for a in self.direction_history]
                        avg_x = np.mean([np.cos(a) for a in angles_rad])
                        avg_y = np.mean([np.sin(a) for a in angles_rad])
                        smoothed_angle = np.degrees(np.arctan2(avg_y, avg_x))
                        
                        alpha = 0.3
                        if self.smoothed_direction is None:
                            self.smoothed_direction = smoothed_angle
                        else:
                            angle_diff = smoothed_angle - self.smoothed_direction
                            if angle_diff > 180:
                                angle_diff -= 360
                            elif angle_diff < -180:
                                angle_diff += 360
                            self.smoothed_direction += alpha * angle_diff
                        
                        arrow_length = 60
                        end_x = int(pos_pixels[0] + arrow_length * np.cos(np.radians(self.smoothed_direction)))
                        end_y = int(pos_pixels[1] + arrow_length * np.sin(np.radians(self.smoothed_direction)))
                        cv2.arrowedLine(display_frame, (int(pos_pixels[0]), int(pos_pixels[1])), (end_x, end_y),
                                      (255, 0, 0), 4, tipLength=0.3)
                        
                        if self.calibrated:
                            recent_mm_positions = [d['position'] for d in self.tracking_data[-window_size:]]
                            curr_mm = recent_mm_positions[-1]
                            start_mm = recent_mm_positions[0]
                            
                            dx_mm = curr_mm[0] - start_mm[0]
                            dy_mm = curr_mm[1] - start_mm[1]
                            distance_mm = np.sqrt(dx_mm**2 + dy_mm**2)
                            
                            time_window = (window_size - 1) / self.fps if self.fps > 0 else (window_size - 1)
                            speed_mms = distance_mm / time_window if time_window > 0 else 0
                            
                            direction_text = f"Direction: {self.smoothed_direction:.1f}° Speed: {speed_mms:.1f}mm/s"
                        else:
                            speed = distance / max(window_size - 1, 1)
                            direction_text = f"Direction: {self.smoothed_direction:.1f}° Speed: {speed:.1f}px/f"
                    else:
                        direction_text = "Direction: Stationary"
                else:
                    direction_text = "Direction: Calculating..."
            else:
                direction_text = "Direction: Initializing..."
            
            panel_height = 150
            cv2.rectangle(display_frame, (0, 0), (600, panel_height), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (0, 0), (600, panel_height), (255, 255, 255), 2)
            
            if self.calibrated:
                cv2.putText(display_frame, f"Position: ({pos_mm[0]:.2f}mm, {pos_mm[1]:.2f}mm)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"(Pixels: {pos_pixels[0]:.1f}, {pos_pixels[1]:.1f})",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(display_frame, f"Position: ({pos_pixels[0]:.1f}px, {pos_pixels[1]:.1f}px)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, "Calibrating...",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(display_frame, direction_text,
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Markers detected: {robot_data['num_markers']}",
                       (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.calibrated:
            cv2.putText(display_frame, "CALIBRATED",
                       (self.width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"Calibrating ({len(self.mat_corners)}/4)",
                       (self.width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 0), 2)
        
        cv2.putText(display_frame, f"Frame: {len(self.tracking_data)}",
                   (self.width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (255, 255, 255), 2)
        
        return display_frame
    
    def process_frame(self, frame_data):
        """Process a single frame - called by thread pool"""
        frame_num, frame = frame_data
        corners, ids = self.detect_markers(frame)
        
        if frame_num < 60:
            self.calibrate_mat(corners, ids)
        
        robot_data = self.get_robot_position(corners, ids, None, None)
        
        return {
            'frame_num': frame_num,
            'frame': frame,
            'corners': corners,
            'ids': ids,
            'robot_data': robot_data
        }
    
    def process_video(self, robot_marker_ids=None, center_marker_id=None, show_preview=True):
        self.initialize_video()
        
        print(f"Processing video: {self.video_path}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Path visualization: {'enabled' if self.show_path else 'disabled'}")
        
        if center_marker_id:
            print(f"Tracking center marker: {center_marker_id}")
        elif robot_marker_ids:
            print(f"Tracking robot markers: {robot_marker_ids}")
        else:
            print("Auto-detecting robot markers")
        
        frame_num = 0
        frames_without_detection = 0
        last_print_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Frame skipping for faster processing
                if frame_num % self.frame_skip != 0:
                    frame_num += 1
                    continue
                
                corners, ids = self.detect_markers(frame)
                
                if frame_num == 0 and ids is not None:
                    print(f"First frame - Detected marker IDs: {ids.flatten().tolist()}")
                
                if frame_num < 60:
                    self.calibrate_mat(corners, ids)
                
                robot_data = self.get_robot_position(corners, ids, robot_marker_ids, center_marker_id)
                
                if robot_data:
                    self.tracking_data.append({
                        'frame': frame_num,
                        'timestamp': frame_num / self.fps if self.fps > 0 else frame_num,
                        'position': robot_data['position'],
                        'position_pixels': robot_data['position_pixels'],
                        'orientation': robot_data['orientation'],
                        'num_markers': robot_data['num_markers']
                    })
                    frames_without_detection = 0
                else:
                    frames_without_detection += 1
                
                display_frame = self.draw_tracking_info(frame, corners, ids, robot_data)
                
                if self.out:
                    self.out.write(display_frame)
                
                if show_preview:
                    display_height = 720
                    if display_frame.shape[0] > display_height:
                        scale = display_height / display_frame.shape[0]
                        display_width = int(display_frame.shape[1] * scale)
                        display_frame = cv2.resize(display_frame, (display_width, display_height))
                    
                    cv2.imshow('Robot Tracking', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nUser stopped processing")
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                frame_num += 1
                
                # Print progress less frequently for speed
                current_time = time.time()
                if current_time - last_print_time > 2:
                    progress = (frame_num / self.frame_count * 100) if self.frame_count > 0 else 0
                    print(f"Processed {frame_num}/{self.frame_count} frames ({progress:.1f}%) - "
                          f"Tracked: {len(self.tracking_data)} positions")
                    last_print_time = current_time
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
            print(f"\n{'='*50}")
            print(f"Processing complete!")
            print(f"Total frames processed: {frame_num}")
            print(f"Tracked positions: {len(self.tracking_data)}")
            print(f"Detection rate: {len(self.tracking_data)/frame_num*100:.1f}%" if frame_num > 0 else "N/A")
            if self.calibrated:
                print(f"Coordinate system: MILLIMETERS (calibrated)")
            else:
                print(f"Coordinate system: PIXELS (not calibrated)")
            print(f"{'='*50}")
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()
    
    def save_tracking_data(self, csv_path):
        """Save tracking data to CSV file"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if self.calibrated:
                writer.writerow([
                    'frame',
                    'timestamp',
                    'position_x_mm',
                    'position_y_mm',
                    'position_x_px',
                    'position_y_px',
                    'orientation',
                    'num_markers'
                ])
                
                for data in self.tracking_data:
                    writer.writerow([
                        data['frame'],
                        f"{data['timestamp']:.4f}",
                        f"{data['position'][0]:.3f}",
                        f"{data['position'][1]:.3f}",
                        f"{data['position_pixels'][0]:.2f}",
                        f"{data['position_pixels'][1]:.2f}",
                        f"{data['orientation']:.2f}",
                        data['num_markers']
                    ])
            else:
                writer.writerow([
                    'frame',
                    'timestamp',
                    'position_x_px',
                    'position_y_px',
                    'orientation',
                    'num_markers'
                ])
                
                for data in self.tracking_data:
                    writer.writerow([
                        data['frame'],
                        f"{data['timestamp']:.4f}",
                        f"{data['position_pixels'][0]:.2f}",
                        f"{data['position_pixels'][1]:.2f}",
                        f"{data['orientation']:.2f}",
                        data['num_markers']
                    ])
        
        print(f"Tracking data saved to: {csv_path}")
    
    def calculate_statistics(self):
        if len(self.tracking_data) < 2:
            print("Not enough tracking data to calculate statistics")
            return None
        
        if self.calibrated:
            positions = np.array([d['position'] for d in self.tracking_data])
            unit = 'mm'
            speed_unit = 'mm/s'
        else:
            positions = np.array([d['position_pixels'] for d in self.tracking_data])
            unit = 'px'
            speed_unit = 'px/s'
        
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        
        timestamps = np.array([d['timestamp'] for d in self.tracking_data])
        time_intervals = np.diff(timestamps)
        time_intervals = time_intervals[time_intervals > 0]
        
        if len(time_intervals) > 0:
            speeds = distances[:len(time_intervals)] / time_intervals
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
        else:
            avg_speed = 0
            max_speed = 0
        
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        stats = {
            'total_distance': float(total_distance),
            'avg_speed': float(avg_speed),
            'max_speed': float(max_speed),
            'duration': float(timestamps[-1] - timestamps[0]),
            'num_positions': len(self.tracking_data),
            'bounding_box': {
                'min_x': float(min_x),
                'min_y': float(min_y),
                'max_x': float(max_x),
                'max_y': float(max_y),
                'width': float(max_x - min_x),
                'height': float(max_y - min_y)
            },
            'unit': unit,
            'speed_unit': speed_unit,
            'calibrated': self.calibrated
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Track robot movement using ArUco markers with real-world coordinates in millimeters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interp.py input.MP4 --center-marker 5
  python interp.py input.MP4 --output tracked.mp4 --csv data.csv --center-marker 5
  python interp.py input.MP4 --robot-ids 4 5 24 47
  python interp.py input.MP4 --mat-size 1100 1700
  python interp.py input.MP4 --center-marker 5 --no-path
  python interp.py input.MP4 --no-preview
  python interp.py input.MP4 --center-marker 5 --speed speed.csv
  python interp.py input.MP4 --center-marker 5 --threads 8 --frame-skip 2

Note: Place ArUco markers with IDs 0-3 at the four corners of your mat for calibration.
        """
    )
    
    parser.add_argument('input_video', help='Path to input MP4 video')
    parser.add_argument('--output', '-o', help='Path to output video file')
    parser.add_argument('--csv', '-c', help='Path to save tracking data CSV')
    parser.add_argument('--speed', '-s', help='Path to save speed data CSV')
    parser.add_argument('--center-marker', type=int,
                       help='Track only this center marker ID (recommended for cross pattern)')
    parser.add_argument('--robot-ids', nargs='+', type=int,
                       help='Marker IDs on the robot (leave empty for auto-detect)')
    parser.add_argument('--mat-size', nargs=2, type=float, default=[1100, 1700],
                       metavar=('WIDTH_MM', 'HEIGHT_MM'),
                       help='Mat dimensions in millimeters (default: 1100 1700)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window (faster processing)')
    parser.add_argument('--no-hull', action='store_true',
                       help='Use simple average instead of convex hull')
    parser.add_argument('--no-path', action='store_true',
                       help='Disable path visualization trail')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of processing threads (default: 4)')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame (default: 1, use 2 for 2x speed)')
    
    args = parser.parse_args()
    
    if not Path(args.input_video).exists():
        print(f"Error: Input video file not found: {args.input_video}")
        sys.exit(1)
    
    tracker = RobotTracker(args.input_video, args.output, 
                          use_convex_hull=not args.no_hull,
                          show_path=not args.no_path,
                          mat_width_mm=args.mat_size[0],
                          mat_height_mm=args.mat_size[1],
                          num_threads=args.threads,
                          frame_skip=args.frame_skip)
    
    tracker.process_video(
        robot_marker_ids=args.robot_ids,
        center_marker_id=args.center_marker,
        show_preview=not args.no_preview
    )
    
    stats = tracker.calculate_statistics()
    if stats:
        print("\n" + "="*50)
        print("MOVEMENT STATISTICS")
        print("="*50)
        print(f"Total distance traveled: {stats['total_distance']:.2f} {stats['unit']}")
        print(f"Average speed: {stats['avg_speed']:.2f} {stats['speed_unit']}")
        print(f"Max speed: {stats['max_speed']:.2f} {stats['speed_unit']}")
        print(f"Duration: {stats['duration']:.2f} seconds")
        print(f"Tracked positions: {stats['num_positions']}")
        print(f"\nMovement area:")
        print(f"  Width: {stats['bounding_box']['width']:.2f} {stats['unit']}")
        print(f"  Height: {stats['bounding_box']['height']:.2f} {stats['unit']}")
        if stats['calibrated']:
            print(f"\nCoordinate system: REAL-WORLD (millimeters)")
        else:
            print(f"\nCoordinate system: PIXELS (calibration failed)")
        print("="*50)
    
    if args.csv:
        tracker.save_tracking_data(args.csv)
        
        if stats:
            stats_path = Path(args.csv).with_suffix('.stats.csv')
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value', 'unit'])
                writer.writerow(['total_distance', f"{stats['total_distance']:.3f}", stats['unit']])
                writer.writerow(['avg_speed', f"{stats['avg_speed']:.3f}", stats['speed_unit']])
                writer.writerow(['max_speed', f"{stats['max_speed']:.3f}", stats['speed_unit']])
                writer.writerow(['duration', f"{stats['duration']:.2f}", 'seconds'])
                writer.writerow(['num_positions', stats['num_positions'], 'count'])
                writer.writerow(['bbox_min_x', f"{stats['bounding_box']['min_x']:.3f}", stats['unit']])
                writer.writerow(['bbox_min_y', f"{stats['bounding_box']['min_y']:.3f}", stats['unit']])
                writer.writerow(['bbox_max_x', f"{stats['bounding_box']['max_x']:.3f}", stats['unit']])
                writer.writerow(['bbox_max_y', f"{stats['bounding_box']['max_y']:.3f}", stats['unit']])
                writer.writerow(['bbox_width', f"{stats['bounding_box']['width']:.3f}", stats['unit']])
                writer.writerow(['bbox_height', f"{stats['bounding_box']['height']:.3f}", stats['unit']])
                writer.writerow(['calibrated', 'yes' if stats['calibrated'] else 'no', ''])
            print(f"Statistics saved to: {stats_path}")
    
    if args.speed:
        tracker.save_speed_data(args.speed)
    elif args.csv:
        speed_csv = Path(args.csv).with_stem(Path(args.csv).stem + '.speed')
        tracker.save_speed_data(speed_csv)


if __name__ == "__main__":
    main()