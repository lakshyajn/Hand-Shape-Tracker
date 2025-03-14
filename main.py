import cv2
import numpy as np
import mediapipe as mp
import time
import math
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull

class PreciseHandTrackingApp:
    def __init__(self):
        # Initialize MediaPipe hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Finger indices in MediaPipe
        self.finger_indices = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20
        }
        
        # Base knuckle indices for detecting raised fingers
        self.base_indices = {
            "thumb": 2,  # thumb IP joint
            "index": 5,  # index MCP joint
            "middle": 9,  # middle MCP joint
            "ring": 13,  # ring MCP joint
            "pinky": 17   # pinky MCP joint
        }
        
        # Drawing parameters
        self.drawing_mode = False
        self.tracking_finger = None
        self.drawing_color = (0, 255, 0)  # Green by default
        self.canvas = None
        self.current_shape = []
        self.completed_shapes = []
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (255, 255, 255)
        self.font_thickness = 2
        
        # Constants
        self.distance_precision = 2  # decimal places
        
        # Shape expiration time (in seconds)
        self.shape_lifetime = 10
        
        self.min_points_for_recognition = 20
        self.shape_types = [
        "line", "triangle", "rectangle", "square", "circle", "diamond", "star", "ellipse", "arrow", "pentagon", "hexagon", "curved_line", 
        "spiral",
        "polygon" 
        ]
        
        # Last drawing
        self.drawing_ended_time = None
        self.last_tracking_finger = None
        
        # Last recognized shape
        self.last_recognized_shape = None
    
    def recognize_shape(self, points):
        """Recognize the shape from a list of points"""
        if len(points) < self.min_points_for_recognition:
            return None, None
        
        points_array = np.array(points)
        
        hull = ConvexHull(points_array)
        hull_points = points_array[hull.vertices]
        centroid = np.mean(points_array, axis=0)
        
        perimeter = cv2.arcLength(np.array(hull_points).astype(np.float32), True)
        area = cv2.contourArea(np.array(hull_points).astype(np.float32))
        
        if perimeter == 0:
            return None, None
        
        # circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Fit shapes
        rect = cv2.minAreaRect(np.array(hull_points).astype(np.float32))
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(np.array(hull_points).astype(np.float32))
        circle_center = (int(x), int(y))
        radius = int(radius)
        
        # how well it fits in a rectangle
        rect_area = cv2.contourArea(box)
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # line
        if len(hull_points) <= 4 and perimeter > 0 and area / perimeter < 5:
            return "line", np.array([points[0], points[-1]])
        
        # circle
        if circularity > 0.8:
            return "circle", (circle_center, radius)
        
        # number of corners using Douglas-Peucker algorithm
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(np.array(hull_points).astype(np.float32), epsilon, True)
        num_corners = len(approx)
        
        # Triangle
        if num_corners == 3:
            return "triangle", approx
        
        # Rectangle or square
        if num_corners == 4:
            width, height = rect[1]
            aspect_ratio = max(width, height) / (min(width, height) + 1e-10)  # Avoid division by zero
            
            if 0.95 <= aspect_ratio <= 1.05 and rectangularity > 0.8:
                return "square", box
            elif rectangularity > 0.8:
                return "rectangle", box
            else:
                return "diamond", box
        
        # Star (10 corners, 5 points)
        if 8 <= num_corners <= 12:
            return "star", approx
        
        # Default
        return "polygon", hull_points
    
    def draw_perfect_shape(self, canvas, shape_type, shape_data, color=(0, 255, 0), thickness=2):
        """Draw a perfect version of the recognized shape"""

        if shape_data is None:
            return canvas
    
        try:
            if shape_type == "line":
                if len(shape_data) >= 2:
                    pt1 = tuple(int(x) for x in shape_data[0])
                    pt2 = tuple(int(x) for x in shape_data[1])
                    cv2.line(canvas, pt1, pt2, color, thickness)
        
            elif shape_type == "circle":
                center, radius = shape_data
                center = tuple(int(x) for x in center)
                radius = int(radius)
                if radius > 0:
                    cv2.circle(canvas, center, radius, color, thickness)
        
            elif shape_type == "ellipse":
                center, axes, angle = shape_data
                center = tuple(int(x) for x in center)
                axes = tuple(int(x) for x in axes)
                angle = float(angle)
                if axes[0] > 0 and axes[1] > 0:
                    cv2.ellipse(canvas, center, axes, angle, 0, 360, color, thickness)
        
            elif shape_type == "arrow":
                if len(shape_data) >= 2:
                    pt1 = tuple(int(x) for x in shape_data[0])
                    pt2 = tuple(int(x) for x in shape_data[1])
                    cv2.arrowedLine(canvas, pt1, pt2, color, thickness, tipLength=0.2)
        
            elif shape_type in ["triangle", "rectangle", "square", "diamond", "star", "pentagon", "hexagon"]:
                if isinstance(shape_data, np.ndarray) and shape_data.size > 0:
                    cv2.drawContours(canvas, [shape_data.astype(np.int32)], 0, color, thickness)
                elif len(shape_data) > 0:
                    points = np.array(shape_data).astype(np.int32)
                    cv2.drawContours(canvas, [points], 0, color, thickness)
        
            elif shape_type == "polygon":
                if len(shape_data) > 0:
                    points = np.array(shape_data).astype(np.int32)
                    cv2.drawContours(canvas, [points], 0, color, thickness)
        
            elif shape_type == "curved_line":
                if len(shape_data) >= 3:
                    points = np.array(shape_data).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
        
            elif shape_type == "spiral":
                if len(shape_data) > 0:
                    points = np.array(shape_data).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
                
        except Exception as e:
            print(f"Error drawing {shape_type}: {e}")
    
        return canvas
    
    def is_finger_raised(self, landmarks, finger_name, height):
        """Determine if a finger is raised/extended based on its position relative to base"""
        if finger_name not in self.finger_indices or finger_name not in self.base_indices:
            return False
            
        tip_idx = self.finger_indices[finger_name]
        base_idx = self.base_indices[finger_name]
        
        # Convert normalized to pixel coordinates
        tip_y = landmarks[tip_idx].y * height
        base_y = landmarks[base_idx].y * height
        
        # For thumb, we need a different method since it doesn't extend like other fingers
        if finger_name == "thumb":
            # Check if thumb tip is sufficiently to the side of the hand
            tip_x = landmarks[tip_idx].x
            wrist_x = landmarks[0].x
            
            # Determine which hand (left or right)
            is_left_hand = landmarks[17].x < landmarks[5].x
            
            if is_left_hand:
                return tip_x < wrist_x  # left for left hand
            else:
                return tip_x > wrist_x  # right for right hand
        
        # For other fingers, check if fingertip is higher (smaller y) than base
        return tip_y < base_y - 20  # small threshold
    
    def get_raised_fingers(self, landmarks, frame_height):
        """Get a list of all raised fingers"""
        raised = []
        for finger_name in self.finger_indices:
            if self.is_finger_raised(landmarks, finger_name, frame_height):
                raised.append(finger_name)
        return raised
    
    def is_fist(self, landmarks, frame_height):
        """Detect if the hand is making a fist (no fingers raised)"""
        raised_fingers = self.get_raised_fingers(landmarks, frame_height)
        return len(raised_fingers) == 0
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points in pixels"""
        if p1 is None or p2 is None:
            return None
        pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return round(pixel_distance, self.distance_precision)
    
    def calculate_fist_volume(self, landmarks, frame_shape):
        """Calculate approximate volume of a fist by finding enclosing sphere"""
        h, w = frame_shape[:2]
        
        # Get all hand landmark coordinates
        points = []
        for landmark in landmarks:
            points.append((int(landmark.x * w), int(landmark.y * h)))
        
        # Find the centroid of all points
        centroid_x = sum(p[0] for p in points) / len(points)
        centroid_y = sum(p[1] for p in points) / len(points)
        centroid = (int(centroid_x), int(centroid_y))
        
        # distance to the furthest point as radius
        max_dist = 0
        for point in points:
            dist = math.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)
            max_dist = max(max_dist, dist)
        
        # volume using sphere formula (4/3 * π * r³)
        radius = max_dist
        volume = (4/3) * math.pi * (radius ** 3)
        
        # Convert to cubic centimeters (assuming 1 pixel ≈ 0.026 cm for typical webcam)
        # This is approximate and would need calibration for accuracy
        pixel_to_cm = 0.026
        volume_cm3 = volume * (pixel_to_cm ** 3)
        
        return centroid, int(radius), round(volume_cm3, 2)
    
    def run(self):
        # webcam
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture image")
                break
                
            # Mirror the frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Initialize canvas if needed
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
            else:
                # Clear the canvas for this frame
                self.canvas = np.zeros_like(frame)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = self.hands.process(rgb_frame)
            
            # Display key information
            cv2.putText(frame, "Raise only 1 finger to track", (10, 30), 
                       self.font, self.font_scale, self.font_color, self.font_thickness)
            cv2.putText(frame, "Raise 2 fingers to measure distance", (10, 60), 
                       self.font, self.font_scale, self.font_color, self.font_thickness)
            cv2.putText(frame, "Make a fist to calculate volume", (10, 90), 
                       self.font, self.font_scale, self.font_color, self.font_thickness)
            
            # Check for and remove expired shapes
            current_time = time.time()
            active_shapes = []
            
            for shape, shape_type, shape_data, timestamp in self.completed_shapes:
                if current_time - timestamp < self.shape_lifetime:
                    active_shapes.append((shape, shape_type, shape_data, timestamp))
                    
                    # Draw the perfected shape on the canvas
                    if shape_type:
                        self.draw_perfect_shape(self.canvas, shape_type, shape_data, self.drawing_color, 2)
                    else:
                        # Draw the original shape if no shape was recognized
                        for i in range(1, len(shape)):
                            cv2.line(self.canvas, shape[i-1], shape[i], self.drawing_color, 2)
            
            self.completed_shapes = active_shapes
            
            # Initialize change in tracking state
            changed_tracking_finger = False
            hand_detected = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get frame dimensions
                    h, w, c = frame.shape
                    
                    # Get raised fingers
                    raised_fingers = self.get_raised_fingers(hand_landmarks.landmark, h)
                    
                    # Mark that we detected a hand
                    hand_detected = True
                    
                    # Check if making a fist
                    if self.is_fist(hand_landmarks.landmark, h):
                        # Calculate and display fist volume
                        centroid, radius, volume = self.calculate_fist_volume(hand_landmarks.landmark, (h, w))
                        
                        # Draw the enclosing circle
                        cv2.circle(frame, centroid, radius, (0, 0, 255), 2)
                        
                        # Display the volume
                        volume_text = f"Volume: {volume} cm³"
                        text_pos = (centroid[0] - 80, centroid[1] - radius - 10)
                        cv2.putText(frame, volume_text, text_pos, 
                                   self.font, self.font_scale, (0, 255, 255), self.font_thickness)
                        
                        # If we were tracking, complete the drawing
                        if self.drawing_mode:
                            # End the current tracking
                            if len(self.current_shape) >= self.min_points_for_recognition:
                                # Recognize the shape
                                shape_type, shape_data = self.recognize_shape(self.current_shape)
                                self.last_recognized_shape = shape_type
                                
                                # Add to completed shapes with current timestamp
                                self.completed_shapes.append((self.current_shape, shape_type, shape_data, time.time()))
                            
                            # Reset tracking
                            self.drawing_mode = False
                            self.drawing_ended_time = time.time()
                            self.last_tracking_finger = self.tracking_finger
                            self.tracking_finger = None
                            self.current_shape = []
                            changed_tracking_finger = True
                    
                    # If exactly one finger is raised, track it
                    elif len(raised_fingers) == 1:
                        finger_name = raised_fingers[0]
                        finger_idx = self.finger_indices[finger_name]
                        
                        # Get fingertip position
                        landmark = hand_landmarks.landmark[finger_idx]
                        px, py = int(landmark.x * w), int(landmark.y * h)
                        
                        # Draw tracking indicator
                        cv2.circle(frame, (px, py), 15, (0, 255, 0), -1)
                        
                        # If we weren't tracking this finger before, start tracking
                        if not self.drawing_mode or self.tracking_finger != finger_name:
                            # If we were tracking before, complete that drawing
                            if self.drawing_mode and len(self.current_shape) >= self.min_points_for_recognition:
                                # Recognize the shape
                                shape_type, shape_data = self.recognize_shape(self.current_shape)
                                self.last_recognized_shape = shape_type
                                
                                # Add to completed shapes with current timestamp
                                self.completed_shapes.append((self.current_shape, shape_type, shape_data, time.time()))
                                
                                # Reset for new tracking
                                self.current_shape = []
                                changed_tracking_finger = True
                            
                            # Start new tracking
                            self.drawing_mode = True
                            self.tracking_finger = finger_name
                        
                        # Add point to current shape
                        self.current_shape.append((px, py))
                        
                        # Draw the current shape as we track
                        if len(self.current_shape) >= 2:
                            for i in range(1, len(self.current_shape)):
                                cv2.line(self.canvas, self.current_shape[i-1], self.current_shape[i], 
                                       self.drawing_color, 2)
                            
                        # Display tracking message
                        track_msg = f"Tracking {finger_name} finger"
                        cv2.putText(frame, track_msg, (px - 70, py - 20), 
                                   self.font, self.font_scale, (255, 0, 255), self.font_thickness)
                                   
                        # Display shape recognition if available
                        if self.last_recognized_shape:
                            cv2.putText(frame, f"Last shape: {self.last_recognized_shape}", (10, 150), 
                                       self.font, self.font_scale, (0, 255, 255), self.font_thickness)
                    
                    # If exactly two fingers are raised, measure distance
                    elif len(raised_fingers) == 2:
                        # If we were tracking, complete the drawing
                        if self.drawing_mode:
                            if len(self.current_shape) >= self.min_points_for_recognition:
                                # Recognize the shape
                                shape_type, shape_data = self.recognize_shape(self.current_shape)
                                self.last_recognized_shape = shape_type
                                
                                # Add to completed shapes with current timestamp
                                self.completed_shapes.append((self.current_shape, shape_type, shape_data, time.time()))
                            
                            # Reset tracking
                            self.drawing_mode = False
                            self.drawing_ended_time = time.time()
                            self.last_tracking_finger = self.tracking_finger
                            self.tracking_finger = None
                            self.current_shape = []
                            changed_tracking_finger = True
                        
                        # Get fingertip positions
                        finger1_name, finger2_name = raised_fingers[0], raised_fingers[1]
                        finger1_idx = self.finger_indices[finger1_name]
                        finger2_idx = self.finger_indices[finger2_name]
                        
                        landmark1 = hand_landmarks.landmark[finger1_idx]
                        landmark2 = hand_landmarks.landmark[finger2_idx]
                        
                        p1 = (int(landmark1.x * w), int(landmark1.y * h))
                        p2 = (int(landmark2.x * w), int(landmark2.y * h))
                        
                        # Draw circles on both fingertips
                        cv2.circle(frame, p1, 10, (255, 0, 0), -1)
                        cv2.circle(frame, p2, 10, (255, 0, 0), -1)
                        
                        # Draw line between fingertips
                        cv2.line(frame, p1, p2, (0, 255, 255), 2)
                        
                        # Calculate and display distance
                        distance = self.calculate_distance(p1, p2)
                        
                        # Convert to cm (approximate)
                        pixel_to_cm = 0.026
                        distance_cm = round(distance * pixel_to_cm, 2)
                        
                        # Calculate midpoint
                        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                        
                        # Display distances
                        cv2.putText(frame, f"{distance} px", midpoint, 
                                   self.font, self.font_scale, self.font_color, self.font_thickness)
                        cv2.putText(frame, f"≈ {distance_cm} cm", 
                                   (midpoint[0], midpoint[1] + 30), 
                                   self.font, self.font_scale, self.font_color, self.font_thickness)
                    
                    # If more than two fingers are raised
                    else:
                        # If we were tracking, complete the drawing
                        if self.drawing_mode:
                            if len(self.current_shape) >= self.min_points_for_recognition:
                                # Recognize the shape
                                shape_type, shape_data = self.recognize_shape(self.current_shape)
                                self.last_recognized_shape = shape_type
                                
                                # Add to completed shapes with current timestamp
                                self.completed_shapes.append((self.current_shape, shape_type, shape_data, time.time()))
                            
                            # Reset tracking
                            self.drawing_mode = False
                            self.drawing_ended_time = time.time()
                            self.last_tracking_finger = self.tracking_finger
                            self.tracking_finger = None
                            self.current_shape = []
                            changed_tracking_finger = True
                        
                        # Display message
                        msg = f"{len(raised_fingers)} fingers raised"
                        cv2.putText(frame, msg, (10, 120), 
                                   self.font, self.font_scale, (0, 165, 255), self.font_thickness)
            
            # If no hand is detected but we were tracking, complete the drawing
            if not hand_detected and self.drawing_mode:
                if len(self.current_shape) >= self.min_points_for_recognition:
                    # Recognize the shape
                    shape_type, shape_data = self.recognize_shape(self.current_shape)
                    self.last_recognized_shape = shape_type
                    
                    # Add to completed shapes with current timestamp
                    self.completed_shapes.append((self.current_shape, shape_type, shape_data, time.time()))
                
                # Reset tracking
                self.drawing_mode = False
                self.drawing_ended_time = time.time()
                self.last_tracking_finger = self.tracking_finger
                self.tracking_finger = None
                self.current_shape = []
                changed_tracking_finger = True
            
            # Draw the current shape while tracking
            if self.drawing_mode and len(self.current_shape) >= 2:
                for i in range(1, len(self.current_shape)):
                    cv2.line(self.canvas, self.current_shape[i-1], self.current_shape[i], 
                           self.drawing_color, 2)
            
            # Combine original frame with the drawing canvas
            combined_frame = cv2.addWeighted(frame, 1.0, self.canvas, 0.7, 0)
            
            # Display information about shape recognition if available
            if changed_tracking_finger and self.last_recognized_shape:
                cv2.putText(combined_frame, f"Detected: {self.last_recognized_shape}", 
                           (10, 180), self.font, 1.0, (0, 255, 255), self.font_thickness)
            
            # Show the frame
            cv2.imshow('Precise Hand Tracking with Shape Recognition', combined_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PreciseHandTrackingApp()
    app.run()
