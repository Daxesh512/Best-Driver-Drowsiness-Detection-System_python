import cv2
import numpy as np
import time
import math

class CameraSimulator:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        
    def generate_realistic_face_frame(self):
        """Generate realistic face frames with varying drowsiness patterns"""
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(40)  # Dark background
        
        current_time = time.time() - self.start_time
        self.frame_count += 1
        
        # Face position with slight movement
        face_x = 320 + int(10 * math.sin(current_time * 0.5))
        face_y = 240 + int(5 * math.cos(current_time * 0.3))
        
        # Draw realistic face shape
        cv2.ellipse(frame, (face_x, face_y), (80, 100), 0, 0, 360, (120, 100, 80), -1)
        cv2.ellipse(frame, (face_x, face_y), (80, 100), 0, 0, 360, (150, 130, 110), 3)
        
        # Calculate dynamic drowsiness parameters
        # Simulate natural blinking and drowsiness cycles
        blink_cycle = math.sin(current_time * 2) * 0.5 + 0.5
        drowsy_cycle = max(0, math.sin(current_time * 0.1)) * 0.8
        
        # Every 30 seconds, simulate drowsiness for 5 seconds
        if (current_time % 30) < 5:
            drowsy_factor = 0.7
        else:
            drowsy_factor = 0.1
        
        # Eye calculations
        base_ear = 0.28
        ear_variation = 0.08 * blink_cycle - drowsy_factor * 0.15
        ear_value = max(0.15, base_ear + ear_variation)
        
        # Mouth calculations  
        base_mar = 0.5
        yawn_cycle = math.sin(current_time * 0.2)
        if (current_time % 45) < 3:  # Yawn every 45 seconds for 3 seconds
            mar_variation = 0.5 * max(0, yawn_cycle)
        else:
            mar_variation = 0.1 * yawn_cycle
        mar_value = max(0.3, base_mar + mar_variation)
        
        # Draw eyes with realistic shapes
        left_eye_center = (face_x - 25, face_y - 20)
        right_eye_center = (face_x + 25, face_y - 20)
        
        # Eye shape based on EAR
        eye_width = 20
        eye_height = int(12 * (ear_value / 0.3))
        
        # Draw eye backgrounds
        cv2.ellipse(frame, left_eye_center, (eye_width, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, right_eye_center, (eye_width, 12), 0, 0, 360, (255, 255, 255), -1)
        
        # Draw pupils
        cv2.ellipse(frame, left_eye_center, (eye_width-5, eye_height), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(frame, right_eye_center, (eye_width-5, eye_height), 0, 0, 360, (50, 50, 50), -1)
        
        # Draw eyebrows
        cv2.ellipse(frame, (left_eye_center[0], left_eye_center[1]-15), (25, 8), 0, 0, 360, (80, 60, 40), -1)
        cv2.ellipse(frame, (right_eye_center[0], right_eye_center[1]-15), (25, 8), 0, 0, 360, (80, 60, 40), -1)
        
        # Draw mouth based on MAR
        mouth_center = (face_x, face_y + 40)
        mouth_width = 30
        mouth_height = int(8 * mar_value)
        
        cv2.ellipse(frame, mouth_center, (mouth_width, mouth_height), 0, 0, 360, (180, 100, 100), -1)
        cv2.ellipse(frame, mouth_center, (mouth_width, mouth_height), 0, 0, 360, (120, 60, 60), 2)
        
        # Draw nose
        nose_points = np.array([
            [face_x-5, face_y],
            [face_x+5, face_y], 
            [face_x, face_y+15]
        ], np.int32)
        cv2.fillPoly(frame, [nose_points], (140, 120, 100))
        
        # Add facial landmarks for detection
        landmarks = []
        
        # Left eye landmarks (6 points)
        for i in range(6):
            angle = (i * 60) * math.pi / 180
            x = int(left_eye_center[0] + (eye_width-2) * math.cos(angle))
            y = int(left_eye_center[1] + eye_height * math.sin(angle))
            landmarks.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Right eye landmarks (6 points)
        for i in range(6):
            angle = (i * 60) * math.pi / 180
            x = int(right_eye_center[0] + (eye_width-2) * math.cos(angle))
            y = int(right_eye_center[1] + eye_height * math.sin(angle))
            landmarks.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Mouth landmarks (8 points)
        for i in range(8):
            angle = (i * 45) * math.pi / 180
            x = int(mouth_center[0] + mouth_width * math.cos(angle))
            y = int(mouth_center[1] + mouth_height * math.sin(angle))
            landmarks.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Add info overlay
        cv2.putText(frame, "REAL-TIME SIMULATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, ear_value, mar_value, landmarks
    
    def read(self):
        """Simulate camera read method"""
        frame, ear, mar, landmarks = self.generate_realistic_face_frame()
        return True, frame
    
    def isOpened(self):
        return True
    
    def release(self):
        pass
    
    def set(self, prop, value):
        pass