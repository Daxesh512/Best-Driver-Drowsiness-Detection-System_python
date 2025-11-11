import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import urllib.request
import os

class DrowsinessDetector:
    def __init__(self):
        """Initialize the drowsiness detector with dlib face detection and landmark prediction"""
        
        # Initialize face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Download shape predictor if not exists
        self.shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(self.shape_predictor_path):
            self.download_shape_predictor()
        
        try:
            self.landmark_predictor = dlib.shape_predictor(self.shape_predictor_path)
        except:
            # If download fails, create a simple fallback detector
            self.landmark_predictor = None
            print("Warning: Could not load shape predictor. Using simplified detection.")
        
        # Eye and mouth landmark indices (68-point model)
        self.LEFT_EYE_INDICES = list(range(36, 42))
        self.RIGHT_EYE_INDICES = list(range(42, 48))
        self.MOUTH_INDICES = list(range(48, 68))
        
        # Detection thresholds
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.7
        self.CONSECUTIVE_FRAMES = 15
        
        # Counters
        self.closed_eyes_counter = 0
        self.yawn_counter = 0
        
        # Status tracking
        self.current_status = "Awake"
        self.confidence = 0.0
        
    def download_shape_predictor(self):
        """Download the shape predictor model if not available"""
        try:
            print("Downloading shape predictor model...")
            # Try multiple sources for the model
            urls = [
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
            ]
            
            for url in urls:
                try:
                    if url.endswith('.bz2'):
                        import bz2
                        compressed_path = self.shape_predictor_path + '.bz2'
                        urllib.request.urlretrieve(url, compressed_path)
                        
                        # Decompress the file
                        with bz2.BZ2File(compressed_path, 'rb') as f_in:
                            with open(self.shape_predictor_path, 'wb') as f_out:
                                f_out.write(f_in.read())
                        os.remove(compressed_path)
                    else:
                        urllib.request.urlretrieve(url, self.shape_predictor_path)
                    
                    print("Shape predictor downloaded successfully!")
                    return
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
                    continue
            
            print("All download attempts failed")
        except Exception as e:
            print(f"Failed to download shape predictor: {e}")
    
    def update_thresholds(self, ear_threshold, mar_threshold, consecutive_frames):
        """Update detection thresholds"""
        self.EAR_THRESHOLD = ear_threshold
        self.MAR_THRESHOLD = mar_threshold
        self.CONSECUTIVE_FRAMES = consecutive_frames
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye landmark
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio (MAR)"""
        # Vertical mouth landmarks
        A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])  # 51, 59
        B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])   # 53, 57
        
        # Horizontal mouth landmark
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])   # 49, 55
        
        # Calculate MAR
        mar = (A + B) / (2.0 * C)
        return mar
    
    def extract_landmarks(self, frame, face):
        """Extract facial landmarks from detected face"""
        if self.landmark_predictor is None:
            return None
            
        landmarks = self.landmark_predictor(frame, face)
        landmarks_points = []
        
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        
        return np.array(landmarks_points)
    
    def draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks on frame"""
        if landmarks is None:
            return frame
            
        # Draw eye landmarks
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        mouth = landmarks[self.MOUTH_INDICES]
        
        # Draw eye contours
        cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
        
        # Draw mouth contour
        cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
        
        # Draw individual points
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
        
        return frame
    
    def advanced_eye_detection(self, frame):
        """Advanced eye detection using multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Haar cascade eye detection
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
        
        # Method 2: Face detection with eye region analysis
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        ear_estimate = 0.25  # Default neutral value
        mar_estimate = 0.5   # Default neutral value
        
        if len(faces) > 0:
            # Use the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Define eye regions (upper 40% of face, left and right thirds)
            eye_region_height = int(h * 0.4)
            eye_width = int(w * 0.3)
            
            left_eye_region = gray[y:y+eye_region_height, x:x+eye_width]
            right_eye_region = gray[y:y+eye_region_height, x+w-eye_width:x+w]
            
            # Analyze eye regions for openness using edge detection
            def analyze_eye_openness(eye_region):
                if eye_region.size == 0:
                    return 0.25
                
                # Apply Gaussian blur and edge detection
                blurred = cv2.GaussianBlur(eye_region, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                
                # Count horizontal edges (indicates open eyes)
                horizontal_edges = 0
                for row in edges:
                    if np.sum(row) > 100:  # Threshold for significant edge
                        horizontal_edges += 1
                
                # More horizontal edges = more open eyes
                openness = min(horizontal_edges / 10.0, 1.0)
                return 0.15 + (openness * 0.2)  # Scale between 0.15 and 0.35
            
            left_ear = analyze_eye_openness(left_eye_region)
            right_ear = analyze_eye_openness(right_eye_region)
            ear_estimate = (left_ear + right_ear) / 2.0
            
            # Analyze mouth region (bottom 30% of face)
            mouth_y_start = y + int(h * 0.6)
            mouth_region = gray[mouth_y_start:y+h, x+int(w*0.25):x+int(w*0.75)]
            
            if mouth_region.size > 0:
                # Detect mouth openness using contour analysis
                blurred_mouth = cv2.GaussianBlur(mouth_region, (5, 5), 0)
                _, thresh = cv2.threshold(blurred_mouth, 60, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour (likely mouth)
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    # Estimate MAR based on contour area
                    mar_estimate = min(area / 1000.0, 1.5)  # Scale appropriately
        
        # Combine with traditional eye detection for better accuracy
        if len(eyes) >= 2:
            ear_estimate = max(ear_estimate, 0.25)
        elif len(eyes) == 1:
            ear_estimate = max(ear_estimate * 0.8, 0.2)
        else:
            ear_estimate = min(ear_estimate * 0.6, 0.2)
        
        return ear_estimate, mar_estimate
    
    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector(gray)
        
        ear_value = 0.3  # Default values
        mar_value = 0.5
        
        # Draw FPS and detection info
        cv2.putText(frame, f"EAR Threshold: {self.EAR_THRESHOLD:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR Threshold: {self.MAR_THRESHOLD:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Draw face rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(gray, face)
            
            if landmarks is not None:
                # Draw landmarks
                frame = self.draw_landmarks(frame, landmarks)
                
                # Calculate EAR for both eyes
                left_eye = landmarks[self.LEFT_EYE_INDICES]
                right_eye = landmarks[self.RIGHT_EYE_INDICES]
                mouth = landmarks[self.MOUTH_INDICES]
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear_value = (left_ear + right_ear) / 2.0
                
                # Calculate MAR
                mar_value = self.calculate_mar(mouth)
                
            else:
                # Fallback to advanced detection
                ear_value, mar_value = self.advanced_eye_detection(frame)
        else:
            # No faces detected - use advanced detection
            ear_value, mar_value = self.advanced_eye_detection(frame)
        
        # Determine status based on EAR and MAR
        status = "Awake"
        confidence = 0.5
        
        # Check for closed eyes
        if ear_value < self.EAR_THRESHOLD:
            self.closed_eyes_counter += 1
            if self.closed_eyes_counter >= self.CONSECUTIVE_FRAMES:
                status = "Eyes Closed"
                confidence = 1.0 - (ear_value / self.EAR_THRESHOLD)
        else:
            self.closed_eyes_counter = 0
        
        # Check for yawning
        if mar_value > self.MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= self.CONSECUTIVE_FRAMES // 2:  # Yawn detection is faster
                if status == "Eyes Closed":
                    status = "Drowsy"  # Both eyes closed and yawning
                else:
                    status = "Drowsy"  # Just yawning
                confidence = max(confidence, mar_value / self.MAR_THRESHOLD - 1.0)
        else:
            self.yawn_counter = 0
        
        # Update current status
        self.current_status = status
        self.confidence = min(confidence, 1.0)
        
        # Draw current metrics on frame
        cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar_value:.3f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw status
        status_color = (0, 255, 0) if status == "Awake" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw counters
        cv2.putText(frame, f"Closed Eyes: {self.closed_eyes_counter}", (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Yawn: {self.yawn_counter}", (10, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame, status, ear_value, mar_value, confidence
