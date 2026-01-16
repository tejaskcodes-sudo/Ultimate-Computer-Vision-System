"""
ULTIMATE COMPUTER VISION SYSTEM - COMPLETE WORKING VERSION
✔ Automatic NumPy version fix
✔ All features working: Face detection, Blink detection, Hand tracking, Emotion detection
✔ High FPS performance
✔ Error handling and user-friendly interface
"""

import sys
import subprocess

# =====================================================
# ---------------- NUMPY VERSION CHECK ----------------
# =====================================================

def check_numpy_version():
    """Ensure NumPy is compatible version"""
    try:
        import numpy as np
        version = np.__version__
        print(f"✓ NumPy version: {version}")
        
        # Check if it's numpy 2.x (incompatible)
        if version.startswith('2.'):
            print("\n⚠️  WARNING: NumPy 2.x detected!")
            print("This version is incompatible with OpenCV.")
            print("\nDowngrading to NumPy 1.24.3...")
            
            try:
                # Uninstall current numpy
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Install compatible version
                subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✓ NumPy downgraded successfully!")
                print("Please restart the program.")
            except:
                print("✗ Automatic fix failed.")
                print("\nPlease run these commands manually:")
                print("  pip uninstall numpy -y")
                print("  pip install numpy==1.24.3")
            
            input("Press Enter to exit...")
            sys.exit(1)
        
        return np
        
    except ImportError:
        print("Installing NumPy 1.24.3...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import numpy as np
        return np

# Check and fix numpy
print("=" * 60)
print("ULTIMATE COMPUTER VISION SYSTEM v3.0")
print("=" * 60)
print("\nChecking dependencies...")
np = check_numpy_version()

# =====================================================
# ---------------- IMPORT OTHER PACKAGES --------------
# =====================================================

try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError:
    print("Installing OpenCV...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python==4.9.0.80"])
    import cv2

try:
    import mediapipe as mp
    print(f"✓ MediaPipe version: {mp.__version__}")
except ImportError:
    print("Installing MediaPipe...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe==0.10.14"])
    import mediapipe as mp

import time
from collections import deque
import math

print("\n" + "="*60)
print("ALL DEPENDENCIES ARE READY!")
print("="*60)

# =====================================================
# ---------------- EYE BLINK DETECTOR -----------------
# =====================================================

class EyeBlinkDetector:
    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        self.counter = 0
        self.total = 0
        self.threshold = 0.23  # Eye Aspect Ratio threshold
        self.is_blinking = False

    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Vertical distances
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Horizontal distance
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C + 1e-6)

    def update(self, landmarks, frame_shape):
        """Update blink detection with new frame"""
        h, w = frame_shape[:2]
        
        # Get eye landmarks
        left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) 
                           for i in self.LEFT_EYE])
        right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) 
                            for i in self.RIGHT_EYE])

        # Calculate EAR for both eyes
        ear_left = self.eye_aspect_ratio(left_eye)
        ear_right = self.eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0
        
        blink_detected = False
        
        # Detect blink (EAR below threshold)
        if ear < self.threshold:
            self.counter += 1
            self.is_blinking = True
        else:
            if self.counter >= 3:  # Minimum frames for a valid blink
                self.total += 1
                blink_detected = True
            self.counter = 0
            self.is_blinking = False

        return ear, blink_detected, self.total, self.is_blinking


# =====================================================
# ---------------- HAND FINGER COUNTER ----------------
# =====================================================

class HandFingerCounter:
    # Finger landmark indices (MediaPipe Hand landmarks)
    FINGER_JOINTS = {
        "Thumb":  (1, 2, 4),    # MCP, IP, Tip
        "Index":  (5, 6, 8),    # MCP, PIP, Tip
        "Middle": (9, 10, 12),  # MCP, PIP, Tip
        "Ring":   (13, 14, 16), # MCP, PIP, Tip
        "Pinky":  (17, 18, 20)  # MCP, PIP, Tip
    }

    def calculate_angle(self, point_a, point_b, point_c):
        """Calculate angle between three points"""
        a = np.array([point_a.x, point_a.y])
        b = np.array([point_b.x, point_b.y])
        c = np.array([point_c.x, point_c.y])
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle

    def count_fingers(self, landmarks, handedness):
        """Count number of raised fingers"""
        # Mirror fix: webcam shows mirrored image
        hand_label = "Left" if handedness == "Right" else "Right"
        
        finger_count = 0
        finger_states = {}
        
        for finger_name, (mcp, pip, tip) in self.FINGER_JOINTS.items():
            # Calculate joint angle
            angle = self.calculate_angle(landmarks[mcp], landmarks[pip], landmarks[tip])
            
            # Different thresholds for thumb vs other fingers
            if finger_name == "Thumb":
                is_open = angle > 150  # Thumb threshold
            else:
                is_open = angle > 160  # Other fingers threshold
            
            finger_states[finger_name] = is_open
            
            if is_open:
                finger_count += 1
        
        return finger_count, hand_label, finger_states


# =====================================================
# ---------------- EMOTION DETECTOR -------------------
# =====================================================

class EmotionDetector:
    def __init__(self):
        self.history = deque(maxlen=10)  # Store last 10 detections
        self.last_emotion = "Neutral"
        
    def detect_emotion(self, landmarks):
        """Detect emotion from facial landmarks"""
        # Mouth openness (for surprise/happy)
        mouth_openness = abs(landmarks[13].y - landmarks[14].y)
        
        # Eye openness (for sad/surprised)
        eye_openness = abs(landmarks[159].y - landmarks[145].y)
        
        # Brow position (for angry)
        brow_lowered = landmarks[65].y < landmarks[158].y
        
        # Emotion logic
        if mouth_openness > 0.05 and eye_openness > 0.03:
            emotion = "Surprised"
        elif mouth_openness > 0.03:
            emotion = "Happy"
        elif eye_openness < 0.015:
            emotion = "Sad"
        elif brow_lowered:
            emotion = "Angry"
        else:
            emotion = "Neutral"
        
        # Add to history and get most common
        self.history.append(emotion)
        
        # Count occurrences and return most frequent
        if self.history:
            return max(set(self.history), key=self.history.count)
        return emotion


# =====================================================
# ---------------- MAIN APPLICATION CLASS -------------
# =====================================================

class UltimateVisionSystem:
    def __init__(self):
        print("\nInitializing vision system components...")
        
        try:
            # Initialize MediaPipe models
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0=short-range, 1=full-range
                min_detection_confidence=0.6
            )
            
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.hand_tracker = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # Drawing utilities
            self.drawing_utils = mp.solutions.drawing_utils
            self.drawing_styles = mp.solutions.drawing_styles
            
            print("✓ MediaPipe models initialized")
            
        except Exception as e:
            print(f"✗ Failed to initialize MediaPipe: {e}")
            input("\nPress Enter to exit...")
            sys.exit(1)
        
        # Initialize custom detectors
        self.blink_detector = EyeBlinkDetector()
        self.finger_counter = HandFingerCounter()
        self.emotion_detector = EmotionDetector()
        
        # Initialize camera
        print("Initializing camera...")
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            print("✗ Cannot access camera")
            print("1. Check if camera is connected")
            print("2. Close other apps using camera")
            print("3. Check camera permissions")
            input("\nPress Enter to exit...")
            sys.exit(1)
        
        # Set camera resolution
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual resolution
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera ready: {width}x{height}")
        
        # FPS calculation
        self.frame_count = 0
        self.current_fps = 0
        self.start_time = time.time()
        
        # Display settings
        self.fullscreen = False
        self.show_info = True
        
        print("\n" + "="*50)
        print("SYSTEM INITIALIZATION COMPLETE!")
        print("="*50)
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:  # Update FPS every second
            self.current_fps = int(self.frame_count / elapsed_time)
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.current_fps
    
    def draw_info_panel(self, frame, ear, blink_count, emotion, is_blinking):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Eye information
        eye_color = (0, 255, 0) if ear > 0.23 else (0, 0, 255)
        cv2.putText(frame, f"Eye Aspect Ratio: {ear:.3f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        
        # Blink counter
        cv2.putText(frame, f"Total Blinks: {blink_count}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Emotion
        emotion_colors = {
            "Happy": (0, 255, 255),
            "Sad": (255, 100, 0),
            "Angry": (0, 0, 255),
            "Surprised": (255, 0, 255),
            "Neutral": (200, 200, 200)
        }
        emotion_color = emotion_colors.get(emotion, (200, 200, 200))
        cv2.putText(frame, f"Emotion: {emotion}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Blink alert
        if is_blinking:
            cv2.putText(frame, "BLINKING!", (w//2 - 100, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*50)
        print("CONTROLS:")
        print("  [Q] - Quit")
        print("  [R] - Reset blink counter")
        print("  [F] - Toggle fullscreen")
        print("  [I] - Toggle info panel")
        print("  [S] - Save screenshot")
        print("="*50)
        print("\nStarting video capture...")
        
        screenshot_count = 0
        
        while True:
            # Read frame from camera
            success, frame = self.video_capture.read()
            if not success:
                print("Failed to capture frame")
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Calculate FPS
            fps = self.calculate_fps()
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe models
            face_detections = self.face_detector.process(rgb_frame)
            face_mesh_results = self.face_mesh.process(rgb_frame)
            hand_results = self.hand_tracker.process(rgb_frame)
            
            frame_height, frame_width = frame.shape[:2]
            
            # ========== FACE DETECTION ==========
            if face_detections.detections:
                for detection in face_detections.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert to pixel coordinates
                    x_min = int(bbox.xmin * frame_width)
                    y_min = int(bbox.ymin * frame_height)
                    x_max = int((bbox.xmin + bbox.width) * frame_width)
                    y_max = int((bbox.ymin + bbox.height) * frame_height)
                    
                    # Draw face bounding box
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), 
                                (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face", (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # ========== FACE MESH & BLINK DETECTION ==========
            ear = 0.0
            blink_detected = False
            blink_count = 0
            emotion = "Neutral"
            is_blinking = False
            
            if face_mesh_results.multi_face_landmarks:
                # Get first face landmarks
                face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                
                # Update blink detector
                ear, blink_detected, blink_count, is_blinking = self.blink_detector.update(
                    face_landmarks, (frame_height, frame_width)
                )
                
                # Detect emotion
                emotion = self.emotion_detector.detect_emotion(face_landmarks)
            
            # ========== HAND TRACKING ==========
            if hand_results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # Get hand label (left/right)
                    if hand_results.multi_handedness:
                        hand_label = hand_results.multi_handedness[idx].classification[0].label
                    else:
                        hand_label = "Unknown"
                    
                    # Count fingers
                    finger_count, display_label, finger_states = self.finger_counter.count_fingers(
                        hand_landmarks.landmark, hand_label
                    )
                    
                    # Draw hand landmarks
                    self.drawing_utils.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.drawing_styles.get_default_hand_landmarks_style(),
                        self.drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Display finger count
                    count_text = f"{display_label} Hand: {finger_count} fingers"
                    cv2.putText(display_frame, count_text,
                              (frame_width - 350, 80 + idx * 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # ========== DRAW INFO PANEL ==========
            if self.show_info:
                display_frame = self.draw_info_panel(
                    display_frame, ear, blink_count, emotion, is_blinking
                )
            
            # Draw controls hint
            cv2.putText(display_frame, "[Q]uit [R]eset [F]ullscreen [I]nfo [S]creenshot",
                       (20, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # ========== DISPLAY WINDOW ==========
            window_title = "Ultimate Computer Vision System"
            
            if self.fullscreen:
                cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_title, 1280, 720)
            
            # Show frame
            cv2.imshow(window_title, display_frame)
            
            # ========== KEYBOARD CONTROLS ==========
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nShutting down...")
                break
            
            elif key == ord('r'):
                self.blink_detector.total = 0
                print("Blink counter reset to 0")
            
            elif key == ord('f'):
                self.fullscreen = not self.fullscreen
                status = "ON" if self.fullscreen else "OFF"
                print(f"Fullscreen: {status}")
            
            elif key == ord('i'):
                self.show_info = not self.show_info
                status = "ON" if self.show_info else "OFF"
                print(f"Info panel: {status}")
            
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        self.video_capture.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print("PROGRAM FINISHED SUCCESSFULLY")
        print(f"Total blinks detected: {self.blink_detector.total}")
        print("="*50)


# =====================================================
# ---------------- PROGRAM ENTRY POINT ----------------
# =====================================================

if __name__ == "__main__":
    try:
        # Create and run the vision system
        vision_system = UltimateVisionSystem()
        vision_system.run()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        
    except Exception as error:
        print(f"\n\nUnexpected error: {error}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nThank you for using Ultimate Computer Vision System!")
        input("Press Enter to exit...")