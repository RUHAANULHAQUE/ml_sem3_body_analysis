import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import csv
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import logging
from typing import Optional, Tuple, Dict, List

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging to both file and console"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"body_language_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

class Config:
    """Configuration with save/load capabilities"""
    
    # Detection confidence
    MIN_DETECTION_CONFIDENCE = 0.3
    MIN_TRACKING_CONFIDENCE = 0.3
    
    STEEPLE_FINGERTIP_DIST = 0.25      
    STEEPLE_PALM_DIST_MIN = 0.03       
    STEEPLE_MIN_FINGERS = 1           

    HOLDING_BALL_HAND_DIST_MIN = 0.30  
    HOLDING_BALL_HAND_DIST_MAX = 0.45  
    HOLDING_BALL_CURVE_THRESHOLD = 0.5
    
    OPEN_PALM_MIN_FINGERS = 2
    OPEN_PALM_EXTENSION = 0.4
    
    PALM_DOWN_Y_THRESHOLD = 0.01
    PALM_DOWN_MIN_FINGERS = 1
    
    POINTING_INDEX_EXTENSION = 0.5
    POINTING_OTHER_CURL = 0.4
    
    BOX_HAND_DISTANCE_MIN = 0.15
    BOX_HAND_DISTANCE_MAX = 0.70
    
    # Performance
    FRAME_SKIP = 1  
    
    # Analysis duration
    ANALYSIS_DURATION_SECONDS = 180
    
    # Posture thresholds
    UPRIGHT_THRESHOLD = 0.20
    # Angle tolerance (radians) for torso verticality - smaller means more upright
    UPRIGHT_ANGLE_THRESHOLD = 0.30
    
    # Stability
    GESTURE_STABILITY_FRAMES = 2
    GESTURE_TRANSITION_SMOOTHING = 3  
    
    # Smoothing
    SMOOTHING_WINDOW = 10
    HISTORY_SIZE = 100
    
    # UI Colors (BGR format)
    COLOR_GOOD = (100, 255, 100)
    COLOR_MEDIUM = (100, 200, 255)
    COLOR_LOW = (150, 100, 255)
    COLOR_NEUTRAL = (200, 200, 200)
    COLOR_ACCENT = (255, 200, 100)
    COLOR_WARNING = (100, 100, 255)
    
    @classmethod
    def save(cls, filename: str = "config_backup.json"):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in cls.__dict__.items() 
                      if not k.startswith('_') and k.isupper()}
        
        config_path = Path(filename)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filename}")
    
    @classmethod
    def load(cls, filename: str = "config_backup.json"):
        """Load configuration from JSON file"""
        config_path = Path(filename)
        if not config_path.exists():
            logger.warning(f"Config file {filename} not found, using defaults")
            return
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        logger.info(f"Configuration loaded from {filename}")

CONFIG = Config()

# ============================================================================
# ERROR MESSAGES & TROUBLESHOOTING
# ============================================================================

class ErrorHandler:
    """Centralized error handling with helpful messages"""
    
    @staticmethod
    def camera_error():
        """Display camera error with troubleshooting"""
        print("\n" + "="*70)
        print("❌ CAMERA ERROR - Cannot open camera")
        print("="*70)
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("  1. Check if camera is connected and not in use by another app")
        print("  2. Try closing other applications that might use the camera")
        print("  3. On Linux, check permissions: ls -l /dev/video*")
        print("  4. Try different camera index: python script.py --camera 1")
        print("  5. Verify drivers are installed (dmesg | grep video)")
        print("\n💡 ALTERNATIVE:")
        print("  • Use a video file: python script.py --input video.mp4")
        print("="*70 + "\n")
    
    @staticmethod
    def mediapipe_error():
        """Display MediaPipe initialization error"""
        print("\n" + "="*70)
        print("❌ MEDIAPIPE ERROR - Failed to initialize")
        print("="*70)
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("  1. Reinstall MediaPipe: pip install --upgrade mediapipe")
        print("  2. Check Python version (requires 3.7+): python --version")
        print("  3. Install dependencies: pip install opencv-python numpy")
        print("  4. Try with --complexity 0 for lighter model")
        print("\n💡 SYSTEM REQUIREMENTS:")
        print("  • Python 3.7 or higher")
        print("  • OpenCV 4.5+")
        print("  • NumPy 1.19+")
        print("  • MediaPipe 0.8.9+")
        print("="*70 + "\n")
    
    @staticmethod
    def frame_read_error():
        """Display frame reading error"""
        print("\n⚠️  Frame read error - Camera may have disconnected")
        print("💡 Try: Reconnect camera or restart the application")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_distance(p1, p2) -> float:
    """Calculate 3D distance between two points"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def smooth_value(history: deque, window: int = 10) -> float:
    """Simple moving average with validation"""
    if not history:
        return 0.0
    values = list(history)[-window:]
    return float(np.mean(values))

def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# ============================================================================
# GESTURE DETECTOR (Enhanced with hold time tracking)
# ============================================================================

class SimpleGestureDetector:
    """Enhanced gesture detection with hold time tracking"""
    
    def __init__(self):
        self.gesture_start_times = {}
        self.current_gesture = "neutral"
    
    @staticmethod
    def get_finger_extension(landmarks, tip_idx, pip_idx, mcp_idx) -> bool:
        """Check if finger is extended - VERY FORGIVING"""
        try:
            tip = landmarks.landmark[tip_idx]
            pip = landmarks.landmark[pip_idx]
            mcp = landmarks.landmark[mcp_idx]
            wrist = landmarks.landmark[0]
            
            tip_to_mcp = calculate_distance(tip, mcp)
            pip_to_mcp = calculate_distance(pip, mcp)
            tip_to_wrist = calculate_distance(tip, wrist)
            mcp_to_wrist = calculate_distance(mcp, wrist)
            
            return (tip_to_mcp > pip_to_mcp * 1.0 or 
                    tip_to_wrist > mcp_to_wrist * 0.6)
        except:
            return False
    
    @staticmethod
    def detect_steeple(hand1, hand2) -> float:
        """Steeple - ULTRA FORGIVING: ANY fingertips somewhat close counts!"""
        if not hand1 or not hand2:
            return 0.0
        
        try:
            # Check ALL possible fingertip pairs
            fingertips = [(4, "Thumb"), (8, "Index"), (12, "Middle"), 
                         (16, "Ring"), (20, "Pinky")]
            
            # Find closest fingertip pair
            closest_distance = float('inf')
            close_pairs = 0
            
            for tip_idx, name in fingertips:
                tip1 = hand1.landmark[tip_idx]
                tip2 = hand2.landmark[tip_idx]
                dist = calculate_distance(tip1, tip2)
                
                if dist < closest_distance:
                    closest_distance = dist
                
                # Count how many fingertips are reasonably close
                if dist < CONFIG.STEEPLE_FINGERTIP_DIST * 1.5:  # Even more forgiving!
                    close_pairs += 1
            
            # VERY GENEROUS - accept if ANY fingertips are somewhat close
            if closest_distance < CONFIG.STEEPLE_FINGERTIP_DIST * 1.3:  # 30% more forgiving
                palm1 = hand1.landmark[0]
                palm2 = hand2.landmark[0]
                palm_dist = calculate_distance(palm1, palm2)
                
                # More forgiving palm distance check
                if palm_dist > CONFIG.STEEPLE_PALM_DIST_MIN * 0.5:  # 50% easier
                    # Calculate confidence with heavy boost
                    fingertip_score = 1.0 - (closest_distance / (CONFIG.STEEPLE_FINGERTIP_DIST * 1.3))
                    palm_score = min(palm_dist / 0.10, 1.0)  # Easier palm scoring
                    pair_bonus = min(close_pairs / 3.0, 1.0)  # Bonus for multiple close pairs
                    
                    # Heavily weighted toward fingertip detection
                    confidence = fingertip_score * 0.7 + palm_score * 0.15 + pair_bonus * 0.15
                    
                    # BIG boost for easier detection
                    return min(confidence * 2.0, 1.0)
            
            return 0.0
        except Exception as e:
            logger.debug(f"Steeple detection error: {e}")
            return 0.0
    
    @staticmethod
    def detect_holding_ball(hand1, hand2) -> float:
        """Holding Ball - ULTRA FORGIVING: Like holding a basketball or sphere"""
        if not hand1 or not hand2:
            return 0.0
        
        try:
            # Use a more robust palm center (average of wrist and MCPs) instead of raw wrist
            def palm_center(hand):
                # use wrist(0), index_mcp(5), middle_mcp(9), ring_mcp(13) if available
                pts = []
                for idx in (0, 5, 9, 13):
                    try:
                        pts.append(hand.landmark[idx])
                    except Exception:
                        pass
                if not pts:
                    return hand.landmark[0]
                # average coordinates
                avg = type(hand.landmark[0])( ) if False else None
                # create a simple object with x,y,z
                class P:
                    pass
                p = P()
                p.x = float(np.mean([pt.x for pt in pts]))
                p.y = float(np.mean([pt.y for pt in pts]))
                p.z = float(np.mean([pt.z for pt in pts]))
                return p

            palm1 = palm_center(hand1)
            palm2 = palm_center(hand2)

            # fingertip points
            index1 = hand1.landmark[8]
            index2 = hand2.landmark[8]
            middle1 = hand1.landmark[12]
            middle2 = hand2.landmark[12]

            # Distance between palm centers
            hand_distance = calculate_distance(palm1, palm2)

            # Accept a wider range but be tolerant based on estimated depth (z)
            z_diff = abs(palm1.z - palm2.z)
            # allow larger vertical tilt
            height_diff = abs(palm1.y - palm2.y)
            if height_diff > 0.40:  # allow more tolerance
                return 0.0

            # Index/middle distances
            index_distance = calculate_distance(index1, index2)
            middle_distance = calculate_distance(middle1, middle2)

            # Hands facing each other characteristic: fingertip distances should be smaller than palm distance
            index_close = index_distance < hand_distance * 0.95
            middle_close = middle_distance < hand_distance * 0.95

            # Compute finger curvature per hand to detect cupping (tip pulled toward palm)
            def finger_curvature(hand, tip_idx, pip_idx, mcp_idx):
                try:
                    tip = hand.landmark[tip_idx]
                    pip = hand.landmark[pip_idx]
                    mcp = hand.landmark[mcp_idx]
                    tip_to_mcp = calculate_distance(tip, mcp)
                    pip_to_mcp = calculate_distance(pip, mcp)
                    # ratio < 1 means tip closer to mcp than pip (curled)
                    return tip_to_mcp / (pip_to_mcp + 1e-6)
                except Exception:
                    return 1.0

            idx_curve_l = finger_curvature(hand1, 8, 6, 5)
            idx_curve_r = finger_curvature(hand2, 8, 6, 5)
            mid_curve_l = finger_curvature(hand1, 12, 10, 9)
            mid_curve_r = finger_curvature(hand2, 12, 10, 9)

            # consider finger 'curved' if ratio significantly less than 1.0
            idx_curved_both = (idx_curve_l < 0.95 and idx_curve_r < 0.95)
            mid_curved_both = (mid_curve_l < 0.95 and mid_curve_r < 0.95)

            if not (index_close or middle_close or idx_curved_both or mid_curved_both):
                # not showing any sign of holding
                return 0.0

            # Scoring: blend palm distance (ideal), curvature agreement, and fingertip proximity
            ideal = 0.22
            distance_score = max(0.0, 1.0 - abs(hand_distance - ideal) / 0.30)
            height_score = max(0.0, 1.0 - (height_diff / 0.40))

            # fingertip proximity score: lower is better (closer)
            finger_prox = 1.0 - (min(index_distance, middle_distance) / max(hand_distance, 1e-6))
            finger_prox = np.clip(finger_prox, 0.0, 1.0)

            # curvature agreement score: average of how curved the relevant fingers are
            curve_vals = []
            if idx_curve_l < 2.0 and idx_curve_r < 2.0:
                curve_vals.append(1.0 - (idx_curve_l + idx_curve_r) / 2.0)
            if mid_curve_l < 2.0 and mid_curve_r < 2.0:
                curve_vals.append(1.0 - (mid_curve_l + mid_curve_r) / 2.0)
            curve_score = np.mean(curve_vals) if curve_vals else 0.0

            confidence = distance_score * 0.45 + curve_score * 0.35 + finger_prox * 0.20

            # small boosts
            if idx_curved_both or mid_curved_both:
                confidence = min(confidence + 0.12, 1.0)

            # Debug logging to assist tuning
            logger.debug(f"HoldingBall metrics: hand_d={hand_distance:.3f}, idx_d={index_distance:.3f}, mid_d={middle_distance:.3f}, idx_curve_l={idx_curve_l:.2f}, idx_curve_r={idx_curve_r:.2f}, mid_curve_l={mid_curve_l:.2f}, mid_curve_r={mid_curve_r:.2f}, conf={confidence:.2f}")

            return float(np.clip(confidence, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"Holding ball detection error: {e}")
            return 0.0
    
    @staticmethod
    def detect_open_palm(hand) -> float:
        """Open palm - VERY EASY: Just 2 fingers extended"""
        try:
            fingers_extended = 0
            finger_indices = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
            
            for tip, pip, mcp in finger_indices:
                if SimpleGestureDetector.get_finger_extension(hand, tip, pip, mcp):
                    fingers_extended += 1
            
            if fingers_extended >= CONFIG.OPEN_PALM_MIN_FINGERS:
                confidence = fingers_extended / 4.0
                return min(confidence * 1.3, 1.0)
            
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def detect_palm_down(hand) -> float:
        """Palm down - SUPER EASY: Just check hand orientation"""
        try:
            wrist = hand.landmark[0]
            middle_mcp = hand.landmark[9]
            y_diff = middle_mcp.y - wrist.y
            
            if y_diff > CONFIG.PALM_DOWN_Y_THRESHOLD:
                confidence = min(y_diff / 0.05, 1.0)
                return min(confidence * 1.5, 1.0)
            elif y_diff > 0:
                return 0.5
            
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def detect_pointing(hand) -> float:
        """Pointing - EASY: Index extended, others curled"""
        try:
            index_extended = SimpleGestureDetector.get_finger_extension(hand, 8, 6, 5)
            if not index_extended:
                return 0.0
            
            fingers_curled = 0
            other_fingers = [(12, 10, 9), (16, 14, 13), (20, 18, 17)]
            
            for tip, pip, mcp in other_fingers:
                if not SimpleGestureDetector.get_finger_extension(hand, tip, pip, mcp):
                    fingers_curled += 1
            
            if fingers_curled >= 1:
                confidence = fingers_curled / 3.0
                return min(confidence * 1.4, 1.0)
            
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def detect_the_box(hand1, hand2) -> float:
        """The Box - EASY: Hands separated horizontally"""
        if not hand1 or not hand2:
            return 0.0
        
        try:
            wrist1 = hand1.landmark[0]
            wrist2 = hand2.landmark[0]
            horizontal_dist = abs(wrist1.x - wrist2.x)
            
            if CONFIG.BOX_HAND_DISTANCE_MIN < horizontal_dist < CONFIG.BOX_HAND_DISTANCE_MAX:
                height_diff = abs(wrist1.y - wrist2.y)
                if height_diff < 0.20:
                    confidence = 1.0 - (height_diff / 0.20)
                    return min(confidence * 1.3, 1.0)
            
            return 0.0
        except:
            return 0.0
    
    def detect_gesture(self, hand_landmarks_list, pose_landmarks=None) -> Tuple[str, float, float]:
        """Main gesture detection - returns (gesture_name, confidence, hold_time)"""
        gesture_scores = {}
        current_time = time.time()
        
        if not hand_landmarks_list:
            self.current_gesture = "neutral"
            return "neutral", 0.0, 0.0
        
        first_hand = hand_landmarks_list[0]
        second_hand = hand_landmarks_list[1] if len(hand_landmarks_list) > 1 else None
        
        # Two-hand gestures (PRIORITIZED - check these first!)
        if second_hand:
            steeple_score = self.detect_steeple(first_hand, second_hand)
            ball_score = self.detect_holding_ball(first_hand, second_hand)
            box_score = self.detect_the_box(first_hand, second_hand)
            
            # Give two-hand gestures a boost to prioritize them
            gesture_scores["steeple"] = steeple_score * 1.2 if steeple_score > 0 else 0
            gesture_scores["holding_ball"] = ball_score * 1.2 if ball_score > 0 else 0
            gesture_scores["the_box"] = box_score * 1.2 if box_score > 0 else 0
            
            # Debug logging for two-hand gestures
            if steeple_score > 0.1 or ball_score > 0.1:
                logger.debug(f"Two-hand scores - Steeple: {steeple_score:.2f}, Ball: {ball_score:.2f}")
        
        # Single hand gestures (lower priority)
        gesture_scores["open_palm"] = self.detect_open_palm(first_hand)
        gesture_scores["palm_down"] = self.detect_palm_down(first_hand)
        gesture_scores["pointing"] = self.detect_pointing(first_hand)
        
        # Get best gesture with LOWER threshold for two-hand gestures
        if gesture_scores:
            best_gesture, confidence = max(gesture_scores.items(), key=lambda x: x[1])
            
            # Lower threshold for two-hand gestures
            threshold = 0.15 if best_gesture in ["steeple", "holding_ball", "the_box"] else 0.25
            
            if confidence > threshold:
                # Track gesture hold time
                if best_gesture != self.current_gesture:
                    self.gesture_start_times[best_gesture] = current_time
                    self.current_gesture = best_gesture
                    logger.info(f"Gesture detected: {best_gesture} (confidence: {confidence:.2f})")
                
                hold_time = current_time - self.gesture_start_times.get(best_gesture, current_time)
                return best_gesture, confidence, hold_time
        
        self.current_gesture = "neutral"
        return "neutral", 0.0, 0.0

# ============================================================================
# ENHANCED ANALYZER
# ============================================================================

class EnhancedBodyLanguageAnalyzer:
    """Enhanced analyzer with all new features"""
    
    def __init__(self, frame_skip: int = 1):
        logger.info("Initializing MediaPipe models...")
        
        try:
            self.pose = mp.solutions.pose.Pose(
                min_detection_confidence=CONFIG.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=CONFIG.MIN_TRACKING_CONFIDENCE,
                model_complexity=0
            )
            
            self.hands = mp.solutions.hands.Hands(
                min_detection_confidence=CONFIG.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=CONFIG.MIN_TRACKING_CONFIDENCE,
                max_num_hands=2,
                model_complexity=0
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            ErrorHandler.mediapipe_error()
            raise
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.gesture_detector = SimpleGestureDetector()
        
        # State
        self.posture_history = deque(maxlen=CONFIG.HISTORY_SIZE)
        self.gesture_history = deque(maxlen=CONFIG.HISTORY_SIZE)
        self.gesture_stability_buffer = deque(maxlen=CONFIG.GESTURE_STABILITY_FRAMES)
        self.gesture_transition_buffer = deque(maxlen=CONFIG.GESTURE_TRANSITION_SMOOTHING)
        self.fps_history = deque(maxlen=30)
        
        self.current_gesture = "neutral"
        self.current_confidence = 0.0
        self.current_hold_time = 0.0
        self.frame_count = 0
        self.processed_frame_count = 0
        self.last_time = time.time()
        self.frame_skip = frame_skip
        
        # CSV tracking
        self.csv_data = []
        
        # Help menu state
        self.show_help = False
        # UI toggles
        self.show_pose = True
        self.show_gesture_tips = True
        # Screenshot flash indicator (timestamp)
        self.screenshot_flash_time = 0.0

        logger.info(f"Analyzer initialized (frame_skip={frame_skip})")
    
    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """Analyze frame with frame skipping for performance"""
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        
        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - self.last_time + 1e-6)
        self.fps_history.append(fps)
        self.last_time = current_time
        
        # Frame skipping for performance
        should_process = (self.frame_count % self.frame_skip == 0)
        
        if should_process:
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            pose_results = self.pose.process(rgb_frame)
            posture_score = self._analyze_posture(pose_results, frame)
            
            # Detect hands and gestures
            hand_results = self.hands.process(rgb_frame)
            gesture, confidence, hold_time = self._analyze_gestures(hand_results, frame)
            
            self.processed_frame_count += 1
            
            # Log to CSV data
            self._log_frame_data(current_time, posture_score, gesture, confidence, hold_time)
        else:
            # Use previous results for skipped frames
            gesture = self.current_gesture
            confidence = self.current_confidence
            hold_time = self.current_hold_time
            posture_score = smooth_value(self.posture_history) if self.posture_history else 0.0
        
        # Always draw UI
        self._draw_ui(frame, posture_score, gesture, confidence, hold_time)
        
        # Draw help menu if active
        if self.show_help:
            self._draw_help_menu(frame)
        
        self.frame_count += 1
        return frame
    
    def _analyze_posture(self, pose_results, frame: np.ndarray) -> float:
        """Improved posture analysis using multiple geometric cues.

        Combines: torso verticality (shoulder-to-hip vector angle), shoulder
        levelness, head alignment and landmark visibility into a single score
        between 0.0 and 1.0. More robust to missing landmarks and noisy frames.
        """
        if not pose_results.pose_landmarks:
            return 0.0

        try:
            landmarks = pose_results.pose_landmarks.landmark

            # Key indices
            L_SHO = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            R_SHO = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
            L_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
            R_HIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
            NOSE = mp.solutions.pose.PoseLandmark.NOSE.value

            # Safely fetch landmarks
            def get(idx):
                try:
                    return landmarks[idx]
                except Exception:
                    return None

            ls = get(L_SHO)
            rs = get(R_SHO)
            lh = get(L_HIP)
            rh = get(R_HIP)
            nose = get(NOSE)

            if not (ls and rs and lh and rh):
                return 0.0

            # Midpoints
            shoulder_mid_x = (ls.x + rs.x) / 2.0
            shoulder_mid_y = (ls.y + rs.y) / 2.0
            hip_mid_x = (lh.x + rh.x) / 2.0
            hip_mid_y = (lh.y + rh.y) / 2.0

            # Horizontal alignment score (existing idea, normalized)
            horiz_diff = abs(shoulder_mid_x - hip_mid_x)
            horiz_score = np.clip(1.0 - (horiz_diff / max(CONFIG.UPRIGHT_THRESHOLD * 4.0, 1e-6)), 0.0, 1.0)

            # Torso vector angle relative to vertical. A perfectly vertical torso -> angle ~ 0
            dx = shoulder_mid_x - hip_mid_x
            dy = shoulder_mid_y - hip_mid_y
            torso_angle = abs(np.arctan2(dx, dy))  # radians; dx small -> angle small
            # Map angle to score: angle <= threshold -> good (1.0), larger angles degrade
            angle_norm = np.clip(1.0 - (torso_angle / (np.pi/2)), 0.0, 1.0)
            angle_score = np.clip((angle_norm / (CONFIG.UPRIGHT_ANGLE_THRESHOLD / (np.pi/2))) , 0.0, 1.0)

            # Shoulder levelness: small y difference between shoulders = level
            shoulder_y_diff = abs(ls.y - rs.y)
            shoulder_level_score = np.clip(1.0 - (shoulder_y_diff / 0.15), 0.0, 1.0)

            # Head alignment: nose should be roughly above shoulder midpoint horizontally
            head_score = 0.5
            if nose:
                head_x_diff = abs(nose.x - shoulder_mid_x)
                head_y_diff = shoulder_mid_y - nose.y  # nose should be above shoulders (positive)
                head_x_score = np.clip(1.0 - (head_x_diff / 0.20), 0.0, 1.0)
                head_y_score = 1.0 if head_y_diff > -0.05 else np.clip(1.0 + head_y_diff / 0.20, 0.0, 1.0)
                head_score = 0.6 * head_x_score + 0.4 * head_y_score

            # Depth coherence: if z difference between shoulders and hips is large, user may be leaning forward/back
            try:
                z_sh_mid = (ls.z + rs.z) / 2.0
                z_hp_mid = (lh.z + rh.z) / 2.0
                z_diff = abs(z_sh_mid - z_hp_mid)
                z_score = np.clip(1.0 - (z_diff / 0.3), 0.0, 1.0)
            except Exception:
                z_score = 0.8

            # Visibility weighting: use pose landmark visibility if available to down-weight unreliable frames
            vis_vals = []
            for lm in (ls, rs, lh, rh, nose):
                try:
                    if lm and hasattr(lm, 'visibility'):
                        vis_vals.append(lm.visibility)
                except Exception:
                    pass
            vis_weight = np.mean(vis_vals) if vis_vals else 0.9

            # Combine the scores with tuned weights
            combined = (
                0.30 * horiz_score +
                0.30 * angle_score +
                0.15 * shoulder_level_score +
                0.15 * head_score +
                0.10 * z_score
            )

            # Weight by visibility
            final_score = np.clip(combined * vis_weight, 0.0, 1.0)

            # Translate into a coarse band similar to previous behavior
            if final_score > 0.75:
                score = 1.0
            elif final_score > 0.45:
                score = 0.5
            else:
                score = final_score * 0.8

            # Save history and optionally draw pose
            self.posture_history.append(score)

            if getattr(self, 'show_pose', True):
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

            return float(score)
        except Exception as e:
            logger.warning(f"Posture analysis error: {e}")
            return 0.0
    
    def _analyze_gestures(self, hand_results, frame: np.ndarray) -> Tuple[str, float, float]:
        """Gesture analysis with smoothing"""
        hand_landmarks_list = []
        
        if hand_results.multi_hand_landmarks:
            hand_landmarks_list = hand_results.multi_hand_landmarks
            
            for hand_landmarks in hand_landmarks_list:
                if getattr(self, 'show_pose', True):
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
        
        gesture, confidence, hold_time = self.gesture_detector.detect_gesture(hand_landmarks_list)
        
        # Stability filtering
        self.gesture_stability_buffer.append((gesture, confidence))
        
        if len(self.gesture_stability_buffer) == CONFIG.GESTURE_STABILITY_FRAMES:
            gestures = [g for g, c in self.gesture_stability_buffer]
            most_common = max(set(gestures), key=gestures.count)
            
            if gestures.count(most_common) >= CONFIG.GESTURE_STABILITY_FRAMES - 1:
                avg_conf = np.mean([c for g, c in self.gesture_stability_buffer if g == most_common])
                
                # Transition smoothing
                self.gesture_transition_buffer.append(most_common)
                if len(set(self.gesture_transition_buffer)) == 1:
                    self.current_gesture = most_common
                    self.current_confidence = avg_conf
                    self.current_hold_time = hold_time
                    self.gesture_history.append(most_common)
        
        return self.current_gesture, self.current_confidence, self.current_hold_time
    
    def _log_frame_data(self, timestamp: float, posture: float, gesture: str, 
                       confidence: float, hold_time: float):
        """Log frame data for CSV export"""
        self.csv_data.append({
            'timestamp': timestamp - self.start_time,
            'frame': self.frame_count,
            'posture_score': posture,
            'gesture': gesture,
            'gesture_confidence': confidence,
            'gesture_hold_time': hold_time,
            'fps': smooth_value(self.fps_history, 10)
        })
    
    def _draw_ui(self, frame: np.ndarray, posture_score: float, gesture: str, 
                 confidence: float, hold_time: float):
        """Enhanced UI with better colors and hold time display"""
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Top panel - Gesture Display with hold time
        if gesture != "neutral" and confidence > 0.25:
            panel_h = 100
            cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            gesture_text = gesture.replace("_", " ").upper()
            hold_text = f"Hold: {hold_time:.1f}s"
            
            # Gesture name
            cv2.putText(frame, gesture_text, (20, 45), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, CONFIG.COLOR_GOOD, 3, cv2.LINE_AA)
            
            # (confidence percent removed by request)
            
            # Hold time
            cv2.putText(frame, hold_text, (w - 200, 45), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, CONFIG.COLOR_MEDIUM, 2, cv2.LINE_AA)
        
        # Bottom left - Enhanced metrics
        metrics_y = h - 150
        self._draw_metric_bar(frame, 20, metrics_y, "POSTURE", 
                             smooth_value(self.posture_history), CONFIG.COLOR_MEDIUM)
        
        metrics_y += 50
        gesture_score = len([g for g in self.gesture_history if g != "neutral"]) / max(len(self.gesture_history), 1)
        self._draw_metric_bar(frame, 20, metrics_y, "GESTURES", 
                             gesture_score, CONFIG.COLOR_LOW)
        
        # Bottom right - Info
        fps = smooth_value(self.fps_history, 10)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, h - 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG.COLOR_NEUTRAL, 2, cv2.LINE_AA)
        
        if self.frame_skip > 1:
            cv2.putText(frame, f"Skip: {self.frame_skip}x", (w - 120, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.COLOR_ACCENT, 1, cv2.LINE_AA)
        
        cv2.putText(frame, "H:Help  Q:Quit  S:Save", (w - 230, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.COLOR_NEUTRAL, 1, cv2.LINE_AA)
        
        # Timer display
        if hasattr(self, 'start_time'):
            elapsed = time.time() - self.start_time
            remaining = max(0, CONFIG.ANALYSIS_DURATION_SECONDS - elapsed)
            timer_text = f"Time: {format_duration(remaining)}"
            cv2.putText(frame, timer_text, (w - 180, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, CONFIG.COLOR_ACCENT, 2, cv2.LINE_AA)

        # Top progress bar for elapsed time
        if hasattr(self, 'start_time') and CONFIG.ANALYSIS_DURATION_SECONDS > 0:
            elapsed = time.time() - self.start_time
            frac = min(max(elapsed / CONFIG.ANALYSIS_DURATION_SECONDS, 0.0), 1.0)
            bar_h = 8
            filled_w = int(w * frac)

            # background bar
            cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 35), -1)
            # filled portion
            if filled_w > 0:
                cv2.rectangle(frame, (0, 0), (filled_w, bar_h), CONFIG.COLOR_ACCENT, -1)
            # percent text
            perc_text = f"{int(frac * 100)}%"
            cv2.putText(frame, perc_text, (10, bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.COLOR_NEUTRAL, 1, cv2.LINE_AA)

        # Screenshot saved badge (brief flash)
        try:
            if hasattr(self, 'screenshot_flash_time') and (time.time() - self.screenshot_flash_time) < 1.2:
                badge_w = 210
                badge_h = 34
                bx = w - badge_w - 10
                by = 10
                overlay = frame.copy()
                cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), CONFIG.COLOR_GOOD, 2)
                cv2.putText(frame, "Screenshot saved", (bx + 12, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG.COLOR_GOOD, 2, cv2.LINE_AA)
        except Exception:
            pass

        # Gesture tips overlay (small, contextual)
        if getattr(self, 'show_gesture_tips', True):
            try:
                gesture_descriptions = {
                    'steeple': 'Steeple: fingertips touch, palms apart',
                    'holding_ball': 'Holding Ball: cup your hands like a ball',
                    'open_palm': 'Open Palm: show palm, 2+ fingers',
                    'palm_down': 'Palm Down: rotate palm downward',
                    'pointing': 'Pointing: extend index finger',
                    'the_box': 'The Box: hands separated like a box'
                }

                if gesture != 'neutral' and gesture in gesture_descriptions and confidence > 0.15:
                    tip = gesture_descriptions.get(gesture, '')
                    tip_w = 520
                    tip_h = 34
                    tx = (w - tip_w) // 2
                    ty = h - 200
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (tx, ty), (tx + tip_w, ty + tip_h), (30, 30, 30), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, tip, (tx + 12, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG.COLOR_NEUTRAL, 1, cv2.LINE_AA)
            except Exception:
                pass
    
    def _draw_metric_bar(self, frame: np.ndarray, x: int, y: int, 
                        label: str, value: float, color: Tuple[int, int, int]):
        """Draw enhanced metric bar"""
        bar_w = 200
        bar_h = 25
        
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.COLOR_NEUTRAL, 1, cv2.LINE_AA)
        
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (30, 30, 40), -1)
        
        filled = int(bar_w * value)
        if filled > 0:
            cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)
        
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 110), 2)
        
        cv2.putText(frame, f"{int(value * 100)}%", (x + bar_w + 10, y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def _draw_help_menu(self, frame: np.ndarray):
        """Draw keyboard help menu"""
        h, w, _ = frame.shape
        menu_w = 600
        menu_h = 450
        menu_x = (w - menu_w) // 2
        menu_y = (h - menu_h) // 2
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), 
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Border
        cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), 
                     CONFIG.COLOR_ACCENT, 3)
        
        # Title
        title_y = menu_y + 40
        cv2.putText(frame, "KEYBOARD SHORTCUTS", (menu_x + 160, title_y), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, CONFIG.COLOR_ACCENT, 2, cv2.LINE_AA)
        
        # Help items
        help_items = [
            ("", ""),  # Spacer
            ("H", "Toggle this help menu"),
            ("Q", "Quit application"),
            ("S", "Save screenshot"),
            ("P", "Toggle pose visualization"),
            ("G", "Toggle gesture tips"),
            ("R", "Reset statistics"),
            ("SPACE", "Pause/Resume analysis"),
            ("", ""),  # Spacer
            ("GESTURES:", ""),
            ("Steeple", "Touch fingertips, palms apart"),
            ("Holding Ball", "Cup hands like basketball"),
            ("Open Palm", "Show palm, 2+ fingers"),
            ("Palm Down", "Orient palm downward"),
            ("Pointing", "Point with index finger"),
            ("The Box", "Hands apart horizontally"),
        ]
        
        text_y = title_y + 50
        for key, desc in help_items:
            if key == "":
                text_y += 15
                continue
            
            if key == "GESTURES:":
                cv2.putText(frame, key, (menu_x + 30, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG.COLOR_GOOD, 2, cv2.LINE_AA)
            elif key in ["Steeple", "Holding Ball", "Open Palm", "Palm Down", "Pointing", "The Box"]:
                cv2.putText(frame, f"• {key}:", (menu_x + 50, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.COLOR_NEUTRAL, 1, cv2.LINE_AA)
                cv2.putText(frame, desc, (menu_x + 200, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.COLOR_MEDIUM, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, key, (menu_x + 30, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG.COLOR_ACCENT, 2, cv2.LINE_AA)
                cv2.putText(frame, desc, (menu_x + 150, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG.COLOR_NEUTRAL, 1, cv2.LINE_AA)
            
            text_y += 35
        
        # Footer
        cv2.putText(frame, "Press H again to close", (menu_x + 200, menu_y + menu_h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.COLOR_NEUTRAL, 1, cv2.LINE_AA)
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """Export session data to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"body_language_report_{timestamp}.csv"
        
        csv_path = Path(filename)
        
        try:
            with open(csv_path, 'w', newline='') as f:
                if self.csv_data:
                    writer = csv.DictWriter(f, fieldnames=self.csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.csv_data)
            
            logger.info(f"CSV report exported to {filename}")
            return str(csv_path)
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return ""

    def export_report(self, csv_file: Optional[str] = None, screenshot_count: int = 0, video_file: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Export a human-readable session summary report to a text file."""
        stats = self.get_summary_stats()

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"body_language_summary_{timestamp}.txt"

        report_path = Path(filename)

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SESSION SUMMARY\n")
                f.write("=" * 50 + "\n")
                f.write(f"Duration:         {format_duration(stats.get('duration', 0))}\n")
                f.write(f"Total Frames:     {stats.get('total_frames', 0)}\n")
                f.write(f"Processed Frames: {stats.get('processed_frames', 0)}\n")
                f.write(f"Avg FPS:          {stats.get('avg_fps', 0):.1f}\n")
                f.write(f"Avg Posture:      {stats.get('avg_posture', 0):.2f} ({int(stats.get('avg_posture', 0)*100)}%)\n")
                f.write(f"Gesture Activity: {stats.get('gesture_activity', 0):.2f} ({int(stats.get('gesture_activity', 0)*100)}%)\n")
                f.write("\nGesture breakdown:\n")

                gesture_counts = stats.get('gesture_counts', {})
                if gesture_counts:
                    for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / max(stats.get('total_frames', 1), 1)) * 100
                        gesture_name = gesture.replace("_", " ").title()
                        f.write(f"  - {gesture_name:15s} {count:5d} frames ({percentage:.1f}%)\n")
                else:
                    f.write("  None detected\n")

                f.write("\nArtifacts:\n")
                if csv_file:
                    f.write(f"  CSV report: {csv_file}\n")
                else:
                    f.write("  CSV report: (none)\n")

                if video_file:
                    f.write(f"  Session video: {video_file}\n")
                else:
                    f.write("  Session video: (none)\n")

                f.write(f"  Screenshots saved: {screenshot_count}\n")
                f.write("\nConfiguration:\n")
                try:
                    # include selected config values for traceability
                    cfg_items = ['MIN_DETECTION_CONFIDENCE', 'MIN_TRACKING_CONFIDENCE', 'FRAME_SKIP', 'ANALYSIS_DURATION_SECONDS']
                    for k in cfg_items:
                        f.write(f"  {k}: {getattr(CONFIG, k, 'n/a')}\n")
                except Exception:
                    pass

            logger.info(f"Summary report exported to {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Failed to export summary report: {e}")
            return ""
    
    def get_summary_stats(self) -> Dict:
        """Generate session summary statistics"""
        return {
            'total_frames': self.frame_count,
            'processed_frames': self.processed_frame_count,
            'avg_fps': smooth_value(self.fps_history, 30),
            'avg_posture': smooth_value(self.posture_history, len(self.posture_history)),
            'gesture_activity': len([g for g in self.gesture_history if g != "neutral"]) / max(len(self.gesture_history), 1),
            'gesture_counts': {gesture: self.gesture_history.count(gesture) 
                             for gesture in set(self.gesture_history) if gesture != "neutral"},
            'duration': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.pose.close()
        self.hands.close()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Body Language Analyzer with MediaPipe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with default settings
  %(prog)s --duration 300           # Run for 5 minutes
  %(prog)s --camera 1               # Use camera index 1
  %(prog)s --skip 2                 # Process every 2nd frame (2x faster)
  %(prog)s --input video.mp4        # Analyze video file
  %(prog)s --config my_config.json  # Load custom configuration
        """
    )
    
    parser.add_argument('--duration', type=int, default=180,
                       help='Analysis duration in seconds (default: 180)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Frame skip factor for performance (1=no skip, 2=2x faster)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input video file instead of camera')
    parser.add_argument('--config', type=str, default=None,
                       help='Load configuration from JSON file')
    parser.add_argument('--save-config', action='store_true',
                       help='Save current configuration to file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV filename')
    parser.add_argument('--save-video', action='store_true',
                       help='Save session video to file')
    parser.add_argument('--video-out', type=str, default=None,
                       help='Output video filename (optional)')
    
    return parser.parse_args()

def main():
    """Main function with enhanced error handling and features"""
    args = parse_arguments()
    
    # Load configuration if specified
    if args.config:
        CONFIG.load(args.config)
    
    # Save configuration if requested
    if args.save_config:
        CONFIG.save()
        logger.info("Configuration saved. Exiting.")
        return
    
    # Update duration from args
    CONFIG.ANALYSIS_DURATION_SECONDS = args.duration
    
    # Print banner
    print("\n" + "="*70)
    print("ENHANCED BODY LANGUAGE ANALYZER - SUPER FORGIVING MODE")
    print("="*70)
    print("\n🎯 GESTURES TO TRY (All Very Easy!):")
    print("  • Steeple       - Touch ANY fingertips together, palms apart")
    print("  • Holding Ball  - Cup hands like holding a basketball")
    print("  • Open Palm     - Show palm with 2+ fingers extended")
    print("  • Palm Down     - Orient palm downward")
    print("  • Pointing      - Point with index finger")
    print("  • The Box       - Hold hands apart like a box")
    print("\n💡 NEW FEATURES:")
    print("  • 2x Performance Boost with frame skipping")
    print("  • Gesture hold time display")
    print("  • CSV report export at end")
    print("  • Keyboard help menu (press H)")
    print("  • Better error messages")
    print("  • Enhanced color scheme")
    print("\n⏱️  DURATION:", format_duration(args.duration))
    if args.skip > 1:
        print(f"⚡ PERFORMANCE: {args.skip}x frame skip enabled")
    print("\n⌨️  CONTROLS:")
    print("  H - Help menu")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("\nStarting", "video analysis..." if args.input else "camera...")
    
    # Open video source
    if args.input:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {args.input}")
            print(f"\n❌ Error: Cannot open video file '{args.input}'")
            print("💡 Check if file exists and is a valid video format")
            return
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {args.camera}")
            ErrorHandler.camera_error()
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize analyzer
    try:
        analyzer = EnhancedBodyLanguageAnalyzer(frame_skip=args.skip)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        cap.release()
        return
    
    screenshot_count = 0
    start_time = time.time()
    
    logger.info("Analysis started")
    print(f"\n✅ Ready! Analysis duration: {format_duration(args.duration)}")
    if args.skip > 1:
        print(f"⚡ Performance mode: Processing every {args.skip} frames\n")

    # Create named window and try to set a friendly title (best-effort)
    window_name = 'Enhanced Body Language Analyzer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        # setWindowTitle may not be available in all builds; ignore errors
        cv2.setWindowTitle(window_name, 'Body Language Analyzer - Enhanced UI')
    except Exception:
        pass
    
    # Prepare video writer if requested
    video_writer = None
    video_file = None
    if args.save_video:
        # determine output filename
        if args.video_out:
            video_file = args.video_out
        else:
            video_file = f"session_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        # try to get FPS and frame size from capture
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
            logger.info(f"Video writer initialized: {video_file} ({width}x{height}@{fps:.1f})")
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            video_writer = None

    try:
        while True:
            # Check if time is up
            elapsed_time = time.time() - start_time
            if elapsed_time >= CONFIG.ANALYSIS_DURATION_SECONDS:
                logger.info(f"Time limit reached: {elapsed_time:.1f}s")
                print(f"\n⏰ Time's up! {format_duration(elapsed_time)} completed.")
                break
            
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                ErrorHandler.frame_read_error()
                break
            
            # Analyze frame
            analyzed_frame = analyzer.analyze_frame(frame)
            
            # Display
            cv2.imshow(window_name, analyzed_frame)

            # Write frame to video if requested
            if video_writer is not None:
                try:
                    video_writer.write(analyzed_frame)
                except Exception as e:
                    logger.debug(f"Failed to write video frame: {e}")
            
            # Progress update every 10 seconds
            if analyzer.frame_count % 300 == 0 and analyzer.frame_count > 0:
                stats = analyzer.get_summary_stats()
                logger.info(f"Progress: {format_duration(elapsed_time)} | "
                          f"FPS: {stats['avg_fps']:.1f} | "
                          f"Gesture: {analyzer.current_gesture}")
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                logger.info("User quit")
                print("\n👋 Quitting...")
                break
            elif key == ord('h') or key == ord('H'):
                analyzer.show_help = not analyzer.show_help
                logger.info(f"Help menu: {'shown' if analyzer.show_help else 'hidden'}")
            elif key == ord('p') or key == ord('P'):
                analyzer.show_pose = not analyzer.show_pose
                logger.info(f"Pose visualization: {'on' if analyzer.show_pose else 'off'}")
            elif key == ord('g') or key == ord('G'):
                analyzer.show_gesture_tips = not analyzer.show_gesture_tips
                logger.info(f"Gesture tips: {'on' if analyzer.show_gesture_tips else 'off'}")
            elif key == ord('s') or key == ord('S'):
                # Save screenshot with timestamp and flash badge
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_count += 1
                filename = f"screenshot_{ts}.png"
                cv2.imwrite(filename, analyzed_frame)
                analyzer.screenshot_flash_time = time.time()
                logger.info(f"Screenshot saved: {filename}")
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        analyzer.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        
        # Get summary statistics
        stats = analyzer.get_summary_stats()
        
        # Export CSV report
        csv_file = analyzer.export_csv(args.output)
        # Export human-readable session summary (include video path if any)
        report_file = analyzer.export_report(csv_file, screenshot_count, video_file)

        # Print summary
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Duration:         {format_duration(stats['duration'])}")
        print(f"Total Frames:     {stats['total_frames']}")
        print(f"Processed Frames: {stats['processed_frames']}")
        print(f"Avg FPS:          {stats['avg_fps']:.1f}")
        print(f"Avg Posture:      {stats['avg_posture']:.2f} ({int(stats['avg_posture']*100)}%)")
        print(f"Gesture Activity: {stats['gesture_activity']:.2f} ({int(stats['gesture_activity']*100)}%)")

        if stats['gesture_counts']:
            print("\nGesture Breakdown:")
            for gesture, count in sorted(stats['gesture_counts'].items(), 
                                        key=lambda x: x[1], reverse=True):
                gesture_name = gesture.replace("_", " ").title()
                percentage = (count / stats['total_frames']) * 100
                print(f"  • {gesture_name:15s} {count:5d} frames ({percentage:.1f}%)")

        if screenshot_count > 0:
            print(f"\nScreenshots:      {screenshot_count}")

        if csv_file:
            print(f"\n📊 CSV Report:    {csv_file}")
        if report_file:
            print(f"\n📝 Summary Report: {report_file}")
        if video_file:
            print(f"\n🎬 Session video:  {video_file}")

        print("\n✅ Analysis complete!")
        print("="*70 + "\n")

        logger.info("Session completed successfully")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
