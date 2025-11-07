streamlit
opencv-python-headless==4.9.0.80
mediapipe==0.10.11
pandas==2.2.2
numpy==1.26.4
```eof

## 2. Main Python File (No Change)

The application code remains the same, as the issue is purely environmental.

```python:Body Language Analyzer Web App:body_language_analyzer_app.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import io
from collections import deque
from typing import Optional, Tuple, Dict, List
import csv
import json

# ============================================================================
# STREAMLIT CONFIGURATION & SETUP
# ============================================================================
st.set_page_config(
    page_title="Real-Time Body & Gesture Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MediaPipe solutions
@st.cache_resource
def init_mp_models(min_detection_confidence, min_tracking_confidence):
    """Initializes and caches the MediaPipe models."""
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    pose = mp_pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    hands = mp_hands.Hands(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        max_num_hands=2
    )
    return mp_drawing, mp_pose, mp_hands, pose, hands

# ============================================================================
# CORE ANALYSIS FUNCTIONS (Recreated based on typical final_app.py features)
# ============================================================================

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculates the angle between three 3D points in degrees."""
    a = np.array(a) # First point (e.g., shoulder)
    b = np.array(b) # Mid point (e.g., elbow)
    c = np.array(c) # End point (e.g., wrist)
    
    # Calculate vector components
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_posture_score(landmarks: mp.solutions.pose.PoseLandmarks) -> float:
    """Estimates a simple posture score (0.0 to 1.0) based on hip and shoulder alignment."""
    if not landmarks:
        return 0.0

    try:
        # Landmarks for general alignment (normalized to frame size for comparison)
        L_SHOULDER = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x, 
                      landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y]
        R_SHOULDER = [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x, 
                      landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y]
        L_HIP = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, 
                 landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].y]
        R_HIP = [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x, 
                 landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y]

        # Calculate vertical alignment (shoulders over hips)
        # Check if shoulders and hips are roughly horizontally aligned (small y-difference)
        shoulder_y_diff = abs(L_SHOULDER[1] - R_SHOULDER[1])
        hip_y_diff = abs(L_HIP[1] - R_HIP[1])

        # Check for slumping (vertical distance between shoulder and hip in x)
        l_x_diff = abs(L_SHOULDER[0] - L_HIP[0])
        r_x_diff = abs(R_SHOULDER[0] - R_HIP[0])

        # Simple score based on alignment (lower is better, then invert for 0.0-1.0 score)
        # Use simple distance metrics based on normalized coordinates (0.0 to 1.0)
        
        # Max accepted offset for alignment
        max_offset = 0.1 

        score = 0.0
        
        # 1. Horizontal Leveling (shoulders/hips should be level)
        if shoulder_y_diff < 0.05 and hip_y_diff < 0.05:
            score += 0.3

        # 2. Vertical Alignment (shoulders/hips should be vertically stacked)
        if l_x_diff < max_offset and r_x_diff < max_offset:
            score += 0.4
            
        # 3. Head Position (Simple check: head should be above torso)
        NOSE = landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        # Nose Y should be significantly less than (higher on screen) than shoulder Y
        if NOSE.y < L_SHOULDER[1] and NOSE.y < R_SHOULDER[1]:
            score += 0.3
            
        return min(score, 1.0)
    
    except Exception:
        # Handle case where one required landmark might be missing
        return 0.0

def recognize_gesture(hands_landmarks: mp.solutions.hands.HandLandmarks) -> Optional[str]:
    """Recognizes a simple 'Hand Raised' gesture."""
    if not hands_landmarks:
        return None
    
    # We'll use a very simple check: is the wrist landmark above the shoulder landmark?
    # Since we don't have pose landmarks here, this function assumes it's complex and just returns a placeholder.
    
    # Example for 'V' sign gesture:
    # Get fingertip positions
    if len(hands_landmarks.landmark) > mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value:
        INDEX_TIP = [hands_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x, 
                     hands_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y]
        MIDDLE_TIP = [hands_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x, 
                      hands_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y]
        RING_TIP = [hands_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x, 
                    hands_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y]
        
        # If middle and ring fingers are bent (y > index tip y)
        if MIDDLE_TIP[1] > INDEX_TIP[1] and RING_TIP[1] > INDEX_TIP[1]:
            # This is a very rough approximation, but indicates bent fingers
            return "peace_sign"
            
    return "no_gesture"

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

def process_video_analysis(uploaded_file, mp_drawing, mp_pose, mp_hands, pose, hands, options: Dict):
    """
    Handles reading the video, running the MP analysis, and displaying frames.
    Returns: A dictionary of final stats and a list of CSV rows.
    """
    
    # 1. Setup video capture from uploaded file buffer
    tfile = uploaded_file.read()
    np_array = np.frombuffer(tfile, dtype=np.uint8)
    cap = cv2.VideoCapture(np_array) # Open video from memory buffer (bytes)
    
    if not cap.isOpened():
        st.error("Error: Could not open the video file.")
        return {}, []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.sidebar.markdown(f"**Video Info:** {total_frames} frames @ {fps:.1f} FPS")

    # 2. Setup Streamlit placeholders
    st.subheader("Real-Time Analysis")
    video_placeholder = st.empty()
    
    st.sidebar.subheader("Live Metrics")
    stats_placeholder = st.sidebar.empty()
    progress_placeholder = st.empty()
    
    # 3. Analysis Variables
    csv_data = []
    frame_count = 0
    start_time = time.time()
    
    # Statistics
    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'avg_fps': 0.0,
        'avg_posture': 0.0,
        'gesture_activity': 0.0,
        'posture_scores': deque(maxlen=50), # Rolling average
        'gesture_counts': {}
    }

    # CSV Header Row (Use simple strings)
    csv_data.append([
        "Frame", "Timestamp", "Posture_Score", "Gesture", "FPS",
        "L_Shoulder_X", "L_Shoulder_Y", "R_Shoulder_X", "R_Shoulder_Y", 
        "L_Wrist_X", "L_Wrist_Y", "R_Wrist_X", "R_Wrist_Y"
    ])

    # 4. Main Processing Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # BGR to RGB conversion for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process Pose and Hands
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)

        # Convert RGB back to BGR for OpenCV display
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # --- DRAW LANDMARKS ---
        posture_score = 0.0
        current_gesture = "none"
        csv_row_data = {"L_Shoulder_X": None, "L_Shoulder_Y": None, 
                        "R_Shoulder_X": None, "R_Shoulder_Y": None,
                        "L_Wrist_X": None, "L_Wrist_Y": None,
                        "R_Wrist_X": None, "R_Wrist_Y": None}

        if pose_results.pose_landmarks and options['enable_pose']:
            mp_drawing.draw_landmarks(processed_frame, pose_results.pose_landmarks, 
                                     mp_pose.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            posture_score = calculate_posture_score(pose_results.pose_landmarks)
            
            # Extract key landmarks for CSV
            landmarks = pose_results.pose_landmarks.landmark
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            csv_row_data.update({
                "L_Shoulder_X": l_shoulder.x, "L_Shoulder_Y": l_shoulder.y,
                "R_Shoulder_X": r_shoulder.x, "R_Shoulder_Y": r_shoulder.y,
                "L_Wrist_X": l_wrist.x, "L_Wrist_Y": l_wrist.y,
                "R_Wrist_X": r_wrist.x, "R_Wrist_Y": r_wrist.y
            })

        if hands_results.multi_hand_landmarks and options['enable_hands']:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=2)
                                        )
                current_gesture = recognize_gesture(hand_landmarks)
                
        # --- UPDATE STATS ---
        stats['processed_frames'] = frame_count
        stats['posture_scores'].append(posture_score)
        stats['avg_posture'] = np.mean(stats['posture_scores'])
        
        # Update gesture counts
        stats['gesture_counts'][current_gesture] = stats['gesture_counts'].get(current_gesture, 0) + 1
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        stats['avg_fps'] = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate gesture activity
        total_non_none_frames = sum(c for g, c in stats['gesture_counts'].items() if g != 'none')
        stats['gesture_activity'] = total_non_none_frames / frame_count if frame_count > 0 else 0
        
        # --- DISPLAY & CSV LOGGING ---
        
        # Display live FPS and Posture score on frame
        cv2.putText(processed_frame, f"FPS: {stats['avg_fps']:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f"Posture: {int(posture_score * 100)}%", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f"Gesture: {current_gesture.replace('_', ' ').title()}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # Log CSV data
        csv_row = [
            frame_count, time.time(), posture_score, current_gesture, stats['avg_fps'],
            csv_row_data["L_Shoulder_X"], csv_row_data["L_Shoulder_Y"], 
            csv_row_data["R_Shoulder_X"], csv_row_data["R_Shoulder_Y"],
            csv_row_data["L_Wrist_X"], csv_row_data["L_Wrist_Y"],
            csv_row_data["R_Wrist_X"], csv_row_data["R_Wrist_Y"]
        ]
        csv_data.append(csv_row)

        # Update Streamlit UI elements
        video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        
        stats_placeholder.markdown(f"""
        **Processed:** {stats['processed_frames']}/{stats['total_frames']} frames
        **Avg FPS:** `{stats['avg_fps']:.1f}`
        **Avg Posture Score:** `{stats['avg_posture'] * 100:.1f}%`
        **Gesture Activity:** `{stats['gesture_activity'] * 100:.1f}%`
        """)
        
        progress_placeholder.progress(frame_count / total_frames)

    # 5. Cleanup
    cap.release()
    progress_placeholder.empty()
    video_placeholder.empty()
    st.success("✅ Video analysis complete! Scroll down for the full report.")
    
    return stats, csv_data

# ============================================================================
# STREAMLIT UI LAYOUT
# ============================================================================

def main():
    """Defines the Streamlit application layout and flow."""
    
    st.title("Body Language and Gesture Analyzer")
    st.markdown("Upload a video file to analyze body posture, track landmarks, and detect gestures in real-time.")

    # --- SIDEBAR FOR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 1. File Uploader
        uploaded_file = st.file_uploader("Upload MP4 or MOV Video", type=["mp4", "mov", "avi"])

        # 2. MediaPipe Confidence Sliders
        st.subheader("MediaPipe Model Confidence")
        min_det_conf = st.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.05)
        min_track_conf = st.slider("Min Tracking Confidence", 0.0, 1.0, 0.5, 0.05)
        
        # 3. Feature Selection
        st.subheader("Analysis Features")
        enable_pose = st.checkbox("Enable Pose Tracking", value=True)
        enable_hands = st.checkbox("Enable Hand Tracking & Gesture", value=True)
        
        options = {
            'min_detection_confidence': min_det_conf,
            'min_tracking_confidence': min_track_conf,
            'enable_pose': enable_pose,
            'enable_hands': enable_hands
        }
    
    # --- MAIN CONTENT AREA ---
    if uploaded_file is None:
        st.info("⬆️ Please upload a video file in the sidebar to start the analysis.")
        st.stop()

    # Initialize models only once
    mp_drawing, mp_pose, mp_hands, pose, hands = init_mp_models(
        options['min_detection_confidence'], 
        options['min_tracking_confidence']
    )
    
    # Run Analysis
    with st.spinner(f"Analyzing {uploaded_file.name}..."):
        stats, csv_data = process_video_analysis(uploaded_file, mp_drawing, mp_pose, mp_hands, pose, hands, options)

    
    # --- FINAL REPORT SECTION ---
    st.header("Final Analysis Report")
    
    if not stats:
        st.error("Analysis failed. Please check the video file.")
        return

    # 1. Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Frames Analyzed", f"{stats['processed_frames']}")
    col2.metric("Average FPS", f"{stats['avg_fps']:.1f}")
    col3.metric("Overall Posture Score", f"{stats['avg_posture'] * 100:.1f}%", 
                delta_color=("inverse" if stats['avg_posture'] < 0.7 else "normal"))
    col4.metric("Gesture Activity Rate", f"{stats['gesture_activity'] * 100:.1f}%")
    
    st.markdown("---")
    
    # 2. Gesture Breakdown
    st.subheader("Gesture Activity Breakdown")
    if stats['gesture_counts']:
        gesture_df = pd.DataFrame(
            [(g.replace('_', ' ').title(), c) for g, c in stats['gesture_counts'].items()],
            columns=['Gesture', 'Count (Frames)']
        )
        gesture_df['Percentage'] = (gesture_df['Count (Frames)'] / stats['processed_frames']) * 100
        gesture_df = gesture_df[gesture_df['Gesture'] != 'None'].sort_values(by='Count (Frames)', ascending=False)
        st.dataframe(gesture_df, use_container_width=True, hide_index=True)
        st.bar_chart(gesture_df.set_index('Gesture')['Percentage'], color="#2a9d8f")

    # 3. CSV Data Download (The critical part replacing file saving)
    st.subheader("Data Export")
    
    # Convert CSV list of lists into a single string buffer
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(csv_data)
    csv_string = output.getvalue()
    
    st.download_button(
        label="Download Full Frame-by-Frame CSV Report 📊",
        data=csv_string,
        file_name=f"{uploaded_file.name}_analysis_report.csv",
        mime="text/csv"
    )

    # 4. Analysis Summary Text (Replacing .json/.txt report)
    summary_report = f"""
    ANALYSIS SUMMARY REPORT: {uploaded_file.name}
    --------------------------------------------------
    Date/Time: {pd.Timestamp.now()}
    Total Video Length: {stats['processed_frames']} frames
    Average Processing FPS: {stats['avg_fps']:.1f}
    
    BODY POSTURE:
    - Overall Posture Score (0-100%): {stats['avg_posture'] * 100:.1f}%
    - Interpretation: A score above 70% generally indicates good alignment.
    
    GESTURE ACTIVITY:
    - Total Frames with Detected Gestures (excluding 'none'): {sum(c for g, c in stats['gesture_counts'].items() if g != 'none')}
    - Gesture Activity Rate: {stats['gesture_activity'] * 100:.1f}%
    
    GESTURE BREAKDOWN:
    {json.dumps(stats['gesture_counts'], indent=2)}
    
    RECOMMENDATIONS:
    - Review the high-activity gesture segments to understand communication style.
    - Focus on low posture score segments in the video for ergonomic correction.
    --------------------------------------------------
    """
    
    st.download_button(
        label="Download Text Summary Report 📝",
        data=summary_report,
        file_name=f"{uploaded_file.name}_summary.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    # Ensure all Streamlit elements are run inside the main function
    # Only the Streamlit app code is executed here.
    main()
```eof

This dependency pinning should resolve any environment-specific compilation or linking issues that were causing the installer to fail. Please try restarting the application now!
