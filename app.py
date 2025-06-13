# Set page config must be the first Streamlit command
import streamlit as st
st.set_page_config(layout="wide")

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import yoga
from math import hypot

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Initialize pygame mixer
pygame.mixer.init()
selection_sound = pygame.mixer.Sound('pop.mp3')
selected_sound = pygame.mixer.Sound('duolingo_correct.mp3')

# Load the black background image
BLACK_BACKGROUND = cv2.imread('black.jpg')
if BLACK_BACKGROUND is None:
    raise RuntimeError("Could not load black.jpg background image")

class GestureDetector:
    def __init__(self):
        self.last_detection_time = time.time()
        self.gesture_history = []
        self.history_size = 5
        self.min_gesture_duration = 1.0  # seconds required to hold pinch
        self.confidence_threshold = 0.7
        self.current_pinch_start = None
        self.current_selection = None
        
        # Initialize MediaPipe hands with better settings
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
    
    def calculate_finger_angles(self, landmarks):
        """Calculate angles between finger joints for better gesture recognition"""
        angles = {}
        
        finger_joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for finger, joints in finger_joints.items():
            point1 = np.array([landmarks.landmark[joints[0]].x, landmarks.landmark[joints[0]].y])
            point2 = np.array([landmarks.landmark[joints[1]].x, landmarks.landmark[joints[1]].y])
            point3 = np.array([landmarks.landmark[joints[2]].x, landmarks.landmark[joints[2]].y])
            
            angle = np.degrees(np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) -
                             np.arctan2(point1[1] - point2[1], point1[0] - point2[0]))
            angles[finger] = angle % 360
            
        return angles

    def detect_pinch(self, landmarks, frame_shape):
        """Enhanced pinch detection with multiple checks"""
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        
        thumb_x, thumb_y = int(thumb_tip.x * frame_shape[1]), int(thumb_tip.y * frame_shape[0])
        index_x, index_y = int(index_tip.x * frame_shape[1]), int(index_tip.y * frame_shape[0])
        middle_x, middle_y = int(middle_tip.x * frame_shape[1]), int(middle_tip.y * frame_shape[0])
        
        thumb_index_dist = hypot(thumb_x - index_x, thumb_y - index_y)
        index_middle_dist = hypot(index_x - middle_x, index_y - middle_y)
        
        angles = self.calculate_finger_angles(landmarks)
        
        is_pinch = (
            thumb_index_dist < 40 and
            index_middle_dist > 50 and
            angles['middle'] > 160 and
            angles['ring'] > 160 and
            angles['pinky'] > 160
        )
        
        return is_pinch, (thumb_x, thumb_y), (index_x, index_y)

    def get_pointer_position(self, landmarks, frame_shape):
        """Get smooth pointer position from index finger"""
        index_tip = landmarks.landmark[8]
        x = int(index_tip.x * frame_shape[1])
        y = int(index_tip.y * frame_shape[0])
        
        if hasattr(self, 'last_pointer_pos'):
            alpha = 0.7
            x = int(alpha * x + (1 - alpha) * self.last_pointer_pos[0])
            y = int(alpha * y + (1 - alpha) * self.last_pointer_pos[1])
        
        self.last_pointer_pos = (x, y)
        return x, y

    def process_frame(self, frame, selection_manager):
        """Process a frame and detect gestures with time-based selection"""
        current_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        display_frame = BLACK_BACKGROUND.copy()
        
        # Draw buttons first
        selection_manager.draw_buttons(display_frame)
        
        if not results.multi_hand_landmarks:
            # Reset pinch timing if hand is lost
            self.current_pinch_start = None
            self.current_selection = None
            return None, current_time, display_frame, None, None
        
        landmarks = results.multi_hand_landmarks[0]
        is_pinch, thumb_pos, index_pos = self.detect_pinch(landmarks, frame.shape)
        pointer_pos = self.get_pointer_position(landmarks, frame.shape)
        
        # Handle selection timing
        selected_option = None
        selection_progress = 0
        
        # Check for button hovering
        for btn in selection_manager.button_rects:
            x, y, w, h = btn["rect"]
            if (x <= pointer_pos[0] <= x + w) and (y <= pointer_pos[1] <= y + h):
                if is_pinch:
                    if self.current_pinch_start is None:
                        # Start timing for new pinch
                        self.current_pinch_start = current_time
                        self.current_selection = btn["option"]
                    elif btn["option"] == self.current_selection:  # Only continue timing if still on same button
                        # Calculate selection progress
                        selection_progress = min(1.0, (current_time - self.current_pinch_start) / self.min_gesture_duration)
                        if selection_progress >= 1.0:
                            selected_option = btn["option"]
                            selected_sound.play()
                else:
                    # Reset timing if pinch is released
                    self.current_pinch_start = None
                    self.current_selection = None
                break
        else:
            # Reset if not hovering over any button
            self.current_pinch_start = None
            self.current_selection = None
        
        # Draw visual feedback
        self.draw_feedback(display_frame, landmarks, pointer_pos, is_pinch, selection_progress)
        
        return ("select" if selected_option else None,
                current_time,
                display_frame,
                selected_option,
                pointer_pos)

    def draw_feedback(self, frame, landmarks, pointer_pos, is_pinch, selection_progress=0):
        """Draw visual feedback for the user with selection progress"""
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(0, 255, 0) if is_pinch else (255, 255, 255),
                thickness=4,
                circle_radius=6
            ),
            mp_drawing.DrawingSpec(
                color=(0, 255, 0) if is_pinch else (255, 255, 255),
                thickness=4
            )
        )
        
        # Draw pointer with dynamic color and size
        pointer_color = (0, 255, 0) if is_pinch else (0, 255, 255)
        pointer_size = 15 if is_pinch else 10
        cv2.circle(frame, pointer_pos, pointer_size, pointer_color, -1)
        
        # Draw selection progress
        if selection_progress > 0:
            # Draw progress circle around pointer
            radius = 25
            end_angle = int(360 * selection_progress)
            cv2.ellipse(frame, pointer_pos, (radius, radius), 
                       -90, 0, end_angle, (0, 255, 0), 2)
            
            # Show progress percentage
            progress_text = f"{int(selection_progress * 100)}%"
            cv2.putText(frame, progress_text,
                       (pointer_pos[0] - 20, pointer_pos[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0), 2)
            
            if selection_progress < 1.0:
                cv2.putText(frame, "Hold to select...",
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Selected!",
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 0), 2)


class SelectionManager:
    def __init__(self):
        self.button_rects = []
        self.selection_cooldown = 0
        self.animation_start = 0
        self.animation_duration = 1.0

    def create_buttons(self, frame_width, frame_height, options):
        button_width = 300
        button_height = 120
        margin = 50
        self.button_rects = []
        
        if len(options) <= 3:
            y = frame_height // 3
            for i, option in enumerate(options):
                x = (frame_width // (len(options) + 1)) * (i + 1) - button_width // 2
                self.button_rects.append({
                    "rect": (x, y, button_width, button_height),
                    "option": option
                })
        else:
            items_per_row = (len(options) + 1) // 2
            for i, option in enumerate(options):
                row = i // items_per_row
                col = i % items_per_row
                y = frame_height // 3 + (row * (button_height + margin))
                x = (frame_width // (items_per_row + 1)) * (col + 1) - button_width // 2
                self.button_rects.append({
                    "rect": (x, y, button_width, button_height),
                    "option": option
                })

    def draw_buttons(self, frame, current_option=None):
        for btn in self.button_rects:
            x, y, w, h = btn["rect"]
            color = (130, 0, 255)
            if btn["option"] == current_option:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            
            text_size = cv2.getTextSize(btn["option"], cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(frame, btn["option"], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

class CameraManager:
    """Context manager for camera handling"""
    def __init__(self):
        self.cap = None
        
    def __enter__(self):
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        return self.cap
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

def selection_page(options, title, instruction, frame_window):
    """Enhanced selection page with robust gesture detection"""
    gesture_detector = GestureDetector()
    selection_manager = SelectionManager()
    current_option = None
    
    placeholder = st.empty()
    
    with placeholder.container():
        col1, col2 = st.columns([6, 1])
        
        with col1:
            main_frame_window = st.empty()
        
        with col2:
            st.markdown(f"### {title}")
            st.write(instruction)
            current_selection = st.empty()
    
    with CameraManager() as cap:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        selection_manager.create_buttons(frame_width, frame_height, options)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to get frame from camera")
                break
            frame=cv2.flip(frame,1)
            gesture, current_time, display_frame, selected, pointer_pos = gesture_detector.process_frame(
                frame, selection_manager
            )
            
            if gesture == "select" and selected:
                placeholder.empty()
                return selected
            
            cv2.putText(display_frame, title,
                       (frame_width//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            cv2.putText(display_frame, "Pinch to select",
                       (frame_width//2 - 80, frame_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            main_frame_window.image(display_frame, channels="BGR", use_container_width=True)
            
            if selected:
                current_option = selected
                current_selection.markdown(f"**Currently selected:** {selected}")


def main():
    st.title("TrainWithAI - Exercise Navigation")
    
    # Add custom CSS with black background
    st.markdown("""
        <style>
        .stApp {
            background-color: black;
            margin: 0;
            padding: 0;
            color: white;
        }
        
        .stMarkdown {
            font-size: 24px !important;
            color: white !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
        
        h1 {
            font-size: 48px !important;
            text-align: center;
        }
        
        h2 {
            font-size: 36px !important;
            text-align: center;
        }
        
        p {
            font-size: 24px !important;
            text-align: center;
            color: white !important;
        }
        
        .stVideo {
            width: 100%;
            height: auto;
        }
        
        .stAlert {
            background-color: #2E2E2E !important;
            color: white !important;
        }
        
        .stButton>button {
            background-color: #4A4A4A !important;
            color: white !important;
        }
        
        .stSelectbox {
            background-color: #2E2E2E !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'language'
    if 'language' not in st.session_state:
        st.session_state.language = None
    if 'exercise' not in st.session_state:
        st.session_state.exercise = None
    
    # Define available exercises
    exercise_functions = {
        'Surya Namaskar': yoga.surya_namaskar,
        'Vinyasana Yoga': yoga.vinyasana_yoga,
        'Hatha Yoga': yoga.hatha_yoga
    }
    
    # Language selection page
    if st.session_state.page == 'language':
        languages = ['English', 'Telugu', 'Tamil', 'Hindi', 'Kanada', 'Malayalam']
        selected_language = selection_page(
            languages,
            "Language Selection",
            "Use pinch gesture to select your preferred language",
            None
        )
        
        if selected_language:
            st.session_state.language = selected_language
            st.session_state.page = 'exercise'
            st.rerun()
    
    # Exercise selection page
    elif st.session_state.page == 'exercise':
        exercises = list(exercise_functions.keys())
        selected_exercise = selection_page(
            exercises,
            "Exercise Selection",
            "Use pinch gesture to select an exercise",
            None
        )
        
        if selected_exercise:
            st.session_state.exercise = selected_exercise
            st.session_state.page = 'training'
            st.rerun()
    
    # Training page
    elif st.session_state.page == 'training':
        # Create header with exercise info
        st.markdown(f"### Selected Language: {st.session_state.language} | Exercise: {st.session_state.exercise}")
        
        # Automatically start the selected exercise
        selected_exercise = st.session_state.exercise
        if selected_exercise in exercise_functions:
            exercise_functions[selected_exercise](st.session_state.language)
        else:
            st.error("Selected exercise is not implemented yet.")

if __name__ == "__main__":
    main()
