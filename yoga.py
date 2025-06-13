# yoga.py

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import onnxruntime
import pickle
import pygame
from time import time
import suryanamskar_au_functions
import vinyasanayoga_au_functions
import hata_au_functions

# Initialize pygame mixer and MediaPipe
pygame.mixer.init()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

class PoseTimer:
    """Class to manage pose timing and timeout handling"""
    def __init__(self):
        self.start_time = None
        self.is_choosing = False
        self.last_gesture_time = 0
        
    def start(self):
        """Start or restart the timer"""
        self.start_time = time()
        
    def get_remaining_time(self):
        """Get remaining time in seconds"""
        if self.start_time is None:
            return 30
        return max(0, 300 - (time() - self.start_time))
    
    def should_show_options(self):
        """Check if we should show timeout options"""
        return self.get_remaining_time() <= 0

class YogaModelManager:
    """Singleton class to manage yoga model loading and caching"""
    _instance = None
    _models = {}
    _encoder = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YogaModelManager, cls).__new__(cls)
            # Initialize ONNX Runtime sessions
            cls._models = {
                'hatha': onnxruntime.InferenceSession('hatha_model.onnx'),
                'surya': onnxruntime.InferenceSession('surya_model.onnx'),
                'vinyasana': onnxruntime.InferenceSession('vinyasana_model.onnx')
            }
            # Load encoder
            with open('onehot_encoder_output.pkl', 'rb') as f:
                cls._encoder = pickle.load(f)
        return cls._instance
    
    def predict(self, model_name, input_data):
        """Make prediction using ONNX model"""
        session = self._models[model_name]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_data.astype(np.float32)})[0]
        return result
    
    def get_model(self, exercise_type):
        """Get model for specific exercise type, loading if necessary"""
        if exercise_type not in self._models:
            try:
                if exercise_type == "surya":
                    self._models[exercise_type] = load_model("surya_model.h5")
                elif exercise_type == "vinyasa":
                    self._models[exercise_type] = load_model("vinyasana_model.h5")
                elif exercise_type == "hatha":
                    self._models[exercise_type] = load_model("hatha_model.h5")
                else:
                    raise ValueError(f"Unknown exercise type: {exercise_type}")
            except Exception as e:
                st.error(f"Failed to load model for {exercise_type}: {str(e)}")
                return None
        
        return self._models[exercise_type]
    
def detect_palm_orientation(landmarks):
    """
    Detect if palm is facing forward or backward based on thumb and pinky positions
    Returns: "front" if palm facing camera, "back" if palm facing away
    """
    # Get thumb, index, and pinky coordinates
    thumb = landmarks.landmark[4]  # Thumb tip
    index = landmarks.landmark[8]  # Index finger tip
    pinky = landmarks.landmark[20]  # Pinky tip
    wrist = landmarks.landmark[0]  # Wrist
    
    # Calculate if thumb is to the left or right of the pinky relative to the wrist
    thumb_side = (thumb.x - wrist.x) - (pinky.x - wrist.x)
    
    # For right hand: 
    # If thumb is to the left of pinky (negative), palm is facing forward
    # If thumb is to the right of pinky (positive), palm is facing backward
    return "front" if thumb_side < 0 else "back"

def draw_instruction_overlay(frame, text_left, text_right):
    """Draw instruction overlay with palm orientation instructions"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw palm icons and text
    # Left instruction (Back of palm - Skip)
    cv2.putText(frame, "✋", (width//4-50, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(frame, text_left, (width//4-120, 120), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
    
    # Right instruction (Palm facing - Continue)
    cv2.putText(frame, "🤚", (width*3//4-50, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(frame, text_right, (width*3//4-80, 120), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
    
    return frame

def handle_timeout_choice(frame, last_gesture_time):
    """Handle timeout choice with palm orientation"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Add instruction overlay
    frame = draw_instruction_overlay(frame, "BACK = SKIP", "FRONT = CONTINUE")
    
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
        )
        
        # Detect palm orientation
        orientation = detect_palm_orientation(landmarks)
        
        # Add visual feedback
        feedback_text = f"Palm: {orientation.upper()}"
        cv2.putText(frame, feedback_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if orientation == "front":
            return "continue", time(), frame
        elif orientation == "back":
            return "skip", time(), frame
    
    return None, last_gesture_time, frame

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
                
        self.cap.set(cv2.CAP_PROP_FPS, 1)
        return self.cap
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

def show_timeout_instructions(choice_display):
    """Display enhanced timeout instructions for palm orientation"""
    choice_display.markdown("""
        <div style='background-color: rgba(0,0,0,0.8); 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center;
                    margin: 20px;
                    border: 2px solid #666;'>
            <h2 style='color: #FF4B4B; margin-bottom: 20px;'>Time's Up!</h2>
            <div style='display: flex; 
                        justify-content: space-around; 
                        align-items: center; 
                        margin-top: 20px;
                        padding: 10px;'>
                <div style='border-right: 2px solid #666; padding-right: 20px;'>
                    <h3 style='color: #FF4B4B; margin-bottom: 10px;'>✋ Back of Palm</h3>
                    <p style='color: white; font-size: 18px;'>to SKIP this pose</p>
                </div>
                <div style='padding-left: 20px;'>
                    <h3 style='color: #00FF00; margin-bottom: 10px;'>Palm Forward 🤚</h3>
                    <p style='color: white; font-size: 18px;'>to CONTINUE practicing</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def surya_namaskar(language):
    """Updated Surya Namaskar implementation with enhanced instructions"""
    model_manager = YogaModelManager()
    model = model_manager.get_model("surya")
    correct_sound = pygame.mixer.Sound('duolingo_correct.mp3')
    end_sound = pygame.mixer.Sound('end_song.mp3')
    
    if model is None:
        return
    
    # Create two columns for side-by-side layout
    left_col, right_col = st.columns(2)
    
    # Left column for pose suggestions
    with left_col:
        st.markdown("### Reference Pose")
        pose_display = st.empty()
    
    # Right column for camera feed and controls
    with right_col:
        st.markdown("### Your Pose")
        frame_window = st.empty()
        timer_display = st.empty()
        choice_display = st.empty()
    
    poses = {
        1: "pranamasana", 2: "hasta uttanasana", 3: "Hastapadasana",
        4: "Ashwa Sanchalanasana left", 5: "stickpose", 6: "Ashtanga namaskara",
        7: "cobra", 8: "adho mukha svanasana", 9: "Ashwa Sanchalanasana right",
        0: "Nothing"
    }
    
    pose_timer = PoseTimer()
    pose_timer.start()
    count = 1
    curr_pos = 1
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with CameraManager() as cap:
            # Display pose suggestion in left column
            with left_col:
                suryanamskar_au_functions.add_gif(1)
                suryanamskar_au_functions.instruction(1, language.lower())
            
            while count <= 12:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to get frame from camera")
                    break
                
                remaining_time = pose_timer.get_remaining_time()
                timer_display.markdown(f"""
                    <div style='text-align: center; 
                               padding: 10px; 
                               background-color: rgba(0,0,0,0.7); 
                               border-radius: 5px;'>
                        <h2 style='color: {'#00FF00' if remaining_time > 10 else '#FF4B4B'}'>
                            Time remaining: {int(remaining_time)} seconds
                        </h2>
                    </div>
                """, unsafe_allow_html=True)
                
                if pose_timer.should_show_options():
                    show_timeout_instructions(choice_display)
                    choice, pose_timer.last_gesture_time, frame = handle_timeout_choice(
                        frame, pose_timer.last_gesture_time)
                    
                    if choice == "skip":
                        count += 1
                        curr_pos = count
                        pose_timer.start()
                        suryanamskar_au_functions.instruction(curr_pos, language)
                        suryanamskar_au_functions.add_gif(curr_pos)
                        choice_display.empty()
                        continue
                    elif choice == "continue":
                        pose_timer.start()
                        choice_display.empty()
                        continue
                
                # Process pose detection and drawing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Process landmarks and make predictions
                    landmarks = results.pose_landmarks.landmark
                    l = []
                    a, b, c = landmarks[0].x, landmarks[0].y, landmarks[0].z
                    for landmark in landmarks:
                        l.extend([landmark.x - a, landmark.y - b, landmark.z - c])
                    
                    l = np.array([l])
                    prediction = model.predict(l)
                    pos = np.argmax(prediction)
                    confidence = prediction[0][pos]
                    
                    # Draw pose information overlay
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (400, 100), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                    
                    cv2.putText(image, f'Target: {poses[curr_pos]}', 
                              (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(image, f'Current: {poses[pos]}', 
                              (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(image, f'Confidence: {confidence:.2f}', 
                              (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    
                    if pos == curr_pos:
                        pose_functions = {
                            1: suryanamskar_au_functions.Pranamasana,
                            2: suryanamskar_au_functions.Hastauttanasana,
                            3: suryanamskar_au_functions.Hastapadasana,
                            4: suryanamskar_au_functions.Ashwa_Sanchalanasana_left,
                            5: suryanamskar_au_functions.stick_pose,
                            6: suryanamskar_au_functions.ashtanga,
                            7: suryanamskar_au_functions.cobra,
                            8: suryanamskar_au_functions.adho_mukha
                        }
                        
                        if curr_pos in pose_functions:
                            status = pose_functions[curr_pos](landmarks, language)
                            
                            if status == "perfect":
                                correct_sound.play()
                                count += 1
                                curr_pos = count
                                pose_timer.start()
                                suryanamskar_au_functions.instruction(curr_pos, language)
                                suryanamskar_au_functions.add_gif(curr_pos)
                
                frame_window.image(image, channels="BGR", use_container_width=True)
                cv2.waitKey(1)
    
    # Show completion
    frame_window.empty()
    timer_display.empty()
    choice_display.empty()
    end_sound.play()
    st.balloons()
    suryanamskar_au_functions.add_success_gif("success3.gif")
    st.success("Congratulations! You have successfully completed all the poses.")

# Similar updates would be applied to vinyasana_yoga() and hatha_yoga() functions


def vinyasana_yoga(language):
    """Vinyasana yoga exercise implementation with timer and gestures"""
    model_manager = YogaModelManager()
    model = model_manager.get_model("vinyasa")
    correct_sound = pygame.mixer.Sound('duolingo_correct.mp3')
    end_sound = pygame.mixer.Sound('end_song.mp3')
    
    if model is None:
        return
    
    # Create two columns for side-by-side layout
    left_col, right_col = st.columns(2)
    
    # Left column for pose suggestions
    with left_col:
        st.markdown("### Reference Pose")
        pose_display = st.empty()
    
    # Right column for camera feed and controls
    with right_col:
        st.markdown("### Your Pose")
        frame_window = st.empty()
        timer_display = st.empty()
        choice_display = st.empty()
    
    poses = {
        0: "Normal",
        1: "ChathurangaDandasana",
        2: "Plank_Pose",
        3: "Cobra",
        4: "AdhoMukha"
    }
    
    pose_timer = PoseTimer()
    pose_timer.start()
    
    curr_pos = 1
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with CameraManager() as cap:
            # Display pose suggestion in left column
            with left_col:
                vinyasanayoga_au_functions.add_gif(1)
                vinyasanayoga_au_functions.instruction(1, language.lower())
            
            while curr_pos <= 4:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to get frame from camera")
                    break
                
                remaining_time = pose_timer.get_remaining_time()
                timer_display.text(f"Time remaining: {int(remaining_time)} seconds")
                
                if pose_timer.should_show_options():
                    choice_display.markdown("### Use hand gestures:\nSwipe Left to Skip\nSwipe Right to Continue")
                    choice, pose_timer.last_gesture_time = handle_timeout_choice(frame, pose_timer.last_gesture_time)
                    
                    if choice == "skip":
                        curr_pos += 1
                        pose_timer.start()
                        if curr_pos <= 4:
                            with left_col:
                                vinyasanayoga_au_functions.instruction(curr_pos, language)
                                vinyasanayoga_au_functions.add_gif(curr_pos)
                        choice_display.empty()
                        continue
                    elif choice == "continue":
                        pose_timer.start()
                        choice_display.empty()
                        continue
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                    landmarks = results.pose_landmarks.landmark
                    l = []
                    a, b, c = landmarks[0].x, landmarks[0].y, landmarks[0].z
                    for landmark in landmarks:
                        l.extend([landmark.x - a, landmark.y - b, landmark.z - c])
                    
                    l = np.array([l])
                    prediction = model.predict(l)
                    pos = np.argmax(prediction)
                    confidence = prediction[0][pos]
                    
                    current_pose = poses.get(pos, "Unknown")
                    target_pose = poses.get(curr_pos, "Unknown")
                    
                    cv2.putText(image, f'Target Pose: {target_pose}', 
                              (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Current Pose: {current_pose}', 
                              (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Confidence: {confidence:.2f}', 
                              (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if pos == curr_pos:
                        pose_functions = {
                            1: vinyasanayoga_au_functions.ChathurangaDandasana,
                            2: vinyasanayoga_au_functions.Plank_Pose,
                            3: vinyasanayoga_au_functions.Cobra,
                            4: vinyasanayoga_au_functions.AdhoMukha
                        }
                        
                        if curr_pos in pose_functions:
                            status = pose_functions[curr_pos](landmarks, language)
                            
                            if status == "perfect":
                                correct_sound.play()
                                curr_pos += 1
                                pose_timer.start()
                                if curr_pos <= 4:
                                    vinyasanayoga_au_functions.instruction(curr_pos, language)
                                    vinyasanayoga_au_functions.add_gif(curr_pos)
                
                frame_window.image(image, channels="BGR")
                cv2.waitKey(1)
    
    # Show completion
    frame_window.empty()
    timer_display.empty()
    choice_display.empty()
    end_sound.play()
    st.balloons()
    vinyasanayoga_au_functions.add_success_gif("success3.gif")
    st.success("Congratulations! You have successfully completed all the poses.")

def hatha_yoga(language):
    """Hatha yoga exercise implementation with timer and gestures"""
    model_manager = YogaModelManager()
    model = model_manager.get_model("hatha")
    correct_sound = pygame.mixer.Sound('duolingo_correct.mp3')
    end_sound = pygame.mixer.Sound('end_song.mp3')
    
    if model is None:
        return
    
    # Create two columns for side-by-side layout
    left_col, right_col = st.columns(2)
    
    # Left column for pose suggestions
    with left_col:
        st.markdown("### Reference Pose")
        pose_display = st.empty()
    
    # Right column for camera feed and controls
    with right_col:
        st.markdown("### Your Pose")
        frame_window = st.empty()
        timer_display = st.empty()
        choice_display = st.empty()
    
    poses = {
        0: "normal",
        1: "Virkshasana",
        2: "Virabhadrasana",
        3: "Hasta_uttanasana",
        4: "adho_mukha",
        5: "Setu_bandhasana",
        6: "Salabhasana"
    }
    
    pose_timer = PoseTimer()
    pose_timer.start()
    
    curr_pos = 1
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with CameraManager() as cap:
            # Display pose suggestion in left column
            with left_col:
                hata_au_functions.add_gif(1)
                hata_au_functions.instruction(1, language.lower())
            
            while curr_pos <= 6:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to get frame from camera")
                    break
                
                remaining_time = pose_timer.get_remaining_time()
                timer_display.text(f"Time remaining: {int(remaining_time)} seconds")
                
                if pose_timer.should_show_options():
                    choice_display.markdown("### Use hand gestures:\nSwipe Left to Skip\nSwipe Right to Continue")
                    choice, pose_timer.last_gesture_time = handle_timeout_choice(frame, pose_timer.last_gesture_time)
                    
                    if choice == "skip":
                        curr_pos += 1
                        pose_timer.start()
                        if curr_pos <= 6:
                            with left_col:
                                hata_au_functions.instruction(curr_pos, language)
                                hata_au_functions.add_gif(curr_pos)
                        choice_display.empty()
                        continue
                    elif choice == "continue":
                        pose_timer.start()
                        choice_display.empty()
                        continue
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                    landmarks = results.pose_landmarks.landmark
                    l = []
                    a, b, c = landmarks[0].x, landmarks[0].y, landmarks[0].z
                    for landmark in landmarks:
                        l.extend([landmark.x - a, landmark.y - b, landmark.z - c])
                    
                    l = np.array([l])
                    prediction = model.predict(l)
                    pos = np.argmax(prediction)
                    confidence = prediction[0][pos]
                    
                    current_pose = poses.get(pos, "Unknown")
                    target_pose = poses.get(curr_pos, "Unknown")
                    
                    cv2.putText(image, f'Target Pose: {target_pose}', 
                              (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Current Pose: {current_pose}', 
                              (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Confidence: {confidence:.2f}', 
                              (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if pos == curr_pos:
                        pose_functions = {
                            1: hata_au_functions.Virkshasana,
                            2: hata_au_functions.Virabhadrasana,
                            3: hata_au_functions.Hasta_uttanasana,
                            4: hata_au_functions.adho_mukha,
                            5: hata_au_functions.Setu_bandhasana,
                            6: hata_au_functions.Salabhasana
                        }
                        
                        if curr_pos in pose_functions:
                            status = pose_functions[curr_pos](landmarks, language)
                            
                            if status == "perfect":
                                correct_sound.play()
                                curr_pos += 1
                                pose_timer.start()
                                if curr_pos <= 6:
                                    hata_au_functions.instruction(curr_pos, language)
                                    hata_au_functions.add_gif(curr_pos)
                
                frame_window.image(image, channels="BGR")
                cv2.waitKey(1)
    
    # Show completion
    frame_window.empty()
    timer_display.empty()
    choice_display.empty()
    end_sound.play()
    st.balloons()
    hata_au_functions.add_success_gif("success3.gif")
    st.success("Congratulations! You have successfully completed all the poses.")
