#importing libraries
import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import pygame
import pyttsx3
import streamlit as st
import base64


mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

pygame.mixer.init()

gif_placeholder = st.empty()



def add_success_gif(path):
    """Display success gif"""
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gifs', path)
        with open(file_path, "rb") as file_:
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")

        gif_placeholder.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="success gif">',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error loading success gif: {str(e)}")

#gifs to train
def add_gif(pose):
    poses = {
        1: '1.gif', 2: '2.gif', 3: '3.gif',
        4: '4.gif'
    }
    
    file_ = open(os.path.join(".", "finalgifs","vinyasanayoga", f'{poses[pose]}'), "rb")  # Path to your GIF file
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    # Set a fixed width and height for the GIF container using inline CSS
    gif_placeholder.markdown(
        f'''
        <div style="width:800px;height:800px;">
            <img src="data:image/gif;base64,{data_url}" alt="{poses[pose]}" style="width:100%;height:100%;">
        </div>
        ''',
        unsafe_allow_html=True,
    )



#play feedback audio file 
def speech(status):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(status)
        pygame.mixer.music.play()



def angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if angle>180:
        angle=360-angle
    return angle


def instruction(pos=1,lang="English"):
    """Play instruction audio"""

            # Structure: audio/languages/instructions/{language}/{position}.mp3
    audio_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'audio',
                'vinyasana_audio',
                f'{lang}',
                'instructions',
                f'{pos}.mp3'
            )
        
    if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()







def Plank_Pose(landmarks,lang):
    message=[]
    elbow_x,elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    wrist_x,wrist_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y

    if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])<160):
        feedback=1
    else:
         return "perfect"
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'vinyasana_audio',
             f'{lang}',
            'Feedback',
            f'{feedback}.mp3'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def Cobra(landmarks,lang):
    message=[]
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])>135):
        feedback=2
    else:
         return "perfect"
    
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'vinyasana_audio',
             f'{lang}',
            'Feedback',
            f'{feedback}.mp3'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def AdhoMukha(landmarks,lang):
    message=[]
    elbow_x,elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    l_ankle_x,l_ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])>90):
        feedback=3
    elif(angle([elbow_x,elbow_y],[shoulder_x,shoulder_y],[hip_x,hip_y])>168):
        feedback=4
    if(angle([hip_x,hip_y],[knee_x,knee_y],[l_ankle_x,l_ankle_y])>168):
        feedback=5
    else:
         return "perfect"
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'vinyasana_audio',
             f'{lang}',
            'Feedback',
            f'{feedback}.mp3'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def ChathurangaDandasana(landmarks,lang):
    message=[]
    elbow_x,elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    wrist_x,wrist_y=landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    Knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y

    if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[Knee_x,knee_y])<160):
        feedback=6
    elif(angle([shoulder_x,shoulder_y],[elbow_x,elbow_y],[wrist_x,wrist_y])>120):
        feedback=7
    else:
        return "perfect"
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'vinyasana_audio',
             f'{lang}',
            'Feedback',
            f'{feedback}.mp3'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None
        

