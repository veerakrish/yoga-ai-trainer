
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

#success gif
def add_success_gif(path):

    file_ = open(path, "rb")  # Path to your GIF file
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    gif_placeholder.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="success gif">',
        unsafe_allow_html=True,
    )

# Function to display GIF based on pose selection
def add_gif(pose):
    poses = {
        1: '1.gif', 2: '2.gif', 3: '3.gif',
        4: '4.gif', 5: '5.gif', 6: '6.gif'
    }
    
    file_ = open(os.path.join(".", "finalgifs","hathayoga", f'{poses[pose]}'), "rb")  # Path to your GIF file
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

def instruction(pos=1, lang='English'):
    """Play instruction audio"""
            # Structure: audio/languages/instructions/{language}/{position}.wav
    audio_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'audio',
                'hata_audio',
                'instructions',
                f'{lang}',
                f'{pos}.wav'
            )
        
    if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
    else:
            st.warning(f"Audio file not found: {audio_path}")

def angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if angle>180:
        angle=360-angle
    return angle

# Position Analysis and feed backfunctions
def Rest_Pose(landmarks,lang):
    return "perfect"

def Virkshasana(landmarks,lang):
    message=[]
    l_hip_x,l_hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    l_knee_x,l_knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    l_ankle_x,l_ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    r_hip_x,r_hip_y=landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    r_knee_x,r_knee_y=landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    r_ankle_x,r_ankle_y=landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    l_shoulder_x,l_shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    l_elbow_x,l_elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    l_wrist_x,l_wrist_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    r_shoulder_x,r_shoulder_y=landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    r_elbow_x,r_elbow_y=landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    r_wrist_x,r_wrist_y=landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    
    if(angle([r_ankle_x,r_ankle_y],[r_knee_x,r_knee_y],[r_hip_x,r_hip_y])>80):
        feedback=5
    elif(angle([l_ankle_x,l_ankle_y],[l_knee_x,l_knee_y],[l_hip_x,l_hip_y])<140):
        feedback=4
    else:
        return "perfect"
    lang=lang.lower()
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'hata_audio',
            'feedbacks',
             f'{lang}',
            f'{feedback}.wav'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def Virabhadrasana(landmarks,lang):
    message=[]
    l_hip_x,l_hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    l_knee_x,l_knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    l_ankle_x,l_ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    r_hip_x,r_hip_y=landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    r_knee_x,r_knee_y=landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    r_ankle_x,r_ankle_y=landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    l_shoulder_x,l_shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    l_elbow_x,l_elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    l_wrist_x,l_wrist_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    r_shoulder_x,r_shoulder_y=landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    r_elbow_x,r_elbow_y=landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    r_wrist_x,r_wrist_y=landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    if(angle([r_ankle_x,r_ankle_y],[r_knee_x,r_knee_y],[r_hip_x,r_hip_y])>118):
        feedback=5
    elif(angle([l_ankle_x,l_ankle_y],[l_knee_x,l_knee_y],[l_hip_x,l_hip_y])<160):
        feedback=4
    elif(angle([r_shoulder_x,r_shoulder_y],[r_elbow_x,r_elbow_y],[r_wrist_x,r_wrist_y])>170 and angle([l_shoulder_x,l_shoulder_y],[l_elbow_x,l_elbow_y],[l_wrist_x,l_wrist_y])>170):
        feedback=6
    else:
        return "perfect"
    lang=lang.lower()
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'hata_audio',
            'feedbacks',
             f'{lang}',
            f'{feedback}.wav'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def Hasta_uttanasana(landmarks,lang):
    message=[]
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    ankle_x,ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    
    if(angle([ankle_x,ankle_y],[knee_x,knee_y],[hip_x,hip_y])>168 and angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])<160):
         return "perfect"
    else:
        if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])>160):
            feedback=1
        elif(angle([ankle_x,ankle_y],[knee_x,knee_y],[hip_x,hip_y])<168):
            feedback=2
    lang=lang.lower()
    extension='wav'
    if lang=='english':
        extension='mp3'

    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'hata_audio',
            'feedbacks',
             f'{lang}',
            f'{feedback}.{extension}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def adho_mukha(landmarks,lang):
    message=[]
    elbow_x,elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    l_ankle_x,l_ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])>90):
        feedback=8
    elif(angle([elbow_x,elbow_y],[shoulder_x,shoulder_y],[hip_x,hip_y])>168):
        feedback=7
    if(angle([hip_x,hip_y],[knee_x,knee_y],[l_ankle_x,l_ankle_y])>168):
        feedback=2
    else:
         return "perfect"
    lang=lang.lower()
    extension='wav'
    if lang=='english':
        extension='mp3'

    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'hata_audio',
            'feedbacks',
             f'{lang}',
            f'{feedback}.{extension}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None
    
def Setu_bandhasana(landmarks,lang):
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    elbow_x,elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    wrist_x,wrist_y=landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    nose_x,nose_y=landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y
    eye_x,eye_y=landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    ankle_x,ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    if(angle([hip_x,hip_y],[knee_x,knee_y],[ankle_x,ankle_y])>120):
        feedback=10
    elif(angle([elbow_x,elbow_y],[shoulder_x,shoulder_y],[hip_x,hip_y])<50):
        feedback=9
    # elif(angle([shoulder_x,shoulder_y],[elbow_x,elbow_y],[wrist_x,wrist_y]<160)):
    #     feedback=11
    else:
        return "perfect"
    lang=lang.lower()
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'hata_audio',
            'feedbacks',
             f'{lang}',
            f'{feedback}.wav'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None  

def Salabhasana(landmarks,lang):
    message=[]
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    ankle_x,ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y    
    if(angle([ankle_x,ankle_y],[hip_x,hip_y],[shoulder_x,shoulder_y])>160):
        feedback=12
        lang=lang.lower()
        feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'hata_audio',
            'feedbacks',
             f'{lang}',
            f'{feedback}.wav'
        )
    elif(angle([ankle_x,ankle_y],[knee_x,knee_y],[hip_x,hip_y])<160):
        lang=lang.lower()
        feedback=2
        extension='wav'
        if lang=='english':
          extension='mp3'

        feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'hata_audio',
            'feedbacks',
             f'{lang}',
            f'{feedback}.{extension}'
        )
    else:
        return "perfect"
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None







