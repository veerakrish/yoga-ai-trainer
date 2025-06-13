
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
        4: '4.gif', 5: '5.gif', 6: '6.gif',
        7: '7.gif', 8: '8.gif', 9: '4.gif',
        10: '3.gif', 11: '2.gif', 12: '1.gif'
    }
    
    file_ = open(os.path.join(".", "finalgifs","suryanamskar", f'{poses[pose]}'), "rb")  # Path to your GIF file
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
def play_music(sound):
    if sound=='correct':
        name="duolingo_correct.mp3"
    else:
        name='end_songs'
    audio_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'audio',
                f'{name}'
            )
    if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
    else:
            st.warning(f"Audio file not found: {audio_path}")  
   

#play feedback audio file 
def speech(status):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(status)
        pygame.mixer.music.play()

def instruction(pos=1, lang='english'):
    """Play instruction audio"""
            # Structure: audio/languages/instructions/{language}/{position}.mp3
    lang=lang.lower()
    extention='wav'
    if(lang=='english'):
            extention='mp3'
    audio_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'audio',
                'surya_audio',
                'instructions',
                f'{lang}',
                f'{pos}.{extention}'
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

#Position Analysis and feed backfunctions
def Pranamasana(landmarks,lang):
    return "perfect"

def Hastauttanasana(landmarks,lang):
    extention='wav'
    lang=lang.lower()
    if(lang=='english'):
            extention='mp3'
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
        feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
        )
        
        tts_thread = threading.Thread(target=speech, args=(feedback_path,))
        tts_thread.start()
        return None

def Hastapadasana(landmarks,lang):  
    message=[]
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    ankle_x,ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y

    if(angle([ankle_x,ankle_y],[knee_x,knee_y],[hip_x,hip_y])<168):
        feedback=2
    elif(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])>83):
        feedback=3
    else:
         return "perfect"

    extention='wav'
    lang=lang.lower()
    if(lang=='english'):
            extention='mp3'
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None


def Ashwa_Sanchalanasana_left(landmarks,lang):
   
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    r_hip_x,r_hip_y=landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    l_hip_x,l_hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    r_knee_x,r_knee_y=landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    l_knee_x,l_knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y    
    l_ankle_x,l_ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    r_ankle_x,r_ankle_y=landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

    if(angle([r_ankle_x,r_ankle_y],[r_knee_x,r_knee_y],[r_hip_x,r_hip_y])<120):
        feedback=3
    elif(angle([r_knee_x,r_knee_y],[r_hip_x,r_hip_y],[l_knee_x,l_knee_y])<90):
        feedback=4
    elif(angle([l_hip_x,l_hip_y],[l_knee_x,l_knee_y],[l_ankle_x,l_ankle_y])>=95 or angle([l_hip_x,l_hip_y],[l_knee_x,l_knee_y],[l_ankle_x,l_ankle_y])<=80 ):
        feedback=5
    else:
         return "perfect"
    
    extention='wav'
    lang=lang.lower()
    if(lang=='english'):
            extention='mp3'
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None



def stick_pose(landmarks,lang):
    message=[]
    elbow_x,elbow_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    wrist_x,wrist_y=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y

    if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])<160):
        feedback=6
    else:
         return "perfect"

    extention='wav'
    lang=lang.lower()
    if(lang=='english'):
            extention='mp3'
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def ashtanga(landmarks,lang):
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    l_ankle_x,l_ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y

    if(angle([l_ankle_x,l_ankle_y],[knee_x,knee_y],[hip_x,hip_y])<100):
        feedback=3
    elif(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])>100):
        feedback=7
    else:
         return "perfect"
    lang=lang.lower()
    extention='wav'
    if(lang=='english'):
            extention='mp3'
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None

def cobra(landmarks,lang):
    message=[]
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_x,hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_x,knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    if(angle([shoulder_x,shoulder_y],[hip_x,hip_y],[knee_x,knee_y])>135):
        feedback=8
    else:
         return "perfect"
    extention='wav'
    lang=lang.lower()
    if(lang=='english'):
            extention='mp3'
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
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
        feedback=11
    elif(angle([elbow_x,elbow_y],[shoulder_x,shoulder_y],[hip_x,hip_y])>168):
        feedback=6
    if(angle([hip_x,hip_y],[knee_x,knee_y],[l_ankle_x,l_ankle_y])>168):
        feedback=2
    else:
         return "perfect"
    extention='wav'
    lang=lang.lower()
    if(lang=='english'):
            extention='mp3'
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None
    


def Ashwa_Sanchalanasana_right(landmarks,lang):
    message=[]
    shoulder_x,shoulder_y=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    l_hip_x,l_hip_y=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    r_hip_x,r_hip_y=landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    l_knee_x,l_knee_y=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    r_knee_x,r_knee_y=landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y    
    r_ankle_x,r_ankle_y=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    l_ankle_x,l_ankle_y=landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

    if(angle([l_ankle_x,l_ankle_y],[l_knee_x,l_knee_y],[l_hip_x,l_hip_y])<120):
        feedback=9
    elif(angle([l_knee_x,l_knee_y],[l_hip_x,l_hip_y],[r_knee_x,r_knee_y])<90):
        feedback=4
    elif(angle([r_hip_x,r_hip_y],[r_knee_x,r_knee_y],[r_ankle_x,r_ankle_y])>=95 or angle([r_hip_x,r_hip_y],[r_knee_x,r_knee_y],[r_ankle_x,r_ankle_y])<=80 ):
        feedback=10
    else:
        return "perfect"
    extention='wav'
    lang=lang.lower()
    if(lang=='english'):
            extention='mp3'
    feedback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audio',
            'surya_audio',
            'feedback',
             f'{lang}',
            f'{feedback}.{extention}'
        )
        
    tts_thread = threading.Thread(target=speech, args=(feedback_path,))
    tts_thread.start()
    return None
    

   
