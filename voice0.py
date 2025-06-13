
import speech_recognition as sr
import pyttsx3
import time
import threading
import queue

class YogaVoiceAssistant:
    def __init__(self, language='en'):
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[0].id)
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1
        self.language = language
        
        # Set up queues for async processing
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Control flags
        self.is_listening = False
        self.listen_thread = None
        
        # Exercise-specific commands
        self.exercise_commands = {
            'start': self.start_exercise,
            'pause': self.pause_exercise,
            'resume': self.resume_exercise,
            'stop': self.stop_exercise,
            'next': self.next_pose,
            'previous': self.previous_pose,
            'help': self.provide_help
        }
        
        # Language mappings for multilingual support
        self.language_codes = {
            'English': 'en-US',
            'Hindi': 'hi-IN',
            'Telugu': 'te-IN',
            'Tamil': 'ta-IN',
            'Kannada': 'kn-IN',
            'Malayalam': 'ml-IN'
        }
    
    def start_listening(self):
        """Start the background listening thread"""
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
    
    def stop_listening(self):
        """Stop the background listening thread"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join()
    
    def _listen_loop(self):
        """Continuous listening loop in background"""
        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    
                    try:
                        lang_code = self.language_codes.get(self.language, 'en-US')
                        text = self.recognizer.recognize_google(audio, language=lang_code)
                        print(f"Recognized: {text}")
                        self.command_queue.put(text.lower())
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError:
                        print("Could not request results")
                        
            except (sr.WaitTimeoutError, Exception) as e:
                print(f"Listening error: {e}")
                continue
    
    def speak(self, text):
        """Convert text to speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
    
    # Exercise control methods
    def start_exercise(self):
        self.speak("Starting exercise session. Please get into position.")
        return "start"
    
    def pause_exercise(self):
        self.speak("Exercise paused. Take a moment to rest.")
        return "pause"
    
    def resume_exercise(self):
        self.speak("Resuming exercise. Get ready.")
        return "resume"
    
    def stop_exercise(self):
        self.speak("Stopping exercise session. Great work today!")
        return "stop"
    
    def next_pose(self):
        self.speak("Moving to next pose.")
        return "next"
    
    def previous_pose(self):
        self.speak("Going back to previous pose.")
        return "previous"
    
    def provide_help(self):
        help_text = """
        Available voice commands:
        - "Start" to begin the exercise
        - "Pause" to pause the current pose
        - "Resume" to continue
        - "Stop" to end the session
        - "Next" for the next pose
        - "Previous" for the previous pose
        - "Help" to hear these instructions
        """
        self.speak(help_text)
        return "help"
    
    def provide_pose_guidance(self, pose_name, instructions):
        """Provide verbal guidance for a specific pose"""
        self.speak(f"Current pose: {pose_name}")
        self.speak(instructions)
    
    def process_commands(self):
        """Process any pending voice commands"""
        try:
            if not self.command_queue.empty():
                command = self.command_queue.get_nowait()
                
                # Check for exercise-specific commands
                for cmd, func in self.exercise_commands.items():
                    if cmd in command:
                        return func()
                
                # Handle general queries
                if "how" in command or "what" in command:
                    self.speak("Let me help you with that. Please try to be more specific with your question.")
                    return "query"
                
            return None
        except queue.Empty:
            return None