import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import time
from datetime import datetime, timedelta
import requests
import io
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import os
import subprocess
from pathlib import Path
from scipy.spatial.distance import euclidean
import queue
import sqlite3
from pathlib import Path
import base64
import streamlit.components.v1 as components
import math
from scipy.spatial.distance import euclidean

# Configure Streamlit page
st.set_page_config(
    page_title="SignLearn Studio",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced YouTube video database with more comprehensive sign coverage
YOUTUBE_VIDEOS = {
    # ALPHABET SIGNS
    'A': {
        'video_id': 'tkMg8g8vVUo',  # ASL Alphabet A-Z
        'start_time': 2,
        'end_time': 6,
        'title': 'ASL Letter A - Fist with Thumb',
        'description': 'Learn to form the letter A in American Sign Language',
        'difficulty': 1,
        'category': 'Alphabet'
    },
    'B': {
        'video_id': 'tkMg8g8vVUo',
        'start_time': 7,
        'end_time': 13,
        'title': 'ASL Letter B - Flat Hand',
        'description': 'Learn to form the letter B with flat hand and tucked thumb',
        'difficulty': 1,
        'category': 'Alphabet'
    },
    'C': {
        'video_id': 'tkMg8g8vVUo',
        'start_time': 14,
        'end_time': 19,
        'title': 'ASL Letter C - Curved Hand',
        'description': 'Learn to form the letter C with curved hand shape',
        'difficulty': 1,
        'category': 'Alphabet'
    },
    'D': {
        'video_id': 'tkMg8g8vVUo',
        'start_time': 20,
        'end_time': 25,
        'title': 'ASL Letter D - Index Finger Up',
        'description': 'Learn to form the letter D with index finger pointing up',
        'difficulty': 1,
        'category': 'Alphabet'
    },
    'E': {
        'video_id': 'tkMg8g8vVUo',
        'start_time': 26,
        'end_time': 31,
        'title': 'ASL Letter E - Bent Fingers',
        'description': 'Learn to form the letter E with bent fingertips',
        'difficulty': 1,
        'category': 'Alphabet'
    },
    
    # GREETINGS AND BASIC SIGNS
    'Hello': {
        'video_id': 'v1desDduz5M',  # Basic greetings
        'start_time': 0,
        'end_time': 15,
        'title': 'ASL Hello - Wave Greeting',
        'description': 'Learn the basic hello wave in ASL',
        'difficulty': 1,
        'category': 'Greetings'
    },
    'Thank You': {
        'video_id': 'v1desDduz5M',
        'start_time': 30,
        'end_time': 45,
        'title': 'ASL Thank You - Chin to Forward',
        'description': 'Learn to say thank you by moving hand from chin forward',
        'difficulty': 1,
        'category': 'Greetings'
    },
    'Please': {
        'video_id': 'v1desDduz5M',
        'start_time': 60,
        'end_time': 75,
        'title': 'ASL Please - Circular Chest Motion',
        'description': 'Learn to say please with circular motion on chest',
        'difficulty': 1,
        'category': 'Greetings'
    },
    'Sorry': {
        'video_id': 'v1desDduz5M',
        'start_time': 90,
        'end_time': 105,
        'title': 'ASL Sorry - Fist on Chest',
        'description': 'Learn to apologize with circular fist motion on chest',
        'difficulty': 1,
        'category': 'Greetings'
    },
    
    # BASIC RESPONSES
    'Yes': {
        'video_id': 'basic_signs_123',
        'start_time': 0,
        'end_time': 12,
        'title': 'ASL Yes - Nodding Fist',
        'description': 'Learn to say yes with nodding fist motion',
        'difficulty': 1,
        'category': 'Basic'
    },
    'No': {
        'video_id': 'basic_signs_123',
        'start_time': 20,
        'end_time': 32,
        'title': 'ASL No - Two Fingers Closing',
        'description': 'Learn to say no with index and middle finger motion',
        'difficulty': 1,
        'category': 'Basic'
    },
    'Maybe': {
        'video_id': 'basic_signs_123',
        'start_time': 45,
        'end_time': 57,
        'title': 'ASL Maybe - Alternating Hands',
        'description': 'Learn maybe with alternating flat hands up and down',
        'difficulty': 2,
        'category': 'Basic'
    },
    
    # DAILY LIFE ESSENTIALS
    'Water': {
        'video_id': 'daily_life_signs',
        'start_time': 0,
        'end_time': 15,
        'title': 'ASL Water - W Shape at Mouth',
        'description': 'Learn water sign with W handshape near mouth',
        'difficulty': 1,
        'category': 'Daily Life'
    },
    'Food': {
        'video_id': 'daily_life_signs',
        'start_time': 25,
        'end_time': 40,
        'title': 'ASL Food - Fingers to Mouth',
        'description': 'Learn food sign by bringing fingertips to mouth',
        'difficulty': 1,
        'category': 'Daily Life'
    },
    'Eat': {
        'video_id': 'daily_life_signs',
        'start_time': 45,
        'end_time': 60,
        'title': 'ASL Eat - Fingertips to Mouth',
        'description': 'Learn eat sign with repeated fingertips to mouth motion',
        'difficulty': 1,
        'category': 'Daily Life'
    },
    'Drink': {
        'video_id': 'daily_life_signs',
        'start_time': 65,
        'end_time': 80,
        'title': 'ASL Drink - C Shape to Mouth',
        'description': 'Learn drink sign with C handshape tilted toward mouth',
        'difficulty': 1,
        'category': 'Daily Life'
    },
    
    # FAMILY SIGNS
    'Mother': {
        'video_id': 'family_signs_asl',
        'start_time': 0,
        'end_time': 15,
        'title': 'ASL Mother - Thumb to Chin',
        'description': 'Learn mother sign with thumb touching chin',
        'difficulty': 1,
        'category': 'Family'
    },
    'Father': {
        'video_id': 'family_signs_asl',
        'start_time': 20,
        'end_time': 35,
        'title': 'ASL Father - Thumb to Forehead',
        'description': 'Learn father sign with thumb touching forehead',
        'difficulty': 1,
        'category': 'Family'
    },
    'Sister': {
        'video_id': 'family_signs_asl',
        'start_time': 40,
        'end_time': 55,
        'title': 'ASL Sister - L Shape Movement',
        'description': 'Learn sister sign with L handshape movement',
        'difficulty': 2,
        'category': 'Family'
    },
    'Brother': {
        'video_id': 'family_signs_asl',
        'start_time': 60,
        'end_time': 75,
        'title': 'ASL Brother - L Shape at Forehead',
        'description': 'Learn brother sign starting from forehead',
        'difficulty': 2,
        'category': 'Family'
    },
    
    # COLORS
    'Red': {
        'video_id': 'colors_asl_tutorial',
        'start_time': 5,
        'end_time': 18,
        'title': 'ASL Red - Index Finger on Lips',
        'description': 'Learn red color sign with index finger on lips moving down',
        'difficulty': 1,
        'category': 'Colors'
    },
    'Blue': {
        'video_id': 'colors_asl_tutorial',
        'start_time': 25,
        'end_time': 38,
        'title': 'ASL Blue - B Handshape Twist',
        'description': 'Learn blue color sign with B handshape twisting motion',
        'difficulty': 1,
        'category': 'Colors'
    },
    'Green': {
        'video_id': 'colors_asl_tutorial',
        'start_time': 45,
        'end_time': 58,
        'title': 'ASL Green - G Handshape Shake',
        'description': 'Learn green color sign with G handshape shaking',
        'difficulty': 1,
        'category': 'Colors'
    },
    
    # EMOTIONS
    'Happy': {
        'video_id': 'emotions_asl_signs',
        'start_time': 0,
        'end_time': 12,
        'title': 'ASL Happy - Upward Chest Motion',
        'description': 'Learn happy sign with upward motion on chest',
        'difficulty': 1,
        'category': 'Emotions'
    },
    'Sad': {
        'video_id': 'emotions_asl_signs',
        'start_time': 18,
        'end_time': 30,
        'title': 'ASL Sad - Downward Face Motion',
        'description': 'Learn sad sign with fingers moving down face',
        'difficulty': 1,
        'category': 'Emotions'
    },
    'Angry': {
        'video_id': 'emotions_asl_signs',
        'start_time': 36,
        'end_time': 48,
        'title': 'ASL Angry - Claw Hand on Face',
        'description': 'Learn angry sign with claw hand moving down face',
        'difficulty': 2,
        'category': 'Emotions'
    },
    
    # NUMBERS (1-10)
    '1': {
        'video_id': 'asl_numbers_1_10',
        'start_time': 5,
        'end_time': 10,
        'title': 'ASL Number 1 - Index Finger',
        'description': 'Learn number one with index finger up',
        'difficulty': 1,
        'category': 'Numbers'
    },
    '2': {
        'video_id': 'asl_numbers_1_10',
        'start_time': 12,
        'end_time': 17,
        'title': 'ASL Number 2 - Two Fingers',
        'description': 'Learn number two with index and middle finger',
        'difficulty': 1,
        'category': 'Numbers'
    },
    '3': {
        'video_id': 'asl_numbers_1_10',
        'start_time': 19,
        'end_time': 24,
        'title': 'ASL Number 3 - Three Fingers',
        'description': 'Learn number three with thumb, index, and middle finger',
        'difficulty': 1,
        'category': 'Numbers'
    },
    '4': {
        'video_id': 'asl_numbers_1_10',
        'start_time': 26,
        'end_time': 31,
        'title': 'ASL Number 4 - Four Fingers',
        'description': 'Learn number four with four fingers up, thumb tucked',
        'difficulty': 1,
        'category': 'Numbers'
    },
    '5': {
        'video_id': 'asl_numbers_1_10',
        'start_time': 33,
        'end_time': 38,
        'title': 'ASL Number 5 - Open Hand',
        'description': 'Learn number five with open hand, all fingers extended',
        'difficulty': 1,
        'category': 'Numbers'
    }
}

# Enhanced video management functions
class VideoManager:
    def __init__(self):
        self.videos = YOUTUBE_VIDEOS
    
    def add_video(self, sign_name, video_info):
        """Add a new video for a sign."""
        self.videos[sign_name] = video_info
        return True
    
    def get_video_info(self, sign_name):
        """Get video information for a specific sign."""
        return self.videos.get(sign_name, None)
    
    def get_videos_by_category(self, category):
        """Get all videos for a specific category."""
        return {k: v for k, v in self.videos.items() if v.get('category') == category}
    
    def get_videos_by_difficulty(self, difficulty):
        """Get all videos filtered by difficulty level."""
        return {k: v for k, v in self.videos.items() if v.get('difficulty') == difficulty}
    
    def search_videos(self, query):
        """Search videos by sign name, title, or description."""
        query = query.lower()
        results = {}
        for sign, info in self.videos.items():
            if (query in sign.lower() or 
                query in info.get('title', '').lower() or 
                query in info.get('description', '').lower()):
                results[sign] = info
        return results
    
    def update_video(self, sign_name, updated_info):
        """Update video information for a sign."""
        if sign_name in self.videos:
            self.videos[sign_name].update(updated_info)
            return True
        return False
    
    def remove_video(self, sign_name):
        """Remove a video for a sign."""
        if sign_name in self.videos:
            del self.videos[sign_name]
            return True
        return False
    
    def get_all_categories(self):
        """Get list of all available categories."""
        categories = set()
        for info in self.videos.values():
            if 'category' in info:
                categories.add(info['category'])
        return sorted(list(categories))
    
    def validate_video_id(self, video_id):
        """Validate if a YouTube video ID exists (basic check)."""
        # Basic validation - YouTube video IDs are typically 11 characters
        return len(video_id) == 11 and video_id.isalnum()


# --- Local Video Download and Playback Utilities ---
def download_youtube_video(video_id, start_time=0, end_time=None, cache_dir=".video_cache"):
    """
    Download a YouTube video by video_id and trim it to the specified start and end time.
    Returns the local file path to the trimmed video.
    Uses yt-dlp and ffmpeg (must be installed in the system).
    """
    os.makedirs(cache_dir, exist_ok=True)
    base_path = Path(cache_dir)
    video_file = base_path / f"{video_id}.mp4"
    trimmed_file = base_path / f"{video_id}_{start_time}_{end_time or 'end'}.mp4"

    # Download if not already present
    if not video_file.exists():
        # Download the best mp4 format using yt-dlp
        ytdlp_cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-f", "mp4",
            "-o", str(video_file)
        ]
        try:
            subprocess.run(ytdlp_cmd, check=True)
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    # Trim if not already present
    if not trimmed_file.exists():
        # Build ffmpeg trim command
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-ss", str(start_time)
        ]
        if end_time:
            duration = int(end_time) - int(start_time)
            ffmpeg_cmd += ["-t", str(duration)]
        ffmpeg_cmd += ["-c:v", "copy", "-c:a", "copy", str(trimmed_file)]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except Exception as e:
            print(f"Error trimming video: {e}")
            return None

    return str(trimmed_file)

import base64

def create_local_video_player(local_video_path, video_info, width="100%", height=400):
    """
    Create a looping HTML5 video player for a local video file with metadata display.
    """
    # Read the video file and encode it as base64
    with open(local_video_path, "rb") as file:
        video_bytes = file.read()
        b64 = base64.b64encode(video_bytes).decode()
    
    # Create HTML with looping video
    video_html = f"""
    <video width="100%" height="{height}" controls autoplay loop>
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    
    # Display the HTML video player
    components.html(video_html, height=height+20)
    
    # Display video metadata
    st.markdown(f"**{video_info.get('title', 'ASL Tutorial')}**")
    st.markdown(f"*{video_info.get('description', '')}*")
    st.markdown(f"**Category:** {video_info.get('category', 'General')} | Level {video_info.get('difficulty', 1)}")
    
    # Optional: Display warning for larger videos that might cause performance issues
    file_size_mb = os.path.getsize(local_video_path) / (1024 * 1024)
    if file_size_mb > 10:  # Warning for videos larger than 10MB
        st.info(f"Video size: {file_size_mb:.1f}MB. Looping large videos may affect performance.")
# Database setup
@st.cache_resource
def init_database():
    """Initialize SQLite database for user progress."""
    db_path = "signlearn.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # Create tables (keeping existing structure)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_points INTEGER DEFAULT 0,
            current_level INTEGER DEFAULT 1,
            streak_days INTEGER DEFAULT 0,
            last_practice_date DATE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            difficulty INTEGER DEFAULT 1,
            signs TEXT NOT NULL,
            description TEXT,
            has_videos BOOLEAN DEFAULT 0
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            lesson_id INTEGER,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            score REAL,
            attempts INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (lesson_id) REFERENCES lessons (id)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS practice_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sign_name TEXT,
            accuracy REAL,
            practice_date DATE,
            duration_seconds INTEGER,
            similarity_score REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    return conn

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    """Initialize MediaPipe hands solution."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands, mp_hands, mp_drawing

class GestureComparator:
    """Compare hand gestures between reference and user."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image):
        """Extract hand landmarks from image."""
        if isinstance(image, np.ndarray):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = np.array(image)
        
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Return first hand landmarks
            landmarks = results.multi_hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        return None
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks for scale, position, and orientation invariance using Procrustes analysis."""
        if not landmarks:
            return None

        points = np.array(landmarks)
        # Center at wrist (landmark 0)
        wrist = points[0]
        centered = points - wrist

        # Scale: use max distance between any two points for better hand size normalization
        dists = np.linalg.norm(centered[:, :2][None, :, :] - centered[:, :2][:, None, :], axis=-1)
        max_dist = np.max(dists)
        if max_dist > 0:
            scaled = centered / max_dist
        else:
            scaled = centered

        # Procrustes alignment: align hand orientation to a canonical direction
        # We'll align the vector from wrist (0) to middle finger MCP (9) to the x-axis
        v = scaled[9][:2]  # 2D vector from wrist to middle finger MCP
        angle = np.arctan2(v[1], v[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        scaled_xy = scaled[:, :2] @ R.T
        # Keep z as is (or set to 0 if only 2D is needed)
        if scaled.shape[1] == 3:
            normalized = np.concatenate([scaled_xy, scaled[:, 2:3]], axis=1)
        else:
            normalized = scaled_xy

        return normalized
    
    def calculate_similarity(self, ref_landmarks, user_landmarks):
        """Calculate similarity between reference and user landmarks."""
        if ref_landmarks is None or user_landmarks is None:
            return 0.0
        
        # Normalize both sets of landmarks
        ref_norm = self.normalize_landmarks(ref_landmarks)
        user_norm = self.normalize_landmarks(user_landmarks)
        
        if ref_norm is None or user_norm is None:
            return 0.0
        
        # Calculate Euclidean distance for each landmark
        distances = []
        for i in range(len(ref_norm)):
            if i < len(user_norm):
                dist = np.linalg.norm(ref_norm[i] - user_norm[i])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Convert distance to similarity percentage
        avg_distance = np.mean(distances)
        similarity = max(0, 100 - (avg_distance * 100))
        
        return min(100, similarity)
    
    def get_gesture_feedback(self, similarity_score):
        """Get feedback based on similarity score."""
        if similarity_score >= 85:
            return "Excellent! 🎉", "success"
        elif similarity_score >= 70:
            return "Good job! 👍", "success"
        elif similarity_score >= 50:
            return "Getting better! 📈", "warning"
        else:
            return "Keep practicing! 💪", "error"

# Enhanced Sign Language Model

# Enhanced Video Processor with gesture comparison
class VideoProcessor:
    def __init__(self, target_sign=None, reference_landmarks=None):
        self.hands, self.mp_hands, self.mp_drawing = load_mediapipe()
        # self.sign_model = SignLanguageModel()  # Removed unused model
        self.gesture_comparator = GestureComparator()
        self.current_prediction = None
        self.confidence = 0.0
        self.similarity_score = 0.0
        self.frame_count = 0
        self.target_sign = target_sign
        self.reference_landmarks = reference_landmarks
        
    def set_reference(self, target_sign, reference_landmarks):
        """Set reference gesture for comparison."""
        self.target_sign = target_sign
        self.reference_landmarks = reference_landmarks
        
    def process(self, frame):
        """Process video frame for hand detection and comparison."""
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract current landmarks
                current_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                
                # Compare with reference if available
                if self.reference_landmarks:
                    self.similarity_score = self.gesture_comparator.calculate_similarity(
                        self.reference_landmarks, current_landmarks
                    )
        
        # Display prediction and similarity
        if self.current_prediction and self.confidence > 0.5:
            cv2.putText(img, f"Detected: {self.current_prediction} ({self.confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.similarity_score > 0:
            color = (0, 255, 0) if self.similarity_score > 70 else (0, 165, 255) if self.similarity_score > 50 else (0, 0, 255)
            cv2.putText(img, f"Similarity: {self.similarity_score:.1f}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.target_sign:
            cv2.putText(img, f"Target: {self.target_sign}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    

def create_youtube_player(video_id, start_time=0, end_time=None, height=400):
    """Embed a YouTube video that loops between start_time and end_time using IFrame API."""
    
    end_time_script = f"""
        var endTime = {end_time};
        var startTime = {start_time};
        function checkTime() {{
            var currentTime = player.getCurrentTime();
            if (currentTime >= endTime) {{
                player.seekTo(startTime);
            }}
        }}
        setInterval(checkTime, 1000);
    """ if end_time is not None else ""

    html_code = f"""
    <div id="player"></div>
    <script>
      var tag = document.createElement('script');
      tag.src = "https://www.youtube.com/iframe_api";
      var firstScriptTag = document.getElementsByTagName('script')[0];
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

      var player;
      function onYouTubeIframeAPIReady() {{
        player = new YT.Player('player', {{
          height: '{height}',
          width: '100%',
          videoId: '{video_id}',
          playerVars: {{
            autoplay: 1,
            controls: 1,
            start: {start_time},
            end: {end_time if end_time is not None else 0},
            modestbranding: 1,
            rel: 0,
            loop: 0,  // We'll control loop manually
          }},
          events: {{
            'onReady': function(event) {{
              event.target.playVideo();
            }}
          }}
        }});
      }}
      {end_time_script}
    </script>
    """
    return html_code



# Enhanced lesson integration with video system
def display_video_tutorial(current_sign):
    """Enhanced video tutorial display function with local download and playback."""
    st.markdown("#### 📹 Video Tutorial")
    video_manager = VideoManager()
    video_info = video_manager.get_video_info(current_sign)
    if video_info and video_info.get('video_id') and len(video_info['video_id']) == 11:
        # Download and play locally
        local_video = download_youtube_video(
            video_info['video_id'],
            start_time=video_info.get('start_time', 0),
            end_time=video_info.get('end_time', None)
        )
        if local_video:
            create_local_video_player(local_video, video_info, height=350)
        else:
            st.warning("Could not download or process the video. Showing YouTube fallback.")
            youtube_url = f"https://www.youtube.com/watch?v={video_info['video_id']}"
            st.markdown(f"[Watch on YouTube]({youtube_url})")
    else:
        # Fallback for signs without videos
        st.info(f"📹 Video tutorial for '{current_sign}' is coming soon!")
        # If you want to show sign info, you can use a static dictionary or remove this block
        all_videos = video_manager.get_videos_by_category('Basic')
        if all_videos:
            st.write("**You might also like these tutorials:**")
            for sign, info in list(all_videos.items())[:3]:
                st.write(f"• {sign} - {info.get('title', '')}")

# Enhanced lesson initialization with video validation
def initialize_enhanced_lessons():
    """Initialize lessons with video validation."""
    video_manager = VideoManager()
    
    enhanced_lessons = [
        {
            'name': 'Complete Alphabet A-E',
            'category': 'Alphabet',
            'difficulty': 1,
            'signs': json.dumps(['A', 'B', 'C', 'D', 'E']),
            'description': 'Master the first five letters with comprehensive video tutorials',
            'has_videos': all(video_manager.get_video_info(sign) for sign in ['A', 'B', 'C', 'D', 'E'])
        },
        {
            'name': 'Essential Greetings',
            'category': 'Greetings',
            'difficulty': 1,
            'signs': json.dumps(['Hello', 'Thank You', 'Please', 'Sorry']),
            'description': 'Learn polite expressions with expert video demonstrations',
            'has_videos': all(video_manager.get_video_info(sign) for sign in ['Hello', 'Thank You', 'Please', 'Sorry'])
        },
        {
            'name': 'Basic Numbers 1-5',
            'category': 'Numbers',
            'difficulty': 1,
            'signs': json.dumps(['1', '2', '3', '4', '5']),
            'description': 'Count from 1 to 5 in ASL with clear video tutorials',
            'has_videos': all(video_manager.get_video_info(sign) for sign in ['1', '2', '3', '4', '5'])
        },
        {
            'name': 'Family Members', 
            'category': 'Family',
            'difficulty': 2,
            'signs': json.dumps(['Mother', 'Father', 'Sister', 'Brother']),
            'description': 'Learn to sign family relationships with detailed videos',
            'has_videos': all(video_manager.get_video_info(sign) for sign in ['Mother', 'Father', 'Sister', 'Brother'])
        },
        {
            'name': 'Colors Basics',
            'category': 'Colors', 
            'difficulty': 1,
            'signs': json.dumps(['Red', 'Blue', 'Green']),
            'description': 'Express colors in ASL with visual video guides',
            'has_videos': all(video_manager.get_video_info(sign) for sign in ['Red', 'Blue', 'Green'])
        },
        {
            'name': 'Emotions Expression',
            'category': 'Emotions',
            'difficulty': 2,
            'signs': json.dumps(['Happy', 'Sad', 'Angry']),
            'description': 'Learn to express emotions with expressive video tutorials',
            'has_videos': all(video_manager.get_video_info(sign) for sign in ['Happy', 'Sad', 'Angry'])
        }
    ]
    
    return enhanced_lessons

# Enhanced Lesson Manager
class LessonManager:
    def __init__(self, db_conn):
        self.db = db_conn
        self.initialize_lessons()
    
    def initialize_lessons(self):
        """Initialize default lessons."""
        default_lessons = [
            {
                'name': 'Alphabet Basics A-C',
                'category': 'Alphabet',
                'difficulty': 1,
                'signs': json.dumps(['A', 'B', 'C']),
                'description': 'Learn the first letters of ASL alphabet with video guidance'
            },
            {
                'name': 'Common Greetings',
                'category': 'Greetings',
                'difficulty': 1,
                'signs': json.dumps(['Hello', 'Thank You', 'Please']),
                'description': 'Essential greeting signs with expert demonstrations'
            },
            {
                'name': 'Basic Responses',
                'category': 'Basic',
                'difficulty': 2,
                'signs': json.dumps(['Yes', 'No']),
                'description': 'Learn to respond with basic yes/no signs'
            },
            {
                'name': 'Daily Life Essentials',
                'category': 'Daily Life',
                'difficulty': 2,
                'signs': json.dumps(['Water', 'Food']),
                'description': 'Essential signs for daily needs with visual tutorials'
            }
        ]
        
        for lesson in default_lessons:
            try:
                self.db.execute("""
                    INSERT OR IGNORE INTO lessons (name, category, difficulty, signs, description)
                    VALUES (?, ?, ?, ?, ?)
                """, (lesson['name'], lesson['category'], lesson['difficulty'], 
                     lesson['signs'], lesson['description']))
                self.db.commit()
            except:
                pass
    
    def get_lessons(self, difficulty=None):
        """Get lessons, optionally filtered by difficulty."""
        query = "SELECT * FROM lessons"
        params = []
        
        if difficulty:
            query += " WHERE difficulty = ?"
            params.append(difficulty)
        
        query += " ORDER BY difficulty, id"
        
        cursor = self.db.execute(query, params)
        return cursor.fetchall()
    
    def get_lesson_by_id(self, lesson_id):
        """Get specific lesson by ID."""
        cursor = self.db.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,))
        return cursor.fetchone()

# User management (keeping existing)
class UserManager:
    def __init__(self, db_conn):
        self.db = db_conn
    
    def create_user(self, username):
        """Create new user."""
        try:
            cursor = self.db.execute("""
                INSERT INTO users (username) VALUES (?)
            """, (username,))
            self.db.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
    
    def get_user(self, username):
        """Get user by username."""
        cursor = self.db.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()
    
    def update_user_progress(self, user_id, points, level=None):
        """Update user points and level."""
        if level:
            self.db.execute("""
                UPDATE users SET total_points = total_points + ?, current_level = ?
                WHERE id = ?
            """, (points, level, user_id))
        else:
            self.db.execute("""
                UPDATE users SET total_points = total_points + ? WHERE id = ?
            """, (points, user_id))
        self.db.commit()
    
    def update_streak(self, user_id):
        """Update user streak."""
        today = datetime.now().date()
        cursor = self.db.execute("""
            SELECT last_practice_date, streak_days FROM users WHERE id = ?
        """, (user_id,))
        result = cursor.fetchone()
        
        if result:
            last_date, current_streak = result
            if last_date:
                last_date = datetime.strptime(last_date, '%Y-%m-%d').date()
                if (today - last_date).days == 1:
                    new_streak = current_streak + 1
                elif (today - last_date).days == 0:
                    new_streak = current_streak
                else:
                    new_streak = 1
            else:
                new_streak = 1
            
            self.db.execute("""
                UPDATE users SET streak_days = ?, last_practice_date = ?
                WHERE id = ?
            """, (new_streak, today.isoformat(), user_id))
            self.db.commit()

# Gamification system (keeping existing)
class GamificationSystem:
    def __init__(self):
        self.level_thresholds = [0, 100, 250, 500, 1000, 2000, 4000, 8000, 15000, 30000]
        self.achievements = {
            'first_sign': {'name': 'First Sign!', 'description': 'Completed your first sign', 'points': 10},
            'perfect_lesson': {'name': 'Perfect!', 'description': 'Completed lesson with 100% accuracy', 'points': 50},
            'week_streak': {'name': 'Week Warrior', 'description': '7 day practice streak', 'points': 100},
            'month_streak': {'name': 'Monthly Master', 'description': '30 day practice streak', 'points': 500},
            'similarity_master': {'name': 'Similarity Master', 'description': 'Achieved 90%+ similarity score', 'points': 75}
        }
    
    def calculate_level(self, points):
        """Calculate user level based on points."""
        for level, threshold in enumerate(self.level_thresholds):
            if points < threshold:
                return level
        return len(self.level_thresholds)
    
    def points_to_next_level(self, points):
        """Calculate points needed for next level."""
        current_level = self.calculate_level(points)
        if current_level < len(self.level_thresholds):
            return self.level_thresholds[current_level] - points
        return 0
    
    def award_points(self, accuracy, difficulty_multiplier=1, similarity_bonus=0):
        """Award points based on accuracy, difficulty, and similarity."""
        base_points = int(accuracy * 100)
        similarity_points = int(similarity_bonus * 0.5)  # Bonus for high similarity
        return (base_points + similarity_points) * difficulty_multiplier

def initialize_session_state():
    """Initialize session state variables."""
    if 'db_conn' not in st.session_state:
        st.session_state.db_conn = init_database()
    
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserManager(st.session_state.db_conn)
    
    if 'lesson_manager' not in st.session_state:
        st.session_state.lesson_manager = LessonManager(st.session_state.db_conn)
    
    if 'gamification' not in st.session_state:
        st.session_state.gamification = GamificationSystem()
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'current_lesson' not in st.session_state:
        st.session_state.current_lesson = None
    
    if 'lesson_progress' not in st.session_state:
        st.session_state.lesson_progress = {}
    
    if 'practice_stats' not in st.session_state:
        st.session_state.practice_stats = []
    
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = None

def create_progress_chart(user_data):
    """Create progress visualization chart."""
    if not user_data:
        return None
    
    # Sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    points = np.cumsum(np.random.randint(0, 50, len(dates)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=points,
        mode='lines+markers',
        name='Total Points',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Learning Progress Over Time",
        xaxis_title="Date",
        yaxis_title="Total Points",
        template="plotly_white",
        height=300
    )
    
    return fig

def main():
    st.title("🤟 SignLearn Studio")
    st.markdown("*AI-Powered Sign Language Learning Platform with Video Tutorials*")
    
    initialize_session_state()
    
    # Sidebar for user management (keeping existing structure)
    with st.sidebar:
        st.header("👤 User Profile")
        
        if st.session_state.current_user is None:
            # User login/registration
            username = st.text_input("Enter Username")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Login", use_container_width=True):
                    user = st.session_state.user_manager.get_user(username)
                    if user:
                        st.session_state.current_user = user
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("User not found!")
            
            with col2:
                if st.button("Register", use_container_width=True):
                    if username:
                        user_id = st.session_state.user_manager.create_user(username)
                        if user_id:
                            user = st.session_state.user_manager.get_user(username)
                            st.session_state.current_user = user
                            st.success(f"Welcome, {username}!")
                            st.rerun()
                        else:
                            st.error("Username already exists!")
                    else:
                        st.error("Please enter a username!")
        
        else:
            # Display user stats (keeping existing)
            user = st.session_state.current_user
            st.markdown(f"**Welcome, {user[1]}!** 🎉")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Level", user[4])
                st.metric("Streak", f"{user[5]} days")
            with col2:
                st.metric("Points", user[3])
                next_level_points = st.session_state.gamification.points_to_next_level(user[3])
                st.metric("To Next Level", next_level_points)
            
            current_level = st.session_state.gamification.calculate_level(user[3])
            if current_level < len(st.session_state.gamification.level_thresholds):
                level_start = st.session_state.gamification.level_thresholds[current_level-1] if current_level > 0 else 0
                level_end = st.session_state.gamification.level_thresholds[current_level]
                progress = (user[3] - level_start) / (level_end - level_start)
                st.progress(progress)
            
            if st.button("Logout"):
                st.session_state.current_user = None
                st.rerun()
    
    # Main content tabs
    if st.session_state.current_user:
        if "active_tab" not in st.session_state:
            st.session_state["active_tab"] = 0
        tabs = st.tabs(
    ["📚 Lessons", "🎯 Practice", "📊 Progress", "🏆 Achievements"]
)
    
        # Lessons Tab (keeping existing structure)
        with tabs[0]:
            st.header("📚 Learning Lessons")
            
            difficulty_filter = st.selectbox("Filter by Difficulty", 
                                           ["All Levels", "Beginner (1)", "Intermediate (2)", "Advanced (3)"])
            
            difficulty = None
            if difficulty_filter != "All Levels":
                difficulty = int(difficulty_filter.split("(")[1].split(")")[0])
            
            lessons = st.session_state.lesson_manager.get_lessons(difficulty)
            
            for i in range(0, len(lessons), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(lessons):
                        lesson = lessons[i + j]
                        with col:
                            with st.container():
                                st.markdown(f"### {lesson[1]}")
                                st.markdown(f"**Category:** {lesson[2]}")
                                st.markdown(f"**Difficulty:** {'⭐' * lesson[3]}")
                                st.markdown(f"**Description:** {lesson[5]}")
                                
                                signs = json.loads(lesson[4])
                                st.markdown(f"**Signs to Learn:** {', '.join(signs)}")
                                
                                if st.button(f"Start Lesson", key=f"lesson_{lesson[0]}"):
                                    st.session_state.current_lesson = lesson
                                    st.session_state.lesson_progress = {
                                        'current_sign': 0,
                                        'signs': signs,
                                        'scores': [],
                                        'similarity_scores': [],
                                        'start_time': time.time()
                                    }
                                    st.success(f"Started: {lesson[1]}")
                                    st.info("Move to the Practice Tab")
                                    #st.rerun()
        
        # Enhanced Practice Tab with YouTube integration
        with tabs[1]:
            st.header("🎯 Practice Session with Video Tutorial")
            
            if st.session_state.current_lesson is None:
                st.info("Please select a lesson from the Lessons tab to start practicing!")
            else:
                lesson = st.session_state.current_lesson
                progress = st.session_state.lesson_progress
                
                if progress['current_sign'] < len(progress['signs']):
                    current_sign = progress['signs'][progress['current_sign']]
                    
                    # Main practice area with video and camera side by side
                    st.markdown(f"### {lesson[1]} - Learning: {current_sign}")
                    
                    # Create two columns for video tutorial and practice camera
                    video_col, camera_col, progress_col = st.columns([1, 1, 0.5])
                    
                    # ...existing code...
                    with video_col:
                                        st.markdown("#### 📹 Video Tutorial")
                                        # Get YouTube video for current sign
                                        if current_sign in YOUTUBE_VIDEOS:
                                            video_info = YOUTUBE_VIDEOS[current_sign]
                                            # Only download if video_id is a valid YouTube ID (11 chars)
                                            if len(video_info['video_id']) == 11:
                                                local_video = download_youtube_video(
                                                    video_info['video_id'],
                                                    start_time=video_info.get('start_time', 0),
                                                    end_time=video_info.get('end_time', None)
                                                )
                                                if local_video:
                                                    create_local_video_player(local_video, video_info, height=350)
                                                else:
                                                    st.warning("Could not download or process the video. Showing YouTube fallback.")
                                                    youtube_url = f"https://www.youtube.com/watch?v={video_info['video_id']}"
                                                    st.markdown(f"[Watch on YouTube]({youtube_url})")
                                            else:
                                                st.info(f"Video tutorial for '{current_sign}' coming soon!")
                                            # Display sign information
                                            # If you want to show sign info, you can use a static dictionary or remove this block
                                        else:
                                            st.info(f"Video tutorial for '{current_sign}' coming soon!")
# ...existing code...
                    
                    # with video_col:
                    #     st.markdown("#### 📹 Video Tutorial")
                        
                        # Get YouTube video for current sign
                        # if current_sign in YOUTUBE_VIDEOS:
                        #     video_info = YOUTUBE_VIDEOS[current_sign]
                        #     youtube_html = create_youtube_player(
                        #         video_info['video_id'], 
                        #         video_info['start_time'],
                        #         height=350
                        #     )
                        #     components.html(youtube_html, height=350)
                            
                        #     st.markdown(f"**{video_info['title']}**")
                            
                        #     # Display sign information
                        #     sign_model = SignLanguageModel()
                        #     if current_sign in sign_model.signs_data:
                        #         sign_info = sign_model.signs_data[current_sign]
                        #         st.markdown(f"**Description:** {sign_info['description']}")
                        #         st.markdown(f"**Category:** {sign_info['category']}")
                        # else:
                        #     st.info(f"Video tutorial for '{current_sign}' coming soon!")
                    
#                     with camera_col:
#                         st.markdown("#### 📷 Your Practice")
#                         st.markdown("Show the sign to your camera:")
                        
#                         # Initialize video processor for current sign
#                         if st.session_state.video_processor is None:
#                             st.session_state.video_processor = VideoProcessor(
#                                 target_sign=current_sign,
#                                 reference_landmarks=None  # In real implementation, extract from video
#                             )
                        
#                         # WebRTC video stream with enhanced processing
#                         # This is the connection between your last code and the new code
# # Your code ends here:
                    with camera_col:
                        st.markdown("#### 📷 Your Practice")
                        st.markdown("Show the sign to your camera:")

                        # --- Extract reference landmarks from the tutorial video ---
                        def extract_reference_landmarks_from_video(video_path):
                            hands, mp_hands, mp_drawing = load_mediapipe()
                            cap = cv2.VideoCapture(video_path)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            middle_frame = frame_count // 2
                            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                            ret, frame = cap.read()
                            cap.release()
                            if not ret:
                                return None
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(frame_rgb)
                            if results.multi_hand_landmarks:
                                landmarks = results.multi_hand_landmarks[0]
                                return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                            return None

                        # --- Initialize video processor with reference landmarks ---
                        if st.session_state.video_processor is None or \
                        st.session_state.video_processor.target_sign != current_sign:
                            reference_landmarks = None
                            if current_sign in YOUTUBE_VIDEOS:
                                video_info = YOUTUBE_VIDEOS[current_sign]
                                if len(video_info['video_id']) == 11:
                                    local_video = download_youtube_video(
                                        video_info['video_id'],
                                        start_time=video_info.get('start_time', 0),
                                        end_time=video_info.get('end_time', None)
                                    )
                                    if local_video:
                                        reference_landmarks = extract_reference_landmarks_from_video(local_video)
                            st.session_state.video_processor = VideoProcessor(
                                target_sign=current_sign,
                                reference_landmarks=reference_landmarks
                            )

                        # --- WebRTC video stream with enhanced processing ---
                        webrtc_ctx = webrtc_streamer(
                            key=f"sign-detection-{current_sign}",
                            mode=WebRtcMode.SENDRECV,
                            rtc_configuration=RTCConfiguration({
                                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                            }),
                            video_frame_callback=st.session_state.video_processor.process,
                            async_processing=True,
                        )

                        # --- Display current detection results and always show similarity score below the camera ---
                        if webrtc_ctx.video_processor:
                            processor = webrtc_ctx.video_processor
                            similarity = getattr(processor, 'similarity_score', 0)
                            detected_sign = getattr(processor, 'current_prediction', None)

                            # Show detected sign if available
                            if detected_sign:
                                st.markdown(f"**Detected:** {detected_sign}")

                            # Always show similarity score and progress message below the camera
                            if similarity > 0:
                                st.markdown(f"### Similarity: {similarity:.1f}%")
                                if similarity >= 90:
                                    st.success("✅ Correct sign! Great job!")
                                elif similarity >= 70:
                                    st.info("Almost there! Keep adjusting your hand.")
                                elif similarity >= 50:
                                    st.warning("Getting closer, try to match the reference more closely.")
                                else:
                                    st.error("Keep practicing!")

                    with progress_col:
                        st.markdown("#### 📊 Progress")
                        
                        # Show lesson progress
                        completed = progress['current_sign']
                        total = len(progress['signs'])
                        st.progress(completed / total)
                        st.markdown(f"**Sign {completed + 1} of {total}**")
                        
                        # Show signs list with status
                        for i, sign in enumerate(progress['signs']):
                            if i < completed:
                                st.markdown(f"✅ {sign}")
                            elif i == completed:
                                st.markdown(f"🔄 **{sign}** (Current)")
                            else:
                                st.markdown(f"⏳ {sign}")
                        
                        # Show current scores if available
                        if progress.get('scores'):
                            avg_score = np.mean(progress['scores'])
                            st.metric("Average Score", f"{avg_score:.1f}%")
                        
                        if progress.get('similarity_scores'):
                            avg_similarity = np.mean(progress['similarity_scores'])
                            st.metric("Average Similarity", f"{avg_similarity:.1f}%")
                    
                    # Practice control buttons
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("✅ Correct Sign", use_container_width=True):
                            # Award points based on similarity if available
                            similarity_score = 0
                            if hasattr(st.session_state.video_processor, 'similarity_score'):
                                similarity_score = st.session_state.video_processor.similarity_score
                            
                            # Calculate score and points
                            base_score = min(100, max(50, similarity_score))  # Minimum 50% for marking correct
                            points = st.session_state.gamification.award_points(
                                base_score / 100, 
                                lesson[3],  # difficulty multiplier
                                similarity_score
                            )
                            
                            # Record progress
                            progress['scores'].append(base_score)
                            if similarity_score > 0:
                                progress['similarity_scores'].append(similarity_score)
                            
                            # Update user points
                            st.session_state.user_manager.update_user_progress(
                                st.session_state.current_user[0], points
                            )
                            
                            # Move to next sign
                            progress['current_sign'] += 1
                            
                            # Show feedback
                            st.success(f"Great! +{points} points earned!")
                            
                            # Check if lesson completed
                            if progress['current_sign'] >= len(progress['signs']):
                                lesson_score = np.mean(progress['scores'])
                                avg_similarity = np.mean(progress['similarity_scores']) if progress['similarity_scores'] else 0
                                
                                # Bonus for high performance
                                if lesson_score >= 90 and avg_similarity >= 85:
                                    bonus_points = 100
                                    st.session_state.user_manager.update_user_progress(
                                        st.session_state.current_user[0], bonus_points
                                    )
                                    st.balloons()
                                    st.success(f"🎉 Lesson completed with excellence! Bonus: +{bonus_points} points!")
                                else:
                                    st.success("🎊 Lesson completed!")
                                
                                # Reset lesson
                                st.session_state.current_lesson = None
                                st.session_state.lesson_progress = {}
                                
                                # Update streak
                                st.session_state.user_manager.update_streak(st.session_state.current_user[0])
                            
                            time.sleep(1)
                            st.rerun()
                            
                    
                    with col2:
                        if st.button("❌ Need Help", use_container_width=True):
                            st.info(f"💡 **Tip for {current_sign}:**")
                            sign_model = SignLanguageModel()
                            if current_sign in sign_model.signs_data:
                                st.info(sign_model.signs_data[current_sign]['description'])
                            
                            # Show helpful hints
                            hints = {
                                'A': "Make a fist and place your thumb on the side of your index finger",
                                'B': "Keep your fingers straight and together, thumb tucked across palm",
                                'C': "Curve your hand like you're holding a small cup",
                                'Hello': "Wave with an open palm, fingers together",
                                'Thank You': "Start with fingers at your chin, then move hand forward",
                                'Please': "Place palm on chest and make circular motions",
                                'Yes': "Make a fist and nod it up and down like a head nodding",
                                'No': "Use index and middle finger like a mouth opening and closing",
                                'Water': "Make a 'W' with three fingers and tap near your mouth",
                                'Food': "Bring fingertips to your mouth repeatedly"
                            }
                            
                            if current_sign in hints:
                                st.info(f"🔍 **Detailed Guide:** {hints[current_sign]}")
                    
                    with col3:
                        if st.button("⏭️ Skip Sign", use_container_width=True):
                            progress['current_sign'] += 1
                            progress['scores'].append(0)  # Record as 0 score
                            
                            if progress['current_sign'] >= len(progress['signs']):
                                st.info("Lesson completed (with skipped signs)")
                                st.session_state.current_lesson = None
                                st.session_state.lesson_progress = {}
                            
                            st.rerun()
                    
                    with col4:
                        if st.button("🔄 Restart Lesson", use_container_width=True):
                            st.session_state.lesson_progress = {
                                'current_sign': 0,
                                'signs': progress['signs'],
                                'scores': [],
                                'similarity_scores': [],
                                'start_time': time.time()
                            }
                            st.rerun()
                    
                    # Advanced features section
                    st.markdown("---")
                    with st.expander("🔧 Advanced Practice Settings"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            similarity_threshold = st.slider(
                                "Similarity Threshold (%)", 
                                min_value=50, max_value=95, value=70,
                                help="Minimum similarity required for automatic recognition"
                            )
                            
                            detection_sensitivity = st.slider(
                                "Detection Sensitivity", 
                                min_value=0.3, max_value=0.8, value=0.5, step=0.1,
                                help="How sensitive the hand detection should be"
                            )
                        
                        with col2:
                            show_landmarks = st.checkbox("Show Hand Landmarks", value=True)
                            show_confidence = st.checkbox("Show Detection Confidence", value=True)
                            
                            practice_mode = st.selectbox(
                                "Practice Mode",
                                ["Standard", "Precision Mode", "Speed Mode"],
                                help="Choose your practice focus"
                            )
                    
                    # Real-time statistics
                    if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'similarity_score'):
                        st.markdown("### 📈 Real-time Performance")
                        
                        # Create a simple performance chart
                        if len(progress.get('similarity_scores', [])) > 1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=progress['similarity_scores'],
                                mode='lines+markers',
                                name='Similarity Score',
                                line=dict(color='#00CC96')
                            ))
                            fig.update_layout(
                                title="Similarity Scores Over Time",
                                yaxis_title="Similarity %",
                                xaxis_title="Attempt",
                                height=200,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Lesson completed screen
                    st.success("🎉 Lesson Completed!")
                    lesson_score = np.mean(progress['scores']) if progress['scores'] else 0
                    avg_similarity = np.mean(progress['similarity_scores']) if progress['similarity_scores'] else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Score", f"{lesson_score:.1f}%")
                    with col2:
                        st.metric("Average Similarity", f"{avg_similarity:.1f}%")
                    with col3:
                        practice_time = time.time() - progress['start_time']
                        st.metric("Practice Time", f"{practice_time/60:.1f} min")
                    
                    if st.button("Start New Lesson"):
                        st.session_state.current_lesson = None
                        st.session_state.lesson_progress = {}
                        st.rerun()
                        st.session_state["active_tab"] = 1  # 0=Lessons, 1=Practice, etc.
        
        # Progress Tab (Enhanced)
        with tabs[2]:
            st.header("📊 Learning Progress")
            
            user = st.session_state.current_user
            
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Points", user[3])
            with col2:
                st.metric("Current Level", user[4])
            with col3:
                st.metric("Practice Streak", f"{user[5]} days")
            with col4:
                next_level = st.session_state.gamification.points_to_next_level(user[3])
                st.metric("To Next Level", next_level)
            
            # Progress visualization
            progress_chart = create_progress_chart(user)
            if progress_chart:
                st.plotly_chart(progress_chart, use_container_width=True)
            
            # Detailed lesson progress
            st.subheader("📚 Lesson Progress")
            
            # Get user's lesson completions
            cursor = st.session_state.db_conn.execute("""
                SELECT l.name, l.category, up.score, up.completed_at, up.attempts
                FROM user_progress up
                JOIN lessons l ON up.lesson_id = l.id
                WHERE up.user_id = ?
                ORDER BY up.completed_at DESC
            """, (user[0],))
            
            completed_lessons = cursor.fetchall()
            
            if completed_lessons:
                df = pd.DataFrame(completed_lessons, 
                                columns=['Lesson', 'Category', 'Score', 'Completed', 'Attempts'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No lessons completed yet. Start your first lesson!")
            
            # Practice session history
            st.subheader("🎯 Recent Practice Sessions")
            
            cursor = st.session_state.db_conn.execute("""
                SELECT sign_name, accuracy, similarity_score, practice_date, duration_seconds
                FROM practice_sessions
                WHERE user_id = ?
                ORDER BY practice_date DESC
                LIMIT 10
            """, (user[0],))
            
            practice_sessions = cursor.fetchall()
            
            if practice_sessions:
                practice_df = pd.DataFrame(practice_sessions,
                                         columns=['Sign', 'Accuracy', 'Similarity', 'Date', 'Duration'])
                st.dataframe(practice_df, use_container_width=True)
                
                # Practice statistics
                avg_accuracy = practice_df['Accuracy'].mean()
                avg_similarity = practice_df['Similarity'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
                with col2:
                    st.metric("Average Similarity", f"{avg_similarity:.1f}%")
            else:
                st.info("Start practicing to see your session history!")
        
        # Achievements Tab
        with tabs[3]:
            st.header("🏆 Achievements & Badges")
            
            user = st.session_state.current_user
            achievements = st.session_state.gamification.achievements
            
            # Calculate user achievements
            user_achievements = []
            
            # Check first sign achievement
            cursor = st.session_state.db_conn.execute("""
                SELECT COUNT(*) FROM practice_sessions WHERE user_id = ?
            """, (user[0],))
            practice_count = cursor.fetchone()[0]
            
            if practice_count > 0:
                user_achievements.append('first_sign')
            
            # Check streak achievements
            if user[5] >= 7:
                user_achievements.append('week_streak')
            if user[5] >= 30:
                user_achievements.append('month_streak')
            
            # Check similarity master
            cursor = st.session_state.db_conn.execute("""
                SELECT MAX(similarity_score) FROM practice_sessions WHERE user_id = ?
            """, (user[0],))
            max_similarity = cursor.fetchone()[0]
            
            if max_similarity and max_similarity >= 90:
                user_achievements.append('similarity_master')
            
            # Check perfect lesson
            cursor = st.session_state.db_conn.execute("""
                SELECT MAX(score) FROM user_progress WHERE user_id = ?
            """, (user[0],))
            max_score = cursor.fetchone()[0]
            
            if max_score and max_score >= 100:
                user_achievements.append('perfect_lesson')
            
            # Display achievements
            cols = st.columns(3)
            
            for i, (achievement_id, achievement) in enumerate(achievements.items()):
                col_idx = i % 3
                with cols[col_idx]:
                    if achievement_id in user_achievements:
                        st.success(f"🏆 **{achievement['name']}**")
                        st.write(achievement['description'])
                        st.write(f"*+{achievement['points']} points*")
                    else:
                        st.info(f"🔒 **{achievement['name']}**")
                        st.write(achievement['description'])
                        st.write(f"*{achievement['points']} points*")
            
            # Achievement progress
            st.subheader("📈 Achievement Progress")
            
            total_possible_points = sum(ach['points'] for ach in achievements.values())
            earned_achievement_points = sum(achievements[ach]['points'] for ach in user_achievements)
            
            achievement_progress = earned_achievement_points / total_possible_points
            st.progress(achievement_progress)
            st.write(f"Achievement Points: {earned_achievement_points}/{total_possible_points}")
            
            # Leaderboard (if multiple users)
            st.subheader("🥇 Leaderboard")
            
            cursor = st.session_state.db_conn.execute("""
                SELECT username, total_points, current_level, streak_days
                FROM users
                ORDER BY total_points DESC
                LIMIT 10
            """)
            
            leaderboard = cursor.fetchall()
            
            if len(leaderboard) > 1:
                leaderboard_df = pd.DataFrame(leaderboard,
                                            columns=['Username', 'Points', 'Level', 'Streak'])
                leaderboard_df.index += 1  # Start ranking from 1
                st.dataframe(leaderboard_df, use_container_width=True)
            else:
                st.info("Be the first to climb the leaderboard!")
    
    else:
        # Welcome screen for non-logged-in users
        st.markdown("""
        ## Welcome to SignLearn Studio! 🤟
        
        **The most advanced AI-powered American Sign Language learning platform**
        
        ### Features:
        - 📹 **Video Tutorials**: Learn from expert ASL instructors with YouTube integration
        - 🤖 **AI Recognition**: Advanced hand gesture detection and comparison
        - 📊 **Real-time Feedback**: Get instant similarity scores and corrections  
        - 🎮 **Gamification**: Earn points, unlock achievements, and track streaks
        - 📚 **Structured Lessons**: Progressive curriculum from beginner to advanced
        - 🎯 **Precision Training**: Compare your gestures with video demonstrations
        
        ### How it works:
        1. **Watch**: Learn signs from integrated YouTube video tutorials
        2. **Practice**: Use your webcam to practice the signs
        3. **Compare**: Our AI compares your gesture with the video reference
        4. **Improve**: Get detailed feedback and similarity scores
        5. **Progress**: Track your learning journey and earn achievements
        
        **Please create an account or login to start your ASL learning journey!**
        """)
        
        # Demo video placeholder
        st.video("https://www.youtube.com/watch?v=pBzJaahJUZU")  # Example ASL video
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Precision Learning
            Our AI analyzes 21 hand landmarks to provide accurate gesture comparison
            """)
        
        with col2:
            st.markdown("""
            ### 📹 Video Integration  
            Learn from professional ASL instructors with embedded YouTube tutorials
            """)
        
        with col3:
            st.markdown("""
            ### 🏆 Gamification
            Stay motivated with points, levels, achievements, and daily streaks
            """)

if __name__ == "__main__":
    main()

