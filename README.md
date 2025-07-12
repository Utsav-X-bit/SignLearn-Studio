# SignLearn Studio

**SignLearn Studio** is an AI-powered sign language learning platform that helps users learn American Sign Language (ASL) through interactive video tutorials and real-time gesture recognition.

## Features

- 📚 **Lessons:** Structured lessons covering ASL alphabets, greetings, daily life, family, colors, emotions, and numbers.
- 🎥 **Video Tutorials:** Each sign comes with a short, focused video demonstration.
- 🤳 **Practice Mode:** Practice signs using your webcam. The app uses AI (MediaPipe) to detect your hand pose and provides instant feedback.
- 🏆 **Gamification:** Earn points, track your progress, and unlock achievements as you learn.
- 📊 **Progress Tracking:** Visualize your learning journey and see which signs you’ve mastered.
- 🔒 **User Profiles:** Track your streak, level, and points.

## How It Works

1. **Select a Lesson:** Browse the available lessons and choose one to start.
2. **Watch & Practice:** Watch the video tutorial for each sign, then practice in front of your webcam.
3. **Get Feedback:** The app compares your hand pose to the reference and gives you a similarity score and feedback.
4. **Advance:** When you perform a sign correctly, move to the next one and keep learning!

## Tech Stack

- [Streamlit](https://streamlit.io/) for the web interface
- [MediaPipe](https://mediapipe.dev/) for real-time hand tracking
- [OpenCV](https://opencv.org/) for video processing
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) and ffmpeg for downloading and trimming YouTube videos
- [Plotly](https://plotly.com/) for progress visualization
- [SQLite](https://www.sqlite.org/) for user data storage

## Getting Started

### Prerequisites

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/) installed on your system

### Installation

```bash
git clone https://github.com/yourusername/SignLearn-Studio.git
cd SignLearn-Studio
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run SignLearn.py
```

## Usage

- **Lessons Tab:** Browse and start lessons.
- **Practice Tab:** Watch the tutorial and practice the sign. Get instant feedback.
- **Progress Tab:** Track your learning stats.
- **Achievements Tab:** See your unlocked achievements.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)

---

**SignLearn Studio** – Empowering everyone to learn sign language, one sign at a time!