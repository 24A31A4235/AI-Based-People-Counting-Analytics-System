# AI-Based People Counting & Analytics System
Note: The counter was fixed to count slow walkers correctly - it now counts 8/8 real crossings on a busy-street test (previously 6/8).
▶ Try it live: https://ai-based-people-counting-analytics-system-production.up.railway.app

A real-time people counting and analytics application featuring high-precision tracking, unique ID assignment, demographic estimation (gender and age), and a modern, dynamic dashboard UI.

## Features

- **Real-Time Detection & Tracking:** Uses YOLOv8 for accurate person detection and a robust tracker (hybrid IoU + ReID) for stable inter-frame associations.
- **Accurate Counting:** Counts people moving across an active line, eliminating double-counting issues using stable ID recognition.
- **Demographic Analytics:** Integrated gender and age estimation modules leveraging OpenCV's DNN framework.
- **Cyberpunk UI Dashboard:** A high-end glassmorphism-inspired PyQt5 dashboard for intuitive real-time monitoring and controls.
- **Data Persistence:** Logs counts and analytics to a JSON file.

## Project Structure

- `main.py` / `run.py`: Entry points for launching the application.
- `ui.py`: Manages the application dashboard and user interface.
- `detector.py`: YOLOv8-based person detection logic.
- `tracker.py`: SORT-based multi-object tracking.
- `reid.py`: Person Re-Identification module for stable tracking across frames.
- `gender_detector.py` / `gender_trainer.py`: Modules for gender classification.
- `counter.py`: Logic for crossing detection and maintaining total counts.
- `config.py`: Centralized configuration settings.
- `requirements.txt`: Python package dependencies.

## Installation

1. Clone the repository.
2. Ensure you have Python installed.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python run.py
   ```

## Usage

- Select the video feed source within the application.
- Adjust the line position and direction as needed.
- Monitor the dashboard to view real-time people counts, unique visitor statistics, and demographic data.

## Note

YOLO model weights (`yolov8m.pt`, `yolov8n.pt`) will automatically be downloaded on the first run if not present in the directory.
