# Weapon Detection Web App

A real-time weapon detection web application built with Flask and YOLOv8. The app captures video from a webcam, performs object detection to identify weapons (e.g., knives, pistols), and displays the feed in a modern web interface with alerts and automatic recording.

## Features

- **Real-time Detection**: Continuous video stream with YOLOv8 model for detecting weapons.
- **Web Interface**: Modern UI with gradient backgrounds, cards, and responsive design.
- **Alerts**: Visual warnings and badges when weapons are detected.
- **Recording**: Automatic saving of annotated frames when detections occur.
- **Icons and Animations**: Enhanced user experience with Font Awesome icons and smooth transitions.

## Requirements

- Python 3.8+
- Flask
- OpenCV
- Ultralytics YOLO
- Webcam access

## Installation

1. Clone or download the project files.
2. Install the required packages:
   ```
   pip install flask opencv-contrib-python ultralytics
   ```
3. Ensure you have the YOLO model file `weapons_yolov8.pt` in the project directory.

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your web browser and go to `http://localhost:5000`.
3. The app will start capturing from your default webcam and display the live feed with detections.

## Project Structure

- `app.py`: Main Flask application file containing the web server, video streaming, and detection logic.
- `weapons_yolov8.pt`: Pre-trained YOLO model for weapon detection (not included; download or train your own).
- `detection_*.jpg`: Saved detection images (generated automatically).

## How It Works

- The Flask app serves a web page with a video feed using MJPEG streaming.
- The `gen()` function captures frames from the webcam, runs YOLO inference, annotates the frame, and yields JPEG images.
- Detection status is polled via AJAX to update the UI with alerts and badges.
- When weapons are detected, frames are saved periodically, and warnings are displayed.

## Customization

- **Model**: Replace `weapons_yolov8.pt` with your own trained model.
- **UI**: Modify the CSS and HTML in `app.py` to customize the appearance.
- **Detection Classes**: Update the model to detect other objects as needed.

## License

This project is for educational purposes. Ensure compliance with local laws regarding surveillance and weapon detection.

## Contributing

Feel free to submit issues or pull requests for improvements.
