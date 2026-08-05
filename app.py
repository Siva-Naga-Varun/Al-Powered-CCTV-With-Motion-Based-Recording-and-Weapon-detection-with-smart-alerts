from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO
import time
import os
from collections import Counter

app = Flask(__name__)

model = YOLO("weapons_yolov8.pt")

detected = False
last_detection = ""
motion_detected = False
bg = None
last_saved_motion = 0
last_saved_weapon = 0
in_motion = False

def gen():
    global detected, last_detection, motion_detected, bg, last_saved_motion, last_saved_weapon, in_motion
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        # Check for detections
        boxes = results[0].boxes
        if len(boxes) > 0:
            names = results[0].names
            classes = [names[int(cls)] for cls in boxes.cls]
            count = Counter(classes)
            last_detection = ', '.join([f"{v} {k}" for k, v in count.items()])
            detected = True
            # Save weapon detection frame
            if time.time() - last_saved_weapon > 5:
                timestamp = int(time.time())
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                last_saved_weapon = time.time()
        else:
            last_detection = "No detections"
            detected = False
        
        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if bg is None:
            bg = gray
        else:
            frame_delta = cv2.absdiff(bg, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion = any(cv2.contourArea(contour) > 500 for contour in contours)
            motion_detected = motion
            if motion and not in_motion:
                timestamp = int(time.time())
                filename = f"motion_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                in_motion = True
            elif not motion:
                in_motion = False
            bg = gray  # Update background
        
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"detected": detected, "detection": last_detection, "motion": motion_detected})

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Weapon Detection</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                width: 100%;
                max-width: 760px;
                background: #ffffff;
                border-radius: 15px;
                padding: 25px;
                margin-top: 30px;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
                display: flex;
                flex-direction: column;
                gap: 20px;
                align-items: center;
            }
            h1 {
                font-family: 'Montserrat', sans-serif;
                font-size: 2.5em;
                font-weight: 700;
                background: linear-gradient(90deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                padding-bottom: 10px;
            }
            .img-container {
                position: relative;
                width: 100%;
                max-width: 720px;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
                border: 2px solid #eee;
            }
            #video-feed {
                width: 100%;
                height: auto;
                display: block;
            }
            #warning {
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                background: linear-gradient(90deg, #ff4d4f, #ff7f50);
                color: #fff;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 1.2em;
                font-weight: bold;
                display: none;
                z-index: 10;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                animation: blink 1s infinite;
                transition: opacity 0.5s;
            }
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            #detections {
                background: linear-gradient(135deg, #fdfbfb, #ebedee);
                border-radius: 8px;
                border: 1px solid #ddd;
                padding: 15px;
                font-size: 1.1em;
                color: #333;
                font-weight: 500;
                width: 100%;
                max-width: 720px;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
            }
            .badge {
                display: inline-block;
                padding: 8px 16px;
                font-weight: 700;
                font-size: 1em;
                border-radius: 50px;
                color: #fff;
                background: #28a745;
                transition: all 0.3s ease;
            }
            .badge.detected {
                background: #ff4d4f;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Real-Time Weapon Detection</h1>
            <div class="img-container">
                <img src="/video_feed" id="video-feed" />
                <div id="warning">WARNING: Weapon Detected!</div>
            </div>
            <div id="detections">Detections: Loading...</div>
            <div class="badge" id="status-badge"><i class="fas fa-shield-alt"></i> Safe</div>
            <div class="badge" id="motion-badge"><i class="fas fa-video"></i> No Motion</div>
        </div>
        <audio id="buzzer" preload="auto">
            <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTGH0fPTgjMGHm7A7+OZURE" type="audio/wav">
        </audio>
        <audio id="loud_buzzer" preload="auto" loop>
            <source src="data:audio/wav;base64,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" type="audio/wav">
        </audio>
        <button onclick="testBuzzer()" style="margin-top: 10px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">Test Buzzer</button>
        <button onclick="testLoudBuzzer()" style="margin-top: 10px; padding: 10px 20px; background: #dc3545; color: white; border: none; border-radius: 5px; cursor: pointer;">Test Loud Buzzer</button>
        <button onclick="stopBuzzer()" style="margin-top: 10px; padding: 10px 20px; background: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer;">Stop Buzzer</button>
        <script>
            let lastDetectionState = false;
            let audioEnabled = false;
            
            function enableAudio() {
                const buzzer = document.getElementById('buzzer');
                buzzer.play().then(() => {
                    buzzer.pause();
                    buzzer.currentTime = 0;
                    audioEnabled = true;
                    console.log('Audio enabled');
                }).catch(e => console.log('Audio enable failed:', e));
            }
            
            function testBuzzer() {
                const buzzer = document.getElementById('buzzer');
                if (!audioEnabled) {
                    enableAudio();
                }
                setTimeout(() => {
                    buzzer.play().catch(e => console.log('Buzzer play failed:', e));
                }, 100);
            }
            
            function testLoudBuzzer() {
                const loudBuzzer = document.getElementById('loud_buzzer');
                if (!audioEnabled) {
                    enableAudio();
                }
                setTimeout(() => {
                    loudBuzzer.play().catch(e => console.log('Loud buzzer play failed:', e));
                }, 100);
            }
            
            function stopBuzzer() {
                const buzzer = document.getElementById('buzzer');
                const loudBuzzer = document.getElementById('loud_buzzer');
                buzzer.pause();
                buzzer.currentTime = 0;
                loudBuzzer.pause();
                loudBuzzer.currentTime = 0;
            }
            
            function checkStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const warning = document.getElementById('warning');
                        const detections = document.getElementById('detections');
                        const badge = document.getElementById('status-badge');
                        const motionBadge = document.getElementById('motion-badge');
                        const buzzer = document.getElementById('buzzer');
                        const loudBuzzer = document.getElementById('loud_buzzer');
                        detections.innerText = 'Detections: ' + data.detection;
                        if (data.detected) {
                            warning.style.display = 'block';
                            warning.style.opacity = 1;
                            badge.classList.add('detected');
                            badge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Weapon Detected!';
                            if (!lastDetectionState) {
                                if (!audioEnabled) {
                                    enableAudio();
                                }
                                setTimeout(() => {
                                    loudBuzzer.play().catch(e => console.log('Loud buzzer play failed:', e));
                                }, 100);
                            }
                        } else {
                            warning.style.opacity = 0;
                            setTimeout(() => { warning.style.display = 'none'; }, 500);
                            badge.classList.remove('detected');
                            badge.innerHTML = '<i class="fas fa-shield-alt"></i> Safe';
                            loudBuzzer.pause();
                            loudBuzzer.currentTime = 0;
                        }
                        if (data.motion) {
                            motionBadge.classList.add('detected');
                            motionBadge.innerHTML = '<i class="fas fa-video"></i> Motion Detected';
                        } else {
                            motionBadge.classList.remove('detected');
                            motionBadge.innerHTML = '<i class="fas fa-video"></i> No Motion';
                        }
                        lastDetectionState = data.detected;
                    })
                    .catch(err => console.error('Error fetching status:', err));
            }
            
            document.addEventListener('click', function() {
                if (!audioEnabled) {
                    enableAudio();
                }
            }, { once: true });
            
            setInterval(checkStatus, 1000);
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
