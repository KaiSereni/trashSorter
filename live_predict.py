import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from predict import GarbageCNN, predict
import time
import base64
import threading
import io
import json
import os
import platform
from flask import Flask, render_template, Response, jsonify, request

classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
          'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = GarbageCNN(num_classes=len(classes))
model.load_state_dict(torch.load('garbage_classification_model.pth', map_location=device))
model = model.to(device)

bin_sort = {
    "Recycle this item.": ["green-glass", "brown-glass", "paper", "white-glass", "metal", "cardboard"],
    "Compost this item if available, otherwise use the trash can.": ["biological"],
    "In Oregon, this can be disposed of in a trash can.": ["battery"],
    "Donate or sell this item.": ["clothes", "shoes"],
    "Put this item in the trash can.": ["trash", "plastic"]
}

app = Flask(__name__, template_folder='templates', static_folder='static')

scanning = False
scan_thread = None
last_prediction = {"class": "", "prob": 0, "bin": ""}
camera = None

ANALYTICS_FILE = 'analytics.json'
analytics_data = {
    "total_scans": 0,
    "feedback_counts": {
        "poor_camera": 0,
        "incorrect_id": 0,
        "incorrect_bin": 0,
        "slow_scan": 0,
        "other": 0,
        "no_problems": 0
    }
}

def load_analytics():
    global analytics_data
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, 'r') as f:
                analytics_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading analytics file: {e}. Resetting analytics.")
            analytics_data = {
                "total_scans": 0,
                "feedback_counts": {
                    "poor_camera": 0,
                    "incorrect_id": 0,
                    "incorrect_bin": 0,
                    "slow_scan": 0,
                    "other": 0,
                    "no_problems": 0
                }
            }
            save_analytics()
    else:
        save_analytics()

def save_analytics():
    try:
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(analytics_data, f, indent=4)
    except IOError as e:
        print(f"Error saving analytics file: {e}")

def get_frame():
    global last_prediction, camera
    
    try:
        if camera is None or not camera.isOpened():
            for idx in [0, 1, -1]:
                camera = cv2.VideoCapture(idx)
                if camera.isOpened():
                    print(f"Successfully opened camera with index {idx}")
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    time.sleep(0.5)
                    break
                else:
                    print(f"Failed to open camera with index {idx}")
            
            if not camera.isOpened():
                print("ERROR: Cannot open webcam after trying multiple indexes")
                return None, "Cannot open webcam"
        
        ret, frame = camera.read()
        
        if not ret or frame is None:
            print("ERROR: Failed to grab frame after multiple attempts")
            return None, "Failed to grab frame"
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        primary_class, primary_prob, results = predict(model, pil_img, preprocess, classes, device)
        
        bin_name = None
        for key, items in bin_sort.items():
            if primary_class in items:
                bin_name = key
                break
        if not bin_name:
            bin_name = "unknown"
        
        last_prediction = {
            "class": primary_class,
            "prob": primary_prob,
            "bin": bin_name
        }
        
        label = f"{primary_class}: {primary_prob:.1f}%"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        bin_label = bin_name if bin_name != "unknown" else "Bin not found"
        cv2.putText(frame, bin_label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes(), None
    except Exception as e:
        print(f"Camera error: {e}")
        return None, f"Camera error: {e}"

def generate_frames():
    global scanning, camera
    
    start_time = time.time()
    scan_duration = 15
    frame_interval = 0.5
    
    try:
        while scanning and time.time() - start_time < scan_duration:
            frame_data, error = get_frame()
            
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            elif error:
                print(f"Error getting frame: {error}")
                time.sleep(frame_interval)
                continue
                
            time.sleep(frame_interval)
            
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        scanning = False
        print("Streaming ended")

def scanning_thread():
    global scanning
    
    time.sleep(15)
    
    scanning = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_scan')
def start_scan():
    global scanning, scan_thread, camera, analytics_data

    if not scanning:
        scanning = True

        if camera is None or not camera.isOpened():
            print("Initializing camera...")
            for idx in [0, 1, -1]:
                camera = cv2.VideoCapture(idx)
                if camera.isOpened():
                    print(f"Successfully opened camera with index {idx}")
                    break
                else:
                    print(f"Failed to open camera with index {idx}")
        
        scan_thread = threading.Thread(target=scanning_thread)
        scan_thread.daemon = True
        scan_thread.start()
        return jsonify({"status": "started"})
    
    return jsonify({"status": "already_running"})

@app.route('/stop_scan')
def stop_scan():
    global scanning, camera, analytics_data

    if scanning:
        analytics_data["total_scans"] += 1
        save_analytics()

    scanning = False
    
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        print("Camera released")
    
    return jsonify({"status": "stopped", "scan_completed": True})

@app.route('/status')
def get_status():
    return jsonify({
        "scanning": scanning,
        "prediction": last_prediction
    })

@app.route('/video_feed')
def video_feed():
    if not scanning:
        return Response(status=204)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    global analytics_data
    data = request.get_json()
    feedback = data.get('feedback')

    if feedback and feedback in analytics_data['feedback_counts']:
        analytics_data['feedback_counts'][feedback] += 1
        save_analytics()
        return jsonify({"status": "success", "analytics": analytics_data})
    elif feedback == 'dismissed':
        return jsonify({"status": "success", "message": "Feedback dismissed"})
    else:
        return jsonify({"status": "error", "message": "Invalid feedback"}), 400

@app.route('/get_analytics')
def get_analytics_data():
    return jsonify(analytics_data)

if __name__ == "__main__":
    print(f"Running on {platform.system()} {platform.release()}")
    print(f"Using device: {device}")
    
    load_analytics()
    
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Trash Sorter</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
        }
        #startBtn {
            background-color: #4CAF50;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            font-size: 24px;
            cursor: pointer;
            margin: 20px 0;
        }
        #startBtn:hover {
            background-color: #45a049;
        }
        #videoFeed, #placeholderImg {
            width: 100%;
            max-width: 640px;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin: 20px 0;
        }
        #prediction {
            font-size: 20px;
            margin: 20px 0;
            color: #333;
        }
        #timer {
            font-size: 18px;
            color: #e74c3c;
            margin: 10px 0;
        }
        .hidden {
            display: none;
        }
        #feedbackModal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 450px;
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            text-align: left;
        }
        #feedbackModal h3 {
            margin-top: 0;
            color: #2c3e50;
            text-align: center;
        }
        #feedbackModal label {
            display: block;
            margin: 10px 0 5px;
        }
        #feedbackModal input[type="radio"] {
            margin-right: 8px;
        }
        #feedbackModal button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 15px;
        }
        #submitFeedbackBtn {
            background-color: #3498db;
            color: white;
            margin-right: 10px;
        }
        #submitFeedbackBtn:hover {
            background-color: #2980b9;
        }
        #closeFeedbackBtn {
            position: absolute;
            top: 10px;
            right: 15px;
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #aaa;
        }
        #closeFeedbackBtn:hover {
            color: #333;
        }
        #modalBackdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            z-index: 999;
        }
        #analyticsSummary {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: left;
            font-size: 14px;
            color: #555;
        }
        #analyticsSummary h3 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 15px;
        }
        #analyticsSummary ul {
            list-style: none;
            padding: 0;
        }
        #analyticsSummary li {
            margin-bottom: 5px;
        }
        #analyticsSummary strong {
            display: inline-block;
            width: 150px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Trash Sorter</h1>
        
        <div id="scannerControls">
            <button id="startBtn">Begin Scan</button>
            <div id="timer" class="hidden">Time remaining: 15s</div>
        </div>
        
        <div id="scannerView">
            <img id="placeholderImg" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgZmlsbD0iI2YyZjJmMiIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMjQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGRvbWluYW50LWJhc2VsaW5lPSJtaWRkbGUiIGZpbGw9IiM2NjY2NjYiPkNhbWVyYSBGZWVkIFdpbGwgQXBwZWFyIEhlcmU8L3RleHQ+PC9zdmc+" alt="Camera Placeholder">
            <div id="videoContainer" class="hidden">
                <img id="videoFeed" src="/video_feed" alt="Camera Feed">
            </div>
            <div id="prediction"></div>
            <div id="cameraStatus"></div>
        </div>

        <div id="analyticsSummary">
            <h3>Scan Analytics</h3>
            <ul id="analyticsList">
                <li>Loading analytics...</li>
            </ul>
        </div>
    </div>

    <div id="modalBackdrop" class="hidden"></div>
    <div id="feedbackModal" class="hidden">
        <button id="closeFeedbackBtn">&times;</button>
        <h3>Scan Complete!</h3>
        <p>Did you experience any issues during the scan?</p>
        <form id="feedbackForm">
            <label><input type="radio" name="feedback" value="no_problems" checked> No problems</label>
            <label><input type="radio" name="feedback" value="poor_camera"> Poor camera quality</label>
            <label><input type="radio" name="feedback" value="incorrect_id"> Incorrect item identification</label>
            <label><input type="radio" name="feedback" value="incorrect_bin"> Incorrect bin suggestion</label>
            <label><input type="radio" name="feedback" value="slow_scan"> Scanning was slow</label>
            <label><input type="radio" name="feedback" value="other"> Other issue</label>
            <button type="button" id="submitFeedbackBtn">Submit Feedback</button>
        </form>
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const videoContainer = document.getElementById('videoContainer');
        const videoFeed = document.getElementById('videoFeed');
        const placeholderImg = document.getElementById('placeholderImg');
        const predictionDiv = document.getElementById('prediction');
        const timerDiv = document.getElementById('timer');
        const cameraStatusDiv = document.getElementById('cameraStatus');
        const feedbackModal = document.getElementById('feedbackModal');
        const modalBackdrop = document.getElementById('modalBackdrop');
        const closeFeedbackBtn = document.getElementById('closeFeedbackBtn');
        const submitFeedbackBtn = document.getElementById('submitFeedbackBtn');
        const feedbackForm = document.getElementById('feedbackForm');
        const analyticsList = document.getElementById('analyticsList');

        let scanInterval = null;
        let timerInterval = null;
        let feedbackTimeout = null;
        let timeLeft = 15;

        startBtn.addEventListener('click', startScanning);
        closeFeedbackBtn.addEventListener('click', () => closeFeedbackModal(true));
        submitFeedbackBtn.addEventListener('click', submitFeedback);

        document.addEventListener('DOMContentLoaded', fetchAnalytics);

        videoFeed.addEventListener('load', () => {
            cameraStatusDiv.textContent = "Camera feed connected";
            cameraStatusDiv.style.color = "green";
        });

        videoFeed.addEventListener('error', (e) => {
            cameraStatusDiv.textContent = "Error loading camera feed. Check console for details.";
            cameraStatusDiv.style.color = "red";
            console.error("Video feed error:", e);
        });

        function startScanning() {
            cameraStatusDiv.textContent = "Connecting to camera...";
            cameraStatusDiv.style.color = "blue";
            
            fetch('/start_scan')
                .then(response => response.json())
                .then(data => {
                    if (data.status === "started") {
                        videoContainer.classList.remove('hidden');
                        placeholderImg.classList.add('hidden');
                        startBtn.disabled = true;
                        
                        const timestamp = new Date().getTime();
                        videoFeed.src = `/video_feed?t=${timestamp}`;
                        
                        timeLeft = 15;
                        timerDiv.textContent = `Time remaining: ${timeLeft}s`;
                        timerDiv.classList.remove('hidden');
                        
                        timerInterval = setInterval(() => {
                            timeLeft--;
                            timerDiv.textContent = `Time remaining: ${timeLeft}s`;
                            
                            if (timeLeft <= 0) {
                                stopScanning();
                            }
                        }, 1000);
                        
                        scanInterval = setInterval(checkStatus, 500);
                    }
                })
                .catch(error => {
                    cameraStatusDiv.textContent = "Failed to start scanning: " + error;
                    cameraStatusDiv.style.color = "red";
                });
        }

        function stopScanning() {
            clearInterval(scanInterval);
            clearInterval(timerInterval);
            
            fetch('/stop_scan')
                .then(response => response.json())
                .then(data => {
                    videoContainer.classList.add('hidden');
                    placeholderImg.classList.remove('hidden');
                    startBtn.disabled = false;
                    timerDiv.classList.add('hidden');
                    showFeedbackModal();
                });
        }

        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.prediction.class) {
                        predictionDiv.innerHTML = `<strong>${data.prediction.class}</strong>: ${data.prediction.prob.toFixed(1)}%<br>${data.prediction.bin}`;
                    }
                    
                    if (!data.scanning) {
                        stopScanning();
                    }
                });
        }

        function showFeedbackModal() {
            feedbackModal.classList.remove('hidden');
            modalBackdrop.classList.remove('hidden');
            feedbackForm.reset();

            clearTimeout(feedbackTimeout);
            feedbackTimeout = setTimeout(() => {
                closeFeedbackModal(false);
            }, 10000);
        }

        function closeFeedbackModal(manualClose) {
            clearTimeout(feedbackTimeout);
            feedbackModal.classList.add('hidden');
            modalBackdrop.classList.add('hidden');
            fetchAnalytics();
        }

        function submitFeedback() {
            clearTimeout(feedbackTimeout);
            const selectedFeedback = feedbackForm.querySelector('input[name="feedback"]:checked');
            if (selectedFeedback) {
                fetch('/submit_feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ feedback: selectedFeedback.value }),
                })
                .then(response => response.json())
                .then(data => {
                    fetchAnalytics();
                });
            }
            feedbackModal.classList.add('hidden');
            modalBackdrop.classList.add('hidden');
        }

        function fetchAnalytics() {
            fetch('/get_analytics')
                .then(response => response.json())
                .then(data => {
                    let html = `<li><strong>Total Scans:</strong> ${data.total_scans}</li>`;
                    for (const [key, value] of Object.entries(data.feedback_counts)) {
                        html += `<li><strong>${key.replace('_', ' ')}:</strong> ${value}</li>`;
                    }
                    analyticsList.innerHTML = html;
                });
        }
    </script>
</body>
</html>
        """)
    
    try:
        app.run(host='0.0.0.0', port=3000, debug=True, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        if camera is not None and camera.isOpened():
            camera.release()
