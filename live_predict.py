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
from flask import Flask, render_template, Response, jsonify

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
    "recycling": ["green-glass", "brown-glass", "paper", "white-glass", "metal", "plastic", "cardboard"],
    "compost": ["biological"],
    "hazardous": ["battery"],
    "clothes": ["clothes", "shoes"],
    "trash": ["trash"]
}

app = Flask(__name__, template_folder='templates', static_folder='static')

scanning = False
scan_thread = None
last_prediction = {"class": "", "prob": 0, "bin": ""}
camera = None

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
        
        bin_label = f"Place in the {bin_name} bin" if bin_name != "unknown" else "Bin not found"
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
    global scanning, scan_thread, camera
    
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
    global scanning, camera
    
    scanning = False
    
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        print("Camera released")
    
    return jsonify({"status": "stopped"})

@app.route('/status')
def get_status():
    return jsonify({
        "scanning": scanning,
        "prediction": last_prediction
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import os
    import platform
    
    print(f"Running on {platform.system()} {platform.release()}")
    print(f"Using device: {device}")
    
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
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const videoContainer = document.getElementById('videoContainer');
        const videoFeed = document.getElementById('videoFeed');
        const placeholderImg = document.getElementById('placeholderImg');
        const predictionDiv = document.getElementById('prediction');
        const timerDiv = document.getElementById('timer');
        const cameraStatusDiv = document.getElementById('cameraStatus');
        
        let scanInterval = null;
        let timerInterval = null;
        let timeLeft = 15;
        
        startBtn.addEventListener('click', startScanning);
        
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
                });
        }
        
        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.prediction.class) {
                        predictionDiv.innerHTML = `<strong>${data.prediction.class}</strong>: ${data.prediction.prob.toFixed(1)}%<br>Place in the <strong>${data.prediction.bin}</strong> bin`;
                    }
                    
                    if (!data.scanning) {
                        stopScanning();
                    }
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
