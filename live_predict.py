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

# Define the classes (must match the order used during training)
classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
          'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the preprocessing transform (same as in training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
model = GarbageCNN(num_classes=len(classes))
model.load_state_dict(torch.load('garbage_classification_model.pth', map_location=device))
model = model.to(device)

# Define bin sorting rules
bin_sort = {
    "recycling": ["green-glass", "brown-glass", "paper", "white-glass", "metal", "plastic", "cardboard"],
    "compost": ["biological"],
    "hazardous": ["battery"],
    "clothes": ["clothes", "shoes"],
    "trash": ["trash"]
}

# Create Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables to manage scanning state
scanning = False
scan_thread = None
last_prediction = {"class": "", "prob": 0, "bin": ""}
camera = None  # Global camera object to avoid repeatedly opening/closing

def get_frame():
    """Capture a frame from the webcam and make a prediction"""
    global last_prediction, camera
    
    try:
        if camera is None or not camera.isOpened():
            # Try different camera indexes for Mac
            for idx in [0, 1, -1]:
                camera = cv2.VideoCapture(idx)
                if camera.isOpened():
                    print(f"Successfully opened camera with index {idx}")
                    # On Mac, set the resolution explicitly
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    # Add a small delay after opening to initialize camera
                    time.sleep(0.5)
                    break
                else:
                    print(f"Failed to open camera with index {idx}")
            
            if not camera.isOpened():
                print("ERROR: Cannot open webcam after trying multiple indexes")
                return None, "Cannot open webcam"
        
        # Read multiple frames to ensure we get a fresh frame (can help with some webcams)
        for _ in range(3):  # Skip a few frames to get a more recent one
            ret, frame = camera.read()
            if not ret:
                print("Warning: Failed to grab frame on attempt, retrying...")
                time.sleep(0.1)
        
        if not ret or frame is None:
            print("ERROR: Failed to grab frame after multiple attempts")
            return None, "Failed to grab frame"
        
        # Convert frame (BGR) to PIL Image (RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Predict
        primary_class, primary_prob, results = predict(model, pil_img, preprocess, classes, device)
        
        # Determine bin
        bin_name = None
        for key, items in bin_sort.items():
            if primary_class in items:
                bin_name = key
                break
        if not bin_name:
            bin_name = "unknown"
        
        # Update last prediction
        last_prediction = {
            "class": primary_class,
            "prob": primary_prob,
            "bin": bin_name
        }
        
        # Overlay prediction on frame
        label = f"{primary_class}: {primary_prob:.1f}%"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Overlay bin suggestion
        bin_label = f"Place in the {bin_name} bin" if bin_name != "unknown" else "Bin not found"
        cv2.putText(frame, bin_label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes(), None
    except Exception as e:
        print(f"Camera error: {e}")
        return None, f"Camera error: {e}"

def generate_frames():
    """Generator function for streaming video frames"""
    global scanning, camera
    
    start_time = time.time()
    scan_duration = 15  # seconds
    frame_count = 0
    
    try:
        while scanning and time.time() - start_time < scan_duration:
            frame_data, error = get_frame()
            
            if frame_data:
                frame_count += 1
                print(f"Generated frame {frame_count}")
                
                # Proper MIME format for multipart streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            elif error:
                print(f"Error getting frame: {error}")
                time.sleep(0.5)  # Wait a bit before retrying
                continue
                
            # Reduce frame rate to avoid overloading the browser
            time.sleep(0.2)
            
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        # End scanning
        scanning = False
        print(f"Streaming ended after {frame_count} frames")

def scanning_thread():
    """Thread function for scanning"""
    global scanning
    
    # Scan for 15 seconds
    time.sleep(15)
    
    # End scanning
    scanning = False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/start_scan')
def start_scan():
    """Start the scanning process"""
    global scanning, scan_thread, camera
    
    if not scanning:
        scanning = True
        
        # Initialize camera here to ensure it's open for the entire session
        if camera is None or not camera.isOpened():
            print("Initializing camera...")
            for idx in [0, 1, -1]:
                camera = cv2.VideoCapture(idx)
                if camera.isOpened():
                    print(f"Successfully opened camera with index {idx}")
                    break
                else:
                    print(f"Failed to open camera with index {idx}")
        
        # Start a thread to end scanning after 15 seconds
        scan_thread = threading.Thread(target=scanning_thread)
        scan_thread.daemon = True
        scan_thread.start()
        return jsonify({"status": "started"})
    
    return jsonify({"status": "already_running"})

@app.route('/stop_scan')
def stop_scan():
    """Stop the scanning process"""
    global scanning, camera
    
    scanning = False
    
    # Release the camera when stopping the scan
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        print("Camera released")
    
    return jsonify({"status": "stopped"})

@app.route('/status')
def get_status():
    """Get the current scanning status and prediction"""
    return jsonify({
        "scanning": scanning,
        "prediction": last_prediction
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Create necessary directories for templates and static files
    import os
    import platform
    
    print(f"Running on {platform.system()} {platform.release()}")
    print(f"Using device: {device}")
    
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Create template files
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
        
        // Add event listeners to detect if video is working
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
            // Reset camera status
            cameraStatusDiv.textContent = "Connecting to camera...";
            cameraStatusDiv.style.color = "blue";
            
            // Call API to start scanning
            fetch('/start_scan')
                .then(response => response.json())
                .then(data => {
                    if (data.status === "started") {
                        // Show camera feed, hide placeholder
                        videoContainer.classList.remove('hidden');
                        placeholderImg.classList.add('hidden');
                        startBtn.disabled = true;
                        
                        // Force reload the video feed src to ensure a fresh connection
                        const timestamp = new Date().getTime();
                        videoFeed.src = `/video_feed?t=${timestamp}`;
                        
                        // Start timer display
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
                        
                        // Poll status for updates
                        scanInterval = setInterval(checkStatus, 500);
                    }
                })
                .catch(error => {
                    cameraStatusDiv.textContent = "Failed to start scanning: " + error;
                    cameraStatusDiv.style.color = "red";
                });
        }
        
        function stopScanning() {
            // Clear intervals
            clearInterval(scanInterval);
            clearInterval(timerInterval);
            
            // Call API to stop scanning
            fetch('/stop_scan')
                .then(response => response.json())
                .then(data => {
                    // Hide camera feed, show placeholder
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
                    // Update prediction display
                    if (data.prediction.class) {
                        predictionDiv.innerHTML = `<strong>${data.prediction.class}</strong>: ${data.prediction.prob.toFixed(1)}%<br>Place in the <strong>${data.prediction.bin}</strong> bin`;
                    }
                    
                    // Check if scanning has stopped
                    if (!data.scanning) {
                        stopScanning();
                    }
                });
        }
    </script>
</body>
</html>
        """)
    
    # Run the Flask app
    try:
        # Turn off reloading when debugging to avoid camera issues
        app.run(host='0.0.0.0', port=3000, debug=True, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        # If camera was opened, make sure to release it
        if camera is not None and camera.isOpened():
            camera.release()
