import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from predict import GarbageCNN, predict

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

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame (BGR) to PIL Image (RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Predict
    primary_class, primary_prob, results = predict(model, pil_img, preprocess, classes, device)

    # Overlay prediction on frame
    label = f"{primary_class}: {primary_prob:.1f}%"
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Determine bin
    bin_name = None
    for key, items in bin_sort.items():
        if primary_class in items:
            bin_name = key
            break
    if bin_name:
        bin_label = f"Place in the {bin_name} bin"
    else:
        bin_label = "Bin not found"

    # Overlay bin suggestion
    cv2.putText(frame, bin_label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    # Show frame
    cv2.imshow('Live Prediction', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
