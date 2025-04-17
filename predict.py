import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

class GarbageCNN(nn.Module):
    def __init__(self, num_classes):
        super(GarbageCNN, self).__init__()
        
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise ValueError(f"Unsupported file format: {ext}. Please use .jpg, .jpeg, or .png files.")
    
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

def predict(model, image, transform, classes, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        top_probs, top_classes = torch.topk(probabilities, 3)
        
        results = []
        for i in range(3):
            class_idx = top_classes[i].item()
            results.append({
                'class': classes[class_idx],
                'probability': top_probs[i].item() * 100
            })
        
        primary_class = classes[outputs.argmax(1).item()]
        primary_prob = probabilities[outputs.argmax(1)].item() * 100
        
        return primary_class, primary_prob, results

def main():
    parser = argparse.ArgumentParser(description='Predict garbage type from image.')
    parser.add_argument('image_path', type=str, help='Path to the image file (jpg, jpeg, or png)')
    parser.add_argument('--model', type=str, default='garbage_classification_model.pth', 
                       help='Path to the trained model file')
    args = parser.parse_args()
    
    classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
              'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        image = load_image(args.image_path)
        print(f"Successfully loaded image: {args.image_path}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        model = GarbageCNN(num_classes=len(classes))
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.to(device)
        print(f"Successfully loaded model from: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        primary_class, primary_prob, results = predict(model, image, preprocess, classes, device)
        
        print("\nPrediction Results:")
        print(f"Primary prediction: {primary_class} with {primary_prob:.2f}% confidence")
        print("\nTop 3 predictions:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['class']}: {result['probability']:.2f}%")
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()