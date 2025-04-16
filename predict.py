import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the model architecture (same as in training)
class GarbageCNN(nn.Module):
    def __init__(self, num_classes):
        super(GarbageCNN, self).__init__()
        
        # Load a pretrained ResNet-18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def load_image(image_path):
    """Load and preprocess an image for prediction."""
    # Check if file exists and is an image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check file extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise ValueError(f"Unsupported file format: {ext}. Please use .jpg, .jpeg, or .png files.")
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

def predict(model, image, transform, classes, device):
    """Make a prediction on a single image."""
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get top 3 predictions
        top_probs, top_classes = torch.topk(probabilities, 3)
        
        results = []
        for i in range(3):
            class_idx = top_classes[i].item()
            results.append({
                'class': classes[class_idx],
                'probability': top_probs[i].item() * 100  # Convert to percentage
            })
        
        # Get the primary prediction
        primary_class = classes[outputs.argmax(1).item()]
        primary_prob = probabilities[outputs.argmax(1)].item() * 100
        
        return primary_class, primary_prob, results

def visualize_prediction(image, primary_class, primary_prob, results):
    """Visualize the image with prediction results."""
    plt.figure(figsize=(10, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display the predictions
    plt.subplot(1, 2, 2)
    
    classes = [result['class'] for result in results]
    probs = [result['probability'] for result in results]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability (%)')
    plt.title('Predictions')
    
    # Add text for main prediction
    plt.figtext(0.5, 0.01, f'Primary prediction: {primary_class} ({primary_prob:.2f}%)', 
               ha='center', fontsize=12, bbox={'facecolor':'lightgreen', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict garbage type from image.')
    parser.add_argument('image_path', type=str, help='Path to the image file (jpg, jpeg, or png)')
    parser.add_argument('--model', type=str, default='garbage_classification_model.pth', 
                       help='Path to the trained model file')
    parser.add_argument('--no_display', action='store_true', 
                       help='Do not display visualization (just print results)')
    args = parser.parse_args()
    
    # Define the classes (must match the order used during training)
    classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
              'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess the image
    try:
        image = load_image(args.image_path)
        print(f"Successfully loaded image: {args.image_path}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Define the preprocessing transform (same as in training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the model
    try:
        model = GarbageCNN(num_classes=len(classes))
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.to(device)
        print(f"Successfully loaded model from: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    try:
        primary_class, primary_prob, results = predict(model, image, preprocess, classes, device)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Primary prediction: {primary_class} with {primary_prob:.2f}% confidence")
        print("\nTop 3 predictions:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['class']}: {result['probability']:.2f}%")
        
        # Visualize results if not disabled
        if not args.no_display:
            visualize_prediction(image, primary_class, primary_prob, results)
            print("\nVisualization saved as 'prediction_result.png'")
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()