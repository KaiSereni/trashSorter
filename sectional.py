import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb
from sklearn.cluster import KMeans
import random
import colorsys

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
    """Load and check image."""
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check file extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise ValueError(f"Unsupported file format: {ext}. Please use .jpg, .jpeg, or .png files.")
    
    # Load the image
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

def segment_image(image, n_segments=50, compactness=20, sigma=5):
    """Segment image into regions with similar texture/color using SLIC algorithm."""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Apply SLIC segmentation
    print(f"Segmenting image into approximately {n_segments} regions...")
    segments = slic(img_array, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)
    
    # Count actual number of segments
    num_segments = len(np.unique(segments))
    print(f"Image segmented into {num_segments} regions")
    
    return segments, num_segments

def extract_segments(image, segments, min_size=500):
    """Extract each segment as a separate image with original shape and black background."""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Get unique segment labels
    unique_segments = np.unique(segments)
    
    # Prepare the result
    segment_images = []
    segment_masks = []
    valid_segment_labels = []
    bounding_boxes = []
    
    # For each segment
    for segment_id in unique_segments:
        # Create a mask for this segment
        mask = segments == segment_id
        
        # Skip if segment is too small
        if np.sum(mask) < min_size:
            continue
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            continue
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Ensure minimum size for classification
        height = rmax - rmin
        width = cmax - cmin
        
        if height < 32 or width < 32:
            continue
        
        # Create a black image with only this segment visible
        segment_img = np.zeros_like(img_array)
        segment_img[mask] = img_array[mask]
        
        # Save the bounding box
        bounding_boxes.append((rmin, rmax, cmin, cmax))
        
        # Save the segment image, mask, and label
        segment_images.append(segment_img)
        segment_masks.append(mask)
        valid_segment_labels.append(segment_id)
    
    print(f"Extracted {len(segment_images)} valid segments for classification")
    return segment_images, segment_masks, valid_segment_labels, bounding_boxes

def classify_segments(model, segment_images, bounding_boxes, transform, classes, device):
    """Classify each segment using the trained model."""
    results = []
    
    print("Classifying segments...")
    for i, (segment_img, bbox) in enumerate(zip(segment_images, bounding_boxes)):
        # Extract the segment from the bounding box
        rmin, rmax, cmin, cmax = bbox
        segment_crop = segment_img[rmin:rmax, cmin:cmax]
        
        # Convert to PIL Image
        segment_pil = Image.fromarray(segment_crop)
        
        # Apply transformations for model
        image_tensor = transform(segment_pil).unsqueeze(0).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            class_idx = outputs.argmax(1).item()
            confidence = probabilities[class_idx].item() * 100
        
        # Store the results
        results.append({
            'segment_id': i,
            'class': classes[class_idx],
            'confidence': confidence
        })
    
    return results

def get_distinct_colors(n):
    """Generate visually distinct colors."""
    colors = []
    for i in range(n):
        # Use golden ratio to spread hues around color wheel
        hue = (i * 0.618033988749895) % 1.0
        # Generate HSV color with max saturation and value
        hsv_color = (hue, 0.9, 0.9)
        # Convert to RGB and scale to 0-255 range
        rgb_color = tuple(int(255 * c) for c in hsv_to_rgb(np.array([[hsv_color]]))[0][0])
        colors.append(rgb_color)
    return colors

def generate_segmentation_map(image, segments, classification_results, valid_segment_labels):
    """Generate an image showing the material of each section."""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create a copy for visualization
    vis_image = img_array.copy()
    
    # Get unique class names from results
    classes = list(set(result['class'] for result in classification_results))
    
    # Create a color mapping for classes
    color_map = dict(zip(classes, get_distinct_colors(len(classes))))
    
    # Create a map from valid_segment_labels to classification_results
    segment_to_class = {}
    for i, result in enumerate(classification_results):
        segment_to_class[valid_segment_labels[i]] = result['class']
    
    # Create a color mask based on segments and their classifications
    color_mask = np.zeros_like(img_array)
    
    for segment_id in np.unique(segments):
        if segment_id in segment_to_class:
            mask = segments == segment_id
            material_class = segment_to_class[segment_id]
            color = color_map[material_class]
            
            # Apply color to this segment in the mask
            color_mask[mask] = color
    
    # Create a semi-transparent overlay
    alpha = 0.5
    segmentation_map = (img_array * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    
    # Mark boundaries between segments
    segmentation_map = mark_boundaries(segmentation_map, segments, color=(1, 1, 1), mode='thick')
    segmentation_map = (segmentation_map * 255).astype(np.uint8)
    
    # Convert to PIL Image for annotation
    segmentation_pil = Image.fromarray(segmentation_map)
    draw = ImageDraw.Draw(segmentation_pil)
    
    # Attempt to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Annotate the segments with text
    for segment_id in np.unique(segments):
        if segment_id in segment_to_class:
            mask = segments == segment_id
            material_class = segment_to_class[segment_id]
            
            # Find center of mass of the segment for text placement
            indices = np.where(mask)
            if len(indices[0]) > 0:
                center_y = int(np.mean(indices[0]))
                center_x = int(np.mean(indices[1]))
                
                # Draw text with contrasting color for readability
                draw.text((center_x, center_y), material_class, fill=(255, 255, 255), font=font)
    
    # Create legend for the colors
    legend_img = Image.new('RGB', (200, 30 * len(classes) + 10), (255, 255, 255))
    legend_draw = ImageDraw.Draw(legend_img)
    
    for i, cls in enumerate(classes):
        color = color_map[cls]
        legend_draw.rectangle([(10, i * 30 + 10), (30, i * 30 + 30)], fill=color)
        legend_draw.text((40, i * 30 + 10), cls, fill=(0, 0, 0), font=font)
    
    # Combine the segmentation map and legend
    result_width = segmentation_pil.width
    result_height = segmentation_pil.height + legend_img.height
    
    result_img = Image.new('RGB', (result_width, result_height), (255, 255, 255))
    result_img.paste(segmentation_pil, (0, 0))
    result_img.paste(legend_img, (0, segmentation_pil.height))
    
    return result_img

def main():
    parser = argparse.ArgumentParser(description='Segment and classify trash materials in an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file (jpg, jpeg, or png)')
    parser.add_argument('--model', type=str, default='garbage_classification_model.pth', 
                       help='Path to the trained model file')
    parser.add_argument('--segments', type=int, default=50, 
                       help='Number of segments to divide the image into')
    parser.add_argument('--output', type=str, default='segmentation_result.png',
                       help='Output image filename')
    parser.add_argument('--compactness', type=float, default=20,
                       help='Compactness parameter for SLIC (higher values make more compact segments)')
    parser.add_argument('--sigma', type=float, default=5,
                       help='Sigma parameter for SLIC (higher values make segments more sensitive to color)')
    parser.add_argument('--min_size', type=int, default=500,
                       help='Minimum size (in pixels) for a segment to be classified')
    
    args = parser.parse_args()
    
    # Define the classes (must match the order used during training)
    classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
              'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the image
    try:
        image = load_image(args.image_path)
        print(f"Successfully loaded image: {args.image_path}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Segment the image
    try:
        segments, num_segments = segment_image(
            image, 
            n_segments=args.segments, 
            compactness=args.compactness, 
            sigma=args.sigma
        )
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return
    
    # Extract individual segments
    try:
        segment_images, segment_masks, valid_segment_labels, bounding_boxes = extract_segments(
            image, 
            segments, 
            min_size=args.min_size
        )
    except Exception as e:
        print(f"Error extracting segments: {e}")
        return
    
    # Define the preprocessing transform
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
    
    # Classify each segment
    try:
        classification_results = classify_segments(
            model, 
            segment_images, 
            bounding_boxes, 
            preprocess, 
            classes, 
            device
        )
    except Exception as e:
        print(f"Error during classification: {e}")
        return
    
    # Generate segmentation map
    try:
        segmentation_map = generate_segmentation_map(
            image, 
            segments, 
            classification_results, 
            valid_segment_labels
        )
        
        # Save the result
        segmentation_map.save(args.output)
        print(f"Segmentation result saved to: {args.output}")
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(np.array(segmentation_map))
        plt.axis('off')
        plt.title('Material Segmentation Result')
        plt.show()
        
    except Exception as e:
        print(f"Error generating segmentation map: {e}")
        return
    
    # Print classification results
    print("\nClassification Results:")
    for result in classification_results:
        print(f"Segment {result['segment_id']+1}: {result['class']} ({result['confidence']:.2f}%)")

if __name__ == "__main__":
    main()