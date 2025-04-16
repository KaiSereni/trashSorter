# Trash Sorter

A computer vision application that helps classify waste items into appropriate disposal bins using a deep learning model. Point your camera at an item, and the system will tell you whether it belongs in recycling, compost, hazardous waste, clothing donation, or regular trash.

## Features

- Real-time trash classification through your webcam
- Classification of 12 different waste categories
- Bin recommendations for proper disposal
- Simple web interface accessible from any browser
- Option to use either the pre-trained model or train your own

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- Flask
- Matplotlib
- NumPy
- tqdm
- kagglehub (for training only)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/trash-sorter.git
   cd trash-sorter
   ```

2. Install the required dependencies:
   ```
   pip install torch torchvision opencv-python flask matplotlib numpy tqdm pillow
   ```

3. Download the pre-trained model from the Releases section of this repository and place it in the project root directory.

## Usage

### Using the Live Webcam Classifier

The easiest way to use the Trash Sorter is through the web interface with your webcam:

1. Start the Flask application:
   ```
   python live_predict.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:3000
   ```

3. Click the "Begin Scan" button and point your camera at a waste item.

4. The application will classify the item and suggest which bin to use for disposal.

### Using the Command-Line Predictor

To classify a single image file:

```
python predict.py path/to/your/image.jpg
```

Optional arguments:
- `--model`: Specify a different model file (default: garbage_classification_model.pth)
- `--no_display`: Prevent the visualization window from appearing

### Training Your Own Model

If you want to train your own model instead of using the pre-trained one:

1. Install the additional training dependency:
   ```
   pip install kagglehub
   ```

2. Run the training script:
   ```
   python train.py
   ```

   This will:
   - Download the garbage classification dataset from Kaggle
   - Train a ResNet18-based model on the dataset
   - Save the trained model as `garbage_classification_model.pth`
   - Generate training curves and accuracy visualizations

## Model Information

- Architecture: Modified ResNet18
- Categories: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
- Bin sorting:
  - Recycling: green-glass, brown-glass, paper, white-glass, metal, plastic, cardboard
  - Compost: biological
  - Hazardous: battery
  - Clothes: clothes, shoes
  - Trash: trash

## Project Structure

- `train.py`: Script for training the classification model
- `predict.py`: Command-line tool for classifying single images
- `live_predict.py`: Web application for real-time classification
- `templates/index.html`: Web interface template (auto-generated)
- `garbage_classification_model.pth`: Pre-trained model file (download from Releases)

## Dataset

The model is trained on the [Garbage Classification dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) from Kaggle, which contains images of various waste items across 12 categories.

## Troubleshooting

- **Webcam issues**: If the camera doesn't work, try using a different camera index by modifying the camera index values in `live_predict.py`.
- **Model loading errors**: Ensure you have downloaded the correct model file from Releases.
- **Training crashes**: Training requires significant RAM and GPU memory. If it crashes, try reducing the batch size in `train.py`.

## License

[MIT License](LICENSE)