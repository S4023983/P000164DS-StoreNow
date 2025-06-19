Rooftop Usable Area Estimation and Obstruction Detection

This project implements a fully automated AI-powered pipeline for rooftop analysis using satellite imagery. The system classifies roof types, predicts optimal solar panel orientation, detects obstructions on roofs, and estimates the final usable rooftop area. It integrates deep learning, computer vision, and geospatial analysis into a user-friendly Streamlit web application.

🌌 Features

1. Address-based and Coordinate-based Geocoding (Google Maps API)
2. Roof Type Classification using EfficientNet-B3 model
3. Google Solar API integration for solar panel orientation and area estimation
4. Rooftop Segmentation using DeepLabV3+ with ResNet-101 backbone
5. Obstruction Detection using Facebook's Segment Anything Model (SAM)
6. Area Computation using GPS coordinates and roof pitch correction
7. Streamlit Web Dashboard for interactive analysis

🔢 Project Structure

S4017535-P000164DS/
│
├── app.py                         # Streamlit web app interface
├── image_classification_estimator.py  # Roof classification module
├── roof_obstruction_estimator.py  # Rooftop segmentation & obstruction detection
├── model_type_segmentation_engine.py  # DeepLabV3+ segmentation training script
├── roof_type_classification_engine.py # EfficientNet-B3 classification training script
├── overlay_visualizer.py          # (Optional) Utility for visualization of segmentation overlays
├── inference.py                   # Additional inference module (for custom standalone inference)
├── model_utils.py                 # Utilities for model loading & evaluation
│
├── dataset/                       # Input datasets (images & masks)
│
├── models/
│   ├── best_model.pth             # Trained EfficientNet model weights
│   ├── checkpoint_256.pth         # Trained segmentation model checkpoint
│   ├── inference.py
│   └── model_utils.py
│
├── myenv/                         # Python virtual environment (optional if using venv)
│
├── roof_obstruction_estimator_images/  # Saved annotated images after analysis
│   └── final_overlay_*.png        # Annotated outputs
│
├── segment-anything/              # Facebook Segment Anything Model (SAM) code
│   ├── segment_anything/
│   ├── assets/
│   ├── scripts/
│   └── notebooks/
│
├── requirements.txt               # List of dependencies
├── label_map.json                 # Classification label mappings
├── roof_type_labels.csv           # Dataset labels for classification
└── .gitignore


🔄 File Descriptions

app.py

1. Streamlit web app entry point.
2. Allows user to enter coordinates and roof angle.
3. Displays satellite image, roof type, Google Solar API orientation (if available), performs roof segmentation, obstruction detection, and shows annotated output.

image_classification_estimator.py

1. Loads EfficientNet-B3 classifier.
2. Fetches satellite images and geocodes addresses.
3. Calls Google Solar API for solar insights.
4. Classifies roof type from satellite image.

roof_obstruction_estimator.py

1. Loads trained DeepLabV3+ segmentation model.
2. Segments roof boundaries.
3. Detects obstructions inside roof area using Facebook's Segment Anything Model (SAM).
4. Calculates usable area after obstruction removal.

model_type_segmentation_engine.py

1. Preprocessing and heavy augmentation for segmentation dataset.
2. Trains DeepLabV3+ model for rooftop segmentation.

roof_type_classification_engine.py

1. Preprocessing, augmentation and training for roof classification.
2. Trains EfficientNet-B3 model.
3. Implements 5-fold Stratified Cross Validation.
4. overlay_visualizer.py
5. Utility for visualizing roof boundaries and obstructions.
6. Saves final annotated image overlays.

model_utils.py

1. Helper functions for resizing and preprocessing.
2. Ensures all images are padded to square shape.

inference.py

1. Runs standalone roof classification prediction using trained EfficientNet model.

label_map.json

1. Label-to-index mapping used by classification model.

roof_type_labels.csv

1. Training labels for classification.

requirements.txt

1. Full list of required Python packages.

🛠️ Installation

1. Clone the Repository

git clone <your-repository-url> 
cd <project-directory>

2. Create Virtual Environment (Recommended)

python -m venv venv
# For Linux/macOS:
source venv/bin/activate
# For Windows:
venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Install Segment Anything (SAM)

git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .

🔑 API Keys Setup:

You need both Google Maps API Key and Google Solar API Key.
Go to Google Cloud Console

Enable:

1. Maps Static API
2. Geocoding API
3. Solar API
4. Setup Billing (mandatory for Solar API access)
5. Replace GOOGLE_API_KEY in: image_classification_estimator.py and roof_obstruction_estimator.py

🕹️ Running the Streamlit App

After installation and API setup, simply run ---> streamlit run app.py

The app will open automatically in your browser.

🔄 Usage Instructions

Enter latitude, longitude coordinates directly in Streamlit interface.

Enter roof tilt angle via the provided slider.

The app performs:

1. Fetched satellite image from Google Maps Static API.
2. Classifies roof type (gable, hip, flat, etc.).
3. Calls Google Solar API for panel orientation insights.
4. Segments roof boundary via DeepLabV3+.
5. Detects obstructions (chimneys, HVAC units) via SAM.
6. Computes total usable rooftop area.
7. Displays annotated results.

🔧 Model Training Commands (VSCode)

Train Roof Classification Model: python roof_type_classification_engine.py

Train Roof Segmentation Model: python model_type_segmentation_engine.py

Test Classification Model Inference: python inference.py

Run Full Pipeline (Outside Streamlit for testing): python roof_obstruction_estimator.py

🔄 Model Training Pipeline commands

1. Roof Type Classification (EfficientNet-B3)

Script: roof_type_classification_engine.py (Command)

Dataset: roof_type_labels.csv + classification images
5-fold cross-validation with data augmentation

2. Rooftop Segmentation (DeepLabV3+)

Script: model_type_segmentation_engine.py (Command)

Dataset: roof_masks.zip (image + binary mask)
Heavy augmentation using Albumentations

3. Obstruction Detection (Segment Anything Model)

Script integrated inside roof_obstruction_estimator.py (Command)
Uses pretrained SAM model from Facebook Meta AI

📜 Acknowledgements

1. Google Maps Platform
2. Google Solar API
3. Facebook Research Segment Anything Model
4. EfficientNet (from efficientnet_pytorch)
5. DeepLabV3+ (from segmentation_models_pytorch)
6. Dataset manually labeled for segmentation

⚠️ Limitations 

1. Google Solar API requires billing.
2. Segment Anything (SAM) is computationally heavy.
3. Models trained on limited data, may require fine-tuning for global generalization.

🌐 Future Work

1. Fully automated dataset annotation pipeline.
2. Fine-tune SAM model for rooftop-specific obstruction accuracy.
3. Full deployment on cloud platforms.
4. Integrate solar potential forecasting using weather datasets.


