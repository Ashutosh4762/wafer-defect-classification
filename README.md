Wafer Defect Classification using MobileNetV3

This project focuses on automatic wafer defect classification using SEM (Scanning Electron Microscope) images. The goal is to accurately classify wafer surface conditions into 8 categories using a lightweight, high-performance deep learning model suitable for edge and real-time deployment.

The solution is designed to be:
✅ Accurate
✅ Lightweight (~6 MB model)
✅ Fast on CPU
✅ Deployment-ready (ONNX supported)

🧪 Problem Statement
Manual inspection of wafer defects is:
Time-consuming
Error-prone
Not scalable
This project automates the inspection process by using a CNN-based classifier trained on grayscale SEM images to identify defect types such as scratches, particles, residues, and more.

🗂️ Defect Classes
The model classifies wafer images into the following 8 classes:
Ball Defects
Clean
Craters
Flakes
Others
Particles
Residues
Scratches


🧠 Model Architecture
Backbone: MobileNetV3-Small
Input: 128 × 128 grayscale images



Why MobileNetV3?

Depthwise separable convolutions
Extremely lightweight
Optimized for low-latency inference
Ideal for edge and industrial use cases
Key Architectural Features
Depthwise + pointwise convolutions
Inverted residual blocks
Efficient channel attention (SE blocks)
ReLU / Hardswish activations
Final fully-connected classification head

🔄 Data Pipeline
1️⃣ Raw Data
data/raw/ organized by class folders (Ball Defects, Clean, Craters, etc.).

2️⃣ Preprocessing
Convert to grayscale
Resize to 128*128
Data augmentation: Flips, Rotations, Brightness & contrast variation
Ensures balanced dataset and generates fixed number of images per class

3️⃣ Dataset Split
Automatically split into:
Train
Validation
Test

🏋️ Training Details
Loss Function: CrossEntropyLoss
Optimizer: Adam
Epochs: 20
Batch Size: Configurable
The model is trained on processed data only, ensuring consistency and reproducibility.



📊 Model Performance

✅ Final Results
Test Accuracy: ~96%
Strong performance on: Clean, Ball Defects, and Scratches.
Metrics Used
Accuracy
Precision
Recall
F1-Score
Confusion Matrix


🖥️ User Interface (Streamlit)
A minimal Streamlit UI is provided to:
Upload single or multiple images
View images directly
Predict defect class for each image
Clean and easy-to-use interface



📦 Model Export & Deployment

ONNX Support
The trained PyTorch model can be exported to ONNX for:
Edge devices
Faster inference
Cross-platform deployment
Model size after export: ~6 MB (No external onnx_data required)


🛠️ Project Structure

wafer-defect-classification/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── models/
│   ├── train/
│   ├── evaluate/
│   └── export/
├── ui/
│   └── app.py
├── models/
│   ├── mobilenet_best.pth
│   └── mobilenet_latest.pth
├── reports/
│   └── confusion_matrix.png
├── config/
│   └── config.yaml
└── README.md



▶️ How to Run
1. Preprocess Data
python -m src.preprocessing.preprocess
python -m src.preprocessing.split_dataset
2. Train Model
python -m src.train.train
3. Evaluate Model
python -m src.evaluate.evaluate
4. Launch UI
python -m streamlit run ui/app.py
5. Export to ONNX
python -m src.export.export_onnx



🏁 Conclusion
This project demonstrates that high-accuracy wafer defect classification can be achieved using a compact, efficient CNN model, making it ideal for industrial, edge, and real-time inspection systems.
