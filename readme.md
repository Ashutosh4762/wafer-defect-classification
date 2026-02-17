🧪 Wafer Defect Classification using MobileNetV3

This project focuses on automatic wafer defect classification using SEM (Scanning Electron Microscope) images.
The goal is to accurately classify wafer surface conditions into 8 defect categories using a lightweight and high-performance deep learning model, suitable for edge and real-time deployment.

✅ Accurate
✅ Lightweight (~6 MB model)
✅ Fast on CPU
✅ Deployment-ready (ONNX supported)

📌 Problem Statement

Manual inspection of wafer defects is:

⏱️ Time-consuming

❌ Error-prone

📉 Not scalable

This project automates wafer inspection using a CNN-based classifier trained on grayscale SEM images to identify defect types such as scratches, particles, residues, craters, and more.

🗂️ Defect Classes

The model classifies wafer images into 8 classes:

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
Input: 128 × 128 grayscale SEM images

Why MobileNetV3?

Depthwise separable convolutions

Extremely lightweight architecture

Optimized for low-latency inference

Ideal for edge & industrial applications

Key Architectural Features

Depthwise + pointwise convolutions

Inverted residual blocks

Squeeze-and-Excitation (SE) attention

ReLU / Hardswish activations

Fully-connected classification head

GeM pooling (improves texture sensitivity)

🔄 Data Pipeline
1️⃣ Raw Data
data/raw/
├── Ball Defects/
├── Clean/
├── Craters/
├── Flakes/
├── Others/
├── Particles/
├── Residues/
└── Scratches/

2️⃣ Preprocessing

Convert images to grayscale

Resize to 128 × 128

Data augmentation:

Horizontal & vertical flips

Rotation

Brightness & contrast variation

Ensures balanced dataset

Generates fixed number of images per class

3️⃣ Dataset Split

Automatically splits processed data into:

Train

Validation

Test

🏋️ Training Details

Loss Function: CrossEntropyLoss

Optimizer: Adam

Epochs: 20

Batch Size: Configurable via config file

Training Data: Processed images only

This ensures reproducibility and consistency.

📊 Model Performance
✅ Final Results

Test Accuracy: ~96%

Strong performance on:

Clean

Ball Defects

Scratches

Metrics Used

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

🖥️ User Interface (Streamlit)

A minimal Streamlit UI is provided to:

Upload single or multiple images

Preview images directly

Predict defect class for each image

Clean, simple, and user-friendly design

📦 Model Export & Deployment
🔁 ONNX Support

The trained PyTorch model can be exported to ONNX for:

Edge devices

Faster inference

Cross-platform deployment

Model size after export: ~6 MB
(No external .onnx_data file required)

🛠️ Project Structure

⚠️ Important: Folder structure must be inside a code block to render correctly.

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
🔧 Preprocess Data
python -m src.preprocessing.preprocess
python -m src.preprocessing.split_dataset

🏋️ Train Model
python -m src.train.train

📊 Evaluate Model
python -m src.evaluate.evaluate

🖥️ Launch UI
streamlit run ui/app.py

📦 Export to ONNX
python -m src.export.export_onnx

🏁 Conclusion

This project demonstrates that high-accuracy wafer defect classification can be achieved using a compact and efficient CNN model.
The solution is well-suited for industrial inspection, edge deployment, and real-time systems, making it both practical and scalable.