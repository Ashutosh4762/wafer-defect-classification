## Wafer Defect Classification using MobileNetV3

This project focuses on automatic wafer defect classification using SEM (Scanning Electron Microscope) images. The goal is to accurately classify wafer surface conditions into 8 categories using a lightweight, high-performance deep learning model suitable for edge and real-time deployment.

The solution is designed to be:
- ✅ Accurate
- ✅ Lightweight (~6 MB model)
- ✅ Fast on CPU
- ✅ Deployment-ready (ONNX supported)

## Problem Statement
Manual inspection of wafer defects is:
Time-consuming
Error-prone
Not scalable
This project automates the inspection process by using a CNN-based classifier trained on grayscale SEM images to identify defect types such as scratches, particles, residues, and more.

## Dataset
Total images planned/current: 1600
No. of classes: 8 
Class list: Scratches, Particles, Ball defects, Craters, Flakes, 
Residues, Clean , Others
Class balance plan: 200 per class
Train/Val/Test split: 80 / 10 / 10
Image type: Grayscale 
Labeling method/source: manual 

One Drive Link for Dataset : https://obtmhl-my.sharepoint.com/:f:/g/personal/ashutosh_kumar_orbitsolutions_net/IgAiGahfY-rSSoWl_s1AeVJZAbeI24zzUZJAjkt_ExsHJFA?e=eaKGSJ


## Model Architecture
Backbone: MobileNetV3-Small
Input: 128 × 128 grayscale images



## Why MobileNetV3?

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

## Data Pipeline
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

## Training Details
```
Loss Function: CrossEntropyLoss
Optimizer: Adam
Epochs: 20
Batch Size: Configurable
The model is trained on processed data only, ensuring consistency and reproducibility.
```


## Model Performance
```
✅ Final Results
Test Accuracy: ~96%
Strong performance on: Clean, Ball Defects, and Scratches.
Metrics Used
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
```
<img width="2000" height="1600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e407e17e-8c9a-46af-95e9-9b09ce5f33c7" />
```
```

## 🖥️ User Interface (Streamlit)
A minimal Streamlit UI is provided to:
Upload single or multiple images
View images directly
Predict defect class for each image
Clean and easy-to-use interface
<img width="1905" height="820" alt="Screenshot 2026-02-07 131138" src="https://github.com/user-attachments/assets/f300f52e-3f09-43ca-a19c-d787e5fba0b6" />

<img width="1848" height="880" alt="image" src="https://github.com/user-attachments/assets/8db3844b-b2e0-4a44-a461-a7cc2de5b038" />


## Model Export & Deployment

ONNX Support
The trained PyTorch model can be exported to ONNX for:
Edge devices
Faster inference
Cross-platform deployment
Model size after export: ~6 MB (No external onnx_data required)


## Project Structure
```
├── config
│   └── config.yaml
│       <- Central configuration file (paths, hyperparameters, runtime settings)
│
├── models
│   ├── label_map.json
│   │   <- Mapping of class indices to wafer defect names
│   └── mobilenet_best.onnx
│       <- Exported ONNX model for edge / CPU deployment
│
├── reports
│   └── confusion_matrix.png
│       <- Model evaluation visualization
│
├── src
│   ├── dataset
│   │   └── wafer_dataset.py
│   │       <- Custom PyTorch Dataset for wafer defect images
│   │
│   ├── preprocessing
│   │   ├── preprocess.py
│   │   │   <- Image preprocessing (resize, normalization, transforms)
│   │   └── split_dataset.py
│   │       <- Train / validation / test dataset splitting logic
│   │
│   ├── models
│   │   └── mobilenet_v3.py
│   │       <- MobileNetV3-Small model architecture definition
│   │
│   ├── train
│   │   └── train.py
│   │       <- Training pipeline, loss calculation, and checkpoint saving
│   │
│   ├── evaluate
│   │   └── evaluate.py
│   │       <- Model evaluation, metrics computation, and testing
│   │
│   ├── export
│   │   └── export_onnx.py
│   │       <- Export trained PyTorch model to ONNX format
│   │
│   └── utils
│       └── config.py
│           <- Utility functions for loading and managing configurations
│
├── ui
│   └── app.py
│       <- Streamlit UI for live wafer defect prediction
│
├── .gitignore
│   <- Specifies files and folders ignored by Git
│
├── README.md
│   <- Complete project documentation
│
└── requirements.txt
    <- Python dependencies required to run the project

```

## How to Run
1.Install Requirements
```
pip install -r requirements.txt
```
2. Preprocess Data
```
python -m src.preprocessing.preprocess
python -m src.preprocessing.split_dataset
```
3. Train Model
```
python -m src.train.train
```
4. Evaluate Model
```
python -m src.evaluate.evaluate
```
5. Launch UI
```
python -m streamlit run ui/app.py
```
6. Export to ONNX
```
python -m src.export.export_onnx
```

## References

1. Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam.  
   **Searching for MobileNetV3.** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.*  
   https://arxiv.org/abs/1905.02244


## Conclusion

This project demonstrates that high-accuracy wafer defect classification can be achieved using a compact, efficient CNN model, making it ideal for industrial, edge, and real-time inspection systems.

