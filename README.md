# Wafer Defect Classification using MobileNetV3

This project focuses on automatic wafer defect classification using SEM (Scanning Electron Microscope) images. The goal is to accurately classify wafer surface conditions into 8 categories using a lightweight, high-performance deep learning model suitable for edge and real-time deployment.

The solution is designed to be:
- ✅ Accurate
- ✅ Lightweight (~6 MB model)
- ✅ Fast on CPU
- ✅ Deployment-ready (ONNX supported)

### Problem Statement
Manual inspection of wafer defects is:
Time-consuming
Error-prone
Not scalable
This project automates the inspection process by using a CNN-based classifier trained on grayscale SEM images to identify defect types such as scratches, particles, residues, and more.

 ### Defect Classes
The model classifies wafer images into the following 8 classes:
Ball Defects
Clean
Craters
Flakes
Others
Particles
Residues
Scratches


### Model Architecture
Backbone: MobileNetV3-Small
Input: 128 × 128 grayscale images



### Why MobileNetV3?

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

### Data Pipeline
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

### Training Details
```
Loss Function: CrossEntropyLoss
Optimizer: Adam
Epochs: 20
Batch Size: Configurable
The model is trained on processed data only, ensuring consistency and reproducibility.
```


### Model Performance
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

### 🖥️ User Interface (Streamlit)
A minimal Streamlit UI is provided to:
Upload single or multiple images
View images directly
Predict defect class for each image
Clean and easy-to-use interface
<img width="1905" height="820" alt="Screenshot 2026-02-07 131138" src="https://github.com/user-attachments/assets/f300f52e-3f09-43ca-a19c-d787e5fba0b6" />

<img width="1909" height="902" alt="Screenshot 2026-02-07 131157" src="https://github.com/user-attachments/assets/b749072f-5a28-4ae8-ba40-6261eea43a7f" />

### Model Export & Deployment

ONNX Support
The trained PyTorch model can be exported to ONNX for:
Edge devices
Faster inference
Cross-platform deployment
Model size after export: ~6 MB (No external onnx_data required)


### Project Structure
```
Project Organization

├── LICENSE
├── README.md                      <- The top-level README for developers using this project.
├── requirements.txt               <- The requirements file for reproducing the environment.
│
├── data
│   ├── processed                  <- The final, canonical dataset used for modeling.
│   └── raw                        <- The original, immutable SEM image dataset.
│
├── src                            <- Source code for use in this project.
│   ├── __init__.py                <- Makes src a Python module
│   ├── config.py                  <- All configuration parameters
│   ├── util.py                    <- Utility/helper functions
│   │
│   ├── data
│   │   └── make_dataset.py        <- Script to generate dataset in required format
│   │
│   └── models                     <- Scripts to train, test and predict using the model
│       ├── network.py             <- MobileNetV3-Small architecture definition
│       ├── loss.py                <- Loss functions used during training
│       ├── train_model.py         <- Training pipeline and checkpoint saving
│       ├── test_model.py          <- Test model performance on test dataset
│       └── predict_model.py       <- Run inference on new images
│
├── ui
│   └── app.py                     <- Streamlit UI for live wafer defect predictions
│
├── weights                        <- Directory for saving model checkpoints
├── logs                           <- Directory for saving terminal outputs/logs
├── inference                      <- Directory where predicted outputs are stored
│
└── reports
    └── confusion_matrix.png       <- Performance visualization
```

### How to Run
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


### Conclusion
This project demonstrates that high-accuracy wafer defect classification can be achieved using a compact, efficient CNN model, making it ideal for industrial, edge, and real-time inspection systems.

