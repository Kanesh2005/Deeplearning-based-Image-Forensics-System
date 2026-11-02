# SPECTRA::FORENSICS — Deep Learning-Based Image Forensics Detection System

A genuinely functional, end-to-end Image Forensics Detection System for detecting tampered digital images using **Error Level Analysis (ELA)** preprocessing and a **Convolutional Neural Network (CNN)** trained on the **CASIA v2.0** image tampering dataset.

Developed for college B.Tech Information Technology Mini Project 

---

## 1. Project Overview & Objectives

- **System Purpose**: Automate digital image tampering detection (Original/Authentic vs. Tampered/Forged) by detecting localized JPEG compression inconsistency residuals.
- **Dataset**: CASIA v2.0 dataset containing `Au` (Authentic) and `Tp` (Tampered) images.
- **Preprocessing**: Error Level Analysis (ELA at JPEG Q=90, scale=15) + 224x224 RGB normalization.
- **Model Architecture**: 4-Block Conv2D + BatchNormalization + ReLU + MaxPooling2D + GlobalAveragePooling2D + Dense + Dropout + Sigmoid Binary Classifier.
- **Web Application**: Flask REST backend connected to an interactive dark forensic dashboard.

---

## 2. Technology Stack

- **Backend**: Python 3.11, Flask
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, Pillow, NumPy
- **Data Analysis & Visualization**: pandas, scikit-learn, matplotlib, seaborn
- **Frontend**: HTML5, CSS3 (Dark Glassmorphism Theme), Vanilla JavaScript (Fetch API)

---

## 3. Architecture & Pipeline Flow

```
CASIA v2.0 Dataset (Au / Tp)
          │
          ▼
   verify_dataset.py
          │
          ▼
  ELA Preprocessing & Resizing (src/ela.py, src/preprocessing.py)
          │
          ▼
  Stratified Dataset Split (src/dataset.py - 70% Train, 15% Val, 15% Test)
          │
          ▼
  CNN Neural Network (src/model.py - 4 Conv Blocks + Sigmoid Head)
          │
          ▼
  Training Script (train.py -> models/cnn_image_forensics.keras)
          │
          ▼
  Evaluation Script (evaluate.py -> results/metrics.json, CM, ROC)
          │
          ▼
  Single-Image Predictor (src/prediction.py)
          │
          ▼
  Flask Web Backend (app.py -> /predict, /metrics, /history)
          │
          ▼
  SPECTRA::FORENSICS Dashboard UI (templates/index.html)
```

---

## 4. Environment & Installation

### Windows PowerShell Commands

```powershell
# 1. Clone/navigate to project folder
cd c:\Users\keshk\OneDrive\Desktop\image

# 2. Create Python 3.11 Virtual Environment using uv or python
uv venv venv --python 3.11

# 3. Activate Virtual Environment
venv\Scripts\activate

# 4. Install Dependencies
uv pip install --link-mode=copy -r requirements.txt --python venv\Scripts\python.exe
```

---

## 5. Step-by-Step Execution Guide

### Step 1: Verify CASIA v2.0 Dataset
```powershell
python verify_dataset.py
```
*Expected Output*: Counts total authentic (`Au`) and tampered (`Tp`) images in `CASIA2/`.

### Step 2: Train the CNN Model
```powershell
python train.py
```
*Outputs*:
- Saves trained model: `models/cnn_image_forensics.keras`
- Saves best loss model: `models/best_model.keras`
- Saves history: `results/training_history.csv`, `training_accuracy.png`, `training_loss.png`

### Step 3: Evaluate Model on Test Dataset
```powershell
python evaluate.py
```
*Outputs*:
- `results/metrics.json`
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/classification_report.txt`

### Step 4: Launch Flask Web Application
```powershell
python app.py
```
*Open Browser at*: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 6. Review-2 Demonstration Checklist (13 Checkpoints)

- [x] **CHECKPOINT 1**: CASIA v2.0 successfully loaded (`CASIA2/`).
- [x] **CHECKPOINT 2**: `Au` (Authentic) and `Tp` (Tampered) images detected.
- [x] **CHECKPOINT 3**: ELA preprocessing generated (`src/ela.py`).
- [x] **CHECKPOINT 4**: CNN architecture created (`src/model.py`).
- [x] **CHECKPOINT 5**: CNN model trained on dataset (`train.py`).
- [x] **CHECKPOINT 6**: Trained model saved (`models/cnn_image_forensics.keras`).
- [x] **CHECKPOINT 7**: Test evaluation completed (`evaluate.py`).
- [x] **CHECKPOINT 8**: Confusion matrix and metrics generated in `results/`.
- [x] **CHECKPOINT 9**: Flask backend starts on `http://127.0.0.1:5000`.
- [x] **CHECKPOINT 10**: User uploads an image via drag-and-drop.
- [x] **CHECKPOINT 11**: Real CNN predicts Original/Tampered.
- [x] **CHECKPOINT 12**: Prediction confidence score displayed.
- [x] **CHECKPOINT 13**: ELA visualization heatmap rendered side-by-side.

---

## 7. Experimental Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **87.78%** |
| **Precision** | **90.48%** |
| **Recall** | **84.44%** |
| **F1 Score** | **87.36%** |
| **ROC-AUC** | **95.11%** |

---

## 8. Project Team

- **Kaneshkara M K**: Deep Learning & Image Processing (Dataset prep, CNN development, ELA preprocessing, model training)
- **Pugalendhi R**: Frontend & Backend (UI design, image upload flow, Flask integration)
- **Prahathish Billgates C**: Database & Documentation (Testing, metrics evaluation, final report)
