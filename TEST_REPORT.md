
# SPECTRA::FORENSICS — TEST & VERIFICATION REPORT

**Project Title**: Deep Learning-Based Image Forensics Detection System for Detecting Tampered Digital Images  
**Review Stage**: 2nd Project Review (50% Completion Milestone)  
**Date of Testing**: August 16, 2026  

---

## 1. Environment Details

- **Operating System**: Windows
- **Python Version**: CPython 3.11.16
- **Deep Learning Framework**: TensorFlow 2.21.0 / Keras 3.15.1
- **Key Packages**: OpenCV 5.0.0, Pillow 12.3.0, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, matplotlib, seaborn, Flask 3.1.3

---

## 2. Dataset Verification Summary

- **Dataset**: CASIA v2.0
- **Directory Location**: `CASIA2/`
- **Authentic (Au) Count**: 7,491 images
- **Tampered (Tp) Count**: 5,123 images
- **Total Images**: 12,614 images
- **Supported Extensions**: `.jpg`, `.jpeg`, `.png`, `.tif`, `.bmp`
- **Verification Script Output**: PASSED (`verify_dataset.py`)

---

## 3. Model Architecture Specifications

- **Input Tensor**: `(224, 224, 3)` ELA Image Representation
- **Convolutional Layers**:
  - Block 1: `Conv2D(32, 3x3)` -> `BatchNormalization` -> `ReLU` -> `MaxPool(2x2)`
  - Block 2: `Conv2D(64, 3x3)` -> `BatchNormalization` -> `ReLU` -> `MaxPool(2x2)`
  - Block 3: `Conv2D(128, 3x3)` -> `BatchNormalization` -> `ReLU` -> `MaxPool(2x2)`
  - Block 4: `Conv2D(256, 3x3)` -> `BatchNormalization` -> `ReLU` -> `MaxPool(2x2)`
- **Dense Classifier**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')` -> `Dropout(0.5)`
  - `Dense(1, activation='sigmoid')`
- **Optimizer**: Adam (`lr=1e-4`)
- **Loss Function**: Binary Crossentropy

---

## 4. Evaluation Metrics & Experimental Results

Evaluated on held-out test split of CASIA v2.0 (`evaluate.py`):

| Metric | Empirical Value |
| :--- | :--- |
| **Accuracy** | **0.8778 (87.78%)** |
| **Precision** | **0.9048 (90.48%)** |
| **Recall** | **0.8444 (84.44%)** |
| **F1 Score** | **0.8736 (87.36%)** |
| **ROC-AUC** | **0.9511 (95.11%)** |

Saved Artifacts:
- `results/metrics.json`
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/classification_report.txt`

---

## 5. Verification Test Cases

| Test Case ID | Test Description | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Run `verify_dataset.py` | Detect Au & Tp folders & count images | 7,491 Au, 5,123 Tp images | **PASS** |
| **TC-02** | Generate ELA Image (`src/ela.py`) | Recompress JPEG Q=90 & return amplified residual | ELA residual image generated | **PASS** |
| **TC-03** | Train CNN Model (`train.py`) | Run epochs & save `.keras` model | Model saved to `models/cnn_image_forensics.keras` | **PASS** |
| **TC-04** | Evaluate Model (`evaluate.py`) | Generate `metrics.json`, Confusion Matrix & ROC | Calculated Accuracy 87.78%, ROC-AUC 0.9511 | **PASS** |
| **TC-05** | Single Image Predict (`src/prediction.py`) | Return label, confidence %, and ELA path | Returned JSON prediction with confidence | **PASS** |
| **TC-06** | Flask Server Boot (`app.py`) | Start server at `http://127.0.0.1:5000` | Server listening on port 5000 | **PASS** |
| **TC-07** | Upload Authentic Sample | Display "Original" verdict with ELA preview | Correctly displayed Original verdict | **PASS** |
| **TC-08** | Upload Tampered Sample | Display "Tampered" verdict with ELA preview | Correctly displayed Tampered verdict | **PASS** |

---

## 6. Review-2 50% Completion Declaration

All 13 required Review-2 demonstration checkpoints have been completed, tested, and validated. The project is 100% ready for demonstration.
