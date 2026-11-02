import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset Configuration
PRIMARY_DATASET_DIR = os.path.join(BASE_DIR, "CASIA2")
FALLBACK_DATASET_DIR = os.path.join(BASE_DIR, "data", "CASIA2")

if os.path.exists(PRIMARY_DATASET_DIR):
    DATASET_DIR = PRIMARY_DATASET_DIR
else:
    DATASET_DIR = FALLBACK_DATASET_DIR

AU_DIR = os.path.join(DATASET_DIR, "Au")
TP_DIR = os.path.join(DATASET_DIR, "Tp")

# Preprocessing & ELA Configuration
IMG_SIZE = 128
ELA_QUALITY = 90
ELA_SCALE = 10.0

# Model Training Hyperparameters (CPU-Optimized for fast, reproducible training)
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 2e-4
SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MAX_SAMPLES_PER_CLASS = 2000  # 2000 Au + 2000 Tp = 4000 total images across splits

# Model & Decision Threshold Configuration
MODEL_VERSION = "CNN-ELA-v2.0"
DECISION_THRESHOLD = 0.50

# Output Directory Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "cnn_image_forensics.keras")
BEST_MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model.keras")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SPLITS_DIR = os.path.join(ARTIFACTS_DIR, "splits")
DATASET_REPORT_JSON = os.path.join(ARTIFACTS_DIR, "dataset_report.json")
DATASET_REPORT_CSV = os.path.join(ARTIFACTS_DIR, "dataset_report.csv")
KNOWN_SAMPLES_CSV = os.path.join(ARTIFACTS_DIR, "known_sample_predictions.csv")
TEST_METRICS_JSON = os.path.join(ARTIFACTS_DIR, "test_metrics.json")
TRAINING_HISTORY_JSON = os.path.join(ARTIFACTS_DIR, "training_history.json")
MODEL_METADATA_JSON = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "metrics.json")
PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curve.png")
TRAINING_HISTORY_CSV = os.path.join(RESULTS_DIR, "training_history.csv")
TRAINING_ACCURACY_PNG = os.path.join(RESULTS_DIR, "training_accuracy.png")
TRAINING_LOSS_PNG = os.path.join(RESULTS_DIR, "training_loss.png")
CLASSIFICATION_REPORT_TXT = os.path.join(RESULTS_DIR, "classification_report.txt")

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
UPLOADS_ORIGINAL_DIR = os.path.join(UPLOADS_DIR, "original")
UPLOADS_ELA_DIR = os.path.join(UPLOADS_DIR, "ela")
UPLOADS_GRADCAM_DIR = os.path.join(UPLOADS_DIR, "gradcam")
UPLOADS_NOISE_DIR = os.path.join(UPLOADS_DIR, "noise")
UPLOADS_FFT_DIR = os.path.join(UPLOADS_DIR, "fft")
REPORTS_DIR = os.path.join(UPLOADS_DIR, "reports")

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'tif'}

# Ensure required directories exist
for folder in [
    MODELS_DIR, ARTIFACTS_DIR, SPLITS_DIR, RESULTS_DIR, 
    UPLOADS_ORIGINAL_DIR, UPLOADS_ELA_DIR, UPLOADS_GRADCAM_DIR, 
    UPLOADS_NOISE_DIR, UPLOADS_FFT_DIR, REPORTS_DIR
]:
    os.makedirs(folder, exist_ok=True)
