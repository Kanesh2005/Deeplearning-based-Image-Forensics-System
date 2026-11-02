import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import keras
    load_keras_model = keras.models.load_model
except ImportError:
    import tensorflow as tf
    load_keras_model = tf.keras.models.load_model

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import get_dataset_splits

def find_optimal_threshold(y_val, val_probs):
    """
    Finds optimal decision threshold using validation set probabilities by maximizing F1 score in [0.40, 0.60].
    NEVER uses test set to select threshold.
    """
    best_threshold = 0.50
    best_f1 = -1.0
    
    thresholds = np.arange(0.40, 0.62, 0.02)
    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(th)
            
    return round(best_threshold, 2), round(float(best_f1), 4)

def evaluate():
    """
    Evaluates trained CNN model on held-out test dataset split.
    Uses validation-tuned decision threshold.
    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, Sensitivity.
    Generates confusion matrix, ROC curve, classification report, and JSON artifacts.
    """
    print("=" * 60)
    print("        SPECTRA::FORENSICS MODEL EVALUATION SCRIPT         ")
    print("=" * 60)

    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        if os.path.exists(config.BEST_MODEL_SAVE_PATH):
            model_path = config.BEST_MODEL_SAVE_PATH
        else:
            print(f"\n[ERROR] Trained model NOT found at {config.MODEL_SAVE_PATH}")
            sys.exit(1)

    print(f"\n[MODEL] Loading trained CNN model from {model_path}...")
    model = load_keras_model(model_path)

    # Load Validation and Test splits
    (X_train, y_train, train_paths), (X_val, y_val, val_paths), (X_test, y_test, test_paths) = get_dataset_splits(
        max_samples_per_class=config.MAX_SAMPLES_PER_CLASS,
        seed=config.SEED,
        use_ela=True,
        return_paths=True
    )

    # 1. Validation-Based Threshold Tuning & Freezing
    print(f"\n[VALIDATION] Tuning decision threshold on {len(X_val)} validation samples...")
    val_probs = model.predict(X_val, batch_size=config.BATCH_SIZE, verbose=0).ravel()
    tuned_threshold, best_val_f1 = find_optimal_threshold(y_val, val_probs)
    print(f"[THRESHOLD TUNING] Optimal Validation Decision Threshold: {tuned_threshold} (Val F1: {best_val_f1})")

    # Freeze threshold artifact BEFORE evaluating test set
    threshold_data = {
        "threshold": tuned_threshold,
        "positive_class": "Tampered"
    }
    threshold_artifact_path = os.path.join(config.ARTIFACTS_DIR, "threshold.json")
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(threshold_artifact_path, 'w') as f:
        json.dump(threshold_data, f, indent=4)
    print(f"[FROZEN THRESHOLD] Saved to {threshold_artifact_path}")

    # Update threshold in config
    config.DECISION_THRESHOLD = tuned_threshold

    # 2. Test Set Inference
    print(f"\n[TEST] Evaluating on {len(X_test)} held-out test samples...")
    y_pred_probs = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1).ravel()
    y_pred_binary = (y_pred_probs >= tuned_threshold).astype(int)

    # 3. Calculate Metrics
    acc = float(accuracy_score(y_test, y_pred_binary))
    prec = float(precision_score(y_test, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_test, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))
    auc = float(roc_auc_score(y_test, y_pred_probs))

    cm = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    from datetime import datetime
    eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics = {
        "model_version": getattr(config, 'MODEL_VERSION', 'CNN-ELA-v2.0'),
        "dataset_name": "CASIA v2.0",
        "decision_threshold": tuned_threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "test_samples": len(y_test),
        "authentic_samples": int(np.sum(y_test == 0)),
        "tampered_samples": int(np.sum(y_test == 1)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "evaluated_at": eval_time
    }

    print("\n--- TEST METRICS REPORT ---")
    print(f"Model Version:      {metrics['model_version']}")
    print(f"Decision Threshold: {metrics['decision_threshold']}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"Sensitivity:        {metrics['sensitivity']:.4f}")
    print(f"Specificity:        {metrics['specificity']:.4f}")
    print("-" * 50)

    # 4. Save JSON Artifacts to artifacts/ and results/
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    artifacts_metrics_json = os.path.join(config.ARTIFACTS_DIR, "metrics.json")
    with open(artifacts_metrics_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(config.TEST_METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(config.METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[SAVED] Metrics saved to {artifacts_metrics_json}, {config.TEST_METRICS_JSON}, and {config.METRICS_JSON_PATH}")

    # 5. Save per-sample predictions CSV to artifacts/test_predictions.csv and results/predictions.csv
    test_predictions_csv = os.path.join(config.ARTIFACTS_DIR, "test_predictions.csv")
    predictions_df = pd.DataFrame({
        'filename': [os.path.basename(p) for p in test_paths],
        'true_label': ['ORIGINAL' if y == 0 else 'TAMPERED' for y in y_test],
        'tampered_probability': [round(float(prob), 4) for prob in y_pred_probs],
        'predicted_label': ['ORIGINAL' if p == 0 else 'TAMPERED' for p in y_pred_binary],
        'is_correct': [bool((prob >= tuned_threshold) == (y == 1)) for prob, y in zip(y_pred_probs, y_test)]
    })
    predictions_df.to_csv(test_predictions_csv, index=False)
    predictions_df.to_csv(config.PREDICTIONS_CSV_PATH, index=False)
    print(f"[SAVED] Test predictions saved to {test_predictions_csv} and {config.PREDICTIONS_CSV_PATH}")

    # 6. Save Classification Report
    clr_report = classification_report(y_test, y_pred_binary, target_names=['Original', 'Tampered'])
    with open(config.CLASSIFICATION_REPORT_TXT, 'w') as f:
        f.write("SPECTRA::FORENSICS - HELD-OUT TEST SET EVALUATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(clr_report)

    # 7. Plot Confusion Matrix
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Original (0)', 'Tampered (1)'],
        yticklabels=['Original (0)', 'Tampered (1)'],
        cbar=True, annot_kws={"size": 14}
    )
    plt.title('SPECTRA::FORENSICS - Confusion Matrix', fontsize=13, pad=12)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('Actual Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    # 8. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(fpr, tpr, color='#00e5ff', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#ff3366', lw=1.5, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('SPECTRA::FORENSICS - Receiver Operating Characteristic (ROC)', fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.ROC_CURVE_PATH, dpi=300)
    plt.close()

    print("\n[SUCCESS] Held-out test evaluation completed successfully!\n")
    return metrics

if __name__ == '__main__':
    evaluate()
