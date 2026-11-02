import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    import keras
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.src.legacy.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from verify_dataset import verify_dataset
from src.dataset import get_dataset_splits
from src.model import build_cnn_model

def train():
    """
    Main reproducible training workflow for the CNN Image Forensics model.
    1. Sets deterministic random seeds.
    2. Audits dataset and generates leak-free train/val/test splits.
    3. Fits model with data augmentation and callbacks.
    4. Saves best model checkpoint, history artifacts, metadata, and metric plots.
    """
    print("=" * 60)
    print("         SPECTRA::FORENSICS CNN MODEL TRAINING         ")
    print("=" * 60)

    # 1. Set reproducible random seeds
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    try:
        tf.random.set_seed(config.SEED)
    except Exception:
        pass

    # 2. Verify Dataset
    verify_dataset()

    # 3. Load Preprocessed ELA Dataset Splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_dataset_splits(
        max_samples_per_class=config.MAX_SAMPLES_PER_CLASS,
        seed=config.SEED,
        use_ela=True
    )

    total_samples = len(X_train) + len(X_val) + len(X_test)
    total_au = np.sum(y_train == 0) + np.sum(y_val == 0) + np.sum(y_test == 0)
    total_tp = np.sum(y_train == 1) + np.sum(y_val == 1) + np.sum(y_test == 1)

    print("\n" + "=" * 60)
    print("              CONTROLLED TRAINING DATASET COUNTS             ")
    print("=" * 60)
    print(f"Total Dataset Count:    {total_samples} images across loaded splits")
    print(f"Train Count:            {len(X_train)} (Au: {np.sum(y_train == 0)}, Tp: {np.sum(y_train == 1)})")
    print(f"Validation Count:       {len(X_val)} (Au: {np.sum(y_val == 0)}, Tp: {np.sum(y_val == 1)})")
    print(f"Held-Out Test Count:    {len(X_test)} (Au: {np.sum(y_test == 0)}, Tp: {np.sum(y_test == 1)})")
    print(f"Class Distribution:     Authentic (0): {total_au}, Tampered (1): {total_tp}")
    print("=" * 60 + "\n")

    # 4. Build Model
    print("\n[MODEL] Initializing CNN Model Architecture...")
    model = build_cnn_model(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        learning_rate=config.LEARNING_RATE
    )
    model.summary()

    # 5. Configure Data Augmentation Generator for ELA Maps
    # (Avoid spatial rotation/shift resampling which destroys 8x8 JPEG block grid boundaries)
    datagen = ImageDataGenerator(
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # 6. Define Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=config.BEST_MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Calculate class weights to handle mild class imbalance
    n_samples = len(y_train)
    n_au = np.sum(y_train == 0)
    n_tp = np.sum(y_train == 1)
    weight_for_0 = (1 / n_au) * (n_samples / 2.0) if n_au > 0 else 1.0
    weight_for_1 = (1 / n_tp) * (n_samples / 2.0) if n_tp > 0 else 1.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    print(f"\n[CLASS WEIGHTS] Original (0): {weight_for_0:.3f}, Tampered (1): {weight_for_1:.3f}")

    # 7. Train Model
    print(f"\n[TRAINING] Starting training for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE, seed=config.SEED),
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Save restored best model
    model.save(config.MODEL_SAVE_PATH)
    model.save(config.BEST_MODEL_SAVE_PATH)
    print(f"[SAVED] Best model saved to {config.MODEL_SAVE_PATH} and {config.BEST_MODEL_SAVE_PATH}")

    # 8. Save Training History to CSV & JSON
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_df = pd.DataFrame(history_dict)
    history_df.to_csv(config.TRAINING_HISTORY_CSV, index_label='epoch')
    
    with open(config.TRAINING_HISTORY_JSON, 'w') as f:
        json.dump(history_dict, f, indent=4)
        
    print(f"[SAVED] Training history saved to {config.TRAINING_HISTORY_CSV} and {config.TRAINING_HISTORY_JSON}")

    # 9. Plot & Save Accuracy Graph
    plt.figure(figsize=(9, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#00ff9d', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#00e5ff', linewidth=2)
    plt.title('SPECTRA::FORENSICS - CNN Training & Validation Accuracy', fontsize=13, pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(config.TRAINING_ACCURACY_PNG, dpi=300)
    plt.close()

    # 10. Plot & Save Loss Graph
    plt.figure(figsize=(9, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='#ff3366', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ff9900', linewidth=2)
    plt.title('SPECTRA::FORENSICS - CNN Training & Validation Loss', fontsize=13, pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Binary Crossentropy Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(config.TRAINING_LOSS_PNG, dpi=300)
    plt.close()

    # 11. Save Model Metadata
    from datetime import datetime
    meta_data = {
        "model_name": "Spectra_CNN_Image_Forensics",
        "model_version": config.MODEL_VERSION,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "CASIA v2.0",
        "total_training_samples": len(X_train),
        "total_validation_samples": len(X_val),
        "total_test_samples": len(X_test),
        "img_size": config.IMG_SIZE,
        "epochs_requested": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "seed": config.SEED,
        "decision_threshold": config.DECISION_THRESHOLD
    }
    with open(config.MODEL_METADATA_JSON, 'w') as f:
        json.dump(meta_data, f, indent=4)
        
    meta_models_path = os.path.join(config.MODELS_DIR, "model_metadata.json")
    with open(meta_models_path, 'w') as f:
        json.dump(meta_data, f, indent=4)

    print(f"[SAVED] Model metadata saved to {config.MODEL_METADATA_JSON}")
    print("\n[SUCCESS] CNN Model Training Completed Successfully!\n")
    return model

if __name__ == '__main__':
    train()
