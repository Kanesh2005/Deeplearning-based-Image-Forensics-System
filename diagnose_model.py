import os
import sys
import glob
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import keras
    load_keras_model = keras.models.load_model
except ImportError:
    import tensorflow as tf
    load_keras_model = tf.keras.models.load_model

import config
from src.preprocessing import preprocess_image

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')

def get_image_files(directory):
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

def main():
    print("=" * 60, flush=True)
    print("        SPECTRA::FORENSICS CNN MODEL DIAGNOSTIC         ", flush=True)
    print("=" * 60, flush=True)

    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        if os.path.exists(config.BEST_MODEL_SAVE_PATH):
            model_path = config.BEST_MODEL_SAVE_PATH

    model_exists = os.path.exists(model_path)
    print(f"MODEL EXISTS: {'YES' if model_exists else 'NO'} ({model_path})", flush=True)

    if not model_exists:
        print("MODEL NOT AVAILABLE — TRAINING REQUIRED", flush=True)
        return

    model = None
    try:
        model = load_keras_model(model_path)
        print("MODEL LOAD: SUCCESS", flush=True)
    except Exception as e:
        print(f"MODEL LOAD: FAIL - {e}", flush=True)
        return

    input_shape = getattr(model, 'input_shape', None)
    output_shape = getattr(model, 'output_shape', None)
    print(f"MODEL INPUT SHAPE: {input_shape}", flush=True)
    print(f"MODEL OUTPUT SHAPE: {output_shape}", flush=True)
    print("=" * 60, flush=True)

    au_images = get_image_files(config.AU_DIR)
    tp_images = get_image_files(config.TP_DIR)

    test_images = []
    if au_images:
        test_images.append((au_images[0], "Original"))
    if tp_images:
        test_images.append((tp_images[0], "Tampered"))
    if len(au_images) > 1:
        test_images.append((au_images[1], "Original"))
    if len(tp_images) > 1:
        test_images.append((tp_images[1], "Tampered"))

    if not test_images:
        print("No sample images found in CASIA2 dataset folder.", flush=True)
        return

    print("\n--- MODEL DIAGNOSTIC INFERENCE TEST ---", flush=True)
    for img_path, true_label in test_images:
        filename = os.path.basename(img_path)
        with Image.open(img_path) as img:
            orig_shape = img.size # (width, height)

        img_tensor = preprocess_image(img_path, target_size=(config.IMG_SIZE, config.IMG_SIZE), use_ela=True)
        input_batch = np.expand_dims(img_tensor, axis=0)

        prob = float(model.predict(input_batch, verbose=0)[0][0])
        threshold = getattr(config, 'DECISION_THRESHOLD', 0.50)
        pred = "Tampered" if prob >= threshold else "Original"

        print(f"\nFilename: {filename} (Ground Truth: {true_label})", flush=True)
        print(f"Original image size: {orig_shape}", flush=True)
        print(f"Input tensor shape:  {img_tensor.shape}", flush=True)
        print(f"Tensor min: {img_tensor.min():.4f}, max: {img_tensor.max():.4f}, mean: {img_tensor.mean():.4f}", flush=True)
        print(f"CNN P(Tampered):    {prob:.4f}", flush=True)
        print(f"CNN P(Original):    {1.0 - prob:.4f}", flush=True)
        print(f"Prediction:         {pred}", flush=True)
        print(f"Correct:            {'YES' if pred.upper() == true_label.upper() else 'NO'}", flush=True)

if __name__ == '__main__':
    main()
