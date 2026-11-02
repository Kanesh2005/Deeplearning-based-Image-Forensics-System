import os
import time
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

import config

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB max batch file upload limit

# Session prediction history in-memory storage
PREDICTION_HISTORY = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Renders the main SPECTRA::FORENSICS dashboard interface."""
    model_exists = os.path.exists(config.MODEL_SAVE_PATH) or os.path.exists(config.BEST_MODEL_SAVE_PATH)
    metrics_exist = os.path.exists(config.TEST_METRICS_JSON) or os.path.exists(config.METRICS_JSON_PATH)
    
    return render_template(
        'index.html',
        model_exists=model_exists,
        metrics_exist=metrics_exist
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint returning system operational status."""
    model_exists = os.path.exists(config.MODEL_SAVE_PATH) or os.path.exists(config.BEST_MODEL_SAVE_PATH)
    metrics_exist = os.path.exists(config.TEST_METRICS_JSON) or os.path.exists(config.METRICS_JSON_PATH)
    return jsonify({
        'status': 'healthy',
        'model_trained': model_exists,
        'metrics_available': metrics_exist,
        'dataset_configured': os.path.exists(config.DATASET_DIR),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles single image upload and runs the 5-layer Multi-Engine forensics prediction pipeline.
    Primary verdict (ORIGINAL vs TAMPERED) is driven strictly by Layer 1 CNN model.
    """
    if not (os.path.exists(config.MODEL_SAVE_PATH) or os.path.exists(config.BEST_MODEL_SAVE_PATH)):
        return jsonify({
            'error': 'Model not trained yet. Please run train.py to train the CNN model.',
            'status': 'model_missing'
        }), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in request.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file format. Supported formats: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400

    try:
        from src.prediction import predict_image

        filename = secure_filename(file.filename)
        timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_")
        saved_filename = timestamp_prefix + filename
        save_path = os.path.join(config.UPLOADS_ORIGINAL_DIR, saved_filename)
        
        file.save(save_path)

        result = predict_image(save_path)
        result['timestamp'] = datetime.now().strftime("%H:%M:%S")

        # Append to prediction history log
        PREDICTION_HISTORY.insert(0, {
            'time': result['timestamp'],
            'file': filename,
            'result': result['prediction'],
            'confidence': result['confidence'],
            'dimensions': result['image_dimensions'],
            'risk_level': result.get('risk_level', 'Low')
        })

        if len(PREDICTION_HISTORY) > 50:
            PREDICTION_HISTORY.pop()

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handles multi-image batch upload and processes forensic inspection."""
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided in request.'}), 400

    files = request.files.getlist('images')
    if not files or len(files) == 0:
        return jsonify({'error': 'No images selected.'}), 400

    saved_paths = []
    try:
        from src.batch_processor import process_image_batch

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_")
                saved_filename = timestamp_prefix + filename
                save_path = os.path.join(config.UPLOADS_ORIGINAL_DIR, saved_filename)
                file.save(save_path)
                saved_paths.append(save_path)

        if not saved_paths:
            return jsonify({'error': 'No valid image files found in batch upload.'}), 400

        batch_result = process_image_batch(saved_paths)

        return jsonify({
            'success': True,
            'batch': batch_result
        })

    except Exception as e:
        return jsonify({'error': f"Batch processing failed: {str(e)}"}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Returns calculated test evaluation metrics from artifacts/test_metrics.json or results/metrics.json."""
    metrics_path = config.TEST_METRICS_JSON if os.path.exists(config.TEST_METRICS_JSON) else config.METRICS_JSON_PATH
    
    if not os.path.exists(metrics_path):
        return jsonify({
            'error': 'Model evaluation has not been completed yet. Please run evaluate.py.',
            'status': 'metrics_missing'
        }), 404

    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            
        metrics_data['confusion_matrix_url'] = '/results/confusion_matrix.png'
        metrics_data['roc_curve_url'] = '/results/roc_curve.png'
        metrics_data['accuracy_plot_url'] = '/results/training_accuracy.png'
        metrics_data['loss_plot_url'] = '/results/training_loss.png'
        
        return jsonify({'success': True, 'metrics': metrics_data})
    except Exception as e:
        return jsonify({'error': f"Failed to load metrics: {str(e)}"}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Returns session prediction history."""
    return jsonify({'success': True, 'history': PREDICTION_HISTORY})


@app.route('/download_report/<path:filename>')
def download_report(filename):
    """Allows downloading of generated PDF audit reports."""
    report_path = os.path.join(config.REPORTS_DIR, filename)
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({'error': 'Report file not found.'}), 404


@app.route('/uploads/<path:subpath>')
def serve_uploads(subpath):
    """Serves uploaded original and generated forensic maps."""
    return send_from_directory(config.UPLOADS_DIR, subpath)


@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serves generated metrics plots and charts."""
    return send_from_directory(config.RESULTS_DIR, filename)


def preload_model():
    """Pre-loads CNN model at server startup to optimize single image inference speed."""
    try:
        from src.prediction import get_model
        model = get_model()
        input_shape = getattr(model, 'input_shape', f"({config.IMG_SIZE}, {config.IMG_SIZE}, 3)")
        threshold = getattr(config, 'DECISION_THRESHOLD', 0.50)
        
        print("\n" + "=" * 60)
        print("          SPECTRA::FORENSICS MODEL LOAD VERIFICATION         ")
        print("=" * 60)
        print("Model loaded successfully into memory.")
        print(f"Model Name:      Spectra_CNN_Image_Forensics ({config.MODEL_VERSION})")
        print(f"Input shape:     {input_shape}")
        print("Output:          Binary classification (Sigmoid Probability)")
        print("Classes:         0 = Original, 1 = Tampered")
        print(f"Threshold:       {threshold}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[Startup Warning] Model pre-load note: {e}\n")

if __name__ == '__main__':
    preload_model()
    print("=" * 60)
    print("      SPECTRA::FORENSICS DEEP LEARNING SYSTEM RUNNING     ")
    print(f"      Server running at: http://127.0.0.1:5000           ")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=True)
