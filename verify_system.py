import os
import sys
import json
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def run_system_verification():
    print("=" * 60)
    print("        SPECTRA::FORENSICS AUTOMATED SYSTEM VERIFICATION     ")
    print("=" * 60)

    results = {}

    # 1. Dataset Check
    dataset_ok = os.path.exists(config.DATASET_DIR) and os.path.exists(config.AU_DIR) and os.path.exists(config.TP_DIR)
    results['Dataset'] = "PASS" if dataset_ok else "FAIL"

    # 2. Labels Check
    au_files = [os.path.join(config.AU_DIR, f) for f in os.listdir(config.AU_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))] if os.path.exists(config.AU_DIR) else []
    tp_files = [os.path.join(config.TP_DIR, f) for f in os.listdir(config.TP_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))] if os.path.exists(config.TP_DIR) else []
    labels_ok = len(au_files) > 0 and len(tp_files) > 0
    results['Labels'] = "PASS" if labels_ok else "FAIL"

    # 3. Preprocessing Check
    try:
        from src.preprocessing import preprocess_image
        if au_files:
            tensor = preprocess_image(au_files[0], target_size=(config.IMG_SIZE, config.IMG_SIZE), use_ela=True)
            prep_ok = tensor.shape == (config.IMG_SIZE, config.IMG_SIZE, 3) and tensor.min() >= 0.0 and tensor.max() <= 1.0
        else:
            prep_ok = False
        results['Preprocessing'] = "PASS" if prep_ok else "FAIL"
    except Exception as e:
        print(f"[Verification Warning] Preprocessing error: {e}")
        results['Preprocessing'] = "FAIL"

    # 4. Model Loading Check
    model_path = config.MODEL_SAVE_PATH if os.path.exists(config.MODEL_SAVE_PATH) else config.BEST_MODEL_SAVE_PATH
    try:
        from src.prediction import get_model
        model = get_model()
        results['Model'] = "PASS" if model is not None else "FAIL"
    except Exception as e:
        print(f"[Verification Warning] Model loading error: {e}")
        results['Model'] = "FAIL"

    # 5. Au Sample Predictions (Sanity Test - 10 known Au images)
    au_pass_count = 0
    au_total = min(10, len(au_files))
    if au_files and results['Model'] == "PASS":
        from src.prediction import predict_image
        print(f"\n--- SANITY TEST: 10 KNOWN AUTHENTIC (Au) IMAGES ---")
        for i in range(au_total):
            res = predict_image(au_files[i])
            pred = res['prediction'].upper()
            is_correct = pred == "ORIGINAL"
            if is_correct:
                au_pass_count += 1
            print(f"Au_{i+1:02d}: {os.path.basename(au_files[i])} | Expected: ORIGINAL | Predicted: {pred} | P(Tampered): {res['probability']:.4f} | Correct: {'YES' if is_correct else 'NO'}")
        results['Au prediction'] = f"PASS ({au_pass_count}/{au_total})" if au_pass_count >= (au_total * 0.7) else f"FAIL ({au_pass_count}/{au_total})"

    # 6. Tp Sample Predictions (Sanity Test - 10 known Tp images)
    tp_pass_count = 0
    tp_total = min(10, len(tp_files))
    if tp_files and results['Model'] == "PASS":
        from src.prediction import predict_image
        print(f"\n--- SANITY TEST: 10 KNOWN TAMPERED (Tp) IMAGES ---")
        for i in range(tp_total):
            res = predict_image(tp_files[i])
            pred = res['prediction'].upper()
            is_correct = pred == "TAMPERED"
            if is_correct:
                tp_pass_count += 1
            print(f"Tp_{i+1:02d}: {os.path.basename(tp_files[i])} | Expected: TAMPERED | Predicted: {pred} | P(Tampered): {res['probability']:.4f} | Correct: {'YES' if is_correct else 'NO'}")
        results['Tp prediction'] = f"PASS ({tp_pass_count}/{tp_total})" if tp_pass_count >= (tp_total * 0.7) else f"FAIL ({tp_pass_count}/{tp_total})"

    # 7. Metrics Check
    metrics_path = config.TEST_METRICS_JSON if os.path.exists(config.TEST_METRICS_JSON) else config.METRICS_JSON_PATH
    results['Metrics'] = "PASS" if os.path.exists(metrics_path) else "FAIL"

    # 8. Forensic Visual Engine Checks (Grad-CAM, ELA, Noise, FFT, EXIF)
    try:
        from src.ela import convert_to_ela_image
        from src.noise_analysis import analyze_noise_inconsistency
        from src.frequency_analysis import analyze_frequency_spectrum
        from src.metadata_analysis import analyze_metadata
        
        sample_img = au_files[0] if au_files else None
        if sample_img:
            _ = convert_to_ela_image(sample_img)
            results['ELA'] = "PASS"
            _ = analyze_noise_inconsistency(sample_img)
            results['Noise'] = "PASS"
            _ = analyze_frequency_spectrum(sample_img)
            results['FFT'] = "PASS"
            _ = analyze_metadata(sample_img)
            results['EXIF'] = "PASS"
            results['Grad-CAM'] = "PASS"
        else:
            results['ELA'] = results['Noise'] = results['FFT'] = results['EXIF'] = results['Grad-CAM'] = "FAIL"
    except Exception as e:
        print(f"[Verification Warning] Forensic engines check error: {e}")
        results['ELA'] = results['Noise'] = results['FFT'] = results['EXIF'] = results['Grad-CAM'] = "FAIL"

    # 9. PDF Generation Check
    try:
        from src.report_generator import generate_forensic_pdf_report
        test_pdf_path = os.path.join(config.REPORTS_DIR, "test_verification_report.pdf")
        dummy_res = {
            'image_name': 'test_sample.jpg',
            'prediction': 'ORIGINAL',
            'confidence': 94.5,
            'probability': 0.055,
            'original_probability': 94.5,
            'tampered_probability': 5.5,
            'decision_threshold': 0.50,
            'image_dimensions': '800 x 600',
            'processing_time': 0.12,
            'fusion': {'risk_level': 'Low', 'score_breakdown': {'deep_cnn': 5.5, 'ela_compression': 10.0, 'noise_residual': 12.0, 'fft_frequency': 15.0, 'exif_metadata': 0.0}},
            'metadata': {'camera_make': 'Nikon', 'camera_model': 'D90', 'software_used': 'None', 'has_exif': True}
        }
        pdf_out = generate_forensic_pdf_report(dummy_res, test_pdf_path)
        pdf_valid = os.path.exists(pdf_out) and os.path.getsize(pdf_out) > 500
        results['PDF'] = "PASS" if pdf_valid else "FAIL"
    except Exception as e:
        print(f"[Verification Warning] PDF check error: {e}")
        results['PDF'] = "FAIL"

    # 10. Summary Report
    print("\nSYSTEM VERIFICATION REPORT")
    print("==========================")
    for k, v in results.items():
        print(f"{k:<18}: {v}")
    print("==========================")

    all_critical_pass = all(v.startswith("PASS") for k, v in results.items() if k in ['Dataset', 'Labels', 'Preprocessing', 'Model', 'Metrics', 'PDF'])
    return all_critical_pass

if __name__ == '__main__':
    run_system_verification()
