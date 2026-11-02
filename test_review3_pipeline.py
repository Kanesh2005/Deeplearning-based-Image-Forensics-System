import os
import glob
import requests
import numpy as np
from PIL import Image

import config

BASE_URL = "http://127.0.0.1:5000"

def test_multi_engine_pipeline():
    print("=" * 70)
    print("      SPECTRA::FORENSICS — REVIEW 3 FINAL PIPELINE TEST SUITE   ")
    print("=" * 70)

    # 1. Test Modules Direct Import & Execution
    print("\n[STEP 1] Testing Forensic Core Modules...")
    
    from src.ela import convert_to_ela_image
    from src.noise_analysis import analyze_noise_inconsistency
    from src.metadata_analysis import analyze_metadata
    from src.frequency_analysis import analyze_frequency_spectrum
    from src.fusion import fuse_forensic_evidence
    from src.report_generator import generate_forensic_pdf_report
    
    print("  [PASS] All 5 Forensic Modules successfully imported.")

    # Find sample image
    sample_images = glob.glob("CASIA2/Au/*.jpg") + glob.glob("CASIA2/Tp/*.jpg")
    if not sample_images:
        # Create a test synthetic image if dataset directory is missing
        os.makedirs("uploads/original", exist_ok=True)
        test_img_path = "uploads/original/test_sample.jpg"
        arr = np.uint8(np.random.rand(300, 300, 3) * 255)
        Image.fromarray(arr).save(test_img_path)
        sample_images = [test_img_path]

    sample = sample_images[0]
    print(f"  [INFO] Running synthetic & unit testing on sample: {sample}")

    # Test ELA
    ela_img = convert_to_ela_image(sample)
    assert isinstance(ela_img, Image.Image), "ELA failed to return PIL image"
    print("  [PASS] ELA Residual Analysis working cleanly.")

    # Test Noise Analysis
    noise_res = analyze_noise_inconsistency(sample)
    assert 'noise_score' in noise_res, "Noise analysis missing score key"
    print(f"  [PASS] Noise Analysis working (Noise Score: {noise_res['noise_score']}%).")

    # Test EXIF Metadata
    meta_res = analyze_metadata(sample)
    assert 'metadata_score' in meta_res, "Metadata analysis missing score key"
    print(f"  [PASS] EXIF Metadata Audit working (Has EXIF: {meta_res['has_exif']}).")

    # Test FFT Frequency Spectrum
    fft_res = analyze_frequency_spectrum(sample)
    assert 'frequency_score' in fft_res, "FFT analysis missing score key"
    print(f"  [PASS] 2D FFT Frequency Analysis working (FFT Score: {fft_res['frequency_score']}%).")

    # Test Fusion
    fusion_res = fuse_forensic_evidence(
        ela_score=40.0,
        deep_cnn_score=85.0,
        noise_score=60.0,
        metadata_score=90.0,
        frequency_score=75.0
    )
    assert fusion_res['verdict'] != "", "Fusion verdict empty"
    print(f"  [PASS] Evidence Fusion Engine working (Verdict: {fusion_res['verdict']}, Confidence: {fusion_res['overall_confidence']}%).")

    # 2. Test Flask Server Endpoints
    print("\n[STEP 2] Testing Flask Web API Server...")
    
    # Test GET /
    res_index = requests.get(f"{BASE_URL}/")
    assert res_index.status_code == 200, f"Index returned status {res_index.status_code}"
    print("  [PASS] GET / (Dashboard UI) -> 200 OK")

    # Test GET /metrics
    res_metrics = requests.get(f"{BASE_URL}/metrics")
    assert res_metrics.status_code == 200, f"Metrics returned status {res_metrics.status_code}"
    metrics = res_metrics.json().get('metrics', {})
    print(f"  [PASS] GET /metrics -> Accuracy: {metrics.get('accuracy')}, ROC-AUC: {metrics.get('roc_auc')}")

    # Test POST /predict
    with open(sample, 'rb') as f:
        res_pred = requests.post(f"{BASE_URL}/predict", files={'image': (sample, f, 'image/jpeg')})
    assert res_pred.status_code == 200, f"Predict returned status {res_pred.status_code}"
    pred_data = res_pred.json().get('data', {})
    print(f"  [PASS] POST /predict (Single Image 5-Layer Analysis):")
    print(f"         - Verdict:     {pred_data.get('prediction')}")
    print(f"         - Confidence:  {pred_data.get('confidence')}%")
    print(f"         - Threat Level: {pred_data.get('risk_level')}")
    print(f"         - Grad-CAM URL: {pred_data.get('gradcam_image_url')}")
    print(f"         - Noise Map:   {pred_data.get('noise_image_url')}")
    print(f"         - FFT Plot:    {pred_data.get('fft_image_url')}")
    print(f"         - PDF Report:  {pred_data.get('report_pdf_url')}")

    # Test PDF Report Download
    report_url = pred_data.get('report_pdf_url')
    if report_url:
        res_pdf = requests.get(f"{BASE_URL}{report_url}")
        assert res_pdf.status_code == 200, f"PDF report download returned {res_pdf.status_code}"
        print(f"  [PASS] GET {report_url} -> PDF Forensic Audit Report Downloaded (200 OK)")

    # Test POST /predict_batch
    if len(sample_images) >= 1:
        files = [('images', (sample_images[0], open(sample_images[0], 'rb'), 'image/jpeg'))]
        if len(sample_images) > 1:
            files.append(('images', (sample_images[1], open(sample_images[1], 'rb'), 'image/jpeg')))
        
        res_batch = requests.post(f"{BASE_URL}/predict_batch", files=files)
        assert res_batch.status_code == 200, f"Predict batch returned {res_batch.status_code}"
        batch_summary = res_batch.json().get('batch', {}).get('summary', {})
        print(f"  [PASS] POST /predict_batch (Batch Pipeline):")
        print(f"         - Total Processed: {res_batch.json().get('batch', {}).get('total_processed')}")
        print(f"         - Avg Confidence:  {batch_summary.get('avg_confidence')}%")

    print("\n" + "=" * 70)
    print("      ALL REVIEW 3 MULTI-ENGINE TESTS PASSED SUCCESSFULLY (100%) ")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    test_multi_engine_pipeline()
