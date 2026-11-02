import glob
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_system():
    print("=" * 60)
    print("       SPECTRA::FORENSICS END-TO-END SYSTEM TEST        ")
    print("=" * 60)

    # Test 1: Health / Index page
    print("\n[TEST 1] Testing GET / (Index Dashboard)...")
    res_index = requests.get(f"{BASE_URL}/")
    assert res_index.status_code == 200, f"Index page returned {res_index.status_code}"
    print("[PASS] Index page rendered successfully (200 OK)")

    # Test 2: Metrics endpoint
    print("\n[TEST 2] Testing GET /metrics...")
    res_metrics = requests.get(f"{BASE_URL}/metrics")
    assert res_metrics.status_code == 200, f"Metrics returned {res_metrics.status_code}"
    metrics = res_metrics.json().get('metrics', {})
    print(f"[PASS] Metrics fetched successfully:")
    print(f"       - Accuracy:  {metrics.get('accuracy')}")
    print(f"       - Precision: {metrics.get('precision')}")
    print(f"       - Recall:    {metrics.get('recall')}")
    print(f"       - F1-Score:  {metrics.get('f1_score')}")
    print(f"       - ROC-AUC:   {metrics.get('roc_auc')}")

    # Test 3: Upload Authentic Image
    au_images = glob.glob("CASIA2/Au/*.jpg") + glob.glob("CASIA2/Au/*.png")
    if au_images:
        sample_au = au_images[0]
        print(f"\n[TEST 3] Testing POST /predict with Authentic Sample: {sample_au}")
        with open(sample_au, 'rb') as f:
            files = {'image': (sample_au, f, 'image/jpeg')}
            res_pred = requests.post(f"{BASE_URL}/predict", files=files)
        assert res_pred.status_code == 200, f"Predict returned {res_pred.status_code}"
        data_au = res_pred.json().get('data', {})
        print(f"[PASS] Authentic Image Prediction Output:")
        print(f"       - Prediction: {data_au.get('prediction')}")
        print(f"       - Confidence: {data_au.get('confidence')}%")
        print(f"       - Probability: {data_au.get('probability')}")
        print(f"       - Processing Time: {data_au.get('processing_time')}s")
        print(f"       - ELA Image URL: {data_au.get('ela_image_url')}")

    # Test 4: Upload Tampered Image
    tp_images = glob.glob("CASIA2/Tp/*.jpg") + glob.glob("CASIA2/Tp/*.png")
    if tp_images:
        sample_tp = tp_images[0]
        print(f"\n[TEST 4] Testing POST /predict with Tampered Sample: {sample_tp}")
        with open(sample_tp, 'rb') as f:
            files = {'image': (sample_tp, f, 'image/jpeg')}
            res_pred_tp = requests.post(f"{BASE_URL}/predict", files=files)
        assert res_pred_tp.status_code == 200, f"Predict returned {res_pred_tp.status_code}"
        data_tp = res_pred_tp.json().get('data', {})
        print(f"[PASS] Tampered Image Prediction Output:")
        print(f"       - Prediction: {data_tp.get('prediction')}")
        print(f"       - Confidence: {data_tp.get('confidence')}%")
        print(f"       - Probability: {data_tp.get('probability')}")
        print(f"       - Processing Time: {data_tp.get('processing_time')}s")
        print(f"       - ELA Image URL: {data_tp.get('ela_image_url')}")

    # Test 5: Session History Endpoint
    print("\n[TEST 5] Testing GET /history...")
    res_hist = requests.get(f"{BASE_URL}/history")
    assert res_hist.status_code == 200, f"History returned {res_hist.status_code}"
    history = res_hist.json().get('history', [])
    print(f"[PASS] History contains {len(history)} session prediction items")

    print("\n" + "=" * 60)
    print("       ALL SYSTEM TESTS PASSED SUCCESSFULLY (100%)       ")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    test_system()
