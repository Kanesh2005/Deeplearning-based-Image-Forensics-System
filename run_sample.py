import os
import sys
import glob
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.prediction import predict_image

def main():
    print("=" * 60)
    print("          SPECTRA::FORENSICS — SAMPLE IMAGE TEST          ")
    print("=" * 60)

    # Pick a sample image from dataset
    au_images = glob.glob(os.path.join(config.AU_DIR, "*.jpg"))
    tp_images = glob.glob(os.path.join(config.TP_DIR, "*.tif")) + glob.glob(os.path.join(config.TP_DIR, "*.jpg"))

    sample_au = au_images[0] if au_images else None
    sample_tp = tp_images[0] if tp_images else None

    print("\n[DIRECT MODEL INFERENCE TEST]")
    if sample_au:
        print(f"\n--- Testing Authentic Sample Image: {os.path.basename(sample_au)} ---")
        res_au = predict_image(sample_au)
        print(f"  File Name:        {res_au['image_name']}")
        print(f"  Image Dimensions: {res_au['image_dimensions']}")
        print(f"  CNN Prediction:   {res_au['prediction']}")
        print(f"  Probability:      {res_au['probability']}")
        print(f"  Confidence:       {res_au['confidence']}%")
        print(f"  Processing Time:  {res_au['processing_time']}s")
        print(f"  ELA Image URL:    {res_au['ela_image_url']}")

    if sample_tp:
        print(f"\n--- Testing Tampered Sample Image: {os.path.basename(sample_tp)} ---")
        res_tp = predict_image(sample_tp)
        print(f"  File Name:        {res_tp['image_name']}")
        print(f"  Image Dimensions: {res_tp['image_dimensions']}")
        print(f"  CNN Prediction:   {res_tp['prediction']}")
        print(f"  Probability:      {res_tp['probability']}")
        print(f"  Confidence:       {res_tp['confidence']}%")
        print(f"  Processing Time:  {res_tp['processing_time']}s")
        print(f"  ELA Image URL:    {res_tp['ela_image_url']}")

    print("\n" + "=" * 60)
    print("      TESTING HTTP FLASK ENDPOINT (http://127.0.0.1:5000/predict)      ")
    print("=" * 60)

    try:
        if sample_au:
            print(f"\nSending POST request with {os.path.basename(sample_au)} to Flask server...")
            with open(sample_au, 'rb') as f:
                files = {'image': (os.path.basename(sample_au), f, 'image/jpeg')}
                response = requests.post("http://127.0.0.1:5000/predict", files=files, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('data', {})
                print(f"[SUCCESS 200 OK] Server Response:")
                print(f"  Prediction:   {data.get('prediction')}")
                print(f"  Probability:  {data.get('probability')}")
                print(f"  Confidence:   {data.get('confidence')}%")
                print(f"  ELA Path:     {data.get('ela_image_url')}")
            else:
                print(f"[HTTP {response.status_code}] Response: {response.text}")
    except Exception as e:
        print(f"[NOTE] Flask server request failed: {e}")
        print("(If Flask server is not running, run 'python app.py' to launch it.)")

    print("\n" + "=" * 60)
    print("                    SAMPLE TEST COMPLETE                  ")
    print("=" * 60)

if __name__ == '__main__':
    main()
