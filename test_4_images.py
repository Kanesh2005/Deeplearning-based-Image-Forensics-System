import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.prediction import predict_image

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')

def get_image_files(directory):
    return [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

def main():
    print("=" * 60)
    print("           PHASE 11 — FINAL 4-IMAGE CNN TEST            ")
    print("=" * 60)

    au_images = get_image_files(config.AU_DIR)
    tp_images = get_image_files(config.TP_DIR)

    if len(au_images) < 2 or len(tp_images) < 2:
        print("[ERROR] Not enough sample images found for 4-image test.")
        return

    # Select 2 Original and 2 Tampered images
    test_set = [
        ("1 Original image", au_images[0]),
        ("2 Tampered image", tp_images[0]),
        ("3 Different Original image", au_images[1]),
        ("4 Different Tampered image", tp_images[1])
    ]

    for label_desc, img_path in test_set:
        result = predict_image(img_path)
        print(f"\n--- Test Item: {label_desc} ---")
        print(f"filename:    {result['image_name']}")
        print(f"prediction:  {result['prediction']}")
        print(f"probability: {result['probability']:.4f}")
        print(f"confidence:  {result['confidence']:.2f}%")

    print("\n" + "=" * 60)
    print("[SUCCESS] 4-Image Verification Completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
