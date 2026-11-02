import os
import sys
import json
import hashlib
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')

def compute_file_hash(filepath):
    """Computes SHA-256 hash for duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_dataset(force_reverify=False):
    """
    Checks for existing verified dataset report artifacts.
    Loads existing report instantly if available, or scans dataset if forced/missing.
    """
    if not force_reverify and os.path.exists(config.DATASET_REPORT_JSON) and os.path.exists(config.DATASET_REPORT_CSV):
        with open(config.DATASET_REPORT_JSON, 'r') as f:
            report_summary = json.load(f)
        print("=" * 60)
        print("        CASIA v2.0 DATASET VERIFICATION & AUDIT         ")
        print("=" * 60)
        print(f"[CACHE] Loaded dataset report from {config.DATASET_REPORT_JSON}")
        print(f"[STATUS] Authentic (Au): {report_summary.get('authentic_count')} images")
        print(f"[STATUS] Tampered (Tp):  {report_summary.get('tampered_count')} images")
        print(f"[TOTAL]  Total Valid:    {report_summary.get('total_valid_images')} images")
        print("-" * 60)
        return report_summary

    print("=" * 60)
    print("        CASIA v2.0 DATASET VERIFICATION & AUDIT         ")
    print("=" * 60)
    print(f"[CONFIG] Target Dataset Directory: {config.DATASET_DIR}")
    
    if not os.path.exists(config.DATASET_DIR):
        print(f"\n[ERROR] Dataset directory NOT found at: {config.DATASET_DIR}")
        sys.exit(1)
        
    au_dir = config.AU_DIR
    tp_dir = config.TP_DIR
    
    if not os.path.exists(au_dir) or not os.path.exists(tp_dir):
        print(f"\n[ERROR] Au or Tp directory missing!")
        sys.exit(1)

    def process_folder(folder_path, label, label_name):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        records = []
        corrupted = 0
        
        for fname in tqdm(files, desc=f"Auditing {label_name} ({folder_path})"):
            fpath = os.path.join(folder_path, fname)
            file_size = os.path.getsize(fpath)
            
            try:
                with Image.open(fpath) as img:
                    img.verify()
                
                with Image.open(fpath) as img:
                    width, height = img.size
                    fmt = img.format or os.path.splitext(fname)[1].replace('.', '').upper()
                    mode = img.mode
                
                file_hash = compute_file_hash(fpath)
                
                records.append({
                    'filename': fname,
                    'filepath': fpath,
                    'label': label,
                    'label_name': label_name,
                    'width': width,
                    'height': height,
                    'format': fmt,
                    'mode': mode,
                    'size_bytes': file_size,
                    'sha256': file_hash,
                    'is_valid': True
                })
            except Exception as e:
                corrupted += 1
                print(f"\n[WARNING] Corrupted image skipped: {fname} ({e})")
                
        return records, corrupted

    au_records, au_corrupted = process_folder(au_dir, 0, 'Authentic')
    tp_records, tp_corrupted = process_folder(tp_dir, 1, 'Tampered')

    all_records = au_records + tp_records
    df = pd.DataFrame(all_records)
    
    duplicate_hashes = df[df.duplicated('sha256', keep=False)]
    num_duplicates = len(duplicate_hashes)

    au_count = len(au_records)
    tp_count = len(tp_records)
    total_valid = au_count + tp_count

    format_counts = df['format'].value_counts().to_dict()

    report_summary = {
        "dataset_name": "CASIA v2.0",
        "dataset_dir": config.DATASET_DIR,
        "total_valid_images": total_valid,
        "authentic_count": au_count,
        "tampered_count": tp_count,
        "corrupted_images_count": au_corrupted + tp_corrupted,
        "duplicate_hash_count": num_duplicates,
        "format_distribution": format_counts,
        "label_mapping": {
            "0": "Authentic (Original)",
            "1": "Tampered"
        }
    }

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.DATASET_REPORT_JSON, 'w') as f:
        json.dump(report_summary, f, indent=4)
    df.to_csv(config.DATASET_REPORT_CSV, index=False)

    print("\n--- VERIFICATION RESULTS ---")
    print(f"[STATUS] Valid Authentic (Au):  {au_count} images")
    print(f"[STATUS] Valid Tampered (Tp):   {tp_count} images")
    print(f"[TOTAL]  Total Valid Images:     {total_valid}")
    print(f"[INFO]   Corrupted Images:       {au_corrupted + tp_corrupted}")
    print(f"[INFO]   Duplicate Hash Records: {num_duplicates}")
    print(f"[INFO]   Formats:                {format_counts}")
    print(f"[SAVED] Dataset report saved to {config.DATASET_REPORT_JSON}")
    print(f"[SAVED] Dataset CSV saved to {config.DATASET_REPORT_CSV}")
    print("-" * 60)

    return report_summary

if __name__ == '__main__':
    verify_dataset(force_reverify=True)
