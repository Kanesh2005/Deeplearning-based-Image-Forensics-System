import sys

def check():
    libraries = [
        'tensorflow',
        'keras',
        'flask',
        'numpy',
        'pandas',
        'cv2',
        'PIL',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'requests'
    ]

    print("=" * 60)
    print("      SPECTRA::FORENSICS — ENVIRONMENT & LIBRARY AUDIT    ")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version}")
    print("-" * 60)

    missing = []
    for lib in libraries:
        try:
            mod = __import__(lib)
            version = getattr(mod, '__version__', 'Installed (No version attr)')
            print(f"  [OK]   {lib:<15} version: {version}")
        except Exception as e:
            print(f"  [FAIL] {lib:<15} ERROR: {e}")
            missing.append(lib)

    print("-" * 60)
    if missing:
        print(f"[WARNING] Missing or failing libraries: {', '.join(missing)}")
    else:
        print("[SUCCESS] All required libraries are installed and importable!")
    print("=" * 60)

if __name__ == '__main__':
    check()
