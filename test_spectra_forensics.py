import os
import unittest
import numpy as np
from PIL import Image

import config
from src.ela import convert_to_ela_image
from src.preprocessing import preprocess_image
from src.gradcam import generate_gradcam_heatmap, save_gradcam_overlay
from src.noise_analysis import analyze_noise_inconsistency
from src.frequency_analysis import analyze_frequency_spectrum
from src.metadata_analysis import analyze_metadata
from src.fusion import fuse_forensic_evidence
from src.report_generator import compute_file_sha256, generate_forensic_pdf_report
from src.model import build_cnn_model

class TestSpectraForensics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_img_dir = os.path.join(config.BASE_DIR, "uploads", "original")
        os.makedirs(cls.test_img_dir, exist_ok=True)
        cls.test_img_path = os.path.join(cls.test_img_dir, "test_sample.jpg")
        cls.invalid_img_path = os.path.join(cls.test_img_dir, "invalid_sample.txt")
        
        # Create a synthetic RGB test image
        img_arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_arr)
        pil_img.save(cls.test_img_path, 'JPEG', quality=95)

        # Create an invalid file
        with open(cls.invalid_img_path, 'w') as f:
            f.write("Invalid image file contents")

    def test_01_ela_conversion(self):
        ela_img = convert_to_ela_image(self.test_img_path, quality=90, scale=15.0)
        self.assertIsInstance(ela_img, Image.Image)
        self.assertEqual(ela_img.mode, 'RGB')
        self.assertEqual(ela_img.size, (300, 300))

    def test_02_sha256_hash(self):
        sha = compute_file_sha256(self.test_img_path)
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 64)

    def test_03_preprocessing(self):
        tensor = preprocess_image(self.test_img_path, target_size=(224, 224), use_ela=True)
        self.assertEqual(tensor.shape, (224, 224, 3))
        self.assertTrue(tensor.min() >= 0.0)
        self.assertTrue(tensor.max() <= 1.0)

    def test_04_noise_residual_analysis(self):
        res = analyze_noise_inconsistency(self.test_img_path)
        self.assertIn('noise_score', res)
        self.assertIsInstance(res['noise_score'], float)
        self.assertTrue(0.0 <= res['noise_score'] <= 100.0)

    def test_05_frequency_fft_analysis(self):
        res = analyze_frequency_spectrum(self.test_img_path)
        self.assertIn('frequency_score', res)
        self.assertIsInstance(res['frequency_score'], float)
        self.assertTrue(0.0 <= res['frequency_score'] <= 100.0)

    def test_06_metadata_analysis(self):
        meta = analyze_metadata(self.test_img_path)
        self.assertIn('has_exif', meta)
        self.assertIn('metadata_score', meta)
        self.assertTrue(0.0 <= meta['metadata_score'] <= 100.0)

    def test_07_evidence_fusion_and_verdict(self):
        # Test low tampered prob -> ORIGINAL
        fusion_orig = fuse_forensic_evidence(
            ela_score=20.0, deep_cnn_score=15.0, noise_score=10.0, metadata_score=0.0, frequency_score=5.0
        )
        self.assertEqual(fusion_orig['verdict'], 'Original')
        self.assertIn(fusion_orig['verdict'], ['Original', 'Tampered'])
        self.assertAlmostEqual(fusion_orig['original_probability'] + fusion_orig['tampered_probability'], 100.0, delta=0.5)

        # Test high tampered prob -> TAMPERED
        fusion_tamp = fuse_forensic_evidence(
            ela_score=80.0, deep_cnn_score=85.0, noise_score=75.0, metadata_score=90.0, frequency_score=60.0
        )
        self.assertEqual(fusion_tamp['verdict'], 'Tampered')
        self.assertIn(fusion_tamp['verdict'], ['Original', 'Tampered'])
        self.assertAlmostEqual(fusion_tamp['original_probability'] + fusion_tamp['tampered_probability'], 100.0, delta=0.5)

    def test_08_cnn_model_architecture(self):
        model = build_cnn_model(input_shape=(224, 224, 3), learning_rate=1e-4)
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 1))

    def test_09_gradcam_generation(self):
        model = build_cnn_model(input_shape=(224, 224, 3), learning_rate=1e-4)
        tensor = preprocess_image(self.test_img_path, target_size=(224, 224), use_ela=True)
        batch = np.expand_dims(tensor, axis=0)
        heatmap = generate_gradcam_heatmap(model, batch)
        self.assertEqual(heatmap.shape, (224, 224))
        out_path = os.path.join(self.test_img_dir, "test_gradcam.jpg")
        ret = save_gradcam_overlay(self.test_img_path, heatmap, out_path)
        self.assertTrue(os.path.exists(ret))

    def test_10_pdf_report_generation(self):
        sample_res = {
            "prediction": "Original",
            "confidence": 85.5,
            "probability": 0.145,
            "original_probability": 85.5,
            "tampered_probability": 14.5,
            "decision_threshold": 0.50,
            "model_version": "CNN-ELA-v1.2",
            "risk_level": "Low",
            "processing_time": 0.45,
            "image_name": "test_sample.jpg",
            "image_dimensions": "300 x 300",
            "metadata": {"camera_make": "TestCam", "camera_model": "X100", "software_used": "None", "has_exif": True},
            "fusion": {"score_breakdown": {"deep_cnn": 14.5, "ela_compression": 20.0, "noise_residual": 10.0, "fft_frequency": 5.0, "exif_metadata": 0.0}}
        }
        pdf_out = os.path.join(self.test_img_dir, "test_report.pdf")
        ret_path = generate_forensic_pdf_report(sample_res, pdf_out)
        self.assertTrue(os.path.exists(ret_path))
        self.assertGreater(os.path.getsize(ret_path), 0)

    def test_11_invalid_image_handling(self):
        with self.assertRaises(Exception):
            preprocess_image(self.invalid_img_path, target_size=(224, 224), use_ela=True)

if __name__ == '__main__':
    unittest.main()
