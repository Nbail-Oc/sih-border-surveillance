"""Unit tests for CLAHE low-light and high-glare preprocessor."""

import unittest
import numpy as np


class TestLPRClaheEnhancement(unittest.TestCase):
    def setUp(self):
        from frigate.analytics.clahe import enhance_license_plate_contrast
        self.enhancer = enhance_license_plate_contrast

    def test_clahe_enhances_low_contrast_image(self):
        """Simulate a dark/low-contrast license plate crop and verify contrast expansion."""
        # Low contrast image with pixel values clustered between 40 and 60
        dark_crop = np.random.randint(40, 60, (48, 320, 3), dtype=np.uint8)
        initial_std = np.std(dark_crop)

        enhanced = self.enhancer(dark_crop, clip_limit=2.0, tile_grid_size=(8, 8))

        self.assertEqual(enhanced.shape, dark_crop.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Enhanced image should have significantly higher variance/standard deviation
        self.assertGreater(np.std(enhanced), initial_std * 1.5)

    def test_clahe_handles_single_channel_grayscale(self):
        """Verify grayscale input is correctly processed and returned."""
        gray_crop = np.random.randint(30, 70, (48, 320), dtype=np.uint8)
        enhanced = self.enhancer(gray_crop)
        self.assertEqual(enhanced.shape, gray_crop.shape)
