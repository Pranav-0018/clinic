"""
Quality Assessment Module
Assesses document blur, skew, contrast, resolution, and high-frequency spectral ratio.
Provides mathematical indicators for clinical document triage.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import fft2, fftshift
from typing import Dict, Any, Tuple


class QualityAssessor:
    """
    Evaluates raw scan quality against clinical document ingestion standards.
    """
    BLUR_THRESHOLD = 120.0        # Laplacian variance threshold
    MIN_CONTRAST_THRESHOLD = 35.0 # RMS contrast threshold
    MAX_SKEW_THRESHOLD = 3.5      # Degrees skew before flagging
    MIN_DPI_ESTIMATE = 150        # Approximate DPI estimate threshold

    @staticmethod
    def _to_grayscale_array(img: Image.Image) -> np.ndarray:
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float32)

    @classmethod
    def calculate_laplacian_variance(cls, gray: np.ndarray) -> float:
        """
        Computes variance of the Laplacian filter response.
        Var(nabla^2 I) is low for blurry scans and high for sharp text/edges.
        """
        # 3x3 discrete 2D Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        lap = ndimage.laplace(gray)
        return float(np.var(lap))

    @classmethod
    def calculate_skew_angle(cls, gray: np.ndarray) -> float:
        """
        Estimates skew angle using projection profile variance across angles [-15, 15] deg.
        Document text lines yield maximal projection variance when perfectly aligned horizontally.
        """
        # Downsample for computational speed
        h, w = gray.shape
        scale = min(1.0, 600.0 / max(h, w))
        if scale < 1.0:
            small = ndimage.zoom(gray, scale, order=1)
        else:
            small = gray

        # Binarize with Otsu-like threshold
        thresh = np.mean(small)
        binary = (small < thresh).astype(np.float32)

        best_score = -1.0
        best_angle = 0.0
        test_angles = np.linspace(-15.0, 15.0, 31)

        for angle in test_angles:
            # Rotate binary image
            rotated = ndimage.rotate(binary, angle, reshape=False, order=0, mode='constant', cval=0.0)
            # Row projection profile
            row_sums = np.sum(rotated, axis=1)
            score = np.var(row_sums)
            if score > best_score:
                best_score = score
                best_angle = angle

        return float(round(best_angle, 2))

    @classmethod
    def calculate_contrast(cls, gray: np.ndarray) -> Dict[str, float]:
        """
        Computes RMS contrast (standard deviation of intensity) and Michelson contrast.
        """
        rms_contrast = float(np.std(gray))
        min_val, max_val = float(np.min(gray)), float(np.max(gray))
        denom = max_val + min_val
        michelson = float((max_val - min_val) / denom) if denom > 0 else 0.0
        return {
            "rms_contrast": round(rms_contrast, 2),
            "michelson_contrast": round(michelson, 3),
            "min_intensity": min_val,
            "max_intensity": max_val
        }

    @classmethod
    def calculate_high_frequency_ratio(cls, gray: np.ndarray) -> float:
        """
        Computes ratio of high-frequency energy using 2D Fast Fourier Transform.
        Blurs attenuate high spatial frequencies in the power spectrum.
        """
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 6

        # Mask center (low frequency)
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        high_freq_mask = dist_from_center > radius

        total_energy = np.sum(magnitude_spectrum) + 1e-9
        high_energy = np.sum(magnitude_spectrum[high_freq_mask])

        ratio = float(high_energy / total_energy)
        return float(round(ratio, 4))

    @classmethod
    def assess(cls, image: Image.Image) -> Dict[str, Any]:
        """
        Performs holistic document quality assessment.
        """
        gray = cls._to_grayscale_array(image)
        w, h = image.size

        lap_var = cls.calculate_laplacian_variance(gray)
        skew_angle = cls.calculate_skew_angle(gray)
        contrast = cls.calculate_contrast(gray)
        hf_ratio = cls.calculate_high_frequency_ratio(gray)

        # Quality scoring heuristics
        is_blurry = lap_var < cls.BLUR_THRESHOLD
        is_skewed = abs(skew_angle) > cls.MAX_SKEW_THRESHOLD
        is_low_contrast = contrast["rms_contrast"] < cls.MIN_CONTRAST_THRESHOLD
        is_low_res = (w < 600 or h < 800)

        # Composite score in [0.0, 1.0]
        score_lap = min(1.0, lap_var / 300.0)
        score_skew = max(0.0, 1.0 - (abs(skew_angle) / 15.0))
        score_contrast = min(1.0, contrast["rms_contrast"] / 60.0)
        score_res = 1.0 if not is_low_res else 0.5

        composite_quality = 0.35 * score_lap + 0.25 * score_skew + 0.25 * score_contrast + 0.15 * score_res
        composite_quality = round(float(np.clip(composite_quality, 0.05, 0.99)), 3)

        status = "EXCELLENT" if composite_quality >= 0.80 else (
            "ACCEPTABLE" if composite_quality >= 0.55 else "POOR_QUALITY_FLAGGED"
        )

        return {
            "status": status,
            "composite_score": composite_quality,
            "resolution": {"width": w, "height": h},
            "laplacian_variance": round(lap_var, 2),
            "is_blurry": is_blurry,
            "skew_angle_deg": skew_angle,
            "is_skewed": is_skewed,
            "contrast": contrast,
            "is_low_contrast": is_low_contrast,
            "hf_ratio": hf_ratio,
            "recommended_actions": [
                action for action, cond in [
                    ("Deskew image by " + str(-skew_angle) + " deg", is_skewed),
                    ("Apply unsharp masking / Laplacian deblur", is_blurry),
                    ("Apply adaptive contrast enhancement / CLAHE", is_low_contrast),
                    ("Warn user: resolution is below 150 DPI target", is_low_res)
                ] if cond
            ]
        }
