"""
Preprocessing Module
Performs deskewing, denoising, contrast stretching, and adaptive Otsu binarization.
Prepares degraded clinical document images for downstream OCR and Layout models.
"""

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage
from typing import Tuple, Dict, Any
from .quality import QualityAssessor


class DocumentPreprocessor:
    """
    Cleans and standardizes clinical documents prior to OCR.
    """

    @staticmethod
    def deskew(image: Image.Image, skew_angle: float) -> Image.Image:
        """
        Rotates image by -skew_angle with white padding to preserve text alignment.
        """
        if abs(skew_angle) < 0.3:
            return image
        # Negative angle to undo skew
        return image.rotate(-skew_angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

    @staticmethod
    def contrast_stretch(gray: np.ndarray) -> np.ndarray:
        """
        Stretches the intensity histogram between the 2nd and 98th percentiles.
        """
        p2, p98 = np.percentile(gray, (2, 98))
        if p98 - p2 < 1e-4:
            return gray
        stretched = np.clip((gray - p2) / (p98 - p2) * 255.0, 0, 255)
        return stretched.astype(np.uint8)

    @staticmethod
    def otsu_binarization(gray_u8: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Computes the optimal binarization threshold via Otsu's method by
        maximizing between-class variance: sigma_b^2(t) = w0(t)*w1(t)*(mu0(t) - mu1(t))^2
        """
        hist, _ = np.histogram(gray_u8, bins=256, range=(0, 256))
        total_pixels = gray_u8.size
        current_max, threshold = 0.0, 128

        sum_total = np.dot(np.arange(256), hist)
        sum_back, weight_back = 0.0, 0.0

        for t in range(256):
            weight_back += hist[t]
            if weight_back == 0:
                continue
            weight_fore = total_pixels - weight_back
            if weight_fore == 0:
                break

            sum_back += t * hist[t]
            mean_back = sum_back / weight_back
            mean_fore = (sum_total - sum_back) / weight_fore

            # Between class variance
            between_variance = weight_back * weight_fore * ((mean_back - mean_fore) ** 2)

            if between_variance > current_max:
                current_max = between_variance
                threshold = t

        binary = (gray_u8 > threshold).astype(np.uint8) * 255
        return binary, threshold

    @classmethod
    def denoise_and_sharpen(cls, image: Image.Image) -> Image.Image:
        """
        Applies median filtering to remove scanner salt-and-pepper noise
        followed by subtle unsharp masking for edge definition.
        """
        filtered = image.filter(ImageFilter.MedianFilter(size=3))
        sharpened = filtered.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        return sharpened

    @classmethod
    def preprocess_pipeline(cls, image: Image.Image, quality_report: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full preprocessing pipeline:
        1. Skew compensation
        2. Denoising & sharpening
        3. Adaptive contrast enhancement
        4. Otsu binarization
        """
        if quality_report is None:
            quality_report = QualityAssessor.assess(image)

        skew_angle = quality_report.get("skew_angle_deg", 0.0)

        # 1. Deskew
        deskewed = cls.deskew(image, skew_angle)

        # 2. Denoise and sharpen
        denoised = cls.denoise_and_sharpen(deskewed)

        # 3. Contrast stretch on grayscale
        gray = np.array(denoised.convert('L'), dtype=np.float32)
        enhanced_gray = cls.contrast_stretch(gray)

        # 4. Otsu binarize
        binary, otsu_thresh = cls.otsu_binarization(enhanced_gray)

        enhanced_uint8 = np.clip(enhanced_gray, 0, 255).astype(np.uint8)
        processed_img = Image.fromarray(enhanced_uint8)
        binary_img = Image.fromarray(binary.astype(np.uint8) if binary.dtype != np.uint8 else binary)

        return {
            "preprocessed_image": processed_img,
            "binary_image": binary_img,
            "otsu_threshold": otsu_thresh,
            "skew_corrected_deg": skew_angle,
            "pipeline_stages_completed": ["deskew", "denoise_median", "unsharp_mask", "contrast_stretch", "otsu_binarize"]
        }
