"""
Feature Extraction Module
Extracts the 9 statistical, morphological, and frequency features defined in the PBL-1 brief:
1. mean_intensity
2. std_intensity
3. edge_density
4. laplacian_var
5. entropy
6. hf_ratio
7. sat_std
8. aspect_ratio
9. n_line_items
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import fft2, fftshift
from typing import Dict, List, Any


class FeatureExtractor:
    """
    Extracts the 9 canonical document features.
    """
    FEATURE_NAMES = [
        "mean_intensity",
        "std_intensity",
        "edge_density",
        "laplacian_var",
        "entropy",
        "hf_ratio",
        "sat_std",
        "aspect_ratio",
        "n_line_items"
    ]

    @classmethod
    def extract_features(cls, image: Image.Image) -> Dict[str, float]:
        """
        Computes all 9 features for an input PIL Image.
        """
        w, h = image.size
        aspect_ratio = float(round(w / float(h), 4)) if h > 0 else 1.0

        # RGB and Grayscale arrays
        rgb = np.array(image.convert('RGB'), dtype=np.float32)
        gray = np.array(image.convert('L'), dtype=np.float32)

        # 1 & 2: Mean and Std of intensity
        mean_intensity = float(round(np.mean(gray), 3))
        std_intensity = float(round(np.std(gray), 3))

        # 3: Edge density via Sobel gradient magnitude
        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        grad_mag = np.hypot(gx, gy)
        edge_thresh = np.percentile(grad_mag, 85)
        edge_density = float(round(np.mean(grad_mag > edge_thresh), 4))

        # 4: Laplacian variance
        lap = ndimage.laplace(gray)
        laplacian_var = float(round(np.var(lap), 3))

        # 5: Shannon Entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_prob = hist / (hist.sum() + 1e-12)
        non_zero = hist_prob[hist_prob > 0]
        entropy = float(round(-np.sum(non_zero * np.log2(non_zero)), 4))

        # 6: High Frequency Energy Ratio (2D FFT)
        f = fft2(gray)
        fshift = fftshift(f)
        mag = np.abs(fshift)
        gh, gw = gray.shape
        cy, cx = gh // 2, gw // 2
        radius = min(gh, gw) // 6
        y, x = np.ogrid[:gh, :gw]
        mask = (x - cx)**2 + (y - cy)**2 > radius**2
        tot_energy = np.sum(mag) + 1e-12
        hf_ratio = float(round(np.sum(mag[mask]) / tot_energy, 4))

        # 7: Saturation standard deviation in HSV color space
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        sat = np.zeros_like(cmax)
        non_zero_mask = cmax > 0
        sat[non_zero_mask] = delta[non_zero_mask] / cmax[non_zero_mask]
        sat_std = float(round(np.std(sat), 4))

        # 9: Morphological estimated line-item count
        # Binarize and run horizontal projection peaks
        thresh = np.mean(gray) * 0.95
        bin_inv = (gray < thresh).astype(np.float32)
        # Morphological horizontal dilation
        horizontal_kernel = np.ones((1, max(5, int(w * 0.03))), dtype=np.float32)
        dilated = ndimage.binary_dilation(bin_inv, structure=horizontal_kernel)
        h_profile = np.sum(dilated, axis=1)
        prof_thresh = np.mean(h_profile) * 0.6
        # Count transitions above threshold
        above = h_profile > prof_thresh
        transitions = np.diff(above.astype(int))
        n_line_items = float(max(1, np.sum(transitions == 1)))

        return {
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "edge_density": edge_density,
            "laplacian_var": laplacian_var,
            "entropy": entropy,
            "hf_ratio": hf_ratio,
            "sat_std": sat_std,
            "aspect_ratio": aspect_ratio,
            "n_line_items": n_line_items
        }

    @classmethod
    def to_vector(cls, features_dict: Dict[str, float]) -> List[float]:
        """Converts feature dictionary to an ordered feature vector."""
        return [features_dict[name] for name in cls.FEATURE_NAMES]
