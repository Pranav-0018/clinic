"""
Document Classification and Priority Triage Module
Implements Unit 2 models (MLE, Logistic Regression, LDA, Decision Tree, Random Forest)
and evaluates calibration, probability distributions, and verification routing.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Any, Tuple
from .features import FeatureExtractor


class DocumentClassifier:
    """
    Multiclass document classifier and priority verification triage engine.
    """
    CLASSES = [
        "Prescription",
        "Discharge Summary",
        "Lab Report",
        "Radiology Report",
        "Medical Invoice / Bill"
    ]

    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    def __init__(self):
        self.scaler = None
        self.models = {}
        self.metadata = {}
        self.active_model_name = "Logistic Regression"
        self._load_or_initialize()

    def _load_or_initialize(self):
        scaler_path = os.path.join(self.MODEL_DIR, "scaler.joblib")
        models_path = os.path.join(self.MODEL_DIR, "models.joblib")
        meta_path = os.path.join(self.MODEL_DIR, "metadata.joblib")

        if os.path.exists(scaler_path) and os.path.exists(models_path):
            self.scaler = joblib.load(scaler_path)
            self.models = joblib.load(models_path)
            if os.path.exists(meta_path):
                self.metadata = joblib.load(meta_path)
        else:
            # Models will be trained if not yet existing
            pass

    @staticmethod
    def compute_gaussian_mle(data_1d: np.ndarray) -> Dict[str, float]:
        """
        Mathematical MLE derivation for Gaussian distribution N(mu, sigma^2):
        mu_mle = (1/n) * sum(x_i)
        sigma2_mle = (1/n) * sum((x_i - mu)^2)   [BIASED: E[sigma2_mle] = (n-1)/n * sigma^2]
        sigma2_unbiased = (1/(n-1)) * sum((x_i - mu)^2)
        """
        n = len(data_1d)
        if n == 0:
            return {"mu_mle": 0.0, "sigma2_mle": 0.0, "sigma2_unbiased": 0.0, "bias_factor": 1.0}
        mu = float(np.mean(data_1d))
        var_mle = float(np.mean((data_1d - mu)**2))
        var_unbiased = float(np.var(data_1d, ddof=1)) if n > 1 else var_mle
        bias_factor = float((n - 1) / n) if n > 1 else 1.0
        return {
            "n_samples": n,
            "mu_mle": round(mu, 4),
            "sigma2_mle": round(var_mle, 4),
            "sigma2_unbiased": round(var_unbiased, 4),
            "bias_factor": round(bias_factor, 4),
            "mle_log_likelihood": round(float(-0.5 * n * (np.log(2 * np.pi * var_mle) + 1.0)), 2) if var_mle > 0 else 0.0
        }

    @staticmethod
    def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE) metric across confidence bins.
        Target: ECE < 0.05.
        """
        bin_limits = np.linspace(0, 1, n_bins + 1)
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true)

        ece = 0.0
        n = len(y_true)

        for i in range(n_bins):
            in_bin = (confidences > bin_limits[i]) & (confidences <= bin_limits[i+1])
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

        return float(round(ece, 4))

    def predict(self, feature_dict: Dict[str, float], model_name: str = None) -> Dict[str, Any]:
        """
        Predicts document category and priority triage queue.
        """
        if not self.models or not self.scaler:
            self._load_or_initialize()

        chosen_name = model_name or self.active_model_name
        model = self.models.get(chosen_name) or next(iter(self.models.values()), None)

        x_vec = np.array([FeatureExtractor.to_vector(feature_dict)])
        x_scaled = self.scaler.transform(x_vec)

        probs = model.predict_proba(x_scaled)[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = self.CLASSES[pred_idx]
        top_prob = float(probs[pred_idx])

        # Class breakdown
        class_probabilities = {
            cls_name: round(float(p), 4) for cls_name, p in zip(self.CLASSES, probs)
        }

        # Priority Verification Queue logic:
        # High-risk if low confidence (< 0.70) or document is prescription/discharge summary with poor quality
        is_priority = (
            top_prob < 0.75 or
            feature_dict.get("laplacian_var", 200) < 110 or
            predicted_class in ["Prescription", "Discharge Summary"] and top_prob < 0.85
        )

        priority_queue = "PRIORITY_VERIFICATION" if is_priority else "STANDARD_AUTOMATION"

        return {
            "predicted_class": predicted_class,
            "confidence": round(top_prob, 4),
            "priority_queue": priority_queue,
            "used_model": chosen_name,
            "class_probabilities": class_probabilities,
            "feature_snapshot": feature_dict,
            "available_models": list(self.models.keys())
        }
