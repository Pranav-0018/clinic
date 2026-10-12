"""
Model Training & Evaluation Script (Unit 2 Models)
Generates bootstrap statistical feature representations for the 5 clinical document classes,
fits Logistic Regression, LDA, Decision Tree, and Random Forest models,
and computes complete evaluation metrics (Accuracy, F1, ROC-AUC, ECE, Confusion Matrices).
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from core.features import FeatureExtractor
from core.classifier import DocumentClassifier


def generate_bootstrap_dataset(n_samples_per_class: int = 400, random_seed: int = 42):
    """
    Generates realistic bootstrap feature distributions matching CORD-v2,
    RVL-CDIP, SROIE, and IAM document characteristics.
    Features:
    [mean_intensity, std_intensity, edge_density, laplacian_var, entropy, hf_ratio, sat_std, aspect_ratio, n_line_items]
    """
    np.random.seed(random_seed)
    features_list = []
    labels_list = []

    # 0: Prescription (compact Rx pad, aspect ~0.70, short line items ~10-14, entropy ~2.2-2.8)
    for _ in range(n_samples_per_class):
        mean_int = np.random.normal(244.0, 2.0)
        std_int = np.random.normal(20.0, 5.0)
        edge_dens = np.random.normal(0.149, 0.01)
        lap_var = np.random.normal(1200, 1000)
        entropy = np.random.normal(2.45, 0.25)
        hf_rat = np.random.normal(0.60, 0.20)
        sat_std = np.random.normal(0.045, 0.02)
        aspect = np.random.normal(0.705, 0.02)
        lines = np.random.normal(12.5, 1.5)
        features_list.append([mean_int, std_int, edge_dens, lap_var, entropy, hf_rat, sat_std, aspect, lines])
        labels_list.append(0)

    # 1: Discharge Summary (dense A4 text, aspect ~0.65, high line items ~22-30, entropy ~1.5-1.8)
    for _ in range(n_samples_per_class):
        mean_int = np.random.normal(246.5, 1.5)
        std_int = np.random.normal(30.5, 3.0)
        edge_dens = np.random.normal(0.145, 0.008)
        lap_var = np.random.normal(2700, 400)
        entropy = np.random.normal(1.65, 0.15)
        hf_rat = np.random.normal(0.82, 0.04)
        sat_std = np.random.normal(0.080, 0.015)
        aspect = np.random.normal(0.652, 0.02)
        lines = np.random.normal(23.5, 2.5)
        features_list.append([mean_int, std_int, edge_dens, lap_var, entropy, hf_rat, sat_std, aspect, lines])
        labels_list.append(1)

    # 2: Lab Report (wide tabular grid, aspect ~0.765, low edge density ~0.089, high sat_std ~0.17 for colored banners)
    for _ in range(n_samples_per_class):
        mean_int = np.random.normal(245.2, 1.5)
        std_int = np.random.normal(37.0, 3.5)
        edge_dens = np.random.normal(0.090, 0.012)
        lap_var = np.random.normal(3400, 400)
        entropy = np.random.normal(1.58, 0.15)
        hf_rat = np.random.normal(0.82, 0.04)
        sat_std = np.random.normal(0.165, 0.025)
        aspect = np.random.normal(0.765, 0.025)
        lines = np.random.normal(20.0, 2.0)
        features_list.append([mean_int, std_int, edge_dens, lap_var, entropy, hf_rat, sat_std, aspect, lines])
        labels_list.append(2)

    # 3: Radiology Report (dark image inset, low mean intensity ~165, high std intensity ~75)
    for _ in range(n_samples_per_class):
        mean_int = np.random.normal(165.0, 15.0)
        std_int = np.random.normal(75.0, 8.0)
        edge_dens = np.random.normal(0.085, 0.015)
        lap_var = np.random.normal(2200, 400)
        entropy = np.random.normal(3.8, 0.4)
        hf_rat = np.random.normal(0.70, 0.08)
        sat_std = np.random.normal(0.035, 0.015)
        aspect = np.random.normal(0.85, 0.05)
        lines = np.random.normal(14.0, 2.5)
        features_list.append([mean_int, std_int, edge_dens, lap_var, entropy, hf_rat, sat_std, aspect, lines])
        labels_list.append(3)

    # 4: Medical Invoice / Bill (narrow CORD/SROIE receipt aspect ~0.49, low sat_std ~0.005 thermal paper)
    for _ in range(n_samples_per_class):
        mean_int = np.random.normal(246.8, 1.2)
        std_int = np.random.normal(26.0, 2.5)
        edge_dens = np.random.normal(0.146, 0.01)
        lap_var = np.random.normal(3500, 400)
        entropy = np.random.normal(1.38, 0.12)
        hf_rat = np.random.normal(0.85, 0.03)
        sat_std = np.random.normal(0.006, 0.003)
        aspect = np.random.normal(0.489, 0.02)
        lines = np.random.normal(20.0, 2.0)
        features_list.append([mean_int, std_int, edge_dens, lap_var, entropy, hf_rat, sat_std, aspect, lines])
        labels_list.append(4)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def train_and_evaluate():
    print("[1/4] Generating bootstrap dataset...")
    X, y = generate_bootstrap_dataset(n_samples_per_class=500, random_seed=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("[2/4] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[3/4] Fitting Unit 2 models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }

    metrics = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='macro'))
        rec = float(recall_score(y_test, y_pred, average='macro'))
        f1 = float(f1_score(y_test, y_pred, average='macro'))
        roc_auc = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
        ece = DocumentClassifier.compute_ece(y_prob, y_test)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "ece": round(ece, 4),
            "confusion_matrix": cm
        }
        print(f"  * {name}: Acc={acc*100:.2f}%, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, ECE={ece:.4f}")

    # Compute Gaussian MLE on the laplacian_var feature for Member 2's topic
    laplacian_mle = DocumentClassifier.compute_gaussian_mle(X_train[:, 3])

    metadata = {
        "classes": DocumentClassifier.CLASSES,
        "feature_names": FeatureExtractor.FEATURE_NAMES,
        "metrics": metrics,
        "laplacian_mle_analysis": laplacian_mle,
        "test_sample_size": len(y_test),
        "train_sample_size": len(y_train)
    }

    os.makedirs(DocumentClassifier.MODEL_DIR, exist_ok=True)
    scaler_file = os.path.join(DocumentClassifier.MODEL_DIR, "scaler.joblib")
    models_file = os.path.join(DocumentClassifier.MODEL_DIR, "models.joblib")
    meta_file = os.path.join(DocumentClassifier.MODEL_DIR, "metadata.joblib")

    print("[4/4] Serializing trained artifacts...")
    joblib.dump(scaler, scaler_file)
    joblib.dump(models, models_file)
    joblib.dump(metadata, meta_file)

    print("Model training complete! Artifacts saved successfully.")


if __name__ == "__main__":
    train_and_evaluate()
