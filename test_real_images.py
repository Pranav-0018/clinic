import os
from PIL import Image
from core.features import FeatureExtractor
from core.classifier import DocumentClassifier

c = DocumentClassifier()

files = [
    ("Real CORD-v2 Receipt", "data/real/real_cord_v2_receipt.png"),
    ("Real SROIE Receipt", "data/real/real_sroie_receipt.png"),
    ("Real RVL-CDIP Sample", "data/real/real_rvl_cdip_sample.png"),
    ("Real IAM Line", "data/real/real_iam_handwriting_line.png"),
]

for label, fpath in files:
    if os.path.exists(fpath):
        img = Image.open(fpath)
        feats = FeatureExtractor.extract_features(img)
        pred = c.predict(feats, model_name="Logistic Regression")
        print(f"{label}: Predicted={pred['predicted_class']} ({pred['confidence']*100:.1f}%)")
