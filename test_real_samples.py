import os
from PIL import Image
from core.features import FeatureExtractor
from core.classifier import DocumentClassifier
from core.ocr_engine import OCREngine

clf = DocumentClassifier()
files = [
    ("data/real/real_cord_v2_receipt.png", "Real CORD-v2 (Receipt)"),
    ("data/real/real_sroie_receipt.png", "Real SROIE (Receipt)"),
    ("data/real/real_rvl_cdip_sample.png", "Real RVL-CDIP (Document)"),
    ("data/real/real_iam_handwriting_line.png", "Real IAM (Handwritten)"),
    ("data/sample_1_prescription_overdose.png", "Clinical Prescription (ICMR Case)")
]

for path, name in files:
    img = Image.open(path)
    feats = FeatureExtractor.extract_features(img)
    res = clf.predict(feats)
    ocr_res = OCREngine.extract_text(img)
    preview = ocr_res['text'][:100].replace('\n', ' ')
    print(f"=== {name} ===")
    print(f"Size: {img.size}, Aspect Ratio: {feats.aspect_ratio:.2f}")
    print(f"Prediction: {res['predicted_class']} (Conf: {res['confidence']:.3f})")
    print(f"OCR words count: {len(ocr_res['tokens'])}, Preview: {preview}\n")
