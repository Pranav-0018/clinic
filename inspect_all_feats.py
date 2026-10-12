import os
from PIL import Image
from core.features import FeatureExtractor
from core.classifier import DocumentClassifier

fnames = {
    1: 'sample_1_prescription_overdose.png',
    2: 'sample_2_discharge_dual_blockade.png',
    3: 'sample_3_pediatric_antibiotic_flag.png',
    4: 'sample_4_lab_report_compliant.png',
    5: 'sample_5_medical_invoice.png'
}
for i in range(1, 6):
    img = Image.open(os.path.join("data", fnames[i]))
    feats = FeatureExtractor.extract_features(img)
    print(f"--- Case {i} ---")
    for k, v in feats.items():
        print(f"  {k}: {v}")
