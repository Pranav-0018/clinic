import os
from PIL import Image
from core.features import FeatureExtractor
from core.classifier import DocumentClassifier

c = DocumentClassifier()
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
    pred = c.predict(feats, model_name='Logistic Regression')
    print(f"Case {i}: Predicted={pred['predicted_class']} ({pred['confidence']*100:.1f}%) | lines={feats['n_line_items']} | aspect={feats['aspect_ratio']} | edge_density={feats['edge_density']} | mean={feats['mean_intensity']}")
