# Clinical Document Classification & STG Compliance Extraction System

### Production Prototype & PBL 1 Evaluation Artifacts
- **Interactive Web Application:** `http://127.0.0.1:8000`
- **Jupyter Notebook:** `Clinical_Document_MLE_Regression_Classification.ipynb`
- **Location:** `C:\Users\Lenovo\.gemini\antigravity\scratch\clinical_stg_prototype`

---

## 1. System Architecture Pipeline

```
Scanned / Uploaded Document
  │
  ├── 1. Quality Assessment Gate (Laplacian blur variance, skew angle, RMS contrast, 2D FFT hf-ratio)
  ├── 2. Document Preprocessing (Deskewing [-15° to +15°], Median denoising, Otsu adaptive binarization)
  ├── 3. 9-Feature Extraction (mean_intensity, std_intensity, edge_density, laplacian_var, entropy, hf_ratio, sat_std, aspect_ratio, n_line_items)
  ├── 4. Unit 2 Document Classification & Priority Routing (Logistic Regression, LDA, Decision Tree, Random Forest)
  ├── 5. Spatial OCR Engine (Word bounding boxes [x1, y1, x2, y2], token confidence tracking)
  ├── 6. Clinical Structured NER (Patient demographics, clinical diagnoses, prescriptions, lab panels, invoice line items)
  ├── 7. STG Compliance Rule Engine (ICMR 2023, MoHFW, WHO guidelines: dosage ceilings, dual RAS blockade contraindications, antibiotic stewardship)
  └── 8. Calibrated Provenance Tracking & Clinician Sign-Off Gate (Human-in-the-loop review for confidence < 0.85)
```

---

## 2. PBL 1 Rubric Mapping (20 Marks)

| Item | Marks | Implementation in Prototype & Notebook |
|---|---|---|
| **1. Problem & Data** | 4 | Real vs surrogate data justification (CORD-v2, SROIE, RVL-CDIP, IAM bootstrap proxies; MIMIC-III/n2c2 DUA governance). |
| **2. EDA & Insights** | 4 | 9 statistical & morphological features, class-wise distributions, Laplacian variance separation, FFT high-frequency energy ratio. |
| **3. Unit 2 Implementation** | 4 | Gaussian MLE hand-worked derivation & biased variance proof, OLS normal equation regression, Logistic Regression, LDA, Decision Tree, KNN. |
| **4. Comparative Analysis** | 4 | 4-model evaluation: Accuracy, Precision, Recall, Macro F1, ROC-AUC (OVR), Expected Calibration Error (ECE < 0.05). |
| **5. Presentation & Viva** | 4 | 4-member presentation split, live web application demonstration, honest limitations defense, clinician sign-off workflow. |

---

## 3. Preloaded Clinical Test Cases

1. **Case 1 (Prescription - T2DM with Metformin Overdose):**
   - Diagnosis: Type 2 Diabetes Mellitus
   - Prescription: Tab Metformin 1500 mg BD (3000 mg/day)
   - **Audit Result:** `CRITICAL ALERT` - Supratherapeutic dose exceeds ICMR maximum safe ceiling of 2550 mg/day (Lactic acidosis risk).
2. **Case 2 (Discharge Summary - Hypertension with Dual RAS Blockade):**
   - Diagnosis: Essential Hypertension
   - Prescription: Tab Telmisartan 40 mg OD + Tab Enalapril 10 mg BD
   - **Audit Result:** `CRITICAL CONTRAINDICATION` - Concurrent ACE-inhibitor + ARB increases acute renal failure and hyperkalemia risk.
3. **Case 3 (Pediatric Prescription - Acute Gastroenteritis Antibiotic Flag):**
   - Diagnosis: Acute Gastroenteritis (Pediatric, 4 yr old)
   - Prescription: Syp Ciprofloxacin 250 mg + Syp ORS
   - **Audit Result:** `STEWARDSHIP ALERT` - Empiric fluoroquinolones contraindicated in simple watery diarrhea without dysentery.
4. **Case 4 (Clinical Lab Report - Compliant Glycemic Panel):**
   - Diagnosis: Type 2 Diabetes Mellitus (HbA1c 8.8%, FBG 184 mg/dL)
   - Prescription: Tab Metformin 1000 mg BD + Tab Glimepiride 1 mg OD
   - **Audit Result:** `FULLY COMPLIANT` - Regimen adheres to ICMR 2023 pharmacotherapy guidelines.
5. **Case 5 (Medical Invoice / Financial Memo):**
   - CORD-v2 / SROIE layout with 5 line items and $480.00 total.
   - **Audit Result:** Standard automation triage routing.

---

## 4. How to Run & Test

```bash
# Activate directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\clinical_stg_prototype

# Run backend and web interface
python run.py

# Open web browser
http://127.0.0.1:8000
```
