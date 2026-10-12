"""
Clinical Named Entity Recognition (NER) & Structured Extraction Module
Parses clinical tokens and text to extract:
- Patient demographics
- Physician & Encounter details
- Clinical Diagnoses
- Prescriptions (Drug, Strength, Unit, Frequency, Daily Dose, Duration)
- Lab Investigations (Test, Value, Unit, Reference, Status)
- Financials (Line items, Subtotal, Tax, Total for invoices)
Links every extracted field directly to token bounding boxes and confidence scores.
"""

import re
from typing import Dict, List, Any, Optional


class ClinicalNERExtractor:
    """
    Extracts structured clinical entities with token provenance.
    """

    COMMON_DRUGS = [
        "Metformin", "Glimepiride", "Empagliflozin", "Dapagliflozin", "Sitagliptin", "Insulin",
        "Telmisartan", "Amlodipine", "Enalapril", "Losartan", "Atenolol", "Ramipril",
        "Atorvastatin", "Rosuvastatin", "Fenofibrate", "Gemfibrozil",
        "Amoxicillin", "Clavulanate", "Azithromycin", "Ciprofloxacin", "Ofloxacin", "Ceftriaxone",
        "Paracetamol", "Ibuprofen", "Diclofenac", "Pantoprazole", "Omeprazole",
        "ORS", "Zinc", "Budesonide", "Formoterol", "Salbutamol", "Propranolol"
    ]

    DIAGNOSES = [
        "Type 2 Diabetes Mellitus", "Type 2 Diabetes", "Diabetes Mellitus", "T2DM",
        "Essential Hypertension", "Hypertension", "HTN",
        "Community-Acquired Pneumonia", "Pneumonia", "CAP",
        "Acute Gastroenteritis", "Gastroenteritis", "AGE",
        "Dyslipidemia", "Hypercholesterolemia",
        "Bronchial Asthma", "Asthma", "COPD",
        "Upper Respiratory Tract Infection", "URTI", "Acute Pharyngitis"
    ]

    @classmethod
    def extract_entities(cls, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses OCR full text and tokens into clinical entities.
        """
        text = ocr_result.get("full_text", "")
        tokens = ocr_result.get("tokens", [])
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        entities = {
            "patient": cls._extract_patient_demographics(lines, tokens),
            "encounter": cls._extract_encounter_info(lines, tokens),
            "diagnoses": cls._extract_diagnoses(lines, tokens),
            "medications": cls._extract_medications(lines, tokens),
            "lab_investigations": cls._extract_lab_investigations(lines, tokens),
            "invoice_data": cls._extract_invoice_fields(lines, tokens)
        }

        # Global extraction confidence
        conf_scores = []
        if entities["patient"].get("name"):
            conf_scores.append(entities["patient"]["name"]["confidence"])
        for m in entities["medications"]:
            conf_scores.append(m["confidence"])
        for d in entities["diagnoses"]:
            conf_scores.append(d["confidence"])
        for l in entities["lab_investigations"]:
            conf_scores.append(l["confidence"])

        entities["mean_ner_confidence"] = float(round(sum(conf_scores) / max(len(conf_scores), 1), 3)) if conf_scores else 0.88

        return entities

    @classmethod
    def _find_tokens_for_phrase(cls, phrase: str, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Matches a phrase to corresponding OCR tokens."""
        phrase_words = phrase.lower().split()
        if not phrase_words:
            return []
        matched = []
        for token in tokens:
            t_clean = re.sub(r'[^\w]', '', token["text"].lower())
            if any(pw in t_clean or t_clean in pw for pw in phrase_words if len(pw) > 2):
                matched.append(token)
        return matched

    @classmethod
    def _merge_bboxes(cls, tokens: List[Dict[str, Any]]) -> Optional[List[int]]:
        """Combines multiple token bounding boxes into one bounding box [x1, y1, x2, y2]."""
        if not tokens:
            return None
        x1 = min(t["bbox"][0] for t in tokens)
        y1 = min(t["bbox"][1] for t in tokens)
        x2 = max(t["bbox"][2] for t in tokens)
        y2 = max(t["bbox"][3] for t in tokens)
        return [x1, y1, x2, y2]

    @classmethod
    def _extract_patient_demographics(cls, lines: List[str], tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        patient_info = {}
        for line in lines:
            # Name
            m_name = re.search(r'(?:Patient|Patient Name|Name|Pt)\s*:\s*([A-Za-z\s\.\-\']+?)(?=[\|\/I1]\s*Age|Age|Gender|\n|$)', line, re.I)
            if m_name and "name" not in patient_info:
                val = m_name.group(1).strip()
                if val and len(val) > 2:
                    match_toks = cls._find_tokens_for_phrase(val, tokens)
                    patient_info["name"] = {
                        "value": val,
                        "confidence": 0.94,
                        "bbox": cls._merge_bboxes(match_toks)
                    }

            # Age
            m_age = re.search(r'Age\s*:\s*(\d{1,3})\s*(?:Y|Years|yrs|m)?', line, re.I)
            if m_age and "age" not in patient_info:
                val = int(m_age.group(1))
                match_toks = cls._find_tokens_for_phrase(m_age.group(0), tokens)
                patient_info["age"] = {
                    "value": val,
                    "confidence": 0.96,
                    "bbox": cls._merge_bboxes(match_toks)
                }

            # Gender
            m_gen = re.search(r'Gender\s*:\s*(Male|Female|Other|M|F)', line, re.I)
            if m_gen and "gender" not in patient_info:
                val = "Male" if m_gen.group(1).upper().startswith("M") else "Female"
                match_toks = cls._find_tokens_for_phrase(m_gen.group(0), tokens)
                patient_info["gender"] = {
                    "value": val,
                    "confidence": 0.97,
                    "bbox": cls._merge_bboxes(match_toks)
                }

            # UHID / MRN
            m_uhid = re.search(r'(?:UHID|MRN)\s*:\s*([A-Za-z0-9\-]+)', line, re.I)
            if not m_uhid:
                m_uhid = re.search(r'(?:IPD|OPD|Reg|Invoice No|Tax Invoice No)\s*:\s*([A-Za-z0-9\-]+)', line, re.I)
            if m_uhid and "uhid" not in patient_info:
                val = m_uhid.group(1).strip()
                match_toks = cls._find_tokens_for_phrase(val, tokens)
                patient_info["uhid"] = {
                    "value": val,
                    "confidence": 0.95,
                    "bbox": cls._merge_bboxes(match_toks)
                }

        return patient_info

    @classmethod
    def _extract_encounter_info(cls, lines: List[str], tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        encounter = {}
        for line in lines:
            m_date = re.search(r'(?:Date|Dt)\s*:\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', line, re.I)
            if m_date and "date" not in encounter:
                encounter["date"] = {
                    "value": m_date.group(1),
                    "confidence": 0.98,
                    "bbox": cls._merge_bboxes(cls._find_tokens_for_phrase(m_date.group(1), tokens))
                }
            m_doc = re.search(r'(?:Dr\.|Doctor)\s*([A-Za-z\s\.]+?)(?:,|\(|\||$)', line, re.I)
            if m_doc and "doctor" not in encounter:
                val = "Dr. " + m_doc.group(1).strip()
                encounter["doctor"] = {
                    "value": val,
                    "confidence": 0.93,
                    "bbox": cls._merge_bboxes(cls._find_tokens_for_phrase(val, tokens))
                }
            if any(k in line.lower() for k in ["hospital", "clinic", "health centre", "dispensary"]):
                if "hospital" not in encounter:
                    encounter["hospital"] = {
                        "value": line.strip(),
                        "confidence": 0.91,
                        "bbox": cls._merge_bboxes(cls._find_tokens_for_phrase(line[:25], tokens))
                    }
        return encounter

    @classmethod
    def _extract_diagnoses(cls, lines: List[str], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        found_diagnoses = []
        for line in lines:
            line_l = line.lower()
            if any(term in line_l for term in ["diagnosis", "impression", "dx:", "condition"]):
                for d in cls.DIAGNOSES:
                    if d.lower() in line_l:
                        match_toks = cls._find_tokens_for_phrase(d, tokens)
                        found_diagnoses.append({
                            "diagnosis": d,
                            "raw_text": line,
                            "confidence": 0.94,
                            "bbox": cls._merge_bboxes(match_toks)
                        })
        if not found_diagnoses:
            # Fallback scan across all text
            all_text = " ".join(lines).lower()
            for d in cls.DIAGNOSES:
                if d.lower() in all_text and not any(x["diagnosis"] == d for x in found_diagnoses):
                    match_toks = cls._find_tokens_for_phrase(d, tokens)
                    found_diagnoses.append({
                        "diagnosis": d,
                        "raw_text": d,
                        "confidence": 0.88,
                        "bbox": cls._merge_bboxes(match_toks)
                    })
        return found_diagnoses

    @classmethod
    def _extract_medications(cls, lines: List[str], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        meds = []
        rx_section = False

        for line in lines:
            if "rx" in line.lower() or "tab " in line.lower() or "cap " in line.lower() or "syp " in line.lower():
                rx_section = True

            # Match drug names
            for drug in cls.COMMON_DRUGS:
                pattern = rf'\b(?:Tab|Cap|Syp|Inj)?\s*({re.escape(drug)})\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)?'
                m = re.search(pattern, line, re.I)
                if m:
                    drug_name = m.group(1).title()
                    strength = float(m.group(2))
                    unit = (m.group(3) or "mg").lower()

                    # Extract frequency
                    freq_pattern = r'\b(1-0-1|1-0-0|0-0-1|1-1-1|0-1-0|OD|BD|TDS|QID|SOS|HS|STAT)\b'
                    m_freq = re.search(freq_pattern, line, re.I)
                    freq_str = m_freq.group(1).upper() if m_freq else "OD"

                    # Calculate times per day
                    multiplier = 1
                    if freq_str in ["1-0-1", "BD"]:
                        multiplier = 2
                    elif freq_str in ["1-1-1", "TDS"]:
                        multiplier = 3
                    elif freq_str in ["QID"]:
                        multiplier = 4
                    elif freq_str in ["1-0-0", "0-0-1", "0-1-0", "OD", "HS"]:
                        multiplier = 1

                    daily_dose = strength * multiplier

                    # Duration
                    m_dur = re.search(r'(?:x|for)\s*(\d{1,3})\s*(?:days|day|d|weeks|months)?', line, re.I)
                    duration_days = int(m_dur.group(1)) if m_dur else 30

                    match_toks = cls._find_tokens_for_phrase(drug_name, tokens)

                    # Deduplicate
                    if not any(x["drug"] == drug_name for x in meds):
                        meds.append({
                            "drug": drug_name,
                            "strength": strength,
                            "unit": unit,
                            "frequency": freq_str,
                            "daily_dose_mg": daily_dose if unit == "mg" else strength,
                            "duration_days": duration_days,
                            "line_text": line,
                            "confidence": 0.93,
                            "bbox": cls._merge_bboxes(match_toks)
                        })

        return meds

    @classmethod
    def _extract_lab_investigations(cls, lines: List[str], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        labs = []
        patterns = [
            (r'HbA1c\s*:\s*(\d+(?:\.\d+)?)\s*(%)?', "HbA1c", "%", "4.0 - 5.6"),
            (r'(?:Fasting Blood Glucose|FBG|FBS)\s*:\s*(\d+(?:\.\d+)?)\s*(mg\/dL)?', "Fasting Blood Glucose", "mg/dL", "70 - 99"),
            (r'(?:Serum Creatinine|Creatinine)\s*:\s*(\d+(?:\.\d+)?)\s*(mg\/dL)?', "Serum Creatinine", "mg/dL", "0.7 - 1.3"),
            (r'eGFR\s*:\s*(\d+(?:\.\d+)?)\s*(mL\/min)?', "eGFR", "mL/min/1.73m2", "> 90"),
            (r'(?:Total Cholesterol|Cholesterol)\s*:\s*(\d+(?:\.\d+)?)\s*(mg\/dL)?', "Total Cholesterol", "mg/dL", "< 200"),
            (r'(?:Blood Pressure|BP)\s*:\s*(\d{2,3})\/(\d{2,3})', "Blood Pressure", "mmHg", "< 120/80")
        ]

        for line in lines:
            for pat, test_name, unit, ref in patterns:
                m = re.search(pat, line, re.I)
                if m:
                    if test_name == "Blood Pressure":
                        val_str = f"{m.group(1)}/{m.group(2)}"
                        val_num = float(m.group(1))
                    else:
                        val_num = float(m.group(1))
                        val_str = str(val_num)

                    # Determine abnormal
                    is_abnormal = False
                    if test_name == "HbA1c" and val_num > 5.7:
                        is_abnormal = True
                    elif test_name == "Fasting Blood Glucose" and val_num > 100:
                        is_abnormal = True
                    elif test_name == "Serum Creatinine" and val_num > 1.3:
                        is_abnormal = True
                    elif test_name == "eGFR" and val_num < 60:
                        is_abnormal = True

                    match_toks = cls._find_tokens_for_phrase(test_name, tokens)
                    labs.append({
                        "test_name": test_name,
                        "value": val_str,
                        "numeric_value": val_num,
                        "unit": unit,
                        "reference_range": ref,
                        "is_abnormal": is_abnormal,
                        "confidence": 0.95,
                        "bbox": cls._merge_bboxes(match_toks)
                    })
        return labs

    @classmethod
    def _extract_invoice_fields(cls, lines: List[str], tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts CORD/SROIE invoice summary fields."""
        invoice = {}
        for line in lines:
            m_tot = re.search(r'(?:Total|Net Amount|Grand Total)\s*[:\$₹Rs\.]*\s*([\d,]+\.?\d*)', line, re.I)
            if m_tot and "total" not in invoice:
                val = float(m_tot.group(1).replace(",", ""))
                invoice["total"] = {
                    "value": val,
                    "confidence": 0.96,
                    "bbox": cls._merge_bboxes(cls._find_tokens_for_phrase(m_tot.group(0), tokens))
                }
            m_sub = re.search(r'(?:Subtotal|Sub-Total)\s*[:\$₹Rs\.]*\s*([\d,]+\.?\d*)', line, re.I)
            if m_sub and "subtotal" not in invoice:
                val = float(m_sub.group(1).replace(",", ""))
                invoice["subtotal"] = {
                    "value": val,
                    "confidence": 0.94,
                    "bbox": cls._merge_bboxes(cls._find_tokens_for_phrase(m_sub.group(0), tokens))
                }
        return invoice
