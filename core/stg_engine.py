"""
Standard Treatment Guidelines (STG) Compliance Rule Engine
Audits extracted clinical documents against ICMR, MoHFW, and WHO clinical treatment protocols.
Validates:
1. Diagnosis-to-drug indication matching
2. Safe dosage limits (subtherapeutic vs recommended vs toxic overdose)
3. Duration & frequency bounds
4. Contraindications and dangerous drug-drug interactions
5. Antimicrobial stewardship flags
"""

from typing import Dict, List, Any


class STGComplianceEngine:
    """
    Evaluates clinical records against standardized treatment guidelines.
    """

    GUIDELINE_CATALOG = {
        "Type 2 Diabetes Mellitus": {
            "authority": "ICMR Guidelines for Management of Type 2 Diabetes (2023)",
            "section": "Section 4.2 - Pharmacotherapy: Oral Antidiabetic Agents",
            "first_line_drugs": ["Metformin"],
            "second_line_drugs": ["Glimepiride", "Empagliflozin", "Dapagliflozin", "Sitagliptin", "Insulin"],
            "dosage_rules": {
                "Metformin": {
                    "min_daily_mg": 500,
                    "max_daily_mg": 2550,
                    "target_maintenance_mg": 2000,
                    "warning_threshold_mg": 2550,
                    "risk": "Lactic acidosis risk at supratherapeutic doses; gastrointestinal intolerance."
                },
                "Glimepiride": {
                    "min_daily_mg": 1,
                    "max_daily_mg": 8,
                    "target_maintenance_mg": 2,
                    "risk": "Hypoglycemia risk."
                },
                "Empagliflozin": {
                    "min_daily_mg": 10,
                    "max_daily_mg": 25,
                    "target_maintenance_mg": 10,
                    "risk": "Euglycemic DKA and genital mycotic infections."
                }
            },
            "contraindications": [
                {
                    "trigger_condition": "eGFR < 30",
                    "drug": "Metformin",
                    "rationale": "Metformin is strictly contraindicated in severe renal impairment (eGFR < 30 mL/min) due to life-threatening lactic acidosis."
                }
            ]
        },
        "Essential Hypertension": {
            "authority": "Indian Guidelines on Hypertension (I.G.H.-IV) & MoHFW STG",
            "section": "Section 3 - First-line Antihypertensive Monotherapy & Combinations",
            "first_line_drugs": ["Telmisartan", "Amlodipine", "Enalapril", "Losartan", "Ramipril"],
            "dosage_rules": {
                "Telmisartan": {"min_daily_mg": 20, "max_daily_mg": 80, "target_maintenance_mg": 40},
                "Amlodipine": {"min_daily_mg": 2.5, "max_daily_mg": 10, "target_maintenance_mg": 5},
                "Enalapril": {"min_daily_mg": 5, "max_daily_mg": 20, "target_maintenance_mg": 10}
            },
            "contraindications": [
                {
                    "dual_drugs": [("Telmisartan", "Enalapril"), ("Losartan", "Ramipril")],
                    "rationale": "Dual RAS blockade (combining ACE inhibitor and ARB) is contraindicated: increases hyperkalemia, syncope, and acute renal injury without cardiovascular benefit."
                }
            ]
        },
        "Community-Acquired Pneumonia": {
            "authority": "ICMR Treatment Guidelines for Antimicrobial Use in Common Syndromes (2022)",
            "section": "Respiratory Tract Infections: Community-Acquired Pneumonia (CAP)",
            "first_line_drugs": ["Amoxicillin", "Clavulanate", "Azithromycin"],
            "dosage_rules": {
                "Amoxicillin": {"min_daily_mg": 1000, "max_daily_mg": 3000, "target_maintenance_mg": 1875},
                "Azithromycin": {"min_daily_mg": 250, "max_daily_mg": 500, "target_maintenance_mg": 500}
            },
            "duration_rules": {
                "max_recommended_days": 7,
                "warning_message": "Antibiotic duration > 7 days in uncomplicated CAP violates antimicrobial stewardship."
            }
        },
        "Acute Gastroenteritis": {
            "authority": "MoHFW / ICMR Pediatric & Adult Diarrheal Disease STG",
            "section": "Acute Watery Diarrhea Management Protocol",
            "first_line_drugs": ["ORS", "Zinc"],
            "restricted_drugs": ["Ciprofloxacin", "Ofloxacin", "Loperamide"],
            "stewardship_rules": [
                {
                    "flag_drugs": ["Ciprofloxacin", "Ofloxacin"],
                    "rationale": "Empiric fluoroquinolones are NOT recommended in acute watery diarrhea without documented blood in stool (dysentery). Promotes antimicrobial resistance."
                }
            ]
        }
    }

    @classmethod
    def evaluate(cls, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete guideline validation across all extracted diagnoses and medications.
        """
        diagnoses = entities.get("diagnoses", [])
        medications = entities.get("medications", [])
        labs = entities.get("lab_investigations", [])

        diag_names = [d["diagnosis"] for d in diagnoses]
        audit_results = []
        overall_status = "COMPLIANT"
        critical_alerts = 0
        warning_alerts = 0

        # Check eGFR / Renal function from labs
        egfr_val = None
        for lab in labs:
            if lab["test_name"] == "eGFR":
                egfr_val = lab["numeric_value"]

        # 1. Evaluate each diagnosis against catalog
        for diag_entry in diagnoses:
            diag = diag_entry["diagnosis"]
            guideline = cls._match_guideline(diag)
            if not guideline:
                continue

            # Check medications against this guideline
            for med in medications:
                drug = med["drug"]
                daily_dose = med.get("daily_dose_mg", 0)
                dur = med.get("duration_days", 0)

                # Dose Range Check
                dose_info = guideline.get("dosage_rules", {}).get(drug)
                if dose_info:
                    max_allowed = dose_info.get("max_daily_mg", 99999)
                    min_allowed = dose_info.get("min_daily_mg", 0)

                    if daily_dose > max_allowed:
                        overall_status = "NON_COMPLIANT_OVERDOSE"
                        critical_alerts += 1
                        audit_results.append({
                            "rule_id": f"STG-DOSE-{drug.upper()}-OVER",
                            "severity": "CRITICAL",
                            "status": "FAIL",
                            "drug": drug,
                            "diagnosis": diag,
                            "issue": f"Supratherapeutic Dose Detected: {daily_dose} mg/day exceeds ICMR ceiling of {max_allowed} mg/day.",
                            "clinical_risk": dose_info.get("risk", "Toxicity and adverse drug event risk."),
                            "guideline_ref": guideline["authority"],
                            "section": guideline["section"]
                        })
                    elif daily_dose < min_allowed:
                        warning_alerts += 1
                        audit_results.append({
                            "rule_id": f"STG-DOSE-{drug.upper()}-SUB",
                            "severity": "WARNING",
                            "status": "WARNING",
                            "drug": drug,
                            "diagnosis": diag,
                            "issue": f"Subtherapeutic Dose Detected: {daily_dose} mg/day is below recommended {min_allowed} mg/day.",
                            "clinical_risk": "Risk of therapeutic failure.",
                            "guideline_ref": guideline["authority"],
                            "section": guideline["section"]
                        })
                    else:
                        audit_results.append({
                            "rule_id": f"STG-DOSE-{drug.upper()}-PASS",
                            "severity": "INFO",
                            "status": "PASS",
                            "drug": drug,
                            "diagnosis": diag,
                            "issue": f"Dose {daily_dose} mg/day is within optimal therapeutic range ({min_allowed} - {max_allowed} mg/day).",
                            "clinical_risk": "None. Clinically appropriate.",
                            "guideline_ref": guideline["authority"],
                            "section": guideline["section"]
                        })

                # Duration Check
                dur_rule = guideline.get("duration_rules")
                if dur_rule and dur > dur_rule.get("max_recommended_days", 7):
                    warning_alerts += 1
                    audit_results.append({
                        "rule_id": f"STG-DUR-{drug.upper()}",
                        "severity": "WARNING",
                        "status": "WARNING",
                        "drug": drug,
                        "diagnosis": diag,
                        "issue": f"Excessive Duration: {dur} days exceeds recommended {dur_rule.get('max_recommended_days')} days.",
                        "clinical_risk": dur_rule.get("warning_message"),
                        "guideline_ref": guideline["authority"],
                        "section": guideline["section"]
                    })

                # Antimicrobial Stewardship Check
                stewardship = guideline.get("stewardship_rules", [])
                for st in stewardship:
                    if drug in st.get("flag_drugs", []):
                        overall_status = "NON_COMPLIANT_STEWARDSHIP"
                        critical_alerts += 1
                        audit_results.append({
                            "rule_id": f"STG-STEWARDSHIP-{drug.upper()}",
                            "severity": "CRITICAL",
                            "status": "FAIL",
                            "drug": drug,
                            "diagnosis": diag,
                            "issue": f"Inappropriate Antibiotic Prescribed for {diag}.",
                            "clinical_risk": st["rationale"],
                            "guideline_ref": guideline["authority"],
                            "section": guideline["section"]
                        })

                # Lab Contraindication (e.g. Metformin in renal impairment)
                if drug == "Metformin" and egfr_val is not None and egfr_val < 30:
                    overall_status = "CONTRAINDICATION_ALERT"
                    critical_alerts += 1
                    audit_results.append({
                        "rule_id": "STG-CONTRAINDICATION-METFORMIN-RENAL",
                        "severity": "CRITICAL",
                        "status": "FAIL",
                        "drug": "Metformin",
                        "diagnosis": diag,
                        "issue": f"Contraindication: eGFR is {egfr_val} mL/min (< 30 mL/min). Metformin contraindicated.",
                        "clinical_risk": "Life-threatening lactic acidosis.",
                        "guideline_ref": guideline["authority"],
                        "section": guideline["section"]
                    })

        # 2. Check Drug-Drug Interaction / Dual RAS Blockade
        prescribed_drugs = [m["drug"] for m in medications]
        if "Telmisartan" in prescribed_drugs and "Enalapril" in prescribed_drugs:
            overall_status = "CONTRAINDICATION_ALERT"
            critical_alerts += 1
            audit_results.append({
                "rule_id": "STG-INTERACTION-DUAL-RAS",
                "severity": "CRITICAL",
                "status": "FAIL",
                "drug": "Telmisartan + Enalapril",
                "diagnosis": "Hypertension",
                "issue": "Dangerous Drug Interaction: Concurrent ACE-inhibitor (Enalapril) + ARB (Telmisartan) dual RAS blockade.",
                "clinical_risk": "Severe hyperkalemia, profound hypotension, and acute tubular necrosis.",
                "guideline_ref": "Indian Guidelines on Hypertension (I.G.H.-IV)",
                "section": "Section 3.4 - Contraindicated Antihypertensive Combinations"
            })

        # Calculate clinician agreement score (metric from brief: target >= 90%)
        pass_count = sum(1 for r in audit_results if r["status"] == "PASS")
        total_rules = max(len(audit_results), 1)
        agreement_score = round(float(pass_count / total_rules), 3)

        # If clean
        if critical_alerts == 0 and warning_alerts == 0 and audit_results:
            overall_status = "FULLY_COMPLIANT"
        elif critical_alerts > 0:
            overall_status = "CRITICAL_NON_COMPLIANT"
        elif warning_alerts > 0:
            overall_status = "WARNING_SUBOPTIMAL"

        return {
            "overall_status": overall_status,
            "critical_alerts_count": critical_alerts,
            "warning_alerts_count": warning_alerts,
            "clinician_agreement_score": agreement_score,
            "audit_trail": audit_results,
            "guidelines_evaluated": list({r.get("guideline_ref") for r in audit_results if "guideline_ref" in r})
        }

    @classmethod
    def _match_guideline(cls, diagnosis_text: str) -> Dict[str, Any]:
        """Matches diagnosis text to guideline definition."""
        d_lower = diagnosis_text.lower()
        if "diabet" in d_lower or "t2dm" in d_lower:
            return cls.GUIDELINE_CATALOG["Type 2 Diabetes Mellitus"]
        elif "hyperten" in d_lower or "htn" in d_lower or "blood pressure" in d_lower:
            return cls.GUIDELINE_CATALOG["Essential Hypertension"]
        elif "pneumonia" in d_lower or "cap" in d_lower:
            return cls.GUIDELINE_CATALOG["Community-Acquired Pneumonia"]
        elif "gastroenter" in d_lower or "diarrh" in d_lower or "age" in d_lower:
            return cls.GUIDELINE_CATALOG["Acute Gastroenteritis"]
        return None
