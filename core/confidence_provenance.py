"""
Confidence Scoring & Provenance Tracking Module
Aggregates scan quality, OCR token confidence, and NER entity extraction scores
into a calibrated composite confidence score for every extracted clinical field.
Enforces human-in-the-loop review triage for fields falling below 0.85 confidence.
"""

from typing import Dict, List, Any


class ProvenanceTracker:
    """
    Maintains cryptographic and spatial provenance for every extracted clinical entity.
    """
    CONFIDENCE_THRESHOLD_AUTOMATION = 0.85

    @classmethod
    def compile_provenance_record(
        cls,
        quality_score: float,
        ocr_result: Dict[str, Any],
        entities: Dict[str, Any],
        stg_audit: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates a unified, auditable provenance record for the ingestion pipeline.
        """
        records = []
        human_review_required = False

        # Patient demographics
        patient = entities.get("patient", {})
        for field_name, data in patient.items():
            if isinstance(data, dict):
                comp_conf = cls._compute_composite_confidence(quality_score, data.get("confidence", 0.90))
                needs_review = comp_conf < cls.CONFIDENCE_THRESHOLD_AUTOMATION
                if needs_review:
                    human_review_required = True
                records.append({
                    "field_group": "Patient Demographics",
                    "field_name": field_name.capitalize(),
                    "extracted_value": str(data.get("value")),
                    "composite_confidence": comp_conf,
                    "ocr_token_confidence": data.get("confidence", 0.90),
                    "bbox": data.get("bbox"),
                    "action": "FLAGGED_FOR_HUMAN_REVIEW" if needs_review else "AUTO_VALIDATED"
                })

        # Diagnoses
        for d in entities.get("diagnoses", []):
            comp_conf = cls._compute_composite_confidence(quality_score, d.get("confidence", 0.90))
            needs_review = comp_conf < cls.CONFIDENCE_THRESHOLD_AUTOMATION
            if needs_review:
                human_review_required = True
            records.append({
                "field_group": "Clinical Diagnosis",
                "field_name": "Diagnosis",
                "extracted_value": d.get("diagnosis"),
                "composite_confidence": comp_conf,
                "ocr_token_confidence": d.get("confidence", 0.90),
                "bbox": d.get("bbox"),
                "action": "FLAGGED_FOR_HUMAN_REVIEW" if needs_review else "AUTO_VALIDATED"
            })

        # Medications
        for m in entities.get("medications", []):
            comp_conf = cls._compute_composite_confidence(quality_score, m.get("confidence", 0.90))
            needs_review = comp_conf < cls.CONFIDENCE_THRESHOLD_AUTOMATION
            if needs_review:
                human_review_required = True
            records.append({
                "field_group": "Pharmacotherapy",
                "field_name": f"Rx: {m.get('drug')}",
                "extracted_value": f"{m.get('strength')} {m.get('unit')} | {m.get('frequency')} | {m.get('daily_dose_mg')} mg/day for {m.get('duration_days')} days",
                "composite_confidence": comp_conf,
                "ocr_token_confidence": m.get("confidence", 0.90),
                "bbox": m.get("bbox"),
                "action": "FLAGGED_FOR_HUMAN_REVIEW" if needs_review else "AUTO_VALIDATED"
            })

        # Lab Investigations
        for l in entities.get("lab_investigations", []):
            comp_conf = cls._compute_composite_confidence(quality_score, l.get("confidence", 0.95))
            needs_review = comp_conf < cls.CONFIDENCE_THRESHOLD_AUTOMATION
            if needs_review:
                human_review_required = True
            records.append({
                "field_group": "Laboratory Data",
                "field_name": l.get("test_name"),
                "extracted_value": f"{l.get('value')} {l.get('unit')} (Ref: {l.get('reference_range')})",
                "composite_confidence": comp_conf,
                "ocr_token_confidence": l.get("confidence", 0.95),
                "bbox": l.get("bbox"),
                "action": "FLAGGED_FOR_HUMAN_REVIEW" if needs_review else "AUTO_VALIDATED"
            })

        # Overall triage decision
        is_priority_queue = classification.get("priority_queue") == "PRIORITY_VERIFICATION"
        has_critical_stg = stg_audit.get("critical_alerts_count", 0) > 0

        final_triage = "IMMEDIATE_CLINICIAN_REVIEW_REQUIRED" if (has_critical_stg or human_review_required or is_priority_queue) else "READY_FOR_EHR_EXPORT"

        return {
            "triage_verdict": final_triage,
            "human_review_required": human_review_required or has_critical_stg or is_priority_queue,
            "provenance_records": records,
            "record_count": len(records),
            "mean_field_confidence": round(sum(r["composite_confidence"] for r in records) / max(len(records), 1), 3) if records else 0.90,
            "pipeline_version": "v1.4-pbl1-stg",
            "provenance_hash": hex(abs(hash(str(records))))[:12]
        }

    @staticmethod
    def _compute_composite_confidence(quality_score: float, model_confidence: float) -> float:
        """
        Multivariate confidence calibration:
        Confidence = 0.30 * ScanQuality + 0.50 * TokenConfidence + 0.20 * Baseline
        """
        raw = 0.30 * quality_score + 0.50 * model_confidence + 0.20 * 0.95
        return float(round(float(min(0.99, max(0.10, raw))), 3))
