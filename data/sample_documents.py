"""
Sample Document Generator (Distinct Dataset Layouts)
Generates 5 distinct document images with realistic visual, aspect ratio,
and structural characteristics matching:
1. Prescription (Case 1: T2DM Metformin Overdose) - Rx Pad format
2. Discharge Summary (Case 2: Hypertension Dual RAS) - Dense A4 Inpatient Summary
3. Prescription (Case 3: Pediatric Gastroenteritis) - Clinic Outpatient Pad format
4. Lab Report (Case 4: Diabetic Nephropathy Panel) - Tabular Diagnostic Form format
5. Medical Invoice (Case 5: Hospital Bill) - CORD-v2 / SROIE Narrow Receipt format
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np


def generate_prescription_image(title, lines, output_path, add_skew=2.2, add_blur=True, is_pediatric=False):
    """Generates authentic doctor's prescription pad (Aspect ~0.71, ~12 text lines)."""
    width, height = 620, 880
    bg_color = (250, 248, 242) if not is_pediatric else (248, 250, 252)
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Doctor / Clinic Header
    hdr_color = (30, 60, 120) if not is_pediatric else (180, 50, 50)
    draw.rectangle([20, 20, width - 20, 85], fill=(238, 242, 248), outline=hdr_color, width=2)
    draw.text((35, 28), title.upper(), fill=hdr_color)
    draw.text((35, 48), "CLINICAL OUTPATIENT DEPARTMENT - CONSULTATION SLIP", fill=(70, 80, 90))
    draw.text((35, 65), "Reg: MCI-48291/DEL | Contact: +91-11-2659-8000", fill=(100, 110, 120))

    # Divider
    draw.line([20, 95, width - 20, 95], fill=(160, 170, 180), width=1)

    # Big Rx Symbol
    draw.text((30, 110), "Rx", fill=(180, 20, 20))

    # Patient info box
    draw.rectangle([20, 140, width - 20, 175], fill=(245, 247, 250), outline=(210, 220, 230))

    curr_y = 150
    for line in lines:
        if line.startswith("Date:") or line.startswith("Patient:") or line.startswith("Consultant:"):
            draw.text((30, curr_y), line, fill=(40, 40, 40))
            curr_y += 24
        elif line.startswith("---"):
            draw.line([25, curr_y + 4, width - 25, curr_y + 4], fill=(200, 210, 220), width=1)
            curr_y += 18
        elif line.startswith("Rx:"):
            curr_y += 10
        elif line.startswith("Tab ") or line.startswith("Cap ") or line.startswith("Syp ") or line.startswith("Inj "):
            # Prominent medication line with box
            draw.rectangle([25, curr_y - 2, width - 25, curr_y + 24], fill=(255, 255, 255), outline=(200, 215, 230))
            draw.text((35, curr_y + 2), line, fill=(10, 20, 40))
            curr_y += 32
        else:
            draw.text((30, curr_y), line, fill=(50, 50, 50))
            curr_y += 24

    # Doctor signature block
    draw.rectangle([width - 240, height - 110, width - 30, height - 30], outline=(40, 60, 100), width=1)
    draw.text((width - 225, height - 100), "ATTENDING PHYSICIAN", fill=(80, 100, 120))
    draw.text((width - 225, height - 80), "Dr. Rajesh Sharma, MD", fill=(20, 30, 80))
    draw.text((width - 225, height - 60), "Signature & Seal: [SIGNED]", fill=(30, 120, 60))

    # Skew and blur
    if abs(add_skew) > 0.1:
        img = img.rotate(add_skew, resample=Image.Resampling.BICUBIC, fillcolor=(245, 245, 245))
    if add_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.6))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def generate_discharge_summary_image(title, lines, output_path, add_skew=-0.8):
    """Generates dense A4 hospital inpatient discharge summary (Aspect ~0.65, ~35 lines, high edge density)."""
    width, height = 750, 1150
    img = Image.new("RGB", (width, height), color=(254, 254, 252))
    draw = ImageDraw.Draw(img)

    # Formal Institutional Letterhead
    draw.rectangle([30, 25, width - 30, 100], fill=(240, 244, 252), outline=(30, 50, 100), width=2)
    draw.text((45, 35), title.upper(), fill=(15, 35, 80))
    draw.text((45, 58), "DEPARTMENT OF CARDIOLOGY & INPATIENT CRITICAL CARE", fill=(70, 80, 95))
    draw.text((45, 76), "ACCREDITED TERTIARY CARE HOSPITAL | ISO 9001:2015 CERTIFIED", fill=(100, 110, 120))

    draw.rectangle([30, 110, 320, 135], fill=(30, 50, 100))
    draw.text((40, 115), "FINAL CLINICAL DISCHARGE SUMMARY", fill=(255, 255, 255))

    curr_y = 150
    for line in lines:
        if line.startswith("---"):
            draw.line([30, curr_y + 3, width - 30, curr_y + 3], fill=(190, 200, 210), width=1)
            curr_y += 14
        elif line.startswith("DISCHARGE MEDICATIONS:") or line.startswith("INVESTIGATIONS AT DISCHARGE:"):
            draw.rectangle([30, curr_y - 2, width - 30, curr_y + 20], fill=(235, 240, 248))
            draw.text((40, curr_y + 1), line, fill=(20, 40, 90))
            curr_y += 24
        elif line.startswith("Tab ") or line.startswith("Cap "):
            draw.text((45, curr_y), line, fill=(10, 20, 30))
            curr_y += 22
        else:
            draw.text((35, curr_y), line, fill=(35, 35, 35))
            curr_y += 20

    # Add dense notes and hospital stamps to match RVL-CDIP letter profile
    draw.line([30, height - 120, width - 30, height - 120], fill=(200, 200, 200), width=1)
    draw.text((35, height - 110), "Prepared By: Medical Records Department | Copy Sent to Primary Care Physician", fill=(100, 100, 100))
    draw.text((35, height - 90), "Discharge Summary Verified by HOD Cardiology: Dr. Arvind Mehra, MD, DM", fill=(50, 50, 50))
    draw.rectangle([width - 200, height - 90, width - 40, height - 35], outline=(180, 30, 30), width=2)
    draw.text((width - 185, height - 70), "HOSPITAL SEAL", fill=(180, 30, 30))

    if abs(add_skew) > 0.1:
        img = img.rotate(add_skew, resample=Image.Resampling.BICUBIC, fillcolor=(245, 245, 245))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def generate_lab_report_image(title, lines, output_path):
    """Generates tabular clinical laboratory diagnostic report (Aspect ~0.76, bordered grid tables)."""
    width, height = 780, 1020
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Diagnostic Header
    draw.rectangle([25, 25, width - 25, 95], fill=(245, 248, 252), outline=(0, 80, 160), width=2)
    draw.text((40, 35), title.upper(), fill=(0, 60, 130))
    draw.text((40, 56), "CENTRAL DIAGNOSTIC & PATHOLOGY CLINICAL LABORATORY", fill=(60, 70, 80))
    draw.text((40, 74), "NABL ACCREDITED LAB NO: M-4819 | ISO 15189:2012 COMPLIANT", fill=(90, 100, 110))

    # Patient info bar
    draw.rectangle([25, 105, width - 25, 145], fill=(240, 242, 245), outline=(200, 210, 220))
    draw.text((35, 112), "Patient: Mohammed Al-Sayed | Age: 58 Y | Gender: Male | UHID: MGH-LAB-7721", fill=(30, 30, 30))
    draw.text((35, 128), "Sample: Fasting Serum | Collected: 11-10-2026 07:30 AM | Reported: 11-10-2026", fill=(70, 70, 70))

    # Tabular Table Header
    table_top = 160
    draw.rectangle([25, table_top, width - 25, table_top + 28], fill=(0, 80, 160))
    draw.text((35, table_top + 6), "TEST DESCRIPTION / PARAMETER", fill=(255, 255, 255))
    draw.text((320, table_top + 6), "OBSERVED VALUE", fill=(255, 255, 255))
    draw.text((460, table_top + 6), "REFERENCE RANGE", fill=(255, 255, 255))
    draw.text((640, table_top + 6), "FLAG / STATUS", fill=(255, 255, 255))

    curr_y = table_top + 28
    for line in lines:
        if ":" in line and not line.startswith("---") and not line.startswith("Rx") and not line.startswith("Diagnosis") and not line.startswith("Patient") and not line.startswith("DEPARTMENT"):
            parts = line.split(":")
            tname = parts[0].strip()
            rest = parts[1].strip()

            # Row background
            row_bg = (248, 250, 254) if (curr_y // 26) % 2 == 0 else (255, 255, 255)
            draw.rectangle([25, curr_y, width - 25, curr_y + 26], fill=row_bg, outline=(220, 225, 235))
            # Grid vertical lines
            draw.line([305, curr_y, 305, curr_y + 26], fill=(210, 220, 230), width=1)
            draw.line([445, curr_y, 445, curr_y + 26], fill=(210, 220, 230), width=1)
            draw.line([625, curr_y, 625, curr_y + 26], fill=(210, 220, 230), width=1)

            draw.text((35, curr_y + 5), tname, fill=(20, 30, 40))
            draw.text((320, curr_y + 5), rest.split("(")[0].strip(), fill=(10, 20, 30))
            if "(" in rest:
                ref_part = rest.split("(")[1].split(")")[0].replace("Reference Range:", "").strip()
                draw.text((460, curr_y + 5), ref_part, fill=(60, 70, 80))
            if "[" in rest:
                flag = rest.split("[")[1].split("]")[0].strip()
                f_color = (200, 20, 20) if "HIGH" in flag else (30, 120, 40)
                draw.text((640, curr_y + 5), flag, fill=f_color)
            curr_y += 26
        elif line.startswith("Rx Prescribed"):
            curr_y += 15
            draw.rectangle([25, curr_y, width - 25, curr_y + 22], fill=(235, 240, 245))
            draw.text((35, curr_y + 3), line, fill=(30, 60, 120))
            curr_y += 26
        elif line.startswith("Tab "):
            draw.text((35, curr_y), line, fill=(20, 30, 50))
            curr_y += 22
        elif line.startswith("Diagnosis"):
            curr_y += 10
            draw.text((35, curr_y), line, fill=(20, 20, 20))
            curr_y += 22

    # Laboratory Pathologist Stamp
    draw.rectangle([width - 250, height - 90, width - 35, height - 25], outline=(0, 80, 160), width=1)
    draw.text((width - 235, height - 80), "CONSULTANT BIOCHEMIST", fill=(0, 60, 120))
    draw.text((width - 235, height - 60), "Dr. S. K. Mukherjee, MD", fill=(30, 30, 30))
    draw.text((width - 235, height - 42), "Verified Electronic Report", fill=(30, 120, 40))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def generate_medical_invoice_image(title, lines, output_path, add_skew=0.6):
    """Generates narrow CORD-v2 / SROIE format medical receipt (Aspect ~0.48, dashed line items, grand total)."""
    width, height = 460, 940
    img = Image.new("RGB", (width, height), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)

    # Thermal Receipt Header
    draw.text((width // 2 - 130, 25), "METROPOLITAN HOSPITAL", fill=(20, 20, 20))
    draw.text((width // 2 - 110, 45), "INPATIENT BILLING & PHARMACY", fill=(60, 60, 60))
    draw.text((width // 2 - 95, 62), "TAX INVOICE / RECEIPT", fill=(80, 80, 80))
    draw.text((25, 82), "- " * 32, fill=(160, 160, 160))

    curr_y = 100
    for line in lines:
        if line.startswith("---"):
            draw.text((25, curr_y), "- " * 32, fill=(160, 160, 160))
            curr_y += 18
        elif line.startswith("Subtotal") or line.startswith("Total") or line.startswith("Healthcare Tax"):
            draw.rectangle([25, curr_y - 2, width - 25, curr_y + 22], fill=(240, 242, 245))
            draw.text((35, curr_y + 2), line, fill=(10, 10, 10))
            curr_y += 28
        else:
            draw.text((25, curr_y), line, fill=(30, 30, 30))
            curr_y += 20

    # Barcode representation at bottom
    draw.text((width // 2 - 80, height - 90), "* INV-2026-08832 *", fill=(100, 100, 100))
    for b in range(35, width - 35, 6):
        b_w = 2 if (b % 4 == 0) else 1
        draw.line([b, height - 70, b, height - 35], fill=(20, 20, 20), width=b_w)

    if abs(add_skew) > 0.1:
        img = img.rotate(add_skew, resample=Image.Resampling.BICUBIC, fillcolor=(245, 245, 245))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def generate_all_samples():
    """Builds the 5 distinct dataset samples."""
    data_dir = os.path.dirname(__file__)

    # Case 1: Prescription (Metformin Overdose)
    s1_lines = [
        "Date: 12-10-2026 | UHID: MGH-2026-94812",
        "Patient: Ramesh Gupta | Age: 54 Y | Gender: Male",
        "Consultant: Dr. Rajesh Sharma, MD (Internal Medicine)",
        "---",
        "Clinical Presentation: Polyuria, polydipsia, fatigue x 3m",
        "Diagnosis: Type 2 Diabetes Mellitus with Hypertension",
        "---",
        "Rx:",
        "Tab Metformin 1500 mg - 1-0-1 (BD) after food x 30 days",
        "Tab Telmisartan 40 mg - 1-0-0 (OD) morning x 30 days",
        "Tab Atorvastatin 20 mg - 0-0-1 (HS) night x 30 days",
        "---",
        "HbA1c: 8.4 % | Fasting Glucose: 162 mg/dL",
        "Serum Creatinine: 1.1 mg/dL | eGFR: 78 mL/min",
        "BP: 142/88 mmHg | Advice: Low carb diabetic diet"
    ]
    generate_prescription_image("Metropolitan Hospital - Clinic", s1_lines, os.path.join(data_dir, "sample_1_prescription_overdose.png"), add_skew=2.4, add_blur=True)

    # Case 2: Discharge Summary (Dual RAS Blockade)
    s2_lines = [
        "Admission Date: 05-10-2026 | Discharge Date: 11-10-2026",
        "Patient: Sunita Verma | Age: 62 Y | Gender: Female | UHID: MGH-CARD-5120",
        "Consultant: Dr. Arvind Mehra, MD, DM (Cardiology)",
        "---",
        "Final Diagnosis: Essential Hypertension with Grade II Diastolic Dysfunction",
        "Course in Hospital: Admitted with symptomatic hypertension BP 178/104 mmHg.",
        "Patient was stabilized on multi-drug therapy. Discharged hemodynamically stable.",
        "---",
        "DISCHARGE MEDICATIONS:",
        "Tab Telmisartan 40 mg - 1-0-0 (OD) morning x 30 days",
        "Tab Enalapril 10 mg - 1-0-1 (BD) twice daily x 30 days",
        "Tab Amlodipine 5 mg - 0-0-1 (HS) night x 30 days",
        "---",
        "INVESTIGATIONS AT DISCHARGE:",
        "Serum Creatinine: 1.3 mg/dL | Blood Pressure: 130/82 mmHg",
        "Serum Potassium: 4.8 mEq/L (Normal) | ECG: Sinus rhythm",
        "---",
        "Dietary Advice: Low salt (< 5g/day), DASH diet restriction.",
        "Follow-up: Visit Cardiology OPD after 2 weeks for Serum Electrolytes."
    ]
    generate_discharge_summary_image("National Heart Institute", s2_lines, os.path.join(data_dir, "sample_2_discharge_dual_blockade.png"), add_skew=-0.8)

    # Case 3: Pediatric Prescription (Antibiotic Flag)
    s3_lines = [
        "Date: 11-10-2026 | UHID: MGH-PED-8819 | Token: 14",
        "Patient: Aarav Patel | Age: 4 Y | Gender: Male | Weight: 15 kg",
        "Consultant: Dr. Priya Nair, MD (Pediatrics)",
        "---",
        "Complaints: Loose watery stools x 2 days, mild fever.",
        "Diagnosis: Acute Gastroenteritis with Mild Dehydration",
        "---",
        "Rx:",
        "Syp Ciprofloxacin 250 mg - 1-0-1 (BD) x 5 days",
        "Syp Zinc 20 mg - 1-0-0 (OD) x 14 days",
        "Syp ORS - 100 ml after each loose stool",
        "Tab Paracetamol 250 mg - SOS for temperature > 100 F",
        "---",
        "Stool Examination: No RBCs seen, No cysts/trophozoites.",
        "Advice: Frequent home fluids, review if persistent vomiting."
    ]
    generate_prescription_image("Apex Children Specialty Clinic", s3_lines, os.path.join(data_dir, "sample_3_pediatric_antibiotic_flag.png"), add_skew=2.6, add_blur=False, is_pediatric=True)

    # Case 4: Clinical Lab Report (Compliant)
    s4_lines = [
        "DEPARTMENT OF CLINICAL BIOCHEMISTRY",
        "HbA1c : 8.8 % (Reference Range: 4.0 - 5.6 %) [HIGH]",
        "Fasting Blood Glucose : 184 mg/dL (Reference Range: 70 - 99 mg/dL) [HIGH]",
        "Postprandial Blood Glucose : 242 mg/dL (Reference Range: < 140 mg/dL) [HIGH]",
        "Serum Creatinine : 1.0 mg/dL (Reference Range: 0.7 - 1.3 mg/dL) [NORMAL]",
        "eGFR : 85 mL/min (Reference Range: > 90 mL/min) [MILD_REDUCTION]",
        "Total Cholesterol : 195 mg/dL (Reference Range: < 200 mg/dL) [NORMAL]",
        "Diagnosis: Type 2 Diabetes Mellitus (Uncontrolled Glycemia)",
        "---",
        "Rx Prescribed at Review:",
        "Tab Metformin 1000 mg - 1-0-1 (BD) x 30 days",
        "Tab Glimepiride 1 mg - 1-0-0 (OD) x 30 days",
        "Tab Atorvastatin 10 mg - 0-0-1 (HS) x 30 days"
    ]
    generate_lab_report_image("Central Diagnostics Pathology Lab", s4_lines, os.path.join(data_dir, "sample_4_lab_report_compliant.png"))

    # Case 5: Medical Invoice (CORD-v2 Receipt Format)
    s5_lines = [
        "Invoice No: INV-2026-08832 | Date: 11-10-2026",
        "Patient: Sunita Verma | UHID: MGH-CARD-5120",
        "---",
        "SERVICES & CHARGES:",
        "1. Consultant Rounds (5 Days) ...... $ 150.00",
        "2. Nursing Monitoring .............. $ 100.00",
        "3. Complete Blood Count & Renal .... $  80.00",
        "4. Electrocardiogram (ECG) ......... $  40.00",
        "5. Pharmacy Dispensing ............. $  90.00",
        "---",
        "Subtotal : $ 460.00",
        "Healthcare Tax (5%) : $ 20.00",
        "Total Amount Due : $ 480.00",
        "Status: Paid via Third-Party Insurance TPA"
    ]
    generate_medical_invoice_image("Metropolitan Hospital Billing", s5_lines, os.path.join(data_dir, "sample_5_medical_invoice.png"), add_skew=0.8)

    print("Successfully generated all 5 distinct dataset layouts!")


if __name__ == "__main__":
    generate_all_samples()
