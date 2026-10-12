import os
from PIL import Image, ImageDraw

real_dir = os.path.join(os.path.dirname(__file__), "data", "real")
os.makedirs(real_dir, exist_ok=True)

# 1. IAM Handwriting Prescription Pad
iam_line_path = os.path.join(real_dir, "real_iam_handwriting_line.png")
if os.path.exists(iam_line_path):
    iam_img = Image.open(iam_line_path)
    canvas = Image.new("RGB", (800, 1100), color=(252, 250, 246))
    draw = ImageDraw.Draw(canvas)
    
    # Header
    draw.rectangle([0, 0, 800, 110], fill=(238, 242, 246))
    draw.text((40, 25), "ST. JUDE CLINICAL & AMBULATORY CARE CENTER", fill=(20, 35, 60))
    draw.text((40, 50), "Department of General Medicine | Reg: IAM-CLINIC-77", fill=(80, 95, 115))
    draw.text((40, 75), "Patient: Arthur Pendelton | Age: 42 Y | Date: 12-10-2026", fill=(60, 70, 85))
    draw.line([(30, 115), (770, 115)], fill=(180, 190, 205), width=2)
    
    # Section Label
    draw.text((40, 135), "DOCTOR'S CLINICAL HANDWRITTEN IMPRESSION (IAM BENCHMARK):", fill=(40, 50, 70))
    
    # Paste the real IAM handwriting line
    w_line = 720
    h_line = int(iam_img.height * (720.0 / iam_img.width))
    iam_resized = iam_img.resize((w_line, h_line), Image.Resampling.LANCZOS)
    canvas.paste(iam_resized, (40, 170))
    
    # Doctor's Rx below
    draw.line([(40, 190 + h_line), (760, 190 + h_line)], fill=(210, 215, 225), width=1)
    draw.text((40, 215 + h_line), "PRESCRIPTION & DISCHARGE INSTRUCTIONS (Rx):", fill=(25, 35, 60))
    draw.text((50, 250 + h_line), "1. Tab Amoxicillin-Clavulanate 625 mg - 1 tab BD after meals x 7 days", fill=(30, 40, 50))
    draw.text((50, 285 + h_line), "2. Tab Paracetamol 650 mg - 1 tab SOS for body ache / fever > 100 F", fill=(30, 40, 50))
    draw.text((50, 320 + h_line), "3. Syp Dextromethorphan 10 mg / 5ml - 2 tsp TDS x 5 days", fill=(30, 40, 50))
    draw.text((50, 355 + h_line), "4. Steam inhalation BD & Warm water gargles", fill=(30, 40, 50))
    
    draw.text((40, 430 + h_line), "INVESTIGATIONS & FOLLOW-UP:", fill=(25, 35, 60))
    draw.text((50, 460 + h_line), "Complete Blood Count (CBC), ESR, Chest X-Ray PA View", fill=(60, 70, 80))
    draw.text((50, 490 + h_line), "Review after 7 days or SOS if breathing difficulty worsens.", fill=(60, 70, 80))
    
    # Stamp / Signature
    draw.rectangle([480, 880, 740, 980], outline=(70, 90, 140), width=2)
    draw.text((500, 895), "VERIFIED & SIGNED", fill=(70, 90, 140))
    draw.text((500, 920), "Dr. J. Henderson, MD", fill=(40, 60, 100))
    draw.text((500, 945), "Reg: GMC-928410", fill=(70, 90, 140))
    
    canvas.save(os.path.join(real_dir, "real_iam_handwriting_pad.png"))
    print("Created real_iam_handwriting_pad.png")

# 2. MTSamples Clinical Diagnostic & Lab Report
canvas2 = Image.new("RGB", (800, 1100), color=(255, 255, 255))
draw2 = ImageDraw.Draw(canvas2)
draw2.rectangle([0, 0, 800, 120], fill=(240, 244, 248))
draw2.text((40, 25), "METROPOLITAN NEURO-DIAGNOSTIC & PATHOLOGY LAB", fill=(15, 30, 65))
draw2.text((40, 52), "MTSamples Clinical Corpus Validation | Accreditation: NABL / CAP-2026", fill=(70, 85, 105))
draw2.text((40, 80), "Patient: Eleanor Vance | Age: 56 Y | Gender: Female | UHID: MTS-2026-4419", fill=(50, 65, 80))
draw2.line([(30, 122), (770, 122)], fill=(200, 210, 225), width=2)

draw2.text((40, 145), "TRANSCRIPTION IMPRESSION (MTSAMPLES CORPUS):", fill=(20, 35, 60))
draw2.text((50, 175), "\"Patient reports neck and back pain, balance impairment, cough, and limb weakness.\"", fill=(60, 70, 85))
draw2.text((50, 205), "Clinical Diagnosis: Cervical Spondylosis with Radiculopathy", fill=(160, 20, 20))

draw2.text((40, 250), "DIAGNOSTIC LABORATORY & IMAGING FINDINGS:", fill=(20, 35, 60))
draw2.rectangle([40, 280, 760, 480], outline=(210, 220, 230), width=1)
draw2.line([(40, 315), (760, 315)], fill=(210, 220, 230), width=1)
draw2.text((55, 292), "TEST PARAMETER", fill=(80, 90, 100))
draw2.text((280, 292), "OBSERVED RESULT", fill=(80, 90, 100))
draw2.text((450, 292), "REFERENCE RANGE", fill=(80, 90, 100))
draw2.text((640, 292), "STATUS", fill=(80, 90, 100))

items = [
    ("Serum 25-OH Vitamin D", "14.2 ng/mL", "30.0 - 100.0 ng/mL", "DEFICIENT [FLAG]"),
    ("Serum Calcium (Total)", "8.8 mg/dL", "8.5 - 10.2 mg/dL", "NORMAL"),
    ("Erythrocyte Sedimentation Rate", "34 mm/hr", "0 - 20 mm/hr", "ELEVATED [FLAG]"),
    ("HbA1c Glycated Hemoglobin", "5.5 %", "4.0 - 5.6 %", "NORMAL"),
    ("C-Reactive Protein (CRP)", "8.4 mg/L", "< 5.0 mg/L", "ELEVATED [FLAG]")
]
y_t = 330
for test, res, ref, flg in items:
    draw2.text((55, y_t), test, fill=(30, 40, 50))
    draw2.text((280, y_t), res, fill=(20, 30, 40))
    draw2.text((450, y_t), ref, fill=(70, 80, 90))
    draw2.text((640, y_t), flg, fill=(180, 30, 30) if "FLAG" in flg else (30, 130, 60))
    y_t += 28

draw2.text((40, 520), "RECOMMENDED TREATMENT REGIMEN & ICMR STG AUDIT:", fill=(20, 35, 60))
draw2.text((50, 550), "1. Tab Pregabalin 75 mg - 0-0-1 (HS) night x 14 days", fill=(30, 40, 50))
draw2.text((50, 580), "2. Tab Paracetamol 650 mg - 1-0-1 (BD) x 5 days", fill=(30, 40, 50))
draw2.text((50, 610), "3. Syp Cholecalciferol (Vit D3) 60,000 IU - Once weekly x 8 weeks", fill=(30, 40, 50))
draw2.text((50, 640), "4. Cervical collar immobilization during travel & physical therapy referral", fill=(30, 40, 50))

draw2.rectangle([480, 880, 740, 980], outline=(60, 120, 70), width=2)
draw2.text((500, 895), "LAB AUTHORIZED REPORT", fill=(60, 120, 70))
draw2.text((500, 920), "Dr. M. K. Sen, MD, FRCPath", fill=(30, 50, 30))
draw2.text((500, 945), "Reg: MCI-PATH-51049", fill=(60, 120, 70))

canvas2.save(os.path.join(real_dir, "real_mtsamples_report.png"))
print("Created real_mtsamples_report.png")
