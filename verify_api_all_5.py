import urllib.request
import json

base_url = "http://127.0.0.1:8000"

req = urllib.request.Request(f"{base_url}/api/samples")
with urllib.request.urlopen(req) as resp:
    samples = json.loads(resp.read().decode("utf-8"))

print("=== SAMPLES LIST FROM API ===")
for s in samples:
    print(f"Case #{s['id']}: {s['title']}")
    print(f"  Source: {s['dataset_source']}")
    print(f"  File: {s['file']}")
    print(f"  Thumbnail length: {len(s.get('thumbnail_base64', ''))} chars\n")

print("=== TESTING 5 ANALYSES VIA /api/analyze ===")
for s in samples:
    sid = s['id']
    payload = json.dumps({"sample_id": sid, "model_name": "Logistic Regression"}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/analyze",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    cls_name = data["classification"]["predicted_class"]
    conf = data["classification"]["confidence"]
    stg_status = data["stg_compliance"]["overall_status"]
    img_b64 = data.get("original_image_base64", "")
    preview = data["ocr"]["full_text"][:75].replace("\n", " ")
    
    print(f"Case #{sid} [{s['dataset_source']}]:")
    print(f"  Class Prediction: {cls_name} (Confidence: {conf*100:.1f}%)")
    print(f"  STG Status      : {stg_status}")
    print(f"  Image Data B64  : {img_b64[:35]}... (Length: {len(img_b64)})")
    print(f"  OCR Transcript  : {preview}...\n")
