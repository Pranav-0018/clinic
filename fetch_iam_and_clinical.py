"""
Fetch IAM-line and MTSamples / FUNSD datasets
"""

import os
from datasets import load_dataset
from PIL import Image

output_dir = os.path.join(os.path.dirname(__file__), "data", "real")


def fetch_iam_and_clinical():
    print("Connecting to Teklia/IAM-line...")
    try:
        ds_iam = load_dataset("Teklia/IAM-line", split="train", streaming=True)
        sample_iam = next(iter(ds_iam))
        img_iam = sample_iam["image"]
        img_path = os.path.join(output_dir, "real_iam_handwriting_line.png")
        img_iam.save(img_path)
        print(f" [SUCCESS] Teklia/IAM-line sample fetched: size={img_iam.size}, saved to {img_path}")
    except Exception as e:
        print(f" [NOTICE] IAM-line notice: {e}")

    print("Connecting to MTSamples or Medical Transcriptions...")
    try:
        # MTSamples is available on Hugging Face as 'tatsu-lab/alpaca' or 'lavita/medical-transcriptions' or 'medical_dialog'
        ds_med = load_dataset("lavita/medical-transcriptions", split="train", streaming=True)
        sample_med = next(iter(ds_med))
        print(f" [SUCCESS] lavita/medical-transcriptions sample fetched: {list(sample_med.keys())}")
        with open(os.path.join(output_dir, "real_mtsamples_transcript.txt"), "w", encoding="utf-8") as f:
            f.write(str(sample_med))
    except Exception as e:
        print(f" [NOTICE] Trying alternate clinical text corpus: {e}")
        try:
            ds_med = load_dataset("gretelai/symptom_to_diagnosis", split="train", streaming=True)
            sample_med = next(iter(ds_med))
            print(f" [SUCCESS] gretelai/symptom_to_diagnosis sample fetched: {sample_med}")
            with open(os.path.join(output_dir, "real_mtsamples_transcript.txt"), "w", encoding="utf-8") as f:
                f.write(str(sample_med))
        except Exception as e2:
            print(f" [NOTICE] Alternate clinical text: {e2}")


if __name__ == "__main__":
    fetch_iam_and_clinical()
