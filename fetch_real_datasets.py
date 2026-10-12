"""
Fetch and integrate real benchmark datasets:
1. CORD-v2 (naver-clova-ix/cord-v2)
2. SROIE (jsdnrs/ICDAR2019-SROIE)
3. RVL-CDIP (chainyo/rvl-cdip or aharley/rvl_cdip - streamed)
4. IAM Handwriting (Teklia/IAM-line)
5. MTSamples (Medical transcribed reports)
"""

import os
import json
from datasets import load_dataset
from PIL import Image

output_dir = os.path.join(os.path.dirname(__file__), "data", "real")
os.makedirs(output_dir, exist_ok=True)


def test_cord_v2():
    print("Connecting to CORD-v2 (naver-clova-ix/cord-v2)...")
    try:
        ds = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)
        sample = next(iter(ds))
        img = sample["image"]
        ground_truth = sample.get("ground_truth", "")
        img_path = os.path.join(output_dir, "real_cord_v2_receipt.png")
        img.save(img_path)
        print(f" [SUCCESS] CORD-v2 sample fetched: size={img.size}, saved to {img_path}")
        return sample
    except Exception as e:
        print(f" [ERROR] CORD-v2 fetch error: {e}")
        return None


def test_sroie():
    print("Connecting to SROIE (jsdnrs/ICDAR2019-SROIE)...")
    try:
        ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split="train", streaming=True)
        sample = next(iter(ds))
        img = sample["image"]
        img_path = os.path.join(output_dir, "real_sroie_receipt.png")
        img.save(img_path)
        print(f" [SUCCESS] SROIE sample fetched: size={img.size}, saved to {img_path}")
        return sample
    except Exception as e:
        print(f" [ERROR] SROIE fetch error: {e}")
        return None


def test_rvl_cdip():
    print("Connecting to RVL-CDIP (chainyo/rvl-cdip)...")
    try:
        ds = load_dataset("chainyo/rvl-cdip", split="train", streaming=True)
        sample = next(iter(ds))
        img = sample["image"]
        label = sample.get("label", "")
        img_path = os.path.join(output_dir, "real_rvl_cdip_sample.png")
        img.save(img_path)
        print(f" [SUCCESS] RVL-CDIP sample fetched: label={label}, size={img.size}, saved to {img_path}")
        return sample
    except Exception as e:
        print(f" [NOTICE] Trying alternate RVL-CDIP (aharley/rvl_cdip)...")
        try:
            ds = load_dataset("aharley/rvl_cdip", split="train", streaming=True)
            sample = next(iter(ds))
            img = sample["image"]
            img_path = os.path.join(output_dir, "real_rvl_cdip_sample.png")
            img.save(img_path)
            print(f" [SUCCESS] aharley/rvl_cdip sample fetched: size={img.size}")
            return sample
        except Exception as e2:
            print(f" [ERROR] RVL-CDIP alternate error: {e2}")
            return None


if __name__ == "__main__":
    c = test_cord_v2()
    s = test_sroie()
    r = test_rvl_cdip()
