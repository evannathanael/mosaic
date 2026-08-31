"""
Pulls a small sample of WildFake images for validation, without downloading
the full dataset. Requires visiting the ModelScope dataset page and clicking
"translate" once in-browser first, per the challenge instructions:
https://modelscope.cn/datasets/hy2628982280/WildFake/summary

Usage:
    python3 -m src.data.pull_wildfake_sample
"""
from pathlib import Path

from modelscope.msdatasets import MsDataset

out_dir = Path("data/raw/wildfake_sample")
out_dir.mkdir(parents=True, exist_ok=True)

ds = MsDataset.load("hy2628982280/WildFake", split="train", use_streaming=True)

count = 0
for example in ds:
    if count == 0:
        print("First example structure:", example)
        print("Keys:", list(example.keys()))
    img = example["image"]
    img.save(out_dir / f"img_{count:03d}.jpg")
    count += 1
    if count >= 100:
        break

print(f"Saved {count} images to {out_dir}")