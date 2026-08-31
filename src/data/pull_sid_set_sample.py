from datasets import load_dataset
from pathlib import Path

out_dir = Path("data/raw/sid_set_sample")
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=500)   # NEW — reduces the risk of grabbing only one source/generator's images in a row

count = 0
for example in ds:
    img = example["image"]
    img.save(out_dir / f"img_{count:03d}.jpg")
    count += 1
    if count >= 300:   # CHANGED from 40
        break

print(f"Saved {count} images to {out_dir}")