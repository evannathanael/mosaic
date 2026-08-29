# Mosaic

A lightweight AI-generated image (AIGC) detector with a repetition/near-duplicate
signal on top — built for TikTok TechJam's AIGC Detection track.

**Core idea:** most projects ask *"is this image AI-generated?"*. Mosaic
also asks *"is AI-generated content being reposted/repeated at a volume that
crowds out other creators?"* — so it can flag repetitive synthetic content for
feed-balancing without penalizing one-off, legitimate AI creativity.

Both signals share **one lightweight backbone** (CLIP ViT-B by default), keeping
the whole pipeline well under the 2B parameter limit.

```
Image → [Shared Backbone] ──→ Classifier head ──→ AI probability + confidence
                          └──→ Embedding ────────→ Cosine similarity → cluster + repetition score
```

## Project structure

```
mosaic/
├── configs/config.yaml        # all tunable settings in one place
├── data/                      # datasets (gitignored except structure)
│   ├── raw/                   # downloaded datasets go here
│   ├── processed/             # cleaned/split data
│   └── near_duplicates/       # generated near-duplicate test variants
├── src/
│   ├── data/
│   │   ├── download.py        # dataset download helpers
│   │   ├── transforms.py      # the 6 required robustness transforms
│   │   ├── dataset.py         # PyTorch Dataset + train/val/test split logic
│   │   └── near_duplicate_gen.py  # builds near-duplicate variant test set
│   ├── models/
│   │   ├── backbone.py        # shared CLIP backbone loader (frozen or fine-tune)
│   │   ├── classifier.py      # classifier head on top of the backbone
│   │   └── train.py           # main training loop (Person A)
│   ├── similarity/
│   │   ├── embeddings.py      # extract embeddings from the shared backbone
│   │   └── clustering.py      # cosine similarity + near-duplicate clustering
│   ├── eval/
│   │   ├── robustness.py      # clean vs. transformed vs. unseen-generator AUC table
│   │   ├── calibration.py     # temperature scaling for calibrated confidence
│   │   ├── error_analysis.py  # false positive / false negative report
│   │   └── shortcut_check.py  # sanity check: is the model learning real signal?
│   ├── inference.py           # end-to-end script: image folder -> required JSON output
│   └── utils.py                # shared helpers (seeding, config loading, logging)
├── app/
│   └── dashboard.py            # Streamlit demo app
├── scripts/                    # thin shell wrappers around the src/ entry points
├── tests/
│   └── test_smoke.py           # quick sanity tests
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

1. **Download & prepare data** (Person 1 — data)
   ```bash
   python src/data/download.py --dataset all --out data/raw
   python src/data/near_duplicate_gen.py --input data/raw/ai --out data/near_duplicates
   ```

2. **Train the classifier** (Person 2 — training)
   ```bash
   python src/models/train.py --config configs/config.yaml
   python src/models/train.py --config configs/config.yaml --experiment rotation_jitter
   ```

3. **Evaluate robustness** (Person 4 — evaluation)
   ```bash
   python src/eval/robustness.py --checkpoint outputs/model_best.pt --out outputs/robustness_table.csv
   python src/eval/calibration.py --checkpoint outputs/model_best.pt
   python src/eval/error_analysis.py --checkpoint outputs/model_best.pt --out outputs/error_analysis.md
   python src/eval/shortcut_check.py --checkpoint outputs/model_best.pt
   ```

4. **Run full inference (required deliverable format)**
   ```bash
   python src/inference.py --input_dir path/to/images --checkpoint outputs/model_best.pt --out outputs/predictions.json
   ```
   Produces JSON in the required format:
   ```json
   {"image_path": "image_01.jpg", "pred": 0.91, "similarity_cluster": 4, "repetition_score": 0.94}
   ```

5. **Launch the demo app** (Person 5 — product/demo)
   ```bash
   streamlit run app/dashboard.py
   ```

## Reproducing results

All experiment settings (backbone choice, augmentation strength, learning rate,
etc.) live in `configs/config.yaml` — change settings there rather than editing
code, so every run is reproducible from a single config file. Each training run
writes its config + metrics to `outputs/<run_name>/`.

## Team ownership (see repo structure above for exact files)

| Area | Owner | Folder |
|---|---|---|
| Data & robustness transforms | Vicky | `src/data/` |
| Core model training | Glory | `src/models/` |
| Similarity & clustering | Chelsea | `src/similarity/` |
| Evaluation & calibration | Evan | `src/eval/` |
| Demo app | Eron | `app/dashboard.py` |

## Limitations & future work

- Parameter budget: shared CLIP ViT-B backbone (~150M params) + small classifier
  head — well under the 2B limit even combined with the similarity component,
  which reuses the same backbone at no extra parameter cost.
- Scope: image-level binary detection only, per the challenge — no video/audio,
  no production deployment, no pixel-level localization.
- See `outputs/error_analysis.md` (generated after evaluation) for a discussion
  of known failure modes and what we'd improve with more time.
