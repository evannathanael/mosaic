# Mosaic

Mosaic is an AI-generated image detection and feed-diversity system built for TikTok TechJam's AIGC Detection track. Most detectors ask only whether an image is synthetic. Mosaic also asks whether substantially the same synthetic image is being reposted often enough to crowd out other creators.

The system combines two signals:

1. **AI-image probability** — a frozen CLIP ViT-L/14 image encoder and a trained MLP classification head estimate whether an image is AI-generated.
2. **Repetition** — normalized CLIP embeddings and cosine-similarity clustering identify near-duplicate images and calculate a repetition score.

```text
Image -> shared CLIP encoder -> AI classifier -> AI probability
                           \-> embedding -----> cluster + repetition score
```

The React frontend presents these signals in a vertical feed. It avoids placing near-duplicates back-to-back, shows why repeated synthetic posts were grouped, and lets a user upload an image for live analysis. The FastAPI backend serves the demo dataset, runs the detector, maintains similarity groups, and can optionally persist uploads to Supabase.

## Project structure

```text
backend/                 FastAPI API, detector integration, uploads, persistence
configs/config.yaml      Reproducible model, training, and clustering settings
data/                    Raw/processed dataset metadata and committed demo images
frontend/                React, TypeScript, and Vite user interface
scripts/                 Training, evaluation, inference, and demo-data helpers
src/data/                Dataset download, cleaning, splitting, and transforms
src/models/              CLIP backbone, classifier heads, and training code
src/similarity/          Embedding, clustering, and threshold evaluation
src/eval/                Robustness, calibration, error, and shortcut checks
tests/                   Data, detector, similarity, and API tests
combined_head.pt         Detector head used by the local FastAPI demo
```

## Setup and installation

### Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Git

A GPU is optional. The application falls back to CPU, although model loading and inference will be slower.

### 1. Clone and create a Python environment

```bash
git clone https://github.com/evannathanael/mosaic.git
cd mosaic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

Install the root requirements as well if you intend to train or run the full offline evaluation pipeline:

```bash
python -m pip install -r requirements.txt
```

### 2. Install the frontend

```bash
cd frontend
npm install
cd ..
```

### 3. Start the backend

From the repository root, with the virtual environment active:

```bash
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

Check that the trained model loaded:

```bash
curl http://127.0.0.1:8000/api/health
```

A working detector reports `"model_ready": true` and `"analysis_mode": "model"`. If it reports `false` and `"mock"`, the API still runs but uploaded images use a filename heuristic rather than the trained detector. The API documentation is available at <http://127.0.0.1:8000/docs>.

### 4. Start the frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open <http://127.0.0.1:5173>. The frontend is currently configured to call the backend at `http://localhost:8000`. Press `Shift+R` in the app to reset the demo feed.

Supabase is not required. Without Supabase environment variables, uploads and metadata remain local. See [backend/README.md](backend/README.md) for optional Supabase setup.

## Reproducing the results

All shared settings and the random seed are stored in [`configs/config.yaml`](configs/config.yaml). The configured seed is `42`, the data split is 80% training / 10% validation / 10% test, and the default similarity threshold is `0.90`.

### Reproduce the committed detector experiment

The current CLIP-head experiment is implemented in `src/models/train_clip_head.py`. Select `cifake`, `sid_set`, `wildfake`, or `combined` using its documented `DATASET` setting. Dataset-specific preparation instructions are in [data/README.md](data/README.md); CIFAKE can be downloaded by the helper, while SID-Set and WildFake require the source files described there.

```bash
# Download supported data and create the cleaned manifests.
python src/data/download.py --dataset cifake --out data/raw
python src/data/clean_cifake.py

# Train and evaluate the selected frozen-CLIP classifier head.
python -m src.models.train_clip_head
```

For the committed CIFAKE run, the repository includes `outputs/cifake_head.pt` and `outputs/cifake_clip_robustness_table.csv`. That run used 200 training and 200 evaluation images per class and produced:

| Condition | Images | Accuracy |
| --- | ---: | ---: |
| Clean | 400 | 80.50% |
| JPEG recompression | 400 | 66.25% |
| Gaussian blur | 400 | 70.75% |
| Random resized crop | 400 | 66.75% |
| Downscale then upscale | 400 | 67.25% |
| Color jitter | 400 | 76.00% |

These are small-subset experiment results, not a claim of production-level or cross-dataset accuracy.

### Reproduce the configurable end-to-end pipeline

The earlier configurable pipeline can be run through the provided scripts:

```bash
# Train the baseline or a named experiment from configs/config.yaml.
./scripts/train.sh baseline
# Alternatives: rotation_jitter or finetune_backbone

# Reproduce calibration, robustness, error-analysis, and shortcut checks.
./scripts/evaluate.sh outputs/baseline/model_best.pt

# Produce AI probabilities, similarity clusters, and repetition scores.
./scripts/run_inference.sh path/to/images outputs/baseline/model_best.pt
```

Inference writes `outputs/predictions.json` with one record per image:

```json
{
  "image_path": "image_01.jpg",
  "pred": 0.91,
  "similarity_cluster": 4,
  "repetition_score": 0.94
}
```

Run the automated checks with:

```bash
python -m pytest -q
cd frontend && npm run build
```

## Limitations and future improvements

- **Limited evaluation scale.** The committed robustness table uses a balanced 400-image evaluation subset. With more time, we would train and evaluate on the full CIFAKE, SID-Set, and WildFake datasets and publish per-source ROC-AUC, precision, recall, calibration error, and confidence intervals.
- **Domain shift.** A detector trained on known generators may perform poorly on unseen models, edited images, screenshots, or platform recompression. We would add newer generators, adversarial transformations, and a strictly held-out generator benchmark.
- **Robustness degradation.** Accuracy falls under JPEG compression, cropping, and resizing. We would tune augmentation using a larger validation set and test fine-tuning part of the CLIP backbone instead of training only the head.
- **Similarity threshold.** The global cosine threshold (`0.90`) is a practical demo setting and may merge visually similar originals or miss heavily edited copies. We would calibrate it on human-labelled duplicate pairs and consider perceptual hashing or a learned similarity model.
- **Cached augmentation trade-off.** Frozen CLIP embeddings make training fast, but each image receives only one sampled augmentation per run. We would cache multiple augmented views or fine-tune end-to-end when compute permits.
- **Demo-scale infrastructure.** Local state is in memory and single-process; resets and restarts discard local metadata. Production use would require authenticated durable storage, background inference, moderation safeguards, monitoring, and privacy/retention controls.
- **Interpretability and fairness.** A probability is not proof that an image is synthetic. We would add uncertainty-aware review flows and audit false positives across image styles, cultures, compression levels, and creator communities before using the score for ranking decisions.

## Team contributions

| Team member | Primary contribution |
| --- | --- |
| Vicky | Data acquisition and cleaning, dataset manifests, robustness transforms |
| Glory | Core model architecture, classifier training, and model integration |
| Chelsea | CLIP embeddings, near-duplicate clustering, repetition scoring, and similarity evaluation |
| Evan | Evaluation, calibration, robustness analysis, and error/shortcut analysis |
| Eron | Product demo, frontend experience, and application integration |

The components were integrated through a shared inference contract so AI classification and repetition analysis can reuse the same visual backbone.
