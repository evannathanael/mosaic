# Mosaic backend

The FastAPI server currently runs in model-independent demo mode. It loads the
small fixture set under `data/dataset/`, serves those images, accepts uploads,
and returns the stable response contract that the frontend can integrate with
before model training is complete.

## Run locally

```powershell
python -m pip install -r backend/requirements.txt
.\scripts\run_backend.ps1
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is
at `/docs`.

## Supabase persistence

The default is local demo storage. To persist metadata in Supabase Postgres
and files in Supabase Storage:

1. Create a Supabase project and a **public** Storage bucket named
   `mosaic-images`.
2. Run [`supabase_schema.sql`](supabase_schema.sql) in the Supabase SQL Editor.
3. Create `D:\mosaic\.env` (it is ignored by Git):

   ```text
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SECRET_KEY=your-server-only-secret-key
   SUPABASE_STORAGE_BUCKET=mosaic-images
   ```

4. Restart the backend. `GET /api/health` should report
   `"storage_mode": "supabase"`.

The backend keeps live ViT-L/14 inference disabled by default on laptop-sized
machines. In `configs/config.yaml`, set both `similarity.enabled` and
`detector.enabled` to `true` only when the machine has enough RAM/pagefile and
the compatible CLIP checkpoint is available. With them disabled, the service
uses the safe exact-hash and filename-heuristic fallbacks without attempting a
large model allocation.

The secret key must stay on the backend and must never be placed in the
frontend. Without these variables, the API continues to use local files and
local metadata.

## Frontend endpoints

* `GET /feed` — seeded demo feed, newest first
* `POST /upload` — accepts one `file` and returns the frontend `Post` shape
* `POST /reset` — restores the seed dataset and returns `{"status":"reset"}`
* `GET /cluster/{cluster_id}` — returns the kept member and suppressed near-copies

## Additional endpoints

* `GET /api/feed` — seeded demo feed
* `POST /api/upload` — upload one image (or a batch using repeated `files`)
* `POST /api/analyze` — batch analysis response with latency and model status
* `GET /api/clusters` — current similarity groups
* `GET /api/account/{account_id}/summary` — account aggregate
* `POST /api/simulate-feed` — suppress repeated-synthetic posts in a copy of a feed
* `POST /api/reset` — reset live uploads to the demo seed
* `GET /api/health` — reports whether the trained model is loaded

With Supabase configured, uploaded images are stored in the configured Storage
bucket; otherwise they are saved under `backend/uploads/`, which is ignored by Git.
Until the detector is available, uploads return `analysis_mode: "mock"` and
`model_ready: false`; this is deliberately explicit so the UI does not present
placeholder scores as trained-model predictions. The inference implementation
can later replace the mock scorer without changing the route schemas.
