## Similarity & Clustering Engine (Person 3)

Owns near-duplicate detection: extracts CLIP ViT-L/14 (OpenAI weights) embeddings
for a folder of images, computes cosine similarity, and clusters near-duplicates
(threshold-based / DBSCAN). Outputs `similarity_cluster` and `repetition_score`
per image.

**Status:** in progress — env set up, embedding extraction + clustering next.

**Depends on:** CLIP ViT-L/14 (OpenAI weights) — no dependency on Person 2's
training pipeline.