# Mosaic Frontend

React / Vite / TypeScript frontend for the Mosaic AIGC-detection demo.

The argument the UI makes in a ~4-minute judge walkthrough: **AI content isn't the
problem, repetition is.** A flood account's near-identical AI images collapse into
"kept + suppressed"; a creator's distinct AI images are untouched. A judge uploads
their own image and watches it score live and land in the feed.

## Quick start

```bash
npm install
npm run dev      # http://localhost:5173  (or next free port)
npm run build    # tsc + vite build -> dist/
```

## Layout

- **Feed** — one post per screen, centered Instagram-style card, vertical
  scroll-snap. `↑` / `↓` or scroll moves exactly one post. Tap the image to expand
  it full-size (Esc / click to close).
- **Floating controls only** — no app bar. Top-left: a small glass pill (brand +
  "Compare accounts"). Bottom-right: the upload button.
- **Cluster sheet** — tap a `1 of N · tap to see` chip; the sheet scales in from
  the chip, shows the kept post plus the suppressed grid with each
  `repetition_score`, and `Kept in feed: 1 · Suppressed: N`.
- **Account comparison** — flood vs creator, derived from the live feed data:
  post counts, % near-copies (mono), a `repetition_score` trend line (flood
  climbs, creator flat), and a plain-language verdict per account.
- **Dev reset** — `Shift+R` calls `/reset` and reloads the feed to the seeded
  state. Use before every dry run.

## Feed ranking (`src/feedRanker.ts`)

The feed order is **not** `uploaded_at` — `rankFeed()` decides what comes next:

1. **First post: random** — any post, duplicate or not, AI or not.
2. **Every post after: the best "next" given the current (previous) post** —
   - never a near-duplicate of the current post (same-cluster back-to-back is
     forbidden; a repair pass guarantees it), and lightly avoided for a few
     posts after;
   - **non-AI prioritised** — demoted by `ai_probability` / `repetition_score`,
     but AI and repeated posts still appear;
   - a proportional-spacing term keeps every cluster (including the repeated-AI
     flood) threaded through the whole feed at ~its share, instead of piling up
     at the end.

Computed once per load (stable while you scroll). Weights live in
`DEFAULT_WEIGHTS`. Uploaded posts are pinned to the top and skip the ranker.

## Animation

All motion goes through `motion` (motion.dev), springs only — `src/motion.ts`
exports `SPRING_UI` (bounce 0) and `SPRING_MOMENTUM` (bounce 0.2, post-drag only).
`useEntrance()` collapses every entrance to a ~200ms opacity fade under
`prefers-reduced-motion`. Floating controls drop their blur under
`prefers-reduced-transparency`.

## About `USE_MOCK`

`USE_MOCK = true` at the top of `src/api.ts` is the **only** switch:

- feed comes from `src/mock/fixtures.ts` (~18-post flood account + ~20-post
  creator, counts kept close so the ranker can interleave them the whole way);
  thumbnails are generated SVG posters
  (`src/mock/mockThumb.ts`), no real likenesses.
- `uploadImage` synthesizes a distinct, plausibly-scored `Post` after ~1.4s and
  is abortable (a second upload cancels the first).
- `reset` restores the fixtures instantly.

Flip to `false` and point `API_BASE` at a running backend — nothing else changes.
Endpoints:

- `GET /feed` → `Post[]` (newest first)
- `POST /upload` (multipart, field `file`) → `Post`
- `POST /reset` → `{ status: "reset" }`

`Post` matches `src/types.ts` (the `clientId` / `status` fields there are
frontend-only and never sent).

## Quality floor

- Works at 390px (mobile-first).
- Visible `:focus-visible` rings; cluster sheet + zoom are keyboard-operable
  (Esc), focus returns to the trigger on close.
- Color is never the only signal — every hue is paired with a label or number.
