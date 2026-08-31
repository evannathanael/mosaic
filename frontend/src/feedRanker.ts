import type { Post } from './types'
import { getClusterKey } from './theme'

/**
 * Feed ranker — decides which post comes next.
 *
 *   • First post: random (any post, duplicate or not, AI or not).
 *   • Every post after that: chosen from the remaining pool to be the best
 *     "next" given the CURRENT post (the one before it):
 *       1. Not a duplicate of the current post — same cluster back-to-back is
 *          strongly penalised (and lightly for a few posts after that).
 *       2. Prioritise non-AI — demote by `ai_probability`; AI posts are still
 *          allowed and still appear.
 *       3. Stay a believable timeline — a spacing term keeps every cluster
 *          (including the repeated-AI flood) appearing at roughly its share of
 *          the feed instead of being pushed to the end.
 *
 * The order is computed once per feed load (a stable sequence), not re-shuffled
 * while you scroll.
 */

export interface RankWeights {
  /** Same cluster as the immediately-previous post. */
  dupVsCurrent: number
  /** Same cluster as a post 2–`window` back (decays). */
  dupInWindow: number
  /** Same author (different image) as a recent post. */
  sameAuthor: number
  /** Gentle continuous demotion scaled by ai_probability (the label terms below
   *  carry most of the ordering now). */
  aiPenalty: number
  /** Demotion scaled by the post's own repetition_score. */
  repetitionPenalty: number
  /** Desirability by diversity_label — the feed leans
   *  original ≳ unique_ai ≫ repeated_synthetic. */
  originalBonus: number
  distinctAiBonus: number
  repeatedSyntheticPenalty: number
  /** Fraction of the fair-share "this cluster is overdue" pull that still applies
   *  to repeated_synthetic posts (< 1 ⇒ AI-repeats resurface less often). */
  repeatedSpacingDamp: number
  /** Anti-clumping: per-post demotion for each consecutive post at the tail with
   *  the *same diversity_label*. Breaks up long single-label blocks; jitter makes
   *  the break length vary. original and unique_ai don't suppress each other. */
  categoryRun: number
  /** Extra demotion when the current post is AI-repeated and so is the candidate
   *  (different cluster) — don't hop from one AI-repeat flood straight to another. */
  aiRepeatChain: number
  /** Weight on the "this cluster is behind its fair share" term. */
  spacingWeight: number
  spacingCap: number
  /** Random tie-break / variation between loads. */
  jitter: number
  window: number
  decay: number
}

export const DEFAULT_WEIGHTS: RankWeights = {
  dupVsCurrent: 4,
  dupInWindow: 1,
  sameAuthor: 0.5,
  aiPenalty: 0.25,
  repetitionPenalty: 0.35,
  originalBonus: 0.7,
  distinctAiBonus: 0.25,
  repeatedSyntheticPenalty: 0.8,
  repeatedSpacingDamp: 0.45,
  categoryRun: 0.4,
  aiRepeatChain: 1.5,
  spacingWeight: 1.6,
  spacingCap: 3,
  jitter: 0.35,
  window: 4,
  decay: 0.5,
}

function scoreCandidate(
  post: Post,
  placed: Post[],
  poolSize: number,
  clusterTotal: number,
  shownCount: number,
  w: RankWeights,
): number {
  let score = 0

  // Rule 2 — lean the feed toward original ≳ unique_ai ≫ repeated_synthetic.
  score -= w.aiPenalty * post.ai_probability
  score -= w.repetitionPenalty * post.repetition_score
  score +=
    post.diversity_label === 'original'
      ? w.originalBonus
      : post.diversity_label === 'unique_ai'
        ? w.distinctAiBonus
        : -w.repeatedSyntheticPenalty

  // Rule 1 — not a duplicate of the current (and recent) post(s).
  const key = getClusterKey(post)
  const recent = placed.slice(-w.window)

  // Anti-clumping — per-post demotion for each consecutive post at the tail with
  // the same diversity_label. Breaks up long single-label blocks; jitter varies
  // the break length. original/unique_ai are different labels so they interleave
  // freely rather than being forced 1:1 against AI.
  let labelRunLen = 0
  for (let i = placed.length - 1; i >= 0; i--) {
    if (placed[i].diversity_label !== post.diversity_label) break
    labelRunLen++
  }
  score -= w.categoryRun * labelRunLen

  // When the current post is an AI near-dup repeat, steer away from another
  // AI-repeat cluster as the next photo (same-cluster is already covered below).
  const current = placed[placed.length - 1]
  if (
    current?.diversity_label === 'repeated_synthetic' &&
    post.diversity_label === 'repeated_synthetic' &&
    getClusterKey(current) !== key
  ) {
    score -= w.aiRepeatChain
  }

  for (let i = 0; i < recent.length; i++) {
    const prev = recent[recent.length - 1 - i] // i = 0 is the current post
    if (getClusterKey(prev) === key) {
      score -= i === 0 ? w.dupVsCurrent : w.dupInWindow * Math.pow(w.decay, i - 1)
    } else if (prev.handle === post.handle) {
      score -= w.sameAuthor * Math.pow(w.decay, i)
    }
  }

  // Rule 3 — keep every cluster near its fair share so AI / repeated posts stay
  // threaded through. deficit > 0 => this cluster is overdue for a slot.
  const expected = clusterTotal * (placed.length / poolSize)
  const deficit = expected - shownCount
  // Asymmetric: being overdue pulls hard; being recently-shown only lightly
  // demotes (so a big cluster recovers quickly and keeps threading through).
  // The overdue pull is damped for AI-repeats so they resurface less often.
  const spacing = w.spacingWeight * Math.max(-1, Math.min(w.spacingCap, deficit))
  score +=
    deficit > 0 && post.diversity_label === 'repeated_synthetic'
      ? spacing * w.repeatedSpacingDamp
      : spacing

  score += Math.random() * w.jitter
  return score
}

/**
 * Given the posts already placed and a pool of candidates, return the index in
 * `pool` of the best "next" post. Cluster totals / fair-share are computed over
 * `placed ∪ pool`, so this works both for the one-shot `rankFeed` loop and for
 * revealing more posts incrementally as the user scrolls.
 */
export function pickNext(
  placed: Post[],
  pool: Post[],
  weights: Partial<RankWeights> = {},
): number {
  const w = { ...DEFAULT_WEIGHTS, ...weights }
  const poolSize = placed.length + pool.length

  const clusterTotal = new Map<string, number>()
  for (const p of [...placed, ...pool]) {
    const k = getClusterKey(p)
    clusterTotal.set(k, (clusterTotal.get(k) ?? 0) + 1)
  }
  const shown = new Map<string, number>()
  for (const p of placed) {
    const k = getClusterKey(p)
    shown.set(k, (shown.get(k) ?? 0) + 1)
  }

  let bestIdx = 0
  let bestScore = -Infinity
  for (let i = 0; i < pool.length; i++) {
    const key = getClusterKey(pool[i])
    const s = scoreCandidate(
      pool[i],
      placed,
      poolSize,
      clusterTotal.get(key) ?? 1,
      shown.get(key) ?? 0,
      w,
    )
    if (s > bestScore) {
      bestScore = s
      bestIdx = i
    }
  }
  return bestIdx
}

export function rankFeed(posts: Post[], weights: Partial<RankWeights> = {}): Post[] {
  if (posts.length <= 1) return posts.map((p) => ({ ...p }))

  const remaining = [...posts]
  const ordered: Post[] = []

  // First post: random.
  const [opener] = remaining.splice(Math.floor(Math.random() * remaining.length), 1)
  ordered.push({ ...opener })

  while (remaining.length) {
    const idx = pickNext(ordered, remaining, weights)
    const [picked] = remaining.splice(idx, 1)
    ordered.push({ ...picked })
  }

  return spreadAdjacentDuplicates(ordered)
}

/**
 * Hard guarantee for rule 1: no two same-cluster posts sit back-to-back. The
 * greedy pass gets close; this repair loop relocates any offending post to the
 * nearest gap whose neighbours are both a different cluster. Only a genuine tail
 * (nothing else of a different cluster remains) can defeat it.
 */
function spreadAdjacentDuplicates(order: Post[]): Post[] {
  const out = [...order]
  const k = (i: number) => (i >= 0 && i < out.length ? getClusterKey(out[i]) : null)

  for (let pass = 0; pass < out.length; pass++) {
    const bad = out.findIndex((_, i) => i > 0 && k(i) === k(i - 1))
    if (bad === -1) break
    const key = k(bad)
    const [moved] = out.splice(bad, 1)

    // Insertion index in the reduced array whose neighbours both differ from
    // `key`; prefer the one closest to where it was.
    let target = -1
    let bestDist = Infinity
    for (let t = 0; t <= out.length; t++) {
      const leftK = t - 1 >= 0 ? getClusterKey(out[t - 1]) : null
      const rightK = t < out.length ? getClusterKey(out[t]) : null
      if (leftK === key || rightK === key) continue
      const dist = Math.abs(t - bad)
      if (dist < bestDist) {
        bestDist = dist
        target = t
      }
    }
    out.splice(target === -1 ? bad : target, 0, moved)
    if (target === -1) break
  }
  return out
}
