import type { Post } from '../types'
import { createPoster } from './mockThumb'
import { FLAG_HUE, getHueStop } from '../theme'

const DAY = 86400
const now = Math.floor(Date.now() / 1000)

/**
 * The demo argument: a flood account whose feed collapses into "one image, many
 * times" (its repetition_score climbs as it ramps up) sharing the SAME timeline
 * as a creator whose AI work stays distinct and flat.
 *
 * Counts are kept close (≈18 repeated vs ≈19 distinct) so the ranker can
 * interleave them the whole way down instead of running out of fresh posts.
 */

const FLOOD_HANDLE = 'trendpulse.ai'
const CREATOR_HANDLE = 'marisol.makes'

// One canonical image the flood keeps re-posting (identical seed => identical art).
const FLOOD_IMAGE = createPoster('trendpulse-canonical-v3', { hue: FLAG_HUE, sat: 78, motif: 'arcs' })

const FLOOD_WARMUP = 3
const FLOOD_REPEATED = 15

const flood: Post[] = Array.from({ length: FLOOD_WARMUP + FLOOD_REPEATED }, (_, i) => {
  const warmup = i < FLOOD_WARMUP // the account's first few posts were still varied
  const t = now - (FLOOD_WARMUP + FLOOD_REPEATED - i) * (DAY / 2) - Math.random() * 3600
  return {
    image_id: `flood_${i}`,
    thumbnail_url: warmup
      ? createPoster(`trendpulse-warmup-${i}`, { hue: (FLAG_HUE + i * 40) % 360, sat: 62 })
      : FLOOD_IMAGE,
    handle: FLOOD_HANDLE,
    ai_probability: 0.86 + Math.random() * 0.1,
    repetition_score: warmup
      ? 0.16 + Math.random() * 0.1
      : Math.min(0.97, 0.5 + (i - FLOOD_WARMUP) / FLOOD_REPEATED * 0.45 + Math.random() * 0.04),
    diversity_label: warmup ? 'unique_ai' : 'repeated_synthetic',
    similarity_cluster: warmup ? 100 + i : 200,
    uploaded_at: Math.floor(t),
  }
})

const CREATOR_COUNT = 20

const creator: Post[] = Array.from({ length: CREATOR_COUNT }, (_, i) => {
  const isAI = i % 3 !== 0 // ~2/3 are AI-assisted but each distinct
  const stop = getHueStop(`solo:creator_${i}`)
  const t = now - (CREATOR_COUNT - i) * DAY - Math.random() * 6 * 3600
  return {
    image_id: `creator_${i}`,
    thumbnail_url: createPoster(`marisol-${i}`, { hue: stop.h + (Math.random() * 20 - 10), sat: 58 }),
    handle: CREATOR_HANDLE,
    ai_probability: isAI ? 0.55 + Math.random() * 0.28 : 0.04 + Math.random() * 0.08,
    repetition_score: 0.05 + Math.random() * 0.09, // flat, low
    diversity_label: isAI ? 'unique_ai' : 'original',
    similarity_cluster: 300 + i,
    uploaded_at: Math.floor(t),
  }
})

export const FIXTURES: Post[] = [...flood, ...creator].sort((a, b) => b.uploaded_at - a.uploaded_at)
