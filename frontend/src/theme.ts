import type { DiversityLabel, Post } from './types'

/**
 * The "signature": every distinct content cluster gets one stop from this fixed
 * spectrum, applied as an ACCENT (dot / trend point), never a full fill.
 * `repeated_synthetic` is exempt — it always uses --flag.
 */
export type HueStop = { name: string; h: number; accent: string; soft: string }

export const HUE_SPECTRUM: HueStop[] = [
  { name: 'coral', h: 358, accent: '#E5484D', soft: '#FCECEC' },
  { name: 'gold', h: 42, accent: '#C99A2E', soft: '#F8F1DF' },
  { name: 'grass', h: 142, accent: '#2E9E5B', soft: '#E7F5EC' },
  { name: 'teal', h: 186, accent: '#0E8F9B', soft: '#E1F3F4' },
  { name: 'sky', h: 218, accent: '#2F6FE4', soft: '#E8EFFC' },
  { name: 'indigo', h: 240, accent: '#5B5BD6', soft: '#EDEDFB' },
  { name: 'violet', h: 270, accent: '#8248D6', soft: '#F2EAFB' },
  { name: 'magenta', h: 322, accent: '#C2298A', soft: '#FBE9F4' },
]

/** Amber family reserved for repeated synthetic content (matches --flag). */
export const FLAG_HUE = 24

const LABEL_COPY: Record<DiversityLabel, string> = {
  original: 'Original',
  unique_ai: 'AI · distinct',
  repeated_synthetic: 'AI · repeated',
}

export function getLabelCopy(label: DiversityLabel): string {
  return LABEL_COPY[label]
}

/** Cluster key: repeated posts share their handle; distinct posts stand alone. */
export function getClusterKey(post: Post): string {
  if (post.diversity_label === 'repeated_synthetic') return `flood:${post.handle}`
  return `solo:${post.image_id}`
}

function hash(key: string): number {
  let h = 0
  for (let i = 0; i < key.length; i++) {
    h = (h << 5) - h + key.charCodeAt(i)
    h |= 0
  }
  return Math.abs(h)
}

export function getHueStop(key: string): HueStop {
  return HUE_SPECTRUM[hash(key) % HUE_SPECTRUM.length]
}

export function getHueForPost(post: Post): HueStop {
  return getHueStop(getClusterKey(post))
}

/** Design tokens mirrored from tokens.css for SVG generation only. */
export const COLORS = {
  paper: '#E9ECF1',
  surface: '#FFFFFF',
  ink: '#14171D',
  muted: '#667085',
  flag: '#B4541C',
  line: '#D3D8E0',
}
