import type { Post } from './types'
import { FIXTURES } from './mock/fixtures'
import { createPoster } from './mock/mockThumb'
import { getHueStop } from './theme'

/**
 * Flip to `false` and point API_BASE at a running backend — this is the ONLY
 * line that changes. Endpoints: GET /feed, POST /upload (multipart `file`), POST /reset.
 */
const USE_MOCK = false
const API_BASE = 'http://localhost:8000'

/** Prefix relative thumbnail_url from the real backend; mock URLs are absolute data URIs. */
export function resolveThumb(url: string): string {
  if (USE_MOCK || url.startsWith('data:') || url.startsWith('http') || url.startsWith('blob:')) return url
  return `${API_BASE}${url.startsWith('/') ? '' : '/'}${url}`
}

let mockFeed: Post[] = [...FIXTURES]

function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('Aborted', 'AbortError'))
    const id = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(id)
      reject(new DOMException('Aborted', 'AbortError'))
    })
  })
}

export async function getFeed(): Promise<Post[]> {
  if (USE_MOCK) {
    await abortableDelay(280)
    return mockFeed.map((p) => ({ ...p }))
  }
  const res = await fetch(`${API_BASE}/feed`)
  if (!res.ok) throw new Error('Failed to fetch feed')
  return res.json()
}

export async function uploadImage(file: File, signal?: AbortSignal): Promise<Post> {
  if (USE_MOCK) {
    await abortableDelay(1400, signal)
    const id = `uploaded_${Date.now()}`
    const stop = getHueStop(`solo:${id}`)
    // A judge's own upload: AI-ish but distinct, so it stays untouched in the feed.
    return {
      image_id: id,
      thumbnail_url: createPoster(id, { hue: stop.h, sat: 60 }),
      handle: 'you',
      ai_probability: 0.58 + Math.random() * 0.34,
      repetition_score: 0.04 + Math.random() * 0.12,
      diversity_label: 'unique_ai',
      uploaded_at: Math.floor(Date.now() / 1000),
    }
  }

  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form, signal })
  if (!res.ok) throw new Error('Upload failed')
  return res.json()
}

export interface ClusterDetail {
  cluster_id: number
  kept_image_id: string
  members: Post[]
  suppressed_image_ids: string[]
}

/** Full member list for a similarity cluster — DB-wide, not just the loaded feed. */
export async function getClusterDetail(clusterId: number): Promise<ClusterDetail> {
  if (USE_MOCK) {
    const members = mockFeed.filter((p) => p.similarity_cluster === clusterId)
    return {
      cluster_id: clusterId,
      kept_image_id: members[0]?.image_id ?? '',
      members,
      suppressed_image_ids: members.slice(1).map((p) => p.image_id),
    }
  }
  const res = await fetch(`${API_BASE}/api/cluster/${clusterId}`)
  if (!res.ok) throw new Error('Failed to fetch cluster')
  return res.json()
}

export async function reset(): Promise<{ status: 'reset' }> {
  if (USE_MOCK) {
    mockFeed = [...FIXTURES]
    return { status: 'reset' }
  }
  const res = await fetch(`${API_BASE}/reset`, { method: 'POST' })
  if (!res.ok) throw new Error('Reset failed')
  return res.json()
}
