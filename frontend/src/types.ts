export type DiversityLabel = 'original' | 'unique_ai' | 'repeated_synthetic'

/** Per-card lifecycle for the optimistic upload flow (frontend-only). */
export type CardStatus = 'scoring' | 'ready' | 'error'

export interface Post {
  image_id: string
  thumbnail_url: string // relative from the real backend; prefix with API base when rendering
  handle: string
  ai_probability: number // 0-1
  repetition_score: number // 0-1
  diversity_label: DiversityLabel
  uploaded_at: number // unix seconds

  /** Frontend-only, never sent to the backend. Stable React key across the temp->real swap. */
  clientId?: string
  /** Frontend-only. Absent means 'ready'. */
  status?: CardStatus
}

export type ViewType = 'feed' | 'compare'

export interface ClusterOrigin {
  x: number
  y: number
}

export type ViewState = { type: 'feed' } | { type: 'compare' }
