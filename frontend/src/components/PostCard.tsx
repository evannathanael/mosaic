import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import type { Post, ClusterOrigin } from '../types'
import { DiversityBadge } from './DiversityBadge'
import { resolveThumb } from '../api'
import { getHueForPost } from '../theme'
import { useEntrance, FADE } from '../motion'

interface Props {
  post: Post
  similarCount: number
  onOpenCluster: (post: Post, origin: ClusterOrigin) => void
  onRetry: (clientId: string) => void
}

export function PostCard({ post, similarCount, onOpenCluster, onRetry }: Props) {
  const chipRef = useRef<HTMLButtonElement>(null)
  const [zoom, setZoom] = useState(false)
  const entrance = useEntrance({
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
  })
  const stop = getHueForPost(post)
  const showChip = post.diversity_label === 'repeated_synthetic' && similarCount > 1
  const src = resolveThumb(post.thumbnail_url)

  useEffect(() => {
    if (!zoom) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setZoom(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoom])

  const openCluster = () => {
    const r = chipRef.current?.getBoundingClientRect()
    onOpenCluster(post, r ? { x: r.x + r.width / 2, y: r.y + r.height / 2 } : { x: 0, y: 0 })
  }

  return (
    <motion.article className="card" data-client-id={post.clientId} {...entrance}>
      <div className="card-top">
        <span className="card-avatar" style={{ backgroundColor: stop.accent }} />
        <span className="card-handle font-display">@{post.handle}</span>
      </div>

      <button type="button" className="card-imgwrap" onClick={() => setZoom(true)} aria-label="Expand image">
        <img className="card-img" src={src} alt="" draggable={false} />
        <span className="card-expand-hint" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
          </svg>
        </span>
      </button>

      <div className="card-bottom">
        <DiversityBadge post={post} onRetry={() => post.clientId && onRetry(post.clientId)} />
        {showChip && (
          <button ref={chipRef} type="button" className="chip font-mono-label" onClick={openCluster}>
            1 of {similarCount} · tap to see
          </button>
        )}
      </div>

      <AnimatePresence>
        {zoom && (
          <motion.div
            className="zoom"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={FADE}
            onClick={() => setZoom(false)}
            role="dialog"
            aria-modal="true"
            aria-label={`Expanded image from @${post.handle}`}
          >
            <img src={src} alt="" />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  )
}
