import { useEffect, useRef } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import type { Post, ClusterOrigin } from '../types'
import { resolveThumb } from '../api'
import { SPRING_UI, FADE } from '../motion'

interface Props {
  trigger: Post | null
  origin: ClusterOrigin | null
  posts: Post[]
  onClose: () => void
}

export function ClusterSheet({ trigger, origin, posts, onClose }: Props) {
  const reduce = useReducedMotion()
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!trigger) return
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [trigger, onClose])

  const group = trigger
    ? posts
        .filter((p) => p.handle === trigger.handle && p.diversity_label === 'repeated_synthetic')
        .sort((a, b) => a.repetition_score - b.repetition_score)
    : []
  const kept = group[0]
  const suppressed = group.slice(1)

  const transformOrigin = origin
    ? `${origin.x}px ${origin.y}px`
    : '50% 100%'

  const sheetMotion = reduce
    ? { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 }, transition: FADE }
    : {
        initial: { opacity: 0, scale: 0.55, filter: 'blur(10px)' },
        animate: { opacity: 1, scale: 1, filter: 'blur(0px)' },
        exit: { opacity: 0, scale: 0.55, filter: 'blur(10px)' },
        transition: SPRING_UI,
      }

  return (
    <AnimatePresence>
      {trigger && kept && (
        <>
          <motion.div
            className="sheet-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={FADE}
            onClick={onClose}
          />
          <motion.div
            className="sheet"
            role="dialog"
            aria-modal="true"
            aria-label={`Similar posts from @${trigger.handle}`}
            style={{ transformOrigin }}
            {...sheetMotion}
          >
            <div className="sheet-head">
              <h2 className="font-display">Same image, {group.length} times</h2>
              <button ref={closeRef} type="button" className="sheet-close" onClick={onClose} aria-label="Close">
                ✕
              </button>
            </div>

            <p className="sheet-stat font-mono-label">
              Kept in feed: 1 · Suppressed: {suppressed.length}
            </p>

            <div className="sheet-kept">
              <img src={resolveThumb(kept.thumbnail_url)} alt="" />
              <div className="kept-body">
                <span className="tag-kept font-mono-label">Kept</span>
                <p className="kept-note">
                  One copy stays in the feed. @{trigger.handle}&rsquo;s {suppressed.length} near-identical
                  re-posts are suppressed — the account isn&rsquo;t.
                </p>
              </div>
            </div>

            <p className="sheet-suppressed-label font-mono-label">Suppressed · repetition score</p>
            <div className="sheet-grid">
              {suppressed.map((p) => (
                <div className="sheet-cell" key={p.image_id}>
                  <img src={resolveThumb(p.thumbnail_url)} alt="" />
                  <span className="mono">{Math.round(p.repetition_score * 100)}%</span>
                </div>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
