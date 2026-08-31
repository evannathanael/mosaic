import { AnimatePresence, motion } from 'motion/react'
import type { Post } from '../types'
import { getHueForPost, getLabelCopy } from '../theme'
import { FADE } from '../motion'

interface Props {
  post: Post
  onRetry?: () => void
}

export function DiversityBadge({ post, onRetry }: Props) {
  const status = post.status ?? 'ready'
  const isRepeated = post.diversity_label === 'repeated_synthetic'
  const stop = getHueForPost(post)

  let key: string
  let node: React.ReactNode

  if (status === 'scoring') {
    key = 'scoring'
    node = (
      <span className="badge badge-scoring font-mono-label">
        <span className="pulse-dot" />
        Scoring…
      </span>
    )
  } else if (status === 'error') {
    key = 'error'
    node = (
      <button type="button" className="badge badge-error font-mono-label" onClick={onRetry}>
        Couldn&rsquo;t score this — tap to retry
      </button>
    )
  } else if (isRepeated) {
    key = 'flag'
    node = <span className="badge badge-flag font-mono-label">AI · repeated</span>
  } else {
    key = `ready-${post.diversity_label}`
    node = (
      <span className="badge font-mono-label">
        <span
          className="badge-dot"
          style={{ background: stop.accent, boxShadow: `0 0 0 3px ${stop.soft}` }}
        />
        {getLabelCopy(post.diversity_label)}
      </span>
    )
  }

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.span
        key={key}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={FADE}
        style={{ display: 'inline-flex' }}
      >
        {node}
      </motion.span>
    </AnimatePresence>
  )
}
