import { useCallback, useEffect, useRef, useState } from 'react'
import type { Post } from '../types'
import { getFeed } from '../api'
import { pickNext, rankFeed } from '../feedRanker'

let seq = 0
const uid = () => `c${Date.now().toString(36)}_${(seq++).toString(36)}`

/** Posts shown before any scroll, and how far ahead of the viewport we keep the feed filled. */
const INITIAL_REVEAL = 12
const PREFETCH = 4
const REVEAL_BATCH = 4

const withMeta = (p: Post): Post => ({ ...p, clientId: p.clientId ?? uid(), status: 'ready' as const })

export function useFeed() {
  const [feed, setFeed] = useState<Post[]>([])
  const [loading, setLoading] = useState(true)
  const [netError, setNetError] = useState(false)
  const feedRef = useRef<HTMLDivElement>(null)
  /** Not-yet-revealed pool, re-scored against the visible tail on each reveal. */
  const queueRef = useRef<Post[]>([])

  const load = useCallback(async () => {
    try {
      const data = await getFeed()
      // rankFeed decides a stable, diversity-aware order; we then stream it in.
      const ordered = rankFeed(data).map(withMeta)
      queueRef.current = ordered.slice(INITIAL_REVEAL)
      setFeed(ordered.slice(0, INITIAL_REVEAL))
      setNetError(false)
    } catch {
      setNetError(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  /** Called as the visible slide advances: keep PREFETCH posts staged ahead,
   *  choosing each next post to be least similar to what's currently on screen. */
  const revealNext = useCallback((visibleIndex: number) => {
    setFeed((prev) => {
      if (queueRef.current.length === 0) return prev
      if (visibleIndex < prev.length - PREFETCH) return prev

      const placed = [...prev]
      for (let n = 0; n < REVEAL_BATCH && queueRef.current.length > 0; n++) {
        const idx = pickNext(placed, queueRef.current)
        const [picked] = queueRef.current.splice(idx, 1)
        placed.push(picked)
      }
      return placed
    })
  }, [])

  const prependCard = useCallback((post: Post) => {
    setFeed((prev) => [post, ...prev])
  }, [])

  const updateCard = useCallback((clientId: string, updater: (p: Post) => Post) => {
    setFeed((prev) => prev.map((p) => (p.clientId === clientId ? updater(p) : p)))
  }, [])

  /** Jump the scroll-snap container back to the freshly prepended card.
   *  Deferred two frames so the new card is laid out before we scroll (a bare
   *  scrollTo(0) races the prepend and scroll-anchoring snaps us back). */
  const scrollToTop = useCallback(() => {
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        feedRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
      }),
    )
  }, [])

  return {
    feed,
    loading,
    netError,
    setNetError,
    feedRef,
    revealNext,
    prependCard,
    updateCard,
    scrollToTop,
    reload: load,
    newClientId: uid,
  }
}
