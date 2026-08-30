import { useCallback, useEffect, useRef, useState } from 'react'
import type { Post } from '../types'
import { getFeed } from '../api'
import { rankFeed } from '../feedRanker'

let seq = 0
const uid = () => `c${Date.now().toString(36)}_${(seq++).toString(36)}`

export function useFeed() {
  const [feed, setFeed] = useState<Post[]>([])
  const [loading, setLoading] = useState(true)
  const [netError, setNetError] = useState(false)
  const feedRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    try {
      const data = await getFeed()
      // Feed order is decided by the ranker, not by uploaded_at.
      setFeed(rankFeed(data).map((p) => ({ ...p, clientId: p.clientId ?? uid(), status: 'ready' as const })))
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
    prependCard,
    updateCard,
    scrollToTop,
    reload: load,
    newClientId: uid,
  }
}
