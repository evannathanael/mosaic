import { useEffect, type RefObject } from 'react'
import type { Post, ClusterOrigin } from '../types'
import { PostCard } from './PostCard'
import { SkeletonFeed } from './SkeletonFeed'
import { getClusterKey } from '../theme'

interface Props {
  posts: Post[]
  loading: boolean
  feedRef: RefObject<HTMLDivElement>
  similarCounts: Map<string, number>
  onOpenCluster: (post: Post, origin: ClusterOrigin) => void
  onRetry: (clientId: string) => void
}

/** ArrowUp / ArrowDown move exactly one slide. Scroll-snap handles the rest. */
function useSlideKeys(feedRef: RefObject<HTMLDivElement>) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
      if (document.querySelector('.zoom, .sheet')) return
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      const feed = feedRef.current
      if (!feed) return
      const slides = Array.from(feed.querySelectorAll<HTMLElement>('.slide'))
      if (!slides.length) return
      e.preventDefault()
      const feedRect = feed.getBoundingClientRect()
      const centerY = feedRect.top + feed.clientHeight / 2
      let cur = slides.findIndex((s) => {
        const r = s.getBoundingClientRect()
        return r.top - 1 <= centerY && r.bottom + 1 >= centerY
      })
      if (cur < 0) cur = 0
      const next =
        e.key === 'ArrowDown'
          ? Math.min(cur + 1, slides.length - 1)
          : Math.max(cur - 1, 0)
      slides[next]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [feedRef])
}

export function Feed({ posts, loading, feedRef, similarCounts, onOpenCluster, onRetry }: Props) {
  useSlideKeys(feedRef)

  if (loading) return <SkeletonFeed />

  return (
    <>
      <div className="feed" ref={feedRef} tabIndex={-1}>
        {posts.length === 0 ? (
          <div className="slide">
            <div className="card feed-empty">
              <p>Upload an image to watch it score live and land in the feed.</p>
            </div>
          </div>
        ) : (
          posts.map((post) => (
            <div className="slide" key={post.clientId ?? post.image_id}>
              <PostCard
                post={post}
                similarCount={similarCounts.get(getClusterKey(post)) ?? 0}
                onOpenCluster={onOpenCluster}
                onRetry={onRetry}
              />
            </div>
          ))
        )}
      </div>
      {posts.length > 1 && (
        <div className="feed-hint font-mono-label" aria-hidden="true">
          ↑ ↓ or scroll
        </div>
      )}
    </>
  )
}
