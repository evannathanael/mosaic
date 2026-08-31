import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Post, ClusterOrigin } from './types'
import { Feed } from './components/Feed'
import { UploadButton } from './components/UploadButton'
import { ClusterSheet } from './components/ClusterSheet'
import { NetBanner } from './components/NetBanner'
import { useFeed } from './hooks/useFeed'
import { useAbortableUpload } from './hooks/useAbortableUpload'
import { reset } from './api'
import { getClusterKey } from './theme'

export function App() {
  const feed = useFeed()
  const { upload } = useAbortableUpload()
  const [cluster, setCluster] = useState<{ post: Post; origin: ClusterOrigin } | null>(null)

  const filesRef = useRef(new Map<string, File>())
  const restoreFocusRef = useRef<HTMLElement | null>(null)
  const objectUrls = useRef<string[]>([])

  useEffect(
    () => () => {
      objectUrls.current.forEach(URL.revokeObjectURL)
    },
    [],
  )

  // Dev-only reset (before every dry run).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.shiftKey && e.code === 'KeyR' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        void reset().then(() => feed.reload())
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [feed])

  const similarCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const p of feed.feed) {
      if (p.diversity_label !== 'repeated_synthetic') continue
      const k = getClusterKey(p)
      counts.set(k, (counts.get(k) ?? 0) + 1)
    }
    return counts
  }, [feed.feed])

  const runUpload = useCallback(
    async (file: File, clientId: string) => {
      feed.updateCard(clientId, (p) => ({ ...p, status: 'scoring' }))
      try {
        const result = await upload(file)
        if (!result) return // superseded by a newer upload
        feed.updateCard(clientId, (prev) => ({
          ...result,
          clientId: prev.clientId,
          thumbnail_url: prev.thumbnail_url, // keep the local preview; no image flicker
          status: 'ready',
        }))
        feed.setNetError(false)
      } catch {
        feed.updateCard(clientId, (prev) => ({ ...prev, status: 'error' }))
        feed.setNetError(true)
      }
    },
    [feed, upload],
  )

  const handleUpload = useCallback(
    (file: File) => {
      const clientId = feed.newClientId()
      const objectUrl = URL.createObjectURL(file)
      objectUrls.current.push(objectUrl)
      filesRef.current.set(clientId, file)

      feed.prependCard({
        image_id: `temp_${clientId}`,
        clientId,
        thumbnail_url: objectUrl,
        handle: 'you',
        ai_probability: 0,
        repetition_score: 0,
        diversity_label: 'unique_ai',
        uploaded_at: Math.floor(Date.now() / 1000),
        status: 'scoring',
      })
      feed.scrollToTop()
      void runUpload(file, clientId)
    },
    [feed, runUpload],
  )

  const handleRetry = useCallback(
    (clientId: string) => {
      const file = filesRef.current.get(clientId)
      if (file) void runUpload(file, clientId)
    },
    [runUpload],
  )

  const openCluster = useCallback((post: Post, origin: ClusterOrigin) => {
    restoreFocusRef.current = document.activeElement as HTMLElement | null
    setCluster({ post, origin })
  }, [])

  const closeCluster = useCallback(() => {
    setCluster(null)
    restoreFocusRef.current?.focus?.()
  }, [])

  const isUploading = feed.feed.some((p) => p.status === 'scoring')

  return (
    <div className="app">
      <Feed
        posts={feed.feed}
        loading={feed.loading}
        feedRef={feed.feedRef}
        similarCounts={similarCounts}
        onOpenCluster={openCluster}
        onRetry={handleRetry}
        onVisibleIndex={feed.revealNext}
      />

      <div className="nav-pill glass" aria-hidden="true">
        <span className="brand">MOSAIC</span>
      </div>

      <UploadButton onFileSelect={handleUpload} isUploading={isUploading} />

      <ClusterSheet
        trigger={cluster?.post ?? null}
        origin={cluster?.origin ?? null}
        posts={feed.feed}
        onClose={closeCluster}
      />

      <NetBanner visible={feed.netError} />
    </div>
  )
}
