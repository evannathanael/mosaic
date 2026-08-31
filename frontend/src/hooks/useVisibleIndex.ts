import { useEffect, useRef, useState, type RefObject } from 'react'

/**
 * Reports the index of the `.slide` currently filling most of the scroll-snap
 * viewport. Used to drive incremental feed reveal as the user scrolls.
 */
export function useVisibleIndex(
  feedRef: RefObject<HTMLElement>,
  onChange?: (index: number) => void,
  ready = true,
): number {
  const [index, setIndex] = useState(0)
  const ratios = useRef<Map<Element, number>>(new Map())
  const cb = useRef(onChange)
  cb.current = onChange

  useEffect(() => {
    const root = feedRef.current
    if (!root || !ready) return

    const recompute = () => {
      const slides = Array.from(root.querySelectorAll<HTMLElement>('.slide'))
      let best = 0
      let bestRatio = -1
      slides.forEach((slide, i) => {
        const r = ratios.current.get(slide) ?? 0
        if (r > bestRatio) {
          bestRatio = r
          best = i
        }
      })
      setIndex((prev) => {
        if (prev !== best) cb.current?.(best)
        return best
      })
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) ratios.current.set(entry.target, entry.intersectionRatio)
        recompute()
      },
      { root, threshold: [0, 0.25, 0.5, 0.75, 1] },
    )

    const mutation = new MutationObserver(() => {
      observer.disconnect()
      root.querySelectorAll('.slide').forEach((slide) => observer.observe(slide))
    })
    mutation.observe(root, { childList: true })
    root.querySelectorAll('.slide').forEach((slide) => observer.observe(slide))

    return () => {
      observer.disconnect()
      mutation.disconnect()
      ratios.current.clear()
    }
  }, [feedRef, ready])

  return index
}
