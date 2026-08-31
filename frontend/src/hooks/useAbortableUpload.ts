import { useCallback, useRef } from 'react'
import { uploadImage } from '../api'
import type { Post } from '../types'

/**
 * Runs one upload at a time. Starting a new upload aborts the in-flight one;
 * the superseded call resolves to `null` so its caller can bail quietly.
 */
export function useAbortableUpload() {
  const controllerRef = useRef<AbortController | null>(null)

  const upload = useCallback(async (file: File): Promise<Post | null> => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller

    try {
      const post = await uploadImage(file, controller.signal)
      if (controller.signal.aborted) return null
      return post
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return null
      throw err
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }, [])

  return { upload }
}
