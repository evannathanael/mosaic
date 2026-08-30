import { useRef } from 'react'
import { motion } from 'motion/react'
import type { ChangeEvent } from 'react'
import { SPRING_UI } from '../motion'

interface Props {
  onFileSelect: (file: File) => void
  isUploading: boolean
}

export function UploadButton({ onFileSelect, isUploading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onFileSelect(file)
    e.target.value = ''
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <motion.button
        type="button"
        className="upload-btn glass"
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
        aria-label="Upload an image"
        whileTap={{ scale: 0.97 }}
        transition={SPRING_UI}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 15V4" />
          <path d="m6.5 9.5 5.5-5.5 5.5 5.5" />
          <path d="M5 20h14" />
        </svg>
        <span className="font-mono-label">{isUploading ? 'Scoring' : 'Upload'}</span>
      </motion.button>
    </>
  )
}
