import { AnimatePresence, motion } from 'motion/react'
import { FADE } from '../motion'

interface Props {
  visible: boolean
  message?: string
}

export function NetBanner({ visible, message = "Can't reach the scorer — showing the last known feed" }: Props) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          role="status"
          className="glass"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 12 }}
          transition={FADE}
          style={{
            position: 'fixed',
            left: '1rem',
            bottom: 'max(1rem, env(safe-area-inset-bottom))',
            zIndex: 44,
            maxWidth: '22rem',
            padding: '0.7rem 0.95rem',
            borderRadius: '0.75rem',
            fontSize: '0.85rem',
            borderColor: 'var(--flag)',
          }}
        >
          {message}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
