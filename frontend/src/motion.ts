import { useReducedMotion } from 'motion/react'

/** Default UI spring — no overshoot. */
export const SPRING_UI = { type: 'spring' as const, bounce: 0, duration: 0.35 }

/** Momentum spring — ONLY after a drag / flick release. */
export const SPRING_MOMENTUM = { type: 'spring' as const, bounce: 0.2, duration: 0.3 }

/** Reduced-motion fallback: a plain opacity cross-fade. */
export const FADE = { duration: 0.2 }

type Entrance = {
  initial: Record<string, number | string>
  animate: Record<string, number | string>
  exit?: Record<string, number | string>
  transition: typeof SPRING_UI | typeof FADE
}

/**
 * Returns entrance props honoring `prefers-reduced-motion`: the rich variant
 * collapses to a ~200ms opacity cross-fade (no scale / blur / overshoot).
 */
export function useEntrance(rich: Omit<Entrance, 'transition'>): Entrance {
  const reduce = useReducedMotion()
  if (reduce) {
    return {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 },
      transition: FADE,
    }
  }
  return { ...rich, transition: SPRING_UI }
}
