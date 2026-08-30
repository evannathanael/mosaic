/**
 * Deterministic generative "posters" — no real likenesses, just bold abstract
 * art so the feed reads as real content. Same seedKey => byte-identical image
 * (that's what makes the flood look like one image re-posted).
 */

function mulberry32(seed: number) {
  return function () {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function seedFrom(str: string): number {
  let h = 2166136261
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

const W = 900
const H = 1350

type Opts = { hue: number; sat?: number; motif?: 'auto' | 'arcs' | 'orbit' | 'bars' | 'grid' }

export function createPoster(seedKey: string, { hue, sat = 72, motif = 'auto' }: Opts): string {
  const rnd = mulberry32(seedFrom(seedKey))
  const pick = <T,>(arr: readonly T[]) => arr[Math.floor(rnd() * arr.length)]
  const range = (a: number, b: number) => a + rnd() * (b - a)
  const hsl = (h: number, s: number, l: number) => `hsl(${((h % 360) + 360) % 360} ${s}% ${l}%)`

  const base = hue
  const comp = base + range(150, 210)
  const near = base + range(20, 55)

  const deep = hsl(base, sat - 12, 16)
  const field = hsl(base, sat, 33)
  const accentA = hsl(near, Math.min(sat + 16, 96), 60)
  const accentB = hsl(comp, Math.min(sat + 10, 92), 62)
  const light = hsl(near, 94, 76)

  const chosen = motif === 'auto' ? pick(['arcs', 'orbit', 'bars', 'grid'] as const) : motif
  let m = ''

  if (chosen === 'arcs') {
    const ox = range(0.05, 0.95) * W
    const oy = range(0.05, 0.95) * H
    const rings = 4 + Math.floor(rnd() * 3)
    for (let i = rings; i >= 1; i--) {
      m += `<circle cx="${ox | 0}" cy="${oy | 0}" r="${(i * range(120, 175)) | 0}" fill="none" stroke="${i % 2 ? accentA : accentB}" stroke-width="${range(14, 34) | 0}" opacity="${(0.9 - i * 0.09).toFixed(2)}"/>`
    }
    m += `<circle cx="${ox | 0}" cy="${oy | 0}" r="${range(40, 90) | 0}" fill="${light}"/>`
  } else if (chosen === 'orbit') {
    m += `<circle cx="${(range(0.28, 0.72) * W) | 0}" cy="${(range(0.24, 0.62) * H) | 0}" r="${range(240, 340) | 0}" fill="${accentA}"/>`
    for (let i = 0; i < 6; i++) {
      m += `<circle cx="${(rnd() * W) | 0}" cy="${(rnd() * H) | 0}" r="${range(28, 92) | 0}" fill="${i % 2 ? accentB : light}" opacity="${range(0.55, 0.95).toFixed(2)}"/>`
    }
  } else if (chosen === 'bars') {
    const n = 4 + Math.floor(rnd() * 4)
    for (let i = 0; i < n; i++) {
      const bw = (W / n) * range(0.5, 0.92)
      const bh = range(0.35, 1) * H
      m += `<rect x="${((W / n) * i + range(2, 26)) | 0}" y="${(H - bh) | 0}" width="${bw | 0}" height="${bh | 0}" rx="${(bw / 2) | 0}" fill="${i % 2 ? accentA : accentB}" opacity="${range(0.72, 1).toFixed(2)}"/>`
    }
    m += `<circle cx="${(range(0.2, 0.8) * W) | 0}" cy="${(range(0.15, 0.4) * H) | 0}" r="${range(60, 120) | 0}" fill="${light}"/>`
  } else {
    const cols = 3 + Math.floor(rnd() * 3)
    const cw = W / cols
    const rows = Math.ceil(H / cw)
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (rnd() > 0.62) continue
        const shape = rnd()
        const fill = [accentA, accentB, light][Math.floor(rnd() * 3)]
        const px = x * cw + cw * 0.1
        const py = y * cw + cw * 0.1
        const s = cw * 0.8
        if (shape < 0.5) m += `<rect x="${px | 0}" y="${py | 0}" width="${s | 0}" height="${s | 0}" rx="${(s * 0.22) | 0}" fill="${fill}" opacity="${range(0.6, 0.95).toFixed(2)}"/>`
        else m += `<circle cx="${(px + s / 2) | 0}" cy="${(py + s / 2) | 0}" r="${(s / 2) | 0}" fill="${fill}" opacity="${range(0.6, 0.95).toFixed(2)}"/>`
      }
    }
  }

  const rot = range(-14, 14).toFixed(1)

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="0.7" y2="1">
<stop offset="0" stop-color="${deep}"/><stop offset="0.5" stop-color="${field}"/><stop offset="1" stop-color="${deep}"/>
</linearGradient>
<radialGradient id="v" cx="0.5" cy="0.42" r="0.75">
<stop offset="0.6" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.34"/>
</radialGradient>
</defs>
<rect width="${W}" height="${H}" fill="url(#bg)"/>
<g transform="rotate(${rot} ${W / 2} ${H / 2})">${m}</g>
<rect width="${W}" height="${H}" fill="url(#v)"/>
</svg>`

  return `data:image/svg+xml,${encodeURIComponent(svg.replace(/\s+/g, ' ').trim())}`
}
