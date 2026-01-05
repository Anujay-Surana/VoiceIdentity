import { useMemo } from 'react'

function WaveformVisualizer({ level = 0 }) {
  const bars = useMemo(() => {
    const count = 32
    return Array.from({ length: count }, (_, i) => {
      // Create a wave pattern that responds to audio level
      const centerDistance = Math.abs(i - count / 2) / (count / 2)
      const baseHeight = 0.2 + (1 - centerDistance) * 0.3
      const variation = Math.sin(i * 0.5 + Date.now() / 200) * 0.2
      const levelBoost = level * (1 - centerDistance * 0.5)
      
      return Math.min(1, Math.max(0.1, baseHeight + variation + levelBoost))
    })
  }, [level])

  return (
    <div className="flex items-center justify-center gap-1 h-16">
      {bars.map((height, i) => (
        <div
          key={i}
          className="waveform-bar w-1.5 bg-gradient-to-t from-violet-600 to-fuchsia-500 rounded-full"
          style={{
            height: `${height * 100}%`,
            opacity: 0.5 + height * 0.5,
          }}
        />
      ))}
    </div>
  )
}

export default WaveformVisualizer
