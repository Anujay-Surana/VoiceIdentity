import { useMemo, useState, useEffect } from 'react'

function WaveformVisualizer({ level = 0 }) {
  const [tick, setTick] = useState(0)
  
  useEffect(() => {
    const interval = setInterval(() => setTick(t => t + 1), 50)
    return () => clearInterval(interval)
  }, [])

  const bars = useMemo(() => {
    const count = 40
    return Array.from({ length: count }, (_, i) => {
      const centerDistance = Math.abs(i - count / 2) / (count / 2)
      const baseHeight = 0.15 + (1 - centerDistance) * 0.25
      const variation = Math.sin(i * 0.4 + tick * 0.2) * 0.15
      const levelBoost = level * (1 - centerDistance * 0.6)
      
      return Math.min(1, Math.max(0.08, baseHeight + variation + levelBoost))
    })
  }, [level, tick])

  return (
    <div className="flex items-center justify-center gap-[3px] h-16">
      {bars.map((height, i) => (
        <div
          key={i}
          className="waveform-bar w-[3px] bg-white rounded-full"
          style={{
            height: `${height * 100}%`,
            opacity: 0.3 + height * 0.5,
          }}
        />
      ))}
    </div>
  )
}

export default WaveformVisualizer
