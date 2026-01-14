import { useState, useRef } from 'react'

function AudioUpload({ config, onProcessed }) {
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const audioRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => setIsDragging(false)

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file?.type.startsWith('audio/')) processFile(file)
    else setError('Please drop an audio file')
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) processFile(file)
  }

  const processFile = async (file) => {
    setIsProcessing(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${config.apiUrl}/api/v1/audio/process`, {
        method: 'POST',
        headers: {
          'X-API-Key': config.apiKey,
          'X-User-Id': config.userId,
        },
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Processing failed')
      }

      const data = await response.json()
      setResult(data)
      
      if (audioRef.current) {
        audioRef.current.src = URL.createObjectURL(file)
      }
      
      onProcessed?.()
    } catch (err) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000)
    return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-white/40 bg-white/5'
            : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        {isProcessing ? (
          <div className="space-y-4">
            <div className="w-12 h-12 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto" />
            <p className="text-white/60">Processing audio...</p>
            <p className="text-sm text-white/30">Running diarization and identification</p>
          </div>
        ) : (
          <>
            <div className="w-12 h-12 bg-white/5 rounded-xl flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            </div>
            <p className="text-base font-medium text-white/80 mb-1">
              Drop audio or click to upload
            </p>
            <p className="text-sm text-white/30">
              WAV, MP3, M4A supported
            </p>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
            <div>
              <h3 className="font-medium">Results</h3>
              <p className="text-sm text-white/40 mt-0.5">
                {result.total_speakers} voice{result.total_speakers !== 1 ? 's' : ''} detected
              </p>
            </div>
            <span className="px-3 py-1 bg-white/5 text-white/60 rounded-full text-sm">
              {result.segments.length} segments
            </span>
          </div>

          {/* Player */}
          <div className="px-6 py-4 bg-white/[0.02]">
            <audio ref={audioRef} controls className="w-full h-10 opacity-70" />
          </div>

          {/* Timeline */}
          <div className="px-6 py-4">
            <div className="relative h-8 bg-white/5 rounded-lg overflow-hidden">
              {result.segments.map((segment) => {
                const totalDuration = result.segments[result.segments.length - 1].end_ms
                const left = (segment.start_ms / totalDuration) * 100
                const width = ((segment.end_ms - segment.start_ms) / totalDuration) * 100
                const opacity = 0.3 + (segment.confidence || 0.5) * 0.5

                return (
                  <div
                    key={segment.segment_id}
                    className="absolute h-full bg-white cursor-pointer hover:opacity-100 transition-opacity"
                    style={{ left: `${left}%`, width: `${width}%`, opacity }}
                    onClick={() => {
                      if (audioRef.current) {
                        audioRef.current.currentTime = segment.start_ms / 1000
                        audioRef.current.play()
                      }
                    }}
                  />
                )
              })}
            </div>
          </div>

          {/* Segments */}
          <div className="max-h-64 overflow-y-auto">
            {result.segments.map((segment) => (
              <div
                key={segment.segment_id}
                className="flex items-center gap-4 px-6 py-3 border-b border-white/5 last:border-0 hover:bg-white/[0.02] cursor-pointer transition-colors"
                onClick={() => {
                  if (audioRef.current) {
                    audioRef.current.currentTime = segment.start_ms / 1000
                    audioRef.current.play()
                  }
                }}
              >
                <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center text-sm font-medium">
                  {(segment.speaker_name || 'U')[0].toUpperCase()}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium">
                    {segment.speaker_name || `Voice ${segment.speaker_id.slice(0, 6)}`}
                    {segment.is_new_speaker && (
                      <span className="ml-2 px-2 py-0.5 bg-white/5 text-white/40 text-xs rounded-full">New</span>
                    )}
                  </p>
                </div>
                <span className="text-xs text-white/30 font-mono">
                  {formatTime(segment.start_ms)} - {formatTime(segment.end_ms)}
                </span>
              </div>
            ))}
          </div>

          {/* Speakers */}
          <div className="px-6 py-4 border-t border-white/5">
            <div className="flex flex-wrap gap-2">
              {[...new Set(result.segments.map(s => s.speaker_id))].map((speakerId) => {
                const segment = result.segments.find(s => s.speaker_id === speakerId)
                return (
                  <div key={speakerId} className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full">
                    <div className="w-2 h-2 rounded-full bg-white/40" />
                    <span className="text-sm text-white/60">
                      {segment.speaker_name || `Voice ${speakerId.slice(0, 6)}`}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AudioUpload
