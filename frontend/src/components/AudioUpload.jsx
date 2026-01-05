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

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('audio/')) {
      processFile(file)
    } else {
      setError('Please drop an audio file')
    }
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      processFile(file)
    }
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
      
      // Create object URL for audio playback
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
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }

  const speakerColors = [
    'bg-violet-500', 'bg-fuchsia-500', 'bg-pink-500', 'bg-rose-500',
    'bg-orange-500', 'bg-amber-500', 'bg-emerald-500', 'bg-cyan-500',
  ]

  const getSpeakerColor = (speakerId, speakers) => {
    const uniqueSpeakers = [...new Set(speakers.map(s => s.speaker_id))]
    const index = uniqueSpeakers.indexOf(speakerId)
    return speakerColors[index % speakerColors.length]
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-violet-500 bg-violet-500/10'
            : 'border-gray-700 hover:border-gray-600 hover:bg-gray-800/50'
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
            <div className="w-16 h-16 border-4 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-gray-400">Processing audio...</p>
            <p className="text-sm text-gray-500">Running speaker diarization and identification</p>
          </div>
        ) : (
          <>
            <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <p className="text-lg font-medium text-gray-300 mb-2">
              Drop audio file here or click to browse
            </p>
            <p className="text-sm text-gray-500">
              Supports WAV, MP3, M4A, and other audio formats
            </p>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400">
          <p className="font-medium">Error</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-lg">Processing Results</h3>
              <p className="text-sm text-gray-500 mt-1">
                {result.total_speakers} speaker{result.total_speakers !== 1 ? 's' : ''} identified
                {result.new_speakers > 0 && ` (${result.new_speakers} new)`}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 bg-violet-600/20 text-violet-400 rounded-full text-sm font-medium">
                {result.segments.length} segments
              </span>
            </div>
          </div>

          {/* Audio Player */}
          <div className="px-6 py-4 bg-gray-800/50">
            <audio ref={audioRef} controls className="w-full" />
          </div>

          {/* Timeline */}
          <div className="px-6 py-4">
            <div className="relative h-12 bg-gray-800 rounded-lg overflow-hidden">
              {result.segments.map((segment, index) => {
                const totalDuration = result.segments[result.segments.length - 1].end_ms
                const left = (segment.start_ms / totalDuration) * 100
                const width = ((segment.end_ms - segment.start_ms) / totalDuration) * 100

                return (
                  <div
                    key={segment.segment_id}
                    className={`absolute h-full ${getSpeakerColor(segment.speaker_id, result.segments)} opacity-80 hover:opacity-100 cursor-pointer transition-opacity`}
                    style={{ left: `${left}%`, width: `${width}%` }}
                    title={`${segment.speaker_name || `Speaker ${segment.speaker_id.slice(0, 4)}`}: ${formatTime(segment.start_ms)} - ${formatTime(segment.end_ms)}`}
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

          {/* Segments List */}
          <div className="px-6 py-4 border-t border-gray-800">
            <h4 className="font-medium text-gray-400 mb-3">Segments</h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {result.segments.map((segment) => (
                <div
                  key={segment.segment_id}
                  className="flex items-center gap-4 p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800 cursor-pointer transition-colors"
                  onClick={() => {
                    if (audioRef.current) {
                      audioRef.current.currentTime = segment.start_ms / 1000
                      audioRef.current.play()
                    }
                  }}
                >
                  <div className={`w-3 h-3 rounded-full ${getSpeakerColor(segment.speaker_id, result.segments)}`} />
                  <div className="flex-1">
                    <p className="font-medium text-sm">
                      {segment.speaker_name || `Speaker ${segment.speaker_id.slice(0, 4)}`}
                      {segment.is_new_speaker && (
                        <span className="ml-2 px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full">
                          New
                        </span>
                      )}
                    </p>
                  </div>
                  <div className="text-sm text-gray-500 font-mono">
                    {formatTime(segment.start_ms)} - {formatTime(segment.end_ms)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="px-6 py-4 border-t border-gray-800">
            <h4 className="font-medium text-gray-400 mb-3">Speakers</h4>
            <div className="flex flex-wrap gap-3">
              {[...new Set(result.segments.map(s => s.speaker_id))].map((speakerId) => {
                const segment = result.segments.find(s => s.speaker_id === speakerId)
                return (
                  <div key={speakerId} className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-full">
                    <div className={`w-3 h-3 rounded-full ${getSpeakerColor(speakerId, result.segments)}`} />
                    <span className="text-sm">
                      {segment.speaker_name || `Speaker ${speakerId.slice(0, 4)}`}
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
