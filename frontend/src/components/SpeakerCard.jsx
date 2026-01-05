import { useState } from 'react'

function SpeakerCard({ speaker, onUpdateName, isActive, config }) {
  const [isEditing, setIsEditing] = useState(false)
  const [name, setName] = useState(speaker.name || '')
  const [isExpanded, setIsExpanded] = useState(false)
  const [transcripts, setTranscripts] = useState([])
  const [isLoadingTranscripts, setIsLoadingTranscripts] = useState(false)
  const [transcriptError, setTranscriptError] = useState(null)

  const handleSave = () => {
    onUpdateName(speaker.id, name)
    setIsEditing(false)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSave()
    } else if (e.key === 'Escape') {
      setName(speaker.name || '')
      setIsEditing(false)
    }
  }

  const fetchTranscripts = async () => {
    if (transcripts.length > 0) return // Already loaded
    
    setIsLoadingTranscripts(true)
    setTranscriptError(null)
    
    try {
      const apiUrl = (config?.apiUrl || 'http://localhost:8000').replace(/\/+$/, '')
      const response = await fetch(
        `${apiUrl}/api/v1/speakers/${speaker.id}/transcripts?limit=20`,
        {
          headers: {
            'X-API-Key': config?.apiKey || '',
            'X-User-ID': config?.userId || '',
            'ngrok-skip-browser-warning': 'true',
          },
        }
      )
      
      if (!response.ok) {
        throw new Error('Failed to fetch transcripts')
      }
      
      const data = await response.json()
      setTranscripts(data.transcripts || [])
    } catch (err) {
      setTranscriptError(err.message)
    } finally {
      setIsLoadingTranscripts(false)
    }
  }

  const toggleExpanded = () => {
    const newExpanded = !isExpanded
    setIsExpanded(newExpanded)
    if (newExpanded) {
      fetchTranscripts()
    }
  }

  const formatDuration = (ms) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`
    }
    return `${seconds}s`
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const speakerColors = [
    'from-violet-500 to-purple-600',
    'from-fuchsia-500 to-pink-600',
    'from-rose-500 to-red-600',
    'from-orange-500 to-amber-600',
    'from-emerald-500 to-teal-600',
    'from-cyan-500 to-blue-600',
    'from-indigo-500 to-violet-600',
    'from-pink-500 to-rose-600',
  ]

  // Generate consistent color based on speaker ID
  const colorIndex = speaker.id.charCodeAt(0) % speakerColors.length
  const gradient = speakerColors[colorIndex]

  return (
    <div
      className={`bg-gray-800/50 border rounded-xl p-5 transition-all duration-300 ${
        isActive
          ? 'border-violet-500 shadow-lg shadow-violet-500/20 scale-[1.02]'
          : 'border-gray-700 hover:border-gray-600'
      }`}
    >
      <div className="flex items-start gap-4">
        {/* Avatar */}
        <div
          className={`w-14 h-14 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-xl font-bold text-white shadow-lg ${
            isActive ? 'animate-pulse-glow' : ''
          }`}
          style={{ color: isActive ? '#a78bfa' : undefined }}
        >
          {(speaker.name || 'U')[0].toUpperCase()}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={handleKeyDown}
                onBlur={handleSave}
                autoFocus
                placeholder="Enter name..."
                className="flex-1 px-3 py-1.5 bg-gray-900 border border-violet-500 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-lg truncate">
                {speaker.name || `Unknown Speaker`}
              </h3>
              <button
                onClick={() => setIsEditing(true)}
                className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
                title="Edit name"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
              </button>
              {!speaker.is_identified && (
                <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full">
                  Unidentified
                </span>
              )}
            </div>
          )}

          <p className="text-sm text-gray-500 mt-1 font-mono">
            ID: {speaker.id.slice(0, 8)}...
          </p>

          <div className="flex items-center gap-4 mt-3 text-sm text-gray-400">
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              <span>{speaker.embedding_count || 0} samples</span>
            </div>
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Last seen {formatDate(speaker.last_seen)}</span>
            </div>
          </div>
        </div>

        {/* Active indicator */}
        {isActive && (
          <div className="flex gap-0.5 items-end h-8">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="w-1.5 bg-violet-500 rounded-full animate-pulse"
                style={{
                  height: `${16 + Math.random() * 16}px`,
                  animationDelay: `${i * 0.1}s`,
                }}
              />
            ))}
          </div>
        )}

        {/* Expand button */}
        <button
          onClick={toggleExpanded}
          className="p-2 text-gray-400 hover:text-gray-300 hover:bg-gray-700 rounded-lg transition-colors"
          title={isExpanded ? 'Hide transcripts' : 'View transcripts'}
        >
          <svg
            className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Transcripts Section */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Recent Transcripts
          </h4>

          {isLoadingTranscripts && (
            <div className="flex items-center justify-center py-4">
              <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="ml-2 text-gray-400 text-sm">Loading transcripts...</span>
            </div>
          )}

          {transcriptError && (
            <div className="text-red-400 text-sm py-2">
              Error: {transcriptError}
            </div>
          )}

          {!isLoadingTranscripts && !transcriptError && transcripts.length === 0 && (
            <div className="text-gray-500 text-sm py-2">
              No transcripts yet for this speaker.
            </div>
          )}

          {!isLoadingTranscripts && transcripts.length > 0 && (
            <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
              {transcripts.map((t) => (
                <div
                  key={t.id}
                  className="bg-gray-900/50 rounded-lg p-3 text-sm"
                >
                  <p className="text-gray-200 leading-relaxed">{t.transcript}</p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                    <span>{formatDuration(t.duration_ms)}</span>
                    {t.created_at && (
                      <>
                        <span>•</span>
                        <span>{formatDate(t.created_at)}</span>
                      </>
                    )}
                    <span>•</span>
                    <span>{Math.round(t.confidence * 100)}% conf</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default SpeakerCard
