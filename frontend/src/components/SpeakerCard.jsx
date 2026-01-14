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
    if (e.key === 'Enter') handleSave()
    else if (e.key === 'Escape') {
      setName(speaker.name || '')
      setIsEditing(false)
    }
  }

  const fetchTranscripts = async () => {
    if (transcripts.length > 0) return
    
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
      
      if (!response.ok) throw new Error('Failed to fetch')
      const data = await response.json()
      setTranscripts(data.transcripts || [])
    } catch (err) {
      setTranscriptError(err.message)
    } finally {
      setIsLoadingTranscripts(false)
    }
  }

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded)
    if (!isExpanded) fetchTranscripts()
  }

  const formatDuration = (ms) => {
    const seconds = Math.floor(ms / 1000)
    if (seconds >= 60) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
    return `${seconds}s`
  }

  const formatDate = (dateString) => {
    if (!dateString) return ''
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  }

  return (
    <div
      className={`bg-white/[0.02] border rounded-xl p-5 transition-all ${
        isActive
          ? 'border-white/30 bg-white/[0.04]'
          : 'border-white/5 hover:border-white/10'
      }`}
    >
      <div className="flex items-start gap-4">
        {/* Avatar */}
        <div
          className={`w-12 h-12 rounded-xl flex items-center justify-center text-lg font-semibold transition-all ${
            isActive ? 'bg-white text-black' : 'bg-white/10 text-white/70'
          }`}
        >
          {(speaker.name || 'U')[0].toUpperCase()}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={handleSave}
              autoFocus
              placeholder="Enter name..."
              className="w-full px-3 py-1.5 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/30 focus:border-white/40"
            />
          ) : (
            <div className="flex items-center gap-2">
              <h3 className="font-medium text-base truncate">
                {speaker.name || `Voice ${speaker.id.slice(0, 6)}`}
              </h3>
              <button
                onClick={() => setIsEditing(true)}
                className="p-1 text-white/30 hover:text-white/60 transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
              </button>
              {!speaker.is_identified && (
                <span className="px-2 py-0.5 bg-white/5 text-white/40 text-xs rounded-full">
                  Unknown
                </span>
              )}
            </div>
          )}

          <p className="text-xs text-white/30 mt-1 font-mono">
            {speaker.id.slice(0, 12)}
          </p>

          <div className="flex items-center gap-4 mt-3 text-xs text-white/40">
            <span>{speaker.embedding_count || 0} samples</span>
            <span>Last: {formatDate(speaker.last_seen)}</span>
          </div>
        </div>

        {/* Active indicator */}
        {isActive && (
          <div className="flex gap-0.5 items-end h-6">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="w-1 bg-white rounded-full animate-pulse"
                style={{ height: `${12 + Math.random() * 12}px`, animationDelay: `${i * 0.1}s` }}
              />
            ))}
          </div>
        )}

        {/* Expand */}
        <button
          onClick={toggleExpanded}
          className="p-2 text-white/30 hover:text-white/60 hover:bg-white/5 rounded-lg transition-all"
        >
          <svg
            className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Transcripts */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-white/5">
          <h4 className="text-xs font-medium text-white/40 mb-3">Recent Transcripts</h4>

          {isLoadingTranscripts && (
            <div className="flex items-center gap-2 py-3">
              <div className="w-4 h-4 border-2 border-white/20 border-t-white/60 rounded-full animate-spin" />
              <span className="text-white/40 text-sm">Loading...</span>
            </div>
          )}

          {transcriptError && (
            <p className="text-red-400/80 text-sm py-2">{transcriptError}</p>
          )}

          {!isLoadingTranscripts && !transcriptError && transcripts.length === 0 && (
            <p className="text-white/30 text-sm py-2">No transcripts yet</p>
          )}

          {!isLoadingTranscripts && transcripts.length > 0 && (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {transcripts.map((t) => (
                <div key={t.id} className="bg-white/[0.03] rounded-lg p-3">
                  <p className="text-sm text-white/70 leading-relaxed">"{t.transcript}"</p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-white/30">
                    <span>{formatDuration(t.duration_ms)}</span>
                    {t.created_at && <span>{formatDate(t.created_at)}</span>}
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
