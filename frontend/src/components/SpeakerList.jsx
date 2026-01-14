import SpeakerCard from './SpeakerCard'

function SpeakerList({ speakers, onUpdateName, onRefresh, activeSpeakers = new Set(), config }) {
  if (speakers.length === 0) {
    return (
      <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-12 text-center">
        <div className="w-14 h-14 bg-white/5 rounded-2xl flex items-center justify-center mx-auto mb-5">
          <svg className="w-7 h-7 text-white/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-white mb-2">No Voices Yet</h3>
        <p className="text-sm text-white/40 max-w-sm mx-auto">
          Start a live recording or upload audio to begin detecting and identifying voices
        </p>
      </div>
    )
  }

  const identifiedSpeakers = speakers.filter(s => s.is_identified)
  const unidentifiedSpeakers = speakers.filter(s => !s.is_identified)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium">All Voices</h2>
          <p className="text-sm text-white/40 mt-0.5">
            {identifiedSpeakers.length} identified · {unidentifiedSpeakers.length} unknown
          </p>
        </div>
        <button
          onClick={onRefresh}
          className="p-2 text-white/40 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Identified */}
      {identifiedSpeakers.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full" />
            <span className="text-xs font-medium text-white/50 uppercase tracking-wide">
              Identified ({identifiedSpeakers.length})
            </span>
          </div>
          <div className="space-y-3">
            {identifiedSpeakers.map((speaker) => (
              <SpeakerCard
                key={speaker.id}
                speaker={speaker}
                onUpdateName={onUpdateName}
                isActive={activeSpeakers.has(speaker.id)}
                config={config}
              />
            ))}
          </div>
        </div>
      )}

      {/* Unidentified */}
      {unidentifiedSpeakers.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="w-1.5 h-1.5 bg-white/30 rounded-full" />
            <span className="text-xs font-medium text-white/50 uppercase tracking-wide">
              Unknown ({unidentifiedSpeakers.length})
            </span>
          </div>
          <div className="space-y-3">
            {unidentifiedSpeakers.map((speaker) => (
              <SpeakerCard
                key={speaker.id}
                speaker={speaker}
                onUpdateName={onUpdateName}
                isActive={activeSpeakers.has(speaker.id)}
                config={config}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default SpeakerList
