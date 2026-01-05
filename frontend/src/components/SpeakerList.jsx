import SpeakerCard from './SpeakerCard'

function SpeakerList({ speakers, onUpdateName, onRefresh, activeSpeakers = new Set(), config }) {
  if (speakers.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-12 text-center">
        <div className="w-20 h-20 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-6">
          <svg className="w-10 h-10 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        </div>
        <h3 className="text-xl font-semibold text-gray-300 mb-2">No Speakers Yet</h3>
        <p className="text-gray-500 max-w-md mx-auto">
          Upload an audio file or start a live recording to begin identifying speakers.
          Speakers will appear here once detected.
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
          <h2 className="text-xl font-semibold">All Speakers</h2>
          <p className="text-sm text-gray-500 mt-1">
            {identifiedSpeakers.length} identified, {unidentifiedSpeakers.length} unknown
          </p>
        </div>
        <button
          onClick={onRefresh}
          className="p-2 text-gray-400 hover:text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
          title="Refresh"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Identified Speakers */}
      {identifiedSpeakers.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            Identified ({identifiedSpeakers.length})
          </h3>
          <div className="grid gap-4">
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

      {/* Unidentified Speakers */}
      {unidentifiedSpeakers.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 bg-amber-500 rounded-full"></span>
            Unidentified ({unidentifiedSpeakers.length})
          </h3>
          <div className="grid gap-4">
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
