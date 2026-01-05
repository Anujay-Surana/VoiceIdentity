import { useState, useEffect } from 'react'
import ConfigPanel from './components/ConfigPanel'
import AudioUpload from './components/AudioUpload'
import LiveRecording from './components/LiveRecording'
import SpeakerList from './components/SpeakerList'

function App() {
  const [activeTab, setActiveTab] = useState('upload')
  const [config, setConfig] = useState({
    apiKey: localStorage.getItem('voiceId_apiKey') || '',
    userId: localStorage.getItem('voiceId_userId') || '',
    apiUrl: localStorage.getItem('voiceId_apiUrl') || 'http://localhost:8000',
  })
  const [speakers, setSpeakers] = useState([])
  const [activeSpeakers, setActiveSpeakers] = useState(new Set())

  // Save config to localStorage
  useEffect(() => {
    localStorage.setItem('voiceId_apiKey', config.apiKey)
    localStorage.setItem('voiceId_userId', config.userId)
    localStorage.setItem('voiceId_apiUrl', config.apiUrl)
  }, [config])

  // Fetch speakers when config changes
  useEffect(() => {
    if (config.apiKey && config.userId) {
      fetchSpeakers()
    }
  }, [config.apiKey, config.userId])

  const fetchSpeakers = async () => {
    try {
      const response = await fetch(`${config.apiUrl.replace(/\/+$/, '')}/api/v1/speakers`, {
        headers: {
          'X-API-Key': config.apiKey,
          'X-User-Id': config.userId,
          'ngrok-skip-browser-warning': 'true',  // Skip ngrok interstitial
        },
      })
      if (response.ok) {
        const data = await response.json()
        setSpeakers(data.speakers || [])
      }
    } catch (error) {
      console.error('Failed to fetch speakers:', error)
    }
  }

  const updateSpeakerName = async (speakerId, name) => {
    try {
      const response = await fetch(`${config.apiUrl.replace(/\/+$/, '')}/api/v1/speakers/${speakerId}`, {
        method: 'PATCH',
        headers: {
          'X-API-Key': config.apiKey,
          'X-User-Id': config.userId,
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({ name }),
      })
      if (response.ok) {
        fetchSpeakers()
      }
    } catch (error) {
      console.error('Failed to update speaker:', error)
    }
  }

  const isConfigured = config.apiKey && config.userId

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
              Voice Identity
            </h1>
          </div>
          <ConfigPanel config={config} setConfig={setConfig} />
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {!isConfigured ? (
          <div className="text-center py-20">
            <div className="w-20 h-20 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-10 h-10 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h2 className="text-2xl font-semibold text-gray-300 mb-2">Configure API Access</h2>
            <p className="text-gray-500 mb-6">Enter your API key and User ID in the settings above to get started</p>
          </div>
        ) : (
          <>
            {/* Tabs */}
            <div className="flex gap-2 mb-8">
              <button
                onClick={() => setActiveTab('upload')}
                className={`px-5 py-2.5 rounded-lg font-medium transition-all ${
                  activeTab === 'upload'
                    ? 'bg-violet-600 text-white shadow-lg shadow-violet-500/25'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-750 hover:text-gray-300'
                }`}
              >
                Upload Audio
              </button>
              <button
                onClick={() => setActiveTab('live')}
                className={`px-5 py-2.5 rounded-lg font-medium transition-all ${
                  activeTab === 'live'
                    ? 'bg-violet-600 text-white shadow-lg shadow-violet-500/25'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-750 hover:text-gray-300'
                }`}
              >
                Live Recording
              </button>
              <button
                onClick={() => setActiveTab('speakers')}
                className={`px-5 py-2.5 rounded-lg font-medium transition-all ${
                  activeTab === 'speakers'
                    ? 'bg-violet-600 text-white shadow-lg shadow-violet-500/25'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-750 hover:text-gray-300'
                }`}
              >
                Speakers ({speakers.length})
              </button>
            </div>

            {/* Tab Content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                {activeTab === 'upload' && (
                  <AudioUpload 
                    config={config} 
                    onProcessed={fetchSpeakers}
                  />
                )}
                {activeTab === 'live' && (
                  <LiveRecording 
                    config={config}
                    speakers={speakers}
                    activeSpeakers={activeSpeakers}
                    setActiveSpeakers={setActiveSpeakers}
                    onNewSpeaker={fetchSpeakers}
                  />
                )}
                {activeTab === 'speakers' && (
                  <SpeakerList 
                    speakers={speakers}
                    onUpdateName={updateSpeakerName}
                    onRefresh={fetchSpeakers}
                    config={config}
                  />
                )}
              </div>

              {/* Sidebar - Active Speakers */}
              <div className="lg:col-span-1">
                <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6 sticky top-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                    Active Speakers
                  </h3>
                  {speakers.length === 0 ? (
                    <p className="text-gray-500 text-sm">No speakers identified yet</p>
                  ) : (
                    <div className="space-y-3">
                      {speakers.slice(0, 8).map((speaker) => (
                        <div
                          key={speaker.id}
                          className={`p-3 rounded-xl transition-all duration-300 ${
                            activeSpeakers.has(speaker.id)
                              ? 'bg-violet-600/20 border border-violet-500/50 shadow-lg shadow-violet-500/10'
                              : 'bg-gray-800/50 border border-transparent'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <div
                              className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                                activeSpeakers.has(speaker.id)
                                  ? 'bg-violet-500 text-white animate-pulse-glow'
                                  : 'bg-gray-700 text-gray-400'
                              }`}
                              style={{ color: activeSpeakers.has(speaker.id) ? '#a78bfa' : undefined }}
                            >
                              {(speaker.name || 'U')[0].toUpperCase()}
      </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-medium truncate">
                                {speaker.name || `Speaker ${speaker.id.slice(0, 4)}`}
                              </p>
                              <p className="text-xs text-gray-500">
                                {speaker.embedding_count || 0} samples
        </p>
      </div>
                            {activeSpeakers.has(speaker.id) && (
                              <div className="flex gap-0.5">
                                {[...Array(3)].map((_, i) => (
                                  <div
                                    key={i}
                                    className="w-1 bg-violet-500 rounded-full animate-pulse"
                                    style={{
                                      height: `${12 + Math.random() * 12}px`,
                                      animationDelay: `${i * 0.1}s`,
                                    }}
                                  />
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}

export default App
