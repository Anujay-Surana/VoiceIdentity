import { useState, useEffect, useCallback } from 'react'
import ConfigPanel from './components/ConfigPanel'
import AudioUpload from './components/AudioUpload'
import LiveRecording from './components/LiveRecording'
import SpeakerList from './components/SpeakerList'

function App() {
  const [activeTab, setActiveTab] = useState('live')
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

  const fetchSpeakers = useCallback(async () => {
    if (!config.apiKey || !config.userId) return
    try {
      const response = await fetch(`${config.apiUrl.replace(/\/+$/, '')}/api/v1/speakers`, {
        headers: {
          'X-API-Key': config.apiKey,
          'X-User-Id': config.userId,
          'ngrok-skip-browser-warning': 'true',
        },
      })
      if (response.ok) {
        const data = await response.json()
        setSpeakers(data.speakers || [])
      }
    } catch (error) {
      console.error('Failed to fetch speakers:', error)
    }
  }, [config.apiKey, config.userId, config.apiUrl])

  // Fetch speakers when config changes
  useEffect(() => {
    let cancelled = false
    const doFetch = async () => {
      if (config.apiKey && config.userId && !cancelled) {
        await fetchSpeakers()
      }
    }
    doFetch()
    return () => { cancelled = true }
  }, [config.apiKey, config.userId, fetchSpeakers])

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

  const tabs = [
    { id: 'live', label: 'Live', icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="3" strokeWidth="2"/>
        <path strokeWidth="2" d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
      </svg>
    )},
    { id: 'upload', label: 'Upload', icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
      </svg>
    )},
    { id: 'speakers', label: `Voices (${speakers.length})`, icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    )},
  ]

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-white/5 sticky top-0 z-50 glass">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-white flex items-center justify-center">
              <svg className="w-4 h-4 text-black" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
            </div>
            <span className="font-semibold text-lg tracking-tight">Voice Identity</span>
          </div>
          <ConfigPanel config={config} setConfig={setConfig} />
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        {!isConfigured ? (
          <div className="text-center py-24 animate-fade-in">
            <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mx-auto mb-6">
              <svg className="w-8 h-8 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h2 className="text-xl font-medium text-white mb-2">Connect Your API</h2>
            <p className="text-white/40 max-w-md mx-auto">
              Enter your API credentials in the settings to start identifying voices
            </p>
          </div>
        ) : (
          <div className="animate-fade-in">
            {/* Tabs */}
            <div className="flex items-center gap-1 p-1 bg-white/5 rounded-xl w-fit mb-8 border border-white/5">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === tab.id
                      ? 'bg-white text-black'
                      : 'text-white/60 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              {/* Main Content */}
              <div className="lg:col-span-8">
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

              {/* Sidebar */}
              <div className="lg:col-span-4">
                <div className="bg-white/[0.02] rounded-2xl border border-white/5 p-5 sticky top-24">
                  <div className="flex items-center gap-2 mb-4">
                    <div className={`w-2 h-2 rounded-full ${speakers.length > 0 ? 'bg-white animate-pulse-subtle' : 'bg-white/20'}`} />
                    <span className="text-sm font-medium text-white/60">Detected Voices</span>
                  </div>
                  
                  {speakers.length === 0 ? (
                    <p className="text-sm text-white/30">No voices detected yet</p>
                  ) : (
                    <div className="space-y-2">
                      {speakers.slice(0, 6).map((speaker) => (
                        <div
                          key={speaker.id}
                          className={`p-3 rounded-xl transition-all duration-300 ${
                            activeSpeakers.has(speaker.id)
                              ? 'bg-white/10 border border-white/20'
                              : 'bg-white/[0.02] border border-transparent hover:bg-white/5'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <div
                              className={`w-9 h-9 rounded-lg flex items-center justify-center text-sm font-semibold transition-all ${
                                activeSpeakers.has(speaker.id)
                                  ? 'bg-white text-black'
                                  : 'bg-white/10 text-white/60'
                              }`}
                            >
                              {(speaker.name || 'U')[0].toUpperCase()}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">
                                {speaker.name || `Voice ${speaker.id.slice(0, 6)}`}
                              </p>
                              <p className="text-xs text-white/30">
                                {speaker.embedding_count || 0} samples
                              </p>
                            </div>
                            {activeSpeakers.has(speaker.id) && (
                              <div className="flex gap-0.5 items-end h-5">
                                {[...Array(3)].map((_, i) => (
                                  <div
                                    key={i}
                                    className="w-0.5 bg-white rounded-full animate-pulse"
                                    style={{
                                      height: `${8 + Math.random() * 12}px`,
                                      animationDelay: `${i * 0.15}s`,
                                    }}
                                  />
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      {speakers.length > 6 && (
                        <button
                          onClick={() => setActiveTab('speakers')}
                          className="w-full py-2 text-sm text-white/40 hover:text-white/60 transition-colors"
                        >
                          View all {speakers.length} voices →
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
