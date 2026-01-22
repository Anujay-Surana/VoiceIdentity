import { useState, useRef, useEffect, useCallback } from 'react'
import WaveformVisualizer from './WaveformVisualizer'

function AudioTestPanel({ config }) {
  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)
  const [recordingDuration, setRecordingDuration] = useState(0)
  
  // Saved recordings
  const [recordings, setRecordings] = useState([])
  const [selectedRecording, setSelectedRecording] = useState(null)
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false)
  const [preprocessingResult, setPreprocessingResult] = useState(null)
  const [embeddingResult, setEmbeddingResult] = useState(null)
  const [error, setError] = useState(null)
  
  // Options
  const [enablePreprocessing, setEnablePreprocessing] = useState(true)
  const [enableVad, setEnableVad] = useState(false)
  
  // Playback
  const [isPlaying, setIsPlaying] = useState(false)
  const [playingType, setPlayingType] = useState(null) // 'original' | 'processed'
  
  // Refs
  const mediaRecorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const streamRef = useRef(null)
  const animationFrameRef = useRef(null)
  const recordingIntervalRef = useRef(null)
  const audioElementRef = useRef(null)
  const chunksRef = useRef([])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording()
      if (audioElementRef.current) {
        audioElementRef.current.pause()
      }
    }
  }, [])

  const startRecording = async () => {
    try {
      setError(null)
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
      })
      streamRef.current = stream

      // Setup audio context for visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }
      
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      
      const source = audioContextRef.current.createMediaStreamSource(stream)
      source.connect(analyserRef.current)

      // Audio level visualization
      const updateAudioLevel = () => {
        if (!analyserRef.current) return
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
        analyserRef.current.getByteFrequencyData(dataArray)
        setAudioLevel(dataArray.reduce((a, b) => a + b) / dataArray.length / 255)
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
      }
      updateAudioLevel()

      // Setup MediaRecorder
      const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']
      const selectedMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || ''
      
      chunksRef.current = []
      const recorder = new MediaRecorder(stream, selectedMimeType ? { mimeType: selectedMimeType } : {})
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }
      
      recorder.onstop = () => {
        if (chunksRef.current.length > 0) {
          const blob = new Blob(chunksRef.current, { type: selectedMimeType || 'audio/webm' })
          const url = URL.createObjectURL(blob)
          const newRecording = {
            id: Date.now(),
            blob,
            url,
            duration: recordingDuration,
            timestamp: new Date().toISOString(),
            name: `Recording ${recordings.length + 1}`,
          }
          setRecordings(prev => [newRecording, ...prev])
          setSelectedRecording(newRecording)
        }
      }
      
      mediaRecorderRef.current = recorder
      recorder.start(100) // Collect data every 100ms
      
      // Duration timer
      setRecordingDuration(0)
      recordingIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 0.1)
      }, 100)
      
      setIsRecording(true)
    } catch (err) {
      setError(err.message || 'Failed to access microphone')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    mediaRecorderRef.current = null

    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current)
    if (recordingIntervalRef.current) clearInterval(recordingIntervalRef.current)
    
    audioContextRef.current?.close()
    audioContextRef.current = null
    analyserRef.current = null

    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null

    setIsRecording(false)
    setAudioLevel(0)
  }

  const deleteRecording = (id) => {
    const recording = recordings.find(r => r.id === id)
    if (recording) {
      URL.revokeObjectURL(recording.url)
    }
    setRecordings(prev => prev.filter(r => r.id !== id))
    if (selectedRecording?.id === id) {
      setSelectedRecording(null)
      setPreprocessingResult(null)
      setEmbeddingResult(null)
    }
  }

  const playAudio = useCallback((type) => {
    if (!audioElementRef.current) {
      audioElementRef.current = new Audio()
    }
    
    const audio = audioElementRef.current
    
    // Stop if already playing
    if (isPlaying && playingType === type) {
      audio.pause()
      audio.currentTime = 0
      setIsPlaying(false)
      setPlayingType(null)
      return
    }
    
    // Get the right source
    let src
    if (type === 'original' && selectedRecording) {
      src = selectedRecording.url
    } else if (type === 'processed' && preprocessingResult?.processed_audio_base64) {
      src = `data:audio/wav;base64,${preprocessingResult.processed_audio_base64}`
    } else if (type === 'original-processed' && preprocessingResult?.original_audio_base64) {
      src = `data:audio/wav;base64,${preprocessingResult.original_audio_base64}`
    }
    
    if (!src) return
    
    audio.src = src
    audio.onended = () => {
      setIsPlaying(false)
      setPlayingType(null)
    }
    audio.play()
    setIsPlaying(true)
    setPlayingType(type)
  }, [selectedRecording, preprocessingResult, isPlaying, playingType])

  const runPreprocessingTest = async () => {
    if (!selectedRecording) return
    
    setIsProcessing(true)
    setError(null)
    setPreprocessingResult(null)
    
    try {
      const formData = new FormData()
      formData.append('file', selectedRecording.blob, 'recording.webm')
      
      const params = new URLSearchParams({
        enable_preprocessing: enablePreprocessing.toString(),
        enable_vad: enableVad.toString(),
      })
      
      const response = await fetch(
        `${config.apiUrl.replace(/\/+$/, '')}/api/v1/audio/test-preprocessing?${params}`,
        {
          method: 'POST',
          headers: {
            'X-API-Key': config.apiKey,
            'X-User-Id': config.userId,
            'ngrok-skip-browser-warning': 'true',
          },
          body: formData,
        }
      )
      
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Preprocessing failed')
      }
      
      const result = await response.json()
      setPreprocessingResult(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const runEmbeddingTest = async () => {
    if (!selectedRecording) return
    
    setIsProcessing(true)
    setError(null)
    setEmbeddingResult(null)
    
    try {
      const formData = new FormData()
      formData.append('file', selectedRecording.blob, 'recording.webm')
      
      const response = await fetch(
        `${config.apiUrl.replace(/\/+$/, '')}/api/v1/audio/test-embedding`,
        {
          method: 'POST',
          headers: {
            'X-API-Key': config.apiKey,
            'X-User-Id': config.userId,
            'ngrok-skip-browser-warning': 'true',
          },
          body: formData,
        }
      )
      
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Embedding extraction failed')
      }
      
      const result = await response.json()
      setEmbeddingResult(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 10)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`
  }

  const MetricCard = ({ label, value, unit, highlight, good }) => (
    <div className={`p-3 rounded-xl ${highlight ? 'bg-white/10 border border-white/20' : 'bg-white/5'}`}>
      <p className="text-xs text-white/40 mb-1">{label}</p>
      <p className={`text-lg font-semibold ${good === true ? 'text-green-400' : good === false ? 'text-red-400' : 'text-white'}`}>
        {value}
        {unit && <span className="text-sm text-white/40 ml-1">{unit}</span>}
      </p>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Recording Section */}
      <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-6">
          <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          </div>
          <div>
            <h3 className="font-medium">Step 1: Record Audio</h3>
            <p className="text-sm text-white/40">Record a voice sample for testing</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          {/* Record Button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
              isRecording
                ? 'bg-red-500 text-white animate-recording'
                : 'bg-white/10 text-white hover:bg-white/20 hover:scale-105'
            }`}
          >
            {isRecording ? (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="6" />
              </svg>
            )}
          </button>

          <div className="flex-1">
            {isRecording ? (
              <>
                <p className="text-sm text-red-400 font-medium mb-2">Recording...</p>
                <p className="text-2xl font-mono">{formatDuration(recordingDuration)}</p>
                <div className="mt-3">
                  <WaveformVisualizer level={audioLevel} />
                </div>
              </>
            ) : (
              <p className="text-sm text-white/40">Tap the button to start recording</p>
            )}
          </div>
        </div>
      </div>

      {/* Saved Recordings */}
      {recordings.length > 0 && (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium">Step 2: Select Recording</h3>
              <p className="text-sm text-white/40">{recordings.length} recording{recordings.length !== 1 ? 's' : ''} saved</p>
            </div>
          </div>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {recordings.map((recording) => (
              <div
                key={recording.id}
                onClick={() => {
                  setSelectedRecording(recording)
                  setPreprocessingResult(null)
                  setEmbeddingResult(null)
                }}
                className={`flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all ${
                  selectedRecording?.id === recording.id
                    ? 'bg-white/10 border border-white/20'
                    : 'bg-white/[0.02] hover:bg-white/5 border border-transparent'
                }`}
              >
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    if (selectedRecording?.id === recording.id) {
                      playAudio('original')
                    }
                  }}
                  className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center hover:bg-white/20 transition-colors"
                >
                  {isPlaying && playingType === 'original' && selectedRecording?.id === recording.id ? (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                      <rect x="6" y="4" width="4" height="16" rx="1" />
                      <rect x="14" y="4" width="4" height="16" rx="1" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  )}
                </button>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{recording.name}</p>
                  <p className="text-xs text-white/40">
                    {formatDuration(recording.duration)} • {new Date(recording.timestamp).toLocaleTimeString()}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteRecording(recording.id)
                  }}
                  className="p-2 text-white/30 hover:text-red-400 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Test Options */}
      {selectedRecording && (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium">Step 3: Configure & Run Tests</h3>
              <p className="text-sm text-white/40">Select options and run audio pipeline tests</p>
            </div>
          </div>

          {/* Options */}
          <div className="flex flex-wrap gap-4 mb-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enablePreprocessing}
                onChange={(e) => setEnablePreprocessing(e.target.checked)}
                className="w-4 h-4 rounded border-white/20 bg-white/5 text-white focus:ring-white/20"
              />
              <span className="text-sm">Noise Reduction & Normalization</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enableVad}
                onChange={(e) => setEnableVad(e.target.checked)}
                className="w-4 h-4 rounded border-white/20 bg-white/5 text-white focus:ring-white/20"
              />
              <span className="text-sm">Voice Activity Detection (VAD)</span>
            </label>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3">
            <button
              onClick={runPreprocessingTest}
              disabled={isProcessing}
              className="flex items-center gap-2 px-4 py-2.5 bg-white/10 hover:bg-white/20 rounded-xl transition-all disabled:opacity-50"
            >
              {isProcessing ? (
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" />
                </svg>
              )}
              Test Preprocessing
            </button>
            <button
              onClick={runEmbeddingTest}
              disabled={isProcessing}
              className="flex items-center gap-2 px-4 py-2.5 bg-white/10 hover:bg-white/20 rounded-xl transition-all disabled:opacity-50"
            >
              {isProcessing ? (
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              )}
              Test Voice Matching
            </button>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Preprocessing Results */}
      {preprocessingResult && (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-6">
            <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center">
              <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="font-medium">Preprocessing Results</h3>
          </div>

          {/* Original vs Processed Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Original */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium text-white/60">Original Audio</h4>
                <button
                  onClick={() => playAudio('original-processed')}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-lg text-xs transition-colors"
                >
                  {isPlaying && playingType === 'original-processed' ? (
                    <>
                      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                        <rect x="6" y="4" width="4" height="16" rx="1" />
                        <rect x="14" y="4" width="4" height="16" rx="1" />
                      </svg>
                      Stop
                    </>
                  ) : (
                    <>
                      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                      Play
                    </>
                  )}
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MetricCard 
                  label="Duration" 
                  value={preprocessingResult.original.duration_seconds} 
                  unit="s" 
                />
                <MetricCard 
                  label="RMS Level" 
                  value={preprocessingResult.original.rms_level} 
                />
                <MetricCard 
                  label="Peak Level" 
                  value={preprocessingResult.original.peak_level}
                  good={!preprocessingResult.original.clipping_detected}
                />
                <MetricCard 
                  label="Silent Ratio" 
                  value={`${(preprocessingResult.original.silent_ratio * 100).toFixed(0)}%`}
                />
              </div>
              {preprocessingResult.original.clipping_detected && (
                <p className="text-xs text-red-400 mt-2">⚠️ Clipping detected in original</p>
              )}
            </div>

            {/* Processed */}
            {preprocessingResult.processed && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-white/60">Processed Audio</h4>
                  <button
                    onClick={() => playAudio('processed')}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-lg text-xs transition-colors"
                  >
                    {isPlaying && playingType === 'processed' ? (
                      <>
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                          <rect x="6" y="4" width="4" height="16" rx="1" />
                          <rect x="14" y="4" width="4" height="16" rx="1" />
                        </svg>
                        Stop
                      </>
                    ) : (
                      <>
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M8 5v14l11-7z" />
                        </svg>
                        Play
                      </>
                    )}
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <MetricCard 
                    label="Quality Score" 
                    value={preprocessingResult.processed.quality_score}
                    highlight
                    good={preprocessingResult.processed.quality_score >= 0.5}
                  />
                  <MetricCard 
                    label="Speech Ratio" 
                    value={`${(preprocessingResult.processed.speech_ratio * 100).toFixed(0)}%`}
                    good={preprocessingResult.processed.speech_ratio >= 0.3}
                  />
                  <MetricCard 
                    label="RMS Level" 
                    value={preprocessingResult.processed.rms_level} 
                  />
                  <MetricCard 
                    label="Acceptable" 
                    value={preprocessingResult.processed.is_acceptable ? 'Yes' : 'No'}
                    good={preprocessingResult.processed.is_acceptable}
                  />
                </div>
                {preprocessingResult.processed.clipping_detected && (
                  <p className="text-xs text-yellow-400 mt-2">⚠️ Clipping detected after processing</p>
                )}
              </div>
            )}
          </div>

          {/* Improvements Summary */}
          {preprocessingResult.improvements && (
            <div className="p-4 bg-white/5 rounded-xl">
              <h4 className="text-sm font-medium mb-2">Processing Summary</h4>
              <div className="flex flex-wrap gap-4 text-sm text-white/60">
                <span>
                  RMS Change: <span className={preprocessingResult.improvements.rms_change > 0 ? 'text-green-400' : 'text-white'}>
                    {preprocessingResult.improvements.rms_change > 0 ? '+' : ''}{(preprocessingResult.improvements.rms_change * 100).toFixed(0)}%
                  </span>
                </span>
                {preprocessingResult.improvements.normalized && (
                  <span className="text-blue-400">✓ Normalized</span>
                )}
                {preprocessingResult.improvements.noise_reduction != null && (
                  <span>
                    Noise Reduction: <span className="text-green-400">{(preprocessingResult.improvements.noise_reduction * 100).toFixed(0)}%</span>
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Embedding/Matching Results */}
      {embeddingResult && (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-6">
            <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
              <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h3 className="font-medium">Voice Matching Results</h3>
          </div>

          {/* Embedding Info */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            <MetricCard label="Duration" value={embeddingResult.duration_seconds} unit="s" />
            <MetricCard label="Quality" value={embeddingResult.quality_score} good={embeddingResult.quality_score >= 0.5} />
            <MetricCard label="Dimensions" value={embeddingResult.embedding_dimensions} />
            <MetricCard label="Threshold" value={embeddingResult.threshold_used} />
          </div>

          {/* Match Result */}
          <div className={`p-4 rounded-xl mb-6 ${
            embeddingResult.match_result.is_match 
              ? 'bg-green-500/10 border border-green-500/20' 
              : 'bg-yellow-500/10 border border-yellow-500/20'
          }`}>
            <div className="flex items-center gap-3">
              {embeddingResult.match_result.is_match ? (
                <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              ) : (
                <div className="w-10 h-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
              )}
              <div>
                <p className="font-medium">
                  {embeddingResult.match_result.is_match ? 'Voice Matched!' : 'No Match Found'}
                </p>
                <p className="text-sm text-white/60">
                  {embeddingResult.match_result.is_match 
                    ? `Confidence: ${(embeddingResult.match_result.confidence * 100).toFixed(1)}%`
                    : 'This voice would create a new speaker profile'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Top Candidates */}
          {embeddingResult.top_candidates.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-white/60 mb-3">Top Matching Speakers</h4>
              <div className="space-y-2">
                {embeddingResult.top_candidates.map((candidate, index) => (
                  <div
                    key={candidate.speaker_id}
                    className={`flex items-center gap-3 p-3 rounded-xl ${
                      index === 0 && embeddingResult.match_result.is_match
                        ? 'bg-green-500/10 border border-green-500/20'
                        : 'bg-white/[0.02]'
                    }`}
                  >
                    <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-sm font-semibold">
                      {index + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {candidate.speaker_name || `Speaker ${candidate.speaker_id.slice(0, 8)}`}
                      </p>
                      <p className="text-xs text-white/40 font-mono">{candidate.speaker_id.slice(0, 12)}...</p>
                    </div>
                    <div className="text-right">
                      <p className={`text-sm font-semibold ${
                        candidate.similarity >= embeddingResult.threshold_used ? 'text-green-400' : 'text-white/60'
                      }`}>
                        {(candidate.similarity * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-white/30">similarity</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      {!selectedRecording && recordings.length === 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { step: '1', icon: '🎤', text: 'Record audio sample' },
            { step: '2', icon: '📁', text: 'Select recording' },
            { step: '3', icon: '⚙️', text: 'Configure options' },
            { step: '4', icon: '🔬', text: 'Run tests & compare' },
          ].map((item) => (
            <div key={item.step} className="flex items-center gap-3 p-4 bg-white/[0.02] border border-white/5 rounded-xl">
              <div className="w-10 h-10 bg-white/10 rounded-lg flex items-center justify-center text-lg">
                {item.icon}
              </div>
              <div>
                <p className="text-xs text-white/40">Step {item.step}</p>
                <p className="text-sm">{item.text}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default AudioTestPanel
