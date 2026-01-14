import { useState, useRef, useEffect, useCallback } from 'react'
import WaveformVisualizer from './WaveformVisualizer'

function LiveRecording({ config, speakers, setActiveSpeakers, onNewSpeaker }) {
  const [isRecording, setIsRecording] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [conversationId, setConversationId] = useState(null)
  const [segments, setSegments] = useState([])
  const [error, setError] = useState(null)
  const [audioLevel, setAudioLevel] = useState(0)

  const wsRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const streamRef = useRef(null)
  const animationFrameRef = useRef(null)

  useEffect(() => {
    return () => {
      stopRecording()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const connectWebSocket = useCallback(() => {
    const baseUrl = config.apiUrl.replace(/\/+$/, '').replace('http', 'ws')
    const ws = new WebSocket(
      `${baseUrl}/api/v1/stream?x_api_key=${encodeURIComponent(config.apiKey)}&x_user_id=${encodeURIComponent(config.userId)}`
    )

    ws.onopen = () => {
      setIsConnected(true)
      setError(null)
      ws.send(JSON.stringify({ action: 'start' }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'conversation_started') {
        setConversationId(data.conversation_id)
      } else if (data.type === 'identification') {
        setSegments(prev => [...prev, ...data.segments])
        const newActiveSpeakers = new Set(data.segments.map(s => s.speaker_id))
        setActiveSpeakers(newActiveSpeakers)
        setTimeout(() => setActiveSpeakers(new Set()), 2000)
        if (data.segments.some(s => s.is_new_speaker)) {
          onNewSpeaker?.()
        }
      } else if (data.type === 'conversation_ended') {
        setConversationId(null)
      }
    }

    ws.onerror = () => {
      setError('Connection failed')
      setIsConnected(false)
    }

    ws.onclose = () => setIsConnected(false)
    wsRef.current = ws
    return ws
  }, [config, setActiveSpeakers, onNewSpeaker])

  const startRecording = async () => {
    try {
      setError(null)
      setSegments([])
      
      if (!window.isSecureContext) {
        throw new Error('HTTPS required for microphone access')
      }
      
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone not supported in this browser')
      }
      
      // Connect WebSocket and wait for it to open
      const ws = connectWebSocket()
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('WebSocket connection timeout')), 10000)
        ws.addEventListener('open', () => {
          clearTimeout(timeout)
          resolve()
        })
        ws.addEventListener('error', () => {
          clearTimeout(timeout)
          reject(new Error('WebSocket connection failed'))
        })
      })
      
      let stream
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          audio: { echoCancellation: true, noiseSuppression: true }
        })
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      }
      streamRef.current = stream

      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }
      
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      
      const source = audioContextRef.current.createMediaStreamSource(stream)
      source.connect(analyserRef.current)

      const updateAudioLevel = () => {
        if (!analyserRef.current) return
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
        analyserRef.current.getByteFrequencyData(dataArray)
        setAudioLevel(dataArray.reduce((a, b) => a + b) / dataArray.length / 255)
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
      }
      updateAudioLevel()

      const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']
      let selectedMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || ''
      
      let isRecordingActive = true
      
      const startNewRecording = () => {
        if (!isRecordingActive || !streamRef.current) return
        
        const recorder = new MediaRecorder(streamRef.current, selectedMimeType ? { mimeType: selectedMimeType } : {})
        const chunks = []
        
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) chunks.push(event.data)
        }
        
        recorder.onstop = async () => {
          console.log('[Recording] Chunk ready:', chunks.length, 'parts, WS state:', wsRef.current?.readyState)
          if (chunks.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
            const blob = new Blob(chunks, { type: selectedMimeType || 'audio/webm' })
            try {
              const buffer = await blob.arrayBuffer()
              console.log('[Recording] Sending', buffer.byteLength, 'bytes')
              wsRef.current.send(buffer)
            } catch (err) {
              console.error('[Recording] Error sending audio:', err)
            }
          } else {
            console.warn('[Recording] Skipped send - WS not open or no chunks')
          }
          if (isRecordingActive) startNewRecording()
        }
        
        recorder.start()
        setTimeout(() => {
          if (recorder.state === 'recording') recorder.stop()
        }, 3000)
      }
      
      startNewRecording()
      
      mediaRecorderRef.current = { 
        stop: () => {
          isRecordingActive = false
        },
        source 
      }
      
      setIsRecording(true)
    } catch (err) {
      setError(err.message || 'Failed to access microphone')
    }
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop?.()
    mediaRecorderRef.current?.source?.disconnect()
    mediaRecorderRef.current = null

    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current)
    audioContextRef.current?.close()
    audioContextRef.current = null
    analyserRef.current = null

    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'end' }))
      wsRef.current.close()
    }
    wsRef.current = null

    setIsRecording(false)
    setIsConnected(false)
    setAudioLevel(0)
    setActiveSpeakers(new Set())
  }

  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000)
    return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      {/* Recording Controls */}
      <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-10">
        <div className="text-center">
          {/* Record Button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto transition-all ${
              isRecording
                ? 'bg-white text-black animate-recording'
                : 'bg-white/10 text-white hover:bg-white/20 hover:scale-105'
            }`}
          >
            {isRecording ? (
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            ) : (
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
            )}
          </button>

          <p className="mt-6 text-base font-medium">
            {isRecording ? 'Recording...' : 'Tap to start'}
          </p>
          
          {/* Status */}
          <div className="flex items-center justify-center gap-4 mt-2 text-sm text-white/40">
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-400' : 'bg-white/20'}`} />
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
            {conversationId && (
              <span className="font-mono text-xs">
                {conversationId.slice(0, 8)}
              </span>
            )}
          </div>
        </div>

        {/* Waveform */}
        {isRecording && (
          <div className="mt-10">
            <WaveformVisualizer level={audioLevel} />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-center text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Segments */}
      {segments.length > 0 && (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-white/5">
            <h3 className="font-medium">Identified Segments</h3>
            <p className="text-sm text-white/40 mt-0.5">{segments.length} detected</p>
          </div>

          <div className="max-h-80 overflow-y-auto">
            {segments.map((segment, index) => {
              const speaker = speakers.find(s => s.id === segment.speaker_id)
              return (
                <div
                  key={index}
                  className={`flex items-center gap-4 px-6 py-4 border-b border-white/5 last:border-0 animate-fade-in ${
                    segment.is_new_speaker ? 'bg-white/[0.03]' : ''
                  }`}
                >
                  <div className="w-10 h-10 bg-white/10 rounded-lg flex items-center justify-center text-sm font-semibold">
                    {(speaker?.name || segment.speaker_name || 'U')[0].toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-medium truncate">
                        {speaker?.name || segment.speaker_name || `Voice ${segment.speaker_id.slice(0, 6)}`}
                      </p>
                      {segment.is_new_speaker && (
                        <span className="px-2 py-0.5 bg-white/10 text-white/60 text-xs rounded-full">
                          New
                        </span>
                      )}
                    </div>
                    {segment.transcript && (
                      <p className="text-sm text-white/50 mt-0.5 truncate">"{segment.transcript}"</p>
                    )}
                  </div>
                  <div className="text-sm text-white/30 font-mono">
                    {formatTime(segment.start_ms)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Instructions */}
      {!isRecording && segments.length === 0 && (
        <div className="grid grid-cols-2 gap-4">
          {[
            { step: '1', text: 'Tap to record' },
            { step: '2', text: 'Speak naturally' },
            { step: '3', text: 'Voices detected' },
            { step: '4', text: 'Auto-saved' },
          ].map((item) => (
            <div key={item.step} className="flex items-center gap-3 p-4 bg-white/[0.02] border border-white/5 rounded-xl">
              <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center text-sm font-medium">
                {item.step}
              </div>
              <span className="text-sm text-white/60">{item.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default LiveRecording
