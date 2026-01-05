import { useState, useRef, useEffect, useCallback } from 'react'
import WaveformVisualizer from './WaveformVisualizer'

// Encode audio samples as WAV format
function encodeWAV(audioBuffers, sampleRate) {
  // Concatenate all buffers
  const totalLength = audioBuffers.reduce((acc, buf) => acc + buf.length, 0)
  const samples = new Float32Array(totalLength)
  let offset = 0
  for (const buffer of audioBuffers) {
    samples.set(buffer, offset)
    offset += buffer.length
  }
  
  // Convert to 16-bit PCM
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  
  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i))
    }
  }
  
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // Subchunk1Size
  view.setUint16(20, 1, true) // AudioFormat (PCM)
  view.setUint16(22, 1, true) // NumChannels (mono)
  view.setUint32(24, sampleRate, true) // SampleRate
  view.setUint32(28, sampleRate * 2, true) // ByteRate
  view.setUint16(32, 2, true) // BlockAlign
  view.setUint16(34, 16, true) // BitsPerSample
  writeString(36, 'data')
  view.setUint32(40, samples.length * 2, true) // Subchunk2Size
  
  // Write samples
  let sampleOffset = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(sampleOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
    sampleOffset += 2
  }
  
  return buffer
}

function LiveRecording({ config, speakers, activeSpeakers, setActiveSpeakers, onNewSpeaker }) {
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording()
    }
  }, [])

  const connectWebSocket = useCallback(() => {
    // Remove trailing slash and convert to WebSocket protocol
    const baseUrl = config.apiUrl.replace(/\/+$/, '').replace('http', 'ws')
    
    // Note: WebSocket doesn't support custom headers directly
    // For ngrok, user must visit the URL in browser first to bypass the warning
    const ws = new WebSocket(
      `${baseUrl}/api/v1/stream?x_api_key=${encodeURIComponent(config.apiKey)}&x_user_id=${encodeURIComponent(config.userId)}`
    )

    ws.onopen = () => {
      setIsConnected(true)
      setError(null)
      // Start a new conversation
      ws.send(JSON.stringify({ action: 'start' }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'conversation_started') {
        setConversationId(data.conversation_id)
      } else if (data.type === 'identification') {
        // Update segments
        setSegments(prev => [...prev, ...data.segments])
        
        // Update active speakers
        const newActiveSpeakers = new Set(data.segments.map(s => s.speaker_id))
        setActiveSpeakers(newActiveSpeakers)
        
        // Clear active speakers after a delay
        setTimeout(() => {
          setActiveSpeakers(new Set())
        }, 2000)
        
        // Check for new speakers
        const hasNewSpeaker = data.segments.some(s => s.is_new_speaker)
        if (hasNewSpeaker) {
          onNewSpeaker?.()
        }
      } else if (data.type === 'conversation_ended') {
        setConversationId(null)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('Connection error')
      setIsConnected(false)
    }

    ws.onclose = () => {
      setIsConnected(false)
    }

    wsRef.current = ws
    return ws
  }, [config, setActiveSpeakers, onNewSpeaker])

  const startRecording = async () => {
    try {
      setError(null)
      setSegments([])
      
      // Check if we're in a secure context (HTTPS required for microphone)
      if (!window.isSecureContext) {
        throw new Error('Microphone requires HTTPS. Please use https:// URL.')
      }
      
      // Check if mediaDevices is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Your browser does not support microphone access. Try Chrome or Safari.')
      }
      
      // Connect WebSocket first
      const ws = connectWebSocket()
      
      // Get microphone access with specific constraints for mobile
      let stream
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
          } 
        })
      } catch (micError) {
        // Try simpler constraints if the first attempt fails
        console.warn('Trying simpler audio constraints:', micError)
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      }
      streamRef.current = stream

      // Set up audio analysis for visualization
      // Must resume AudioContext on mobile (requires user gesture)
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }
      
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      
      const source = audioContextRef.current.createMediaStreamSource(stream)
      source.connect(analyserRef.current)

      // Start audio level monitoring
      const updateAudioLevel = () => {
        if (!analyserRef.current) return
        
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
        analyserRef.current.getByteFrequencyData(dataArray)
        
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length
        setAudioLevel(average / 255)
        
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
      }
      updateAudioLevel()

      // Use MediaRecorder for better mobile compatibility
      // Try different MIME types for cross-browser support
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
        'audio/ogg;codecs=opus',
        'audio/wav',
      ]
      
      let selectedMimeType = ''
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType
          break
        }
      }
      
      console.log('Using MIME type:', selectedMimeType || 'default')
      
      const mediaRecorder = new MediaRecorder(stream, selectedMimeType ? { mimeType: selectedMimeType } : {})
      
      // Collect chunks and send every 3 seconds
      let chunks = []
      
      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }
      
      // Send audio every 3 seconds
      const sendInterval = setInterval(async () => {
        if (chunks.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          const blob = new Blob(chunks, { type: selectedMimeType || 'audio/webm' })
          chunks = []
          
          // Convert to WAV for the backend
          try {
            const arrayBuffer = await blob.arrayBuffer()
            const audioContext = new (window.AudioContext || window.webkitAudioContext)()
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
            
            // Convert to WAV
            const wavBuffer = encodeWAV([audioBuffer.getChannelData(0)], audioBuffer.sampleRate)
            wsRef.current.send(wavBuffer)
            
            audioContext.close()
          } catch (err) {
            console.error('Error converting audio:', err)
            // Try sending raw blob as fallback
            const arrayBuffer = await blob.arrayBuffer()
            wsRef.current.send(arrayBuffer)
          }
        }
      }, 3000)
      
      mediaRecorder.start(1000) // Collect data every second
      
      // Store for cleanup
      mediaRecorderRef.current = { mediaRecorder, sendInterval, source }
      
      setIsRecording(true)
    } catch (err) {
      console.error('Failed to start recording:', err)
      setError(err.message || 'Failed to access microphone')
    }
  }

  const stopRecording = () => {
    // Stop MediaRecorder
    if (mediaRecorderRef.current?.mediaRecorder) {
      mediaRecorderRef.current.mediaRecorder.stop()
    }
    if (mediaRecorderRef.current?.sendInterval) {
      clearInterval(mediaRecorderRef.current.sendInterval)
    }
    if (mediaRecorderRef.current?.source) {
      mediaRecorderRef.current.source.disconnect()
    }
    mediaRecorderRef.current = null

    // Stop audio analysis
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
    }
    audioContextRef.current = null
    analyserRef.current = null

    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
    }
    streamRef.current = null

    // End conversation and close WebSocket
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
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      {/* Recording Controls */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8">
        <div className="text-center">
          {/* Record Button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto transition-all duration-300 ${
              isRecording
                ? 'bg-red-600 hover:bg-red-500 shadow-lg shadow-red-500/30 scale-110'
                : 'bg-violet-600 hover:bg-violet-500 shadow-lg shadow-violet-500/30 hover:scale-105'
            }`}
          >
            {isRecording ? (
              <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            )}
          </button>

          <p className="mt-4 text-lg font-medium">
            {isRecording ? 'Recording...' : 'Click to Start Recording'}
          </p>
          
          {/* Status */}
          <div className="flex items-center justify-center gap-4 mt-2 text-sm">
            <div className={`flex items-center gap-1.5 ${isConnected ? 'text-green-400' : 'text-gray-500'}`}>
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-600'}`}></span>
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
            {conversationId && (
              <div className="text-gray-500">
                Conversation: {conversationId.slice(0, 8)}...
              </div>
            )}
          </div>
        </div>

        {/* Audio Level Visualization */}
        {isRecording && (
          <div className="mt-8">
            <WaveformVisualizer level={audioLevel} />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-center">
            {error}
          </div>
        )}
      </div>

      {/* Live Segments */}
      {segments.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-800">
            <h3 className="font-semibold text-lg">Live Identifications</h3>
            <p className="text-sm text-gray-500 mt-1">
              {segments.length} segment{segments.length !== 1 ? 's' : ''} detected
            </p>
          </div>

          <div className="px-6 py-4 max-h-96 overflow-y-auto">
            <div className="space-y-2">
              {segments.map((segment, index) => {
                const speaker = speakers.find(s => s.id === segment.speaker_id)
                return (
                  <div
                    key={index}
                    className={`flex items-center gap-4 p-3 rounded-lg animate-fade-in ${
                      segment.is_new_speaker ? 'bg-amber-500/10 border border-amber-500/30' : 'bg-gray-800/50'
                    }`}
                  >
                    <div className="w-10 h-10 bg-violet-600 rounded-full flex items-center justify-center text-white font-bold">
                      {(speaker?.name || segment.speaker_name || 'U')[0].toUpperCase()}
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">
                        {speaker?.name || segment.speaker_name || `Speaker ${segment.speaker_id.slice(0, 4)}`}
                        {segment.is_new_speaker && (
                          <span className="ml-2 px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full">
                            New Speaker
                          </span>
                        )}
                      </p>
                      <p className="text-sm text-gray-500">
                        {formatTime(segment.start_ms)} - {formatTime(segment.end_ms)}
                      </p>
                    </div>
                    <div className="text-sm text-gray-500">
                      {Math.round(segment.confidence * 100)}% confidence
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      {!isRecording && segments.length === 0 && (
        <div className="bg-gray-800/30 border border-gray-800 rounded-2xl p-6">
          <h4 className="font-medium text-gray-300 mb-3">How it works</h4>
          <ol className="space-y-2 text-sm text-gray-500">
            <li className="flex items-start gap-2">
              <span className="w-5 h-5 bg-violet-600 rounded-full flex items-center justify-center text-xs text-white flex-shrink-0 mt-0.5">1</span>
              Click the microphone button to start recording
            </li>
            <li className="flex items-start gap-2">
              <span className="w-5 h-5 bg-violet-600 rounded-full flex items-center justify-center text-xs text-white flex-shrink-0 mt-0.5">2</span>
              Speak or have a conversation - audio is streamed in real-time
            </li>
            <li className="flex items-start gap-2">
              <span className="w-5 h-5 bg-violet-600 rounded-full flex items-center justify-center text-xs text-white flex-shrink-0 mt-0.5">3</span>
              Speakers light up in the sidebar when identified
            </li>
            <li className="flex items-start gap-2">
              <span className="w-5 h-5 bg-violet-600 rounded-full flex items-center justify-center text-xs text-white flex-shrink-0 mt-0.5">4</span>
              New speakers are automatically added to your database
            </li>
          </ol>
        </div>
      )}
    </div>
  )
}

export default LiveRecording
