import { useState } from 'react'
import { createPortal } from 'react-dom'

function ConfigPanel({ config, setConfig }) {
  const [isOpen, setIsOpen] = useState(false)
  const [tempConfig, setTempConfig] = useState(config)

  const handleSave = () => {
    setConfig(tempConfig)
    setIsOpen(false)
  }

  const isConfigured = config.apiKey && config.userId

  const modal = isOpen ? createPortal(
    <div 
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* Backdrop */}
      <div 
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
        }}
        onClick={() => setIsOpen(false)}
      />
      
      {/* Modal */}
      <div 
        style={{
          position: 'relative',
          width: '100%',
          maxWidth: '400px',
          margin: '16px',
          backgroundColor: '#0a0a0a',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '16px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
        }}
      >
        <div style={{ padding: '24px', borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
          <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'white', margin: 0 }}>API Configuration</h3>
          <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.4)', marginTop: '4px' }}>Connect to your Voice Identity backend</p>
        </div>

        <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
              API URL
            </label>
            <input
              type="text"
              value={tempConfig.apiUrl}
              onChange={(e) => setTempConfig({ ...tempConfig, apiUrl: e.target.value })}
              placeholder="https://api.example.com"
              style={{
                width: '100%',
                padding: '12px 16px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                color: 'white',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
              API Key
            </label>
            <input
              type="password"
              value={tempConfig.apiKey}
              onChange={(e) => setTempConfig({ ...tempConfig, apiKey: e.target.value })}
              placeholder="Enter your API key"
              style={{
                width: '100%',
                padding: '12px 16px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                color: 'white',
                fontSize: '14px',
                fontFamily: 'monospace',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
              User ID
            </label>
            <input
              type="text"
              value={tempConfig.userId}
              onChange={(e) => setTempConfig({ ...tempConfig, userId: e.target.value })}
              placeholder="your-user-id"
              style={{
                width: '100%',
                padding: '12px 16px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                color: 'white',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>
        </div>

        <div style={{ 
          padding: '24px', 
          borderTop: '1px solid rgba(255, 255, 255, 0.05)', 
          display: 'flex', 
          gap: '12px' 
        }}>
          <button
            onClick={() => setIsOpen(false)}
            style={{
              flex: 1,
              padding: '12px 16px',
              backgroundColor: 'transparent',
              border: 'none',
              borderRadius: '12px',
              color: 'rgba(255, 255, 255, 0.6)',
              fontSize: '14px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            style={{
              flex: 1,
              padding: '12px 16px',
              backgroundColor: 'white',
              border: 'none',
              borderRadius: '12px',
              color: 'black',
              fontSize: '14px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Save
          </button>
        </div>
      </div>
    </div>,
    document.body
  ) : null

  return (
    <>
      <button
        onClick={() => {
          setTempConfig(config)
          setIsOpen(!isOpen)
        }}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all border ${
          isConfigured
            ? 'bg-white/5 border-white/10 text-white/80 hover:bg-white/10'
            : 'bg-white text-black border-white hover:bg-white/90'
        }`}
      >
        {isConfigured ? (
          <>
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full" />
            Connected
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            Connect
          </>
        )}
      </button>
      {modal}
    </>
  )
}

export default ConfigPanel
