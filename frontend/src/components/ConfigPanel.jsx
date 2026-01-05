import { useState } from 'react'

function ConfigPanel({ config, setConfig }) {
  const [isOpen, setIsOpen] = useState(false)
  const [tempConfig, setTempConfig] = useState(config)

  const handleSave = () => {
    setConfig(tempConfig)
    setIsOpen(false)
  }

  const isConfigured = config.apiKey && config.userId

  return (
    <div className="relative">
      <button
        onClick={() => {
          setTempConfig(config)
          setIsOpen(!isOpen)
        }}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
          isConfigured
            ? 'bg-green-600/20 text-green-400 border border-green-500/30 hover:bg-green-600/30'
            : 'bg-amber-600/20 text-amber-400 border border-amber-500/30 hover:bg-amber-600/30'
        }`}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        <span className="text-sm font-medium">
          {isConfigured ? 'Connected' : 'Configure'}
        </span>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          
          {/* Panel */}
          <div className="absolute right-0 top-full mt-2 w-96 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl z-50 animate-fade-in">
            <div className="p-4 border-b border-gray-800">
              <h3 className="font-semibold text-lg">API Configuration</h3>
              <p className="text-sm text-gray-500 mt-1">Connect to your Voice Identity backend</p>
            </div>

            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  API URL
                </label>
                <input
                  type="text"
                  value={tempConfig.apiUrl}
                  onChange={(e) => setTempConfig({ ...tempConfig, apiUrl: e.target.value })}
                  placeholder="http://localhost:8000"
                  className="w-full px-4 py-2.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  API Key
                </label>
                <input
                  type="password"
                  value={tempConfig.apiKey}
                  onChange={(e) => setTempConfig({ ...tempConfig, apiKey: e.target.value })}
                  placeholder="Enter your API key"
                  className="w-full px-4 py-2.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors font-mono text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  User ID
                </label>
                <input
                  type="text"
                  value={tempConfig.userId}
                  onChange={(e) => setTempConfig({ ...tempConfig, userId: e.target.value })}
                  placeholder="e.g., user-123"
                  className="w-full px-4 py-2.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors"
                />
              </div>
            </div>

            <div className="p-4 border-t border-gray-800 flex gap-3">
              <button
                onClick={() => setIsOpen(false)}
                className="flex-1 px-4 py-2.5 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-750 transition-colors font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="flex-1 px-4 py-2.5 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors font-medium"
              >
                Save
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default ConfigPanel
