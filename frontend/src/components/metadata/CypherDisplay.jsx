import { useState } from 'react'

/**
 * Copy icon component
 */
function CopyIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
      />
    </svg>
  )
}

/**
 * Check icon for copied state
 */
function CheckIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  )
}

/**
 * Cypher query display component
 *
 * Shows the Cypher query in a monospace code block with copy functionality
 */
export function CypherDisplay({ query }) {
  const [copied, setCopied] = useState(false)

  if (!query) return null

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(query)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback for older browsers
      const textArea = document.createElement('textarea')
      textArea.value = query
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand('copy')
      document.body.removeChild(textArea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 border-b border-gray-200">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Query</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors"
          title={copied ? 'Copied!' : 'Copy query'}
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>

      {/* Query content */}
      <div className="p-3 bg-gray-900 overflow-x-auto">
        <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap break-words">
          {query}
        </pre>
      </div>
    </div>
  )
}
