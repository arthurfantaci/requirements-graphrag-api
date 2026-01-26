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
 * Shows the Cypher query in a terminal-style code block with copy functionality
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
    <div className="border border-black/10 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-terminal-bg border-b border-black/10">
        <span className="text-xs font-medium text-terracotta uppercase tracking-widest">Query</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-terminal-text/60 hover:text-terminal-text transition-colors"
          title={copied ? 'Copied!' : 'Copy query'}
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>

      {/* Query content */}
      <div className="p-3 bg-terminal-bg overflow-x-auto">
        <pre className="text-sm text-terminal-green font-mono whitespace-pre-wrap break-words">
          {query}
        </pre>
      </div>
    </div>
  )
}
