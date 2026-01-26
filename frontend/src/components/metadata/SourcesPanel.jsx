import { useState } from 'react'
import { Tooltip } from '../ui/Tooltip'

/**
 * Info icon for tooltip trigger
 */
function InfoIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  )
}

/**
 * Chevron icon component
 */
function ChevronIcon({ isOpen }) {
  return (
    <svg
      className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-90' : ''}`}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
  )
}

/**
 * External link icon
 */
function LinkIcon() {
  return (
    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
      />
    </svg>
  )
}

/**
 * Relevance indicator bar
 */
function RelevanceBar({ score }) {
  if (score === undefined || score === null) return null

  const percentage = Math.round(score * 100)

  return (
    <div className="flex items-center gap-2 text-xs text-charcoal-muted">
      <div className="w-16 h-1.5 bg-emerald-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-emerald-500 rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span>{percentage}%</span>
    </div>
  )
}

/**
 * Single source item
 */
function SourceItem({ source }) {
  const { title, url, relevance_score } = source

  return (
    <div className="flex items-start justify-between gap-2 py-2 border-b border-black/5 last:border-b-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-charcoal-light truncate">{title || 'Untitled Source'}</p>
        <RelevanceBar score={relevance_score} />
      </div>
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 p-1 text-charcoal-muted hover:text-emerald-600 transition-colors"
          title="Open source"
        >
          <LinkIcon />
        </a>
      )}
    </div>
  )
}

/**
 * Collapsible sources panel component
 *
 * Shows a summary header that expands to reveal full source list
 */
export function SourcesPanel({ sources }) {
  const [isOpen, setIsOpen] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="border border-emerald-200 rounded-lg overflow-hidden bg-emerald-50/30">
      {/* Header - always visible */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-emerald-700 hover:bg-emerald-50 transition-colors"
      >
        <ChevronIcon isOpen={isOpen} />
        <span className="font-medium">Sources ({sources.length})</span>
        <Tooltip
          title="Article Sources"
          description="Source articles used to generate this response. The bar and percentage indicate semantic relevance—how closely each source's content matches your question based on vector similarity."
          color="emerald"
          position="top"
        >
          <span className="text-charcoal-muted hover:text-emerald-600 transition-colors">
            <InfoIcon />
          </span>
        </Tooltip>
      </button>

      {/* Expandable content */}
      {isOpen && (
        <div className="px-3 pb-2 border-t border-emerald-100 bg-ivory-light">
          {sources.map((source, index) => (
            <SourceItem key={`source-${index}`} source={source} />
          ))}
        </div>
      )}
    </div>
  )
}
