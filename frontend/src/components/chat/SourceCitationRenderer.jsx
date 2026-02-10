import { useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { TooltipPortal, TooltipContent } from '../ui/Tooltip'

/**
 * Single source citation link with tooltip
 *
 * Displays as a styled link that shows source details on hover
 * and opens the source URL on click.
 */
function SourceCitation({ sourceNumber, source }) {
  const [showTooltip, setShowTooltip] = useState(false)
  const linkRef = useRef(null)

  if (!source) {
    // Source not found, render as plain text
    return <span className="text-charcoal-muted">[Source {sourceNumber}]</span>
  }

  const { title, url, relevance_score } = source
  const percentage = relevance_score ? Math.round(relevance_score * 100) : null

  return (
    <>
      <a
        ref={linkRef}
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 hover:border-emerald-300 transition-colors no-underline"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {sourceNumber}
      </a>

      <TooltipPortal targetRef={linkRef} show={showTooltip} position="top">
        <TooltipContent
          title={title || `Source ${sourceNumber}`}
          description={
            percentage
              ? `${percentage}% relevance. Click to open source article.`
              : 'Click to open source article.'
          }
          position="top"
          color="emerald"
        />
      </TooltipPortal>
    </>
  )
}

/**
 * Parse and render source citations within text
 *
 * Finds patterns like [Source 1], [Source 1, Source 4], [Sources 1-3]
 * and replaces them with clickable SourceCitation components.
 */
function renderTextWithCitations(text, sources) {
  if (!text || typeof text !== 'string') return text

  // Pattern matches any bracketed content starting with "Source" or "Sources"
  // Examples: [Source 1], [Source 1, Source 4], [Sources 1, 2, 3], [Source 1 and Source 2]
  // Captures everything inside the brackets for further parsing
  const citationPattern = /\[(Sources?[^\]]+)\]/gi

  const parts = []
  let lastIndex = 0
  let match

  // Reset lastIndex for global regex
  citationPattern.lastIndex = 0

  while ((match = citationPattern.exec(text)) !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }

    // Parse the source numbers from the full matched content
    const fullContent = match[1]
    const sourceNumbers = extractSourceNumbers(fullContent)

    // Create citation links
    parts.push(
      <span key={match.index} className="inline-flex items-center">
        <span className="text-charcoal-muted">[</span>
        {sourceNumbers.map((num, idx) => (
          <span key={num}>
            {idx > 0 && <span className="text-charcoal-muted">, </span>}
            <SourceCitation
              sourceNumber={num}
              source={sources?.[num - 1]} // Convert 1-indexed to 0-indexed
            />
          </span>
        ))}
        <span className="text-charcoal-muted">]</span>
      </span>
    )

    lastIndex = match.index + match[0].length
  }

  // Add remaining text after last citation
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts.length > 0 ? parts : text
}

/**
 * Extract source numbers from citation content
 * Handles various formats:
 * - "Source 1" → [1]
 * - "Source 1, Source 4" → [1, 4]
 * - "Sources 1, 2, 3" → [1, 2, 3]
 * - "Source 1 and Source 2" → [1, 2]
 * - "Sources 1-3" → [1, 2, 3]
 */
function extractSourceNumbers(str) {
  const numbers = []

  // First, check for ranges like "1-3" or "1–3" (en-dash)
  const rangePattern = /(\d+)\s*[-–]\s*(\d+)/g
  let rangeMatch
  while ((rangeMatch = rangePattern.exec(str)) !== null) {
    const start = parseInt(rangeMatch[1], 10)
    const end = parseInt(rangeMatch[2], 10)
    if (!isNaN(start) && !isNaN(end)) {
      for (let i = start; i <= end; i++) {
        numbers.push(i)
      }
    }
  }

  // Then extract all standalone numbers (not part of a range we already processed)
  // Remove range patterns first to avoid double-counting
  const strWithoutRanges = str.replace(/(\d+)\s*[-–]\s*(\d+)/g, '')
  const numberPattern = /\d+/g
  let numMatch
  while ((numMatch = numberPattern.exec(strWithoutRanges)) !== null) {
    const num = parseInt(numMatch[0], 10)
    if (!isNaN(num)) {
      numbers.push(num)
    }
  }

  return [...new Set(numbers)].sort((a, b) => a - b) // Remove duplicates and sort
}

/**
 * Markdown renderer with clickable source citations
 *
 * Renders markdown content and transforms [Source N] references
 * into styled, clickable links that show source details on hover.
 *
 * @param {string} content - Markdown content to render
 * @param {Array} sources - Array of source objects with title, url, relevance_score
 */
export function SourceCitationRenderer({ content, sources }) {
  return (
    <ReactMarkdown
      components={{
        // Override text rendering to handle citations
        p: ({ children }) => (
          <p>
            {processChildren(children, sources)}
          </p>
        ),
        li: ({ children }) => (
          <li>
            {processChildren(children, sources)}
          </li>
        ),
        // Keep other elements as-is but process their text content
        strong: ({ children }) => (
          <strong>{processChildren(children, sources)}</strong>
        ),
        em: ({ children }) => (
          <em>{processChildren(children, sources)}</em>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

/**
 * Recursively process children to find and transform text with citations
 */
function processChildren(children, sources) {
  if (typeof children === 'string') {
    return renderTextWithCitations(children, sources)
  }

  if (Array.isArray(children)) {
    return children.map((child, index) => {
      if (typeof child === 'string') {
        return <span key={index}>{renderTextWithCitations(child, sources)}</span>
      }
      return child
    })
  }

  return children
}
