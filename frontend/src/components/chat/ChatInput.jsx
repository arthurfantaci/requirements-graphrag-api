import { useRef, useState } from 'react'

const CHAR_LIMIT = 500
const CHAR_WARNING = 400

/**
 * Chat input form component
 *
 * - ↑/↓ cycles through this session's submitted queries (held in a ref; not persisted).
 * - Character counter appears while typing; turns amber at CHAR_WARNING. No hard cap.
 */
export function ChatInput({ onSend, disabled }) {
  const [input, setInput] = useState('')
  const historyRef = useRef([])
  // -1 = not navigating history (live edit). Otherwise index into historyRef.current.
  const cursorRef = useRef(-1)

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || disabled) return
    historyRef.current.push(trimmed)
    cursorRef.current = -1
    onSend(input)
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowUp') {
      const history = historyRef.current
      if (history.length === 0) return
      e.preventDefault()
      const nextCursor =
        cursorRef.current === -1
          ? history.length - 1
          : Math.max(0, cursorRef.current - 1)
      cursorRef.current = nextCursor
      setInput(history[nextCursor])
    } else if (e.key === 'ArrowDown') {
      if (cursorRef.current === -1) return
      e.preventDefault()
      const history = historyRef.current
      const nextCursor = cursorRef.current + 1
      if (nextCursor >= history.length) {
        cursorRef.current = -1
        setInput('')
      } else {
        cursorRef.current = nextCursor
        setInput(history[nextCursor])
      }
    }
  }

  const handleChange = (e) => {
    // Editing exits history-navigation mode so further ↑ starts fresh from the newest entry.
    if (cursorRef.current !== -1) cursorRef.current = -1
    setInput(e.target.value)
  }

  return (
    <footer className="bg-ivory-light border-t border-black/10 p-4">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex items-center gap-4">
        <input
          type="text"
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about requirements management..."
          aria-label="Ask a question about requirements management"
          className="flex-1 px-4 py-3 border border-black/15 rounded-lg focus:outline-none focus:ring-2 focus:ring-terracotta focus:border-transparent"
          disabled={disabled}
        />
        {input.length > 0 && (
          <span
            className={`text-xs tabular-nums ${
              input.length >= CHAR_WARNING ? 'text-amber-600' : 'text-charcoal-muted'
            }`}
            aria-live="polite"
          >
            {input.length} / {CHAR_LIMIT}
          </span>
        )}
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className="w-10 h-10 flex items-center justify-center bg-terracotta text-white rounded-full hover:bg-terracotta-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex-shrink-0 self-center"
          aria-label="Send"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
        </button>
      </form>
    </footer>
  )
}
