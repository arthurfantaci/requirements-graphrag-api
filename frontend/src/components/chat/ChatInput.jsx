import { useState } from 'react'

/**
 * Chat input form component
 */
export function ChatInput({ onSend, disabled }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || disabled) return
    onSend(input)
    setInput('')
  }

  return (
    <footer className="bg-ivory-light border-t border-black/10 p-4">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex gap-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about requirements management..."
          className="flex-1 px-4 py-3 border border-black/15 rounded-lg focus:outline-none focus:ring-2 focus:ring-terracotta focus:border-transparent"
          disabled={disabled}
        />
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className="px-6 py-3 bg-terracotta text-white rounded-lg font-medium hover:bg-terracotta-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Send
        </button>
      </form>
    </footer>
  )
}
