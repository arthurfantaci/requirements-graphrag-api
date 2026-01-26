import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Thumbs up icon
 */
function ThumbsUpIcon({ filled }) {
  return (
    <svg
      className="w-4 h-4"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"
      />
    </svg>
  )
}

/**
 * Thumbs down icon
 */
function ThumbsDownIcon({ filled }) {
  return (
    <svg
      className="w-4 h-4"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5"
      />
    </svg>
  )
}

/**
 * Feedback bar component with thumbs up/down
 *
 * Sends feedback to the backend which forwards to LangSmith
 */
export function FeedbackBar({ runId, messageId, disabled }) {
  const [feedback, setFeedback] = useState(null) // 'positive' | 'negative' | null
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showThanks, setShowThanks] = useState(false)

  const submitFeedback = async (score) => {
    if (!runId || isSubmitting) return

    const isPositive = score === 1
    setFeedback(isPositive ? 'positive' : 'negative')
    setIsSubmitting(true)

    try {
      const response = await fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_id: runId,
          score: score, // 1 for positive, 0 for negative
          message_id: messageId,
        }),
      })

      if (response.ok) {
        setShowThanks(true)
        setTimeout(() => setShowThanks(false), 3000)
      }
    } catch (error) {
      console.error('Failed to submit feedback:', error)
      // Keep the UI state even if submission fails
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleThumbsUp = () => {
    if (feedback !== 'positive') {
      submitFeedback(1)
    }
  }

  const handleThumbsDown = () => {
    if (feedback !== 'negative') {
      submitFeedback(0)
    }
  }

  // Don't render if no runId (can't correlate feedback)
  if (!runId) return null

  return (
    <div className="flex items-center gap-2 pt-2 border-t border-black/5">
      {/* Feedback buttons */}
      <div className="flex items-center gap-1">
        <button
          onClick={handleThumbsUp}
          disabled={disabled || isSubmitting}
          className={`p-1.5 rounded transition-colors ${
            feedback === 'positive'
              ? 'text-green-600 bg-green-50'
              : 'text-charcoal-muted hover:text-green-600 hover:bg-green-50'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          title="Helpful"
          aria-label="Mark as helpful"
        >
          <ThumbsUpIcon filled={feedback === 'positive'} />
        </button>
        <button
          onClick={handleThumbsDown}
          disabled={disabled || isSubmitting}
          className={`p-1.5 rounded transition-colors ${
            feedback === 'negative'
              ? 'text-red-600 bg-red-50'
              : 'text-charcoal-muted hover:text-red-600 hover:bg-red-50'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          title="Not helpful"
          aria-label="Mark as not helpful"
        >
          <ThumbsDownIcon filled={feedback === 'negative'} />
        </button>
      </div>

      {/* Thank you message */}
      {showThanks && (
        <span className="text-xs text-charcoal-muted animate-fade-in">
          Thanks for your feedback!
        </span>
      )}

      {/* Feedback label when not yet submitted */}
      {!feedback && !showThanks && (
        <span className="text-xs text-charcoal-muted">Was this helpful?</span>
      )}
    </div>
  )
}
