import { useState } from 'react'
import { FeedbackModal } from './FeedbackModal'
import { apiFetch } from '../../utils/api'

// Copy icon (outline style like Claude Desktop)
function CopyIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  )
}

// Thumbs up icon (outline style)
function ThumbsUpIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
    </svg>
  )
}

// Thumbs down icon (outline style)
function ThumbsDownIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
    </svg>
  )
}

/**
 * Response action bar with Copy, Thumbs Up, Thumbs Down
 * Styled similar to Claude Desktop
 */
export function ResponseActions({ content, runId, messageId }) {
  const [copied, setCopied] = useState(false)
  const [feedbackGiven, setFeedbackGiven] = useState(null) // 'positive' | 'negative' | null
  const [modalOpen, setModalOpen] = useState(false)
  const [pendingFeedback, setPendingFeedback] = useState(null) // 'positive' | 'negative'

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback
      const textArea = document.createElement('textarea')
      textArea.value = content || ''
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand('copy')
      document.body.removeChild(textArea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleThumbsUp = () => {
    if (feedbackGiven) return
    setPendingFeedback('positive')
    setModalOpen(true)
  }

  const handleThumbsDown = () => {
    if (feedbackGiven) return
    setPendingFeedback('negative')
    setModalOpen(true)
  }

  const submitFeedback = async (details) => {
    setModalOpen(false)
    const isPositive = pendingFeedback === 'positive'
    setFeedbackGiven(pendingFeedback)

    if (!runId) return

    try {
      await apiFetch('/feedback', {
        method: 'POST',
        body: JSON.stringify({
          run_id: runId,
          score: isPositive ? 1.0 : 0.0,
          comment: details || null, // Goes to feedback_notes in LangSmith
          message_id: messageId,
          category: isPositive ? 'positive' : 'negative',
        }),
      })
    } catch (error) {
      console.error('Failed to submit feedback:', error)
    }
  }

  const cancelFeedback = () => {
    setModalOpen(false)
    setPendingFeedback(null)
  }

  return (
    <>
      <div className="flex items-center gap-1 pt-2 border-t border-black/5">
        {/* Copy button */}
        <button
          onClick={handleCopy}
          className="p-1.5 text-charcoal-muted hover:text-charcoal-light rounded transition-colors"
          title={copied ? 'Copied!' : 'Copy'}
        >
          {copied ? (
            <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <CopyIcon />
          )}
        </button>

        {/* Thumbs up */}
        <button
          onClick={handleThumbsUp}
          disabled={feedbackGiven !== null}
          className={`p-1.5 rounded transition-colors ${
            feedbackGiven === 'positive'
              ? 'text-green-600'
              : feedbackGiven
              ? 'text-charcoal-muted/60 cursor-not-allowed'
              : 'text-charcoal-muted hover:text-charcoal-light'
          }`}
          title="Good response"
        >
          <ThumbsUpIcon />
        </button>

        {/* Thumbs down */}
        <button
          onClick={handleThumbsDown}
          disabled={feedbackGiven !== null}
          className={`p-1.5 rounded transition-colors ${
            feedbackGiven === 'negative'
              ? 'text-red-600'
              : feedbackGiven
              ? 'text-charcoal-muted/60 cursor-not-allowed'
              : 'text-charcoal-muted hover:text-charcoal-light'
          }`}
          title="Bad response"
        >
          <ThumbsDownIcon />
        </button>
      </div>

      {/* Feedback Modal */}
      <FeedbackModal
        isOpen={modalOpen}
        isPositive={pendingFeedback === 'positive'}
        onSubmit={submitFeedback}
        onCancel={cancelFeedback}
      />
    </>
  )
}
