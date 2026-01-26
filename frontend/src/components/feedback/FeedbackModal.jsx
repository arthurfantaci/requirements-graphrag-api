import { useState } from 'react'

/**
 * Feedback modal for collecting optional details
 */
export function FeedbackModal({ isOpen, isPositive, onSubmit, onCancel }) {
  const [details, setDetails] = useState('')

  if (!isOpen) return null

  const handleSubmit = () => {
    onSubmit(details)
    setDetails('')
  }

  const handleCancel = () => {
    setDetails('')
    onCancel()
  }

  const placeholder = isPositive
    ? 'What was satisfying about this response?'
    : 'What was unsatisfying about this response?'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/30" onClick={handleCancel} />

      {/* Modal */}
      <div className="relative bg-ivory-light rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-black/10">
          <h2 className="text-lg font-semibold text-charcoal">Feedback</h2>
          <button
            onClick={handleCancel}
            className="text-charcoal-muted hover:text-charcoal-light transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4">
          <label className="block text-sm text-charcoal-light mb-2">
            Please provide details: (optional)
          </label>
          <textarea
            value={details}
            onChange={(e) => setDetails(e.target.value)}
            placeholder={placeholder}
            rows={4}
            className="w-full px-3 py-2 border border-black/15 rounded-lg focus:outline-none focus:ring-2 focus:ring-terracotta focus:border-transparent resize-none"
          />
          <p className="mt-3 text-xs text-charcoal-muted">
            Submitting this feedback will help improve future responses.
          </p>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-black/5">
          <button
            onClick={handleCancel}
            className="px-4 py-2 text-sm text-charcoal-light hover:text-charcoal transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="px-4 py-2 text-sm bg-terracotta text-white rounded-md hover:bg-terracotta-dark transition-colors"
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  )
}
