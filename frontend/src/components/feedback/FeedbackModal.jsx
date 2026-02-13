import { useState, useEffect, useMemo } from 'react'

/**
 * Intent-specific rubric dimensions.
 *
 * Each intent type gets purpose-built rating dimensions so annotators
 * evaluate what matters for that response type.
 */
const RUBRIC_DIMENSIONS = {
  explanatory: [
    { key: 'accuracy', label: 'Accuracy', description: 'Factually correct information' },
    { key: 'completeness', label: 'Completeness', description: 'Covers the topic adequately' },
    { key: 'citation_quality', label: 'Citation Quality', description: 'Sources are relevant and specific' },
  ],
  structured: [
    { key: 'cypher_quality', label: 'Query Quality', description: 'Well-formed and efficient Cypher' },
    { key: 'result_correctness', label: 'Result Correctness', description: 'Results answer the question' },
  ],
  conversational: [
    { key: 'coherence', label: 'Coherence', description: 'Logically consistent response' },
    { key: 'helpfulness', label: 'Helpfulness', description: 'Useful and actionable answer' },
  ],
}

/**
 * Star rating component for a single rubric dimension
 */
function RubricRating({ dimension, value, onChange }) {
  const stars = [1, 2, 3, 4, 5]

  return (
    <div className="flex items-center justify-between py-1.5">
      <div className="flex-1 min-w-0 mr-3">
        <span className="text-sm text-charcoal">{dimension.label}</span>
        <span className="block text-xs text-charcoal-muted truncate">{dimension.description}</span>
      </div>
      <div className="flex gap-0.5 shrink-0">
        {stars.map((star) => (
          <button
            key={star}
            type="button"
            onClick={() => onChange(star / 5)}
            className={`w-6 h-6 rounded transition-colors ${
              value != null && star <= value * 5
                ? 'text-terracotta'
                : 'text-charcoal-muted/30 hover:text-charcoal-muted/60'
            }`}
            aria-label={`${star} of 5 for ${dimension.label}`}
          >
            <svg viewBox="0 0 20 20" fill="currentColor" className="w-full h-full">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
          </button>
        ))}
      </div>
    </div>
  )
}

/**
 * Feedback modal for collecting optional details and intent-specific rubric scores
 */
export function FeedbackModal({ isOpen, isPositive, intent, onSubmit, onCancel }) {
  const [details, setDetails] = useState('')
  const [rubricScores, setRubricScores] = useState({})

  const dimensions = useMemo(
    () => (intent && RUBRIC_DIMENSIONS[intent]) || [],
    [intent]
  )

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setDetails('')
      setRubricScores({})
    }
  }, [isOpen])

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onCancel()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onCancel])

  if (!isOpen) return null

  const handleRubricChange = (key, value) => {
    setRubricScores((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = () => {
    onSubmit({ comment: details, rubricScores })
    setDetails('')
    setRubricScores({})
  }

  const handleCancel = () => {
    setDetails('')
    setRubricScores({})
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
      <div role="dialog" aria-modal="true" aria-label="Feedback" className="relative bg-ivory-light rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-black/10">
          <h2 className="text-lg font-semibold text-charcoal">Feedback</h2>
          <button
            onClick={handleCancel}
            aria-label="Close"
            className="text-charcoal-muted hover:text-charcoal-light transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4">
          {/* Intent-specific rubric */}
          {dimensions.length > 0 && (
            <div>
              <label className="block text-sm text-charcoal-light mb-2">
                Rate this response: (optional)
              </label>
              <div className="space-y-1">
                {dimensions.map((dim) => (
                  <RubricRating
                    key={dim.key}
                    dimension={dim}
                    value={rubricScores[dim.key] ?? null}
                    onChange={(val) => handleRubricChange(dim.key, val)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Free-text comment */}
          <div>
            <label className="block text-sm text-charcoal-light mb-2">
              {dimensions.length > 0 ? 'Additional comments: (optional)' : 'Please provide details: (optional)'}
            </label>
            <textarea
              value={details}
              onChange={(e) => setDetails(e.target.value)}
              placeholder={placeholder}
              rows={dimensions.length > 0 ? 3 : 4}
              autoFocus={dimensions.length === 0}
              className="w-full px-3 py-2 border border-black/15 rounded-lg focus:outline-none focus:ring-2 focus:ring-terracotta focus:border-transparent resize-none"
            />
          </div>

          <p className="text-xs text-charcoal-muted">
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
