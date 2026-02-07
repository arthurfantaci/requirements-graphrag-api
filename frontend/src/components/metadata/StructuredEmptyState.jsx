/**
 * Empty state display for structured queries that return 0 results.
 *
 * Shows the backend message (if provided) with a suggestion to rephrase
 * as a natural language question for broader search coverage.
 */
export function StructuredEmptyState({ message }) {
  return (
    <div className="border border-amber-200 bg-amber-50 rounded-lg p-4 flex items-start gap-3">
      <svg
        className="w-5 h-5 text-amber-500 mt-0.5 shrink-0"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth={1.5}
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"
        />
      </svg>
      <div className="text-sm text-amber-800">
        <p>{message || 'No matching results found for this query.'}</p>
      </div>
    </div>
  )
}
