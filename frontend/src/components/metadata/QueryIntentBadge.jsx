/**
 * Badge component to display query intent type
 *
 * Shows "GraphRAG" for explanatory, "Cypher" for structured, or "Chat" for conversational
 */
export function QueryIntentBadge({ intent }) {
  if (!intent) return null

  const variants = {
    structured: { label: 'Cypher', classes: 'bg-purple-100 text-purple-800' },
    explanatory: { label: 'GraphRAG', classes: 'bg-blue-100 text-blue-800' },
    conversational: { label: 'Chat', classes: 'bg-green-100 text-green-800' },
  }

  const variant = variants[intent] ?? variants.explanatory

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variant.classes}`}
    >
      {variant.label}
    </span>
  )
}
