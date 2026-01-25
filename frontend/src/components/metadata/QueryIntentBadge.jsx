/**
 * Badge component to display query intent type
 *
 * Shows "GraphRAG" for explanatory queries or "Cypher" for structured queries
 */
export function QueryIntentBadge({ intent }) {
  if (!intent) return null

  const isStructured = intent === 'structured'

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
        isStructured
          ? 'bg-purple-100 text-purple-800'
          : 'bg-blue-100 text-blue-800'
      }`}
    >
      {isStructured ? 'Cypher' : 'GraphRAG'}
    </span>
  )
}
