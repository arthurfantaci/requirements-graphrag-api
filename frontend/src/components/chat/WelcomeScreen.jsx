const QUICK_START_PROMPTS = [
  {
    text: 'What is requirements traceability?',
    hint: null,
  },
  {
    text: 'How does requirements management help reduce project risk?',
    hint: null,
  },
  {
    text: 'List all webinars about traceability',
    hint: 'This triggers a Cypher query against the knowledge graph',
  },
]

/**
 * Welcome screen shown before any messages are sent
 *
 * Provides app description, feature highlights, quick-start prompts,
 * and an architecture summary. Disappears after the first message.
 */
export function WelcomeScreen({ onQuickStart }) {
  return (
    <div className="max-w-2xl mx-auto space-y-6 py-4">
      {/* Title + description */}
      <div>
        <h2 className="font-heading text-2xl text-charcoal mb-2">
          Requirements Management Assistant
        </h2>
        <p className="text-sm text-charcoal-light leading-relaxed">
          An AI-powered assistant that answers questions about requirements management
          using GraphRAG technology and Jama Software&apos;s{' '}
          <em>Essential Guide to Requirements Management and Traceability</em>.
          It combines vector search with Neo4j knowledge graph traversal for richer,
          more contextual answers.
        </p>
      </div>

      {/* What This Demonstrates */}
      <div className="border border-black/10 rounded-lg bg-ivory-light p-4">
        <h3 className="text-xs font-medium text-terracotta uppercase tracking-widest mb-3">
          What This Demonstrates
        </h3>
        <ul className="space-y-1.5 text-sm text-charcoal-light">
          <li className="flex items-start gap-2">
            <span className="text-terracotta mt-0.5">&#8226;</span>
            Graph-enhanced retrieval from a Neo4j knowledge graph
          </li>
          <li className="flex items-start gap-2">
            <span className="text-terracotta mt-0.5">&#8226;</span>
            Real-time SSE streaming with source attribution
          </li>
          <li className="flex items-start gap-2">
            <span className="text-terracotta mt-0.5">&#8226;</span>
            Natural language to Cypher query translation
          </li>
          <li className="flex items-start gap-2">
            <span className="text-terracotta mt-0.5">&#8226;</span>
            Color-coded concept ontology (Definitions, Challenges, Best Practices)
          </li>
          <li className="flex items-start gap-2">
            <span className="text-terracotta mt-0.5">&#8226;</span>
            LangSmith tracing and evaluation pipeline
          </li>
        </ul>
      </div>

      {/* Try These Questions */}
      <div className="border border-black/10 rounded-lg bg-ivory-light p-4">
        <h3 className="text-xs font-medium text-terracotta uppercase tracking-widest mb-3">
          Try These Questions
        </h3>
        <div className="space-y-2">
          {QUICK_START_PROMPTS.map(({ text, hint }) => (
            <div key={text}>
              <button
                onClick={() => onQuickStart(text)}
                className="w-full text-left px-4 py-2.5 rounded-lg border border-black/10 bg-ivory text-sm text-charcoal hover:bg-ivory-medium hover:border-black/15 transition-colors"
              >
                {text}
              </button>
              {hint && (
                <p className="text-xs text-charcoal-muted mt-1 ml-4">{hint}</p>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Architecture summary */}
      <div className="rounded-lg overflow-hidden border border-black/10">
        <div className="px-3 py-2 bg-terminal-bg flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-green-500/70" />
          <span className="ml-2 text-xs text-terminal-text/60 font-mono">architecture</span>
        </div>
        <div className="px-4 py-3 bg-terminal-bg font-mono text-xs leading-relaxed">
          <p className="text-terminal-green">React + Vite</p>
          <p className="text-terminal-text/40 ml-2">&darr;</p>
          <p className="text-terminal-blue">FastAPI (Railway)</p>
          <p className="text-terminal-text/40 ml-2">&darr;</p>
          <p className="text-terminal-green">Neo4j AuraDB</p>
          <p className="text-terminal-text/50 mt-2">
            LangChain &middot; neo4j-graphrag &middot; LangSmith Tracing
          </p>
        </div>
      </div>
    </div>
  )
}
