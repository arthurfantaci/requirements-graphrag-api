import { useState } from 'react'

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
    hint: 'Triggers a Cypher query',
  },
]

function ChevronIcon({ isOpen }) {
  return (
    <svg
      className={`w-3.5 h-3.5 transition-transform ${isOpen ? 'rotate-90' : ''}`}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

function SidebarSection({ title, defaultOpen = false, children }) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border-b border-black/10">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-4 py-2.5 text-xs font-medium text-charcoal uppercase tracking-widest hover:bg-ivory-medium/50 transition-colors"
      >
        <ChevronIcon isOpen={isOpen} />
        {title}
      </button>
      {isOpen && <div className="px-4 pb-3">{children}</div>}
    </div>
  )
}

export function Sidebar({ isOpen, onClose, onQuickStart }) {
  return (
    <aside
      className={`
        bg-ivory-light border-r border-black/10 flex flex-col
        fixed inset-y-0 left-0 z-50 w-72 transition-transform duration-200
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        md:static md:z-0 md:w-56 md:shrink-0 md:transition-none
        ${isOpen ? 'md:translate-x-0' : 'md:hidden'}
      `}
    >
      {/* Mobile close button */}
      <div className="flex items-center justify-end px-4 py-3 border-b border-black/10 md:hidden">
        <button
          onClick={onClose}
          className="text-charcoal-muted hover:text-charcoal transition-colors"
          aria-label="Close sidebar"
        >
          <CloseIcon />
        </button>
      </div>

      {/* Sections */}
      <div className="flex-1 overflow-y-auto">
        <SidebarSection title="About" defaultOpen>
          <p className="text-xs text-charcoal-light leading-relaxed">
            AI-powered assistant using GraphRAG and Jama Software&apos;s{' '}
            <em>Essential Guide to Requirements Management and Traceability</em>.
          </p>
          <ul className="mt-2 space-y-1 text-xs text-charcoal-light">
            <li className="flex items-start gap-1.5">
              <span className="text-terracotta mt-0.5">&#8226;</span>
              Graph-enhanced retrieval (Neo4j)
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-terracotta mt-0.5">&#8226;</span>
              Real-time SSE streaming
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-terracotta mt-0.5">&#8226;</span>
              Natural language to Cypher
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-terracotta mt-0.5">&#8226;</span>
              LangSmith observability
            </li>
          </ul>
        </SidebarSection>

        <SidebarSection title="Quick Start" defaultOpen>
          <div className="space-y-1.5">
            {QUICK_START_PROMPTS.map(({ text, hint }) => (
              <div key={text}>
                <button
                  onClick={() => {
                    onQuickStart(text)
                    onClose()
                  }}
                  className="w-full text-left px-3 py-2 rounded border border-black/10 bg-ivory text-xs text-charcoal hover:bg-ivory-medium hover:border-black/15 transition-colors"
                >
                  {text}
                </button>
                {hint && (
                  <p className="text-[10px] text-charcoal-muted mt-0.5 ml-3">{hint}</p>
                )}
              </div>
            ))}
          </div>
        </SidebarSection>

        <SidebarSection title="Architecture" defaultOpen={false}>
          <div className="rounded overflow-hidden border border-black/10">
            <div className="px-2 py-1.5 bg-terminal-bg flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-red-500/70" />
              <span className="w-2 h-2 rounded-full bg-yellow-500/70" />
              <span className="w-2 h-2 rounded-full bg-green-500/70" />
              <span className="ml-1.5 text-[10px] text-terminal-text/60 font-mono">arch</span>
            </div>
            <div className="px-3 py-2 bg-terminal-bg font-mono text-[10px] leading-relaxed">
              <p className="text-terminal-green">React + Vite</p>
              <p className="text-terminal-text/40 ml-1.5">&darr;</p>
              <p className="text-terminal-blue">FastAPI (Railway)</p>
              <p className="text-terminal-text/40 ml-1.5">&darr;</p>
              <p className="text-terminal-green">Neo4j AuraDB</p>
              <p className="text-terminal-text/50 mt-1.5">
                LangChain &middot; LangSmith
              </p>
            </div>
          </div>
        </SidebarSection>
      </div>

      {/* Footer */}
      <div className="border-t border-black/10 px-4 py-2.5 text-[10px] text-charcoal-muted text-center">
        Norfolk AI/BI &middot; GraphRAG
      </div>
    </aside>
  )
}
