import { useState } from 'react'
import { useSSEChat } from './hooks/useSSEChat'
import { MessageList, ChatInput, WelcomeScreen } from './components/chat'
import { Sidebar } from './components/sidebar'

function SidebarToggleIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth={1.5} />
      <line x1="9" y1="3" x2="9" y2="21" strokeWidth={1.5} />
    </svg>
  )
}

function App() {
  const { messages, isLoading, sendMessage } = useSSEChat()
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const toggleSidebar = () => setSidebarOpen((prev) => !prev)
  const closeSidebar = () => setSidebarOpen(false)

  return (
    <div className="min-h-screen bg-ivory flex flex-col">
      {/* Header */}
      <header className="bg-ivory-light border-b border-black/10 px-6 py-4 relative z-10">
        {/* Top bar: branding + nav */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <button
              onClick={toggleSidebar}
              className="text-charcoal-muted hover:text-charcoal transition-colors"
              aria-label="Toggle sidebar"
            >
              <SidebarToggleIcon />
            </button>
            <span className="font-heading text-2xl text-charcoal">
              Norfolk <span className="text-terracotta">AI/BI</span>
            </span>
          </div>
          <nav className="flex items-center gap-4 text-xs text-charcoal-muted">
            <a
              href="https://github.com/arthurfantaci/requirements-graphrag-api"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-charcoal transition-colors"
            >
              GitHub
            </a>
            <a
              href="https://norfolkaibi.com"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-charcoal transition-colors"
            >
              &larr; Norfolk AI/BI
            </a>
          </nav>
        </div>

        {/* Title + subtitle */}
        <h1 className="font-heading text-2xl text-charcoal">
          Requirements Management Assistant
        </h1>
        <p className="text-base text-charcoal-muted">
          Ask questions about Jama Software&apos;s{' '}
          <a
            href="https://www.jamasoftware.com/requirements-management-guide"
            target="_blank"
            rel="noopener noreferrer"
            className="font-bold italic text-jama-orange hover:text-jama-orange-dark hover:underline"
          >
            The Essential Guide to Requirements Management and Traceability
          </a>
        </p>
      </header>

      {/* Body: sidebar + main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Mobile backdrop */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/30 z-40 md:hidden"
            onClick={closeSidebar}
          />
        )}

        {/* Sidebar */}
        <Sidebar
          isOpen={sidebarOpen}
          onClose={closeSidebar}
          onQuickStart={sendMessage}
        />

        {/* Main content column */}
        <div className="flex-1 flex flex-col min-w-0">
          <main className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <WelcomeScreen onQuickStart={sendMessage} />
            ) : (
              <MessageList messages={messages} />
            )}
          </main>

          {/* Input */}
          <ChatInput onSend={sendMessage} disabled={isLoading} />

          {/* Footer */}
          <footer className="bg-ivory-light border-t border-black/10 px-6 py-2 text-center text-xs text-charcoal-muted">
            &copy; 2026 Norfolk AI/BI &middot; GraphRAG &middot; Neo4j &middot; LangChain &middot; LangSmith
          </footer>
        </div>
      </div>
    </div>
  )
}

export default App
