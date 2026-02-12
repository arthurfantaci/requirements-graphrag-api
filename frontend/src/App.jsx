import { useState } from 'react'
import { Analytics } from '@vercel/analytics/react'
import { SpeedInsights } from '@vercel/speed-insights/react'
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

function GitHubIcon() {
  return (
    <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
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
              Norfolk <span className="text-norfolk-red">AI | BI</span>
            </span>
          </div>
          <nav className="flex items-center gap-4 text-xs text-charcoal-muted">
            <a
              href="https://github.com/arthurfantaci/requirements-graphrag-api"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-charcoal transition-colors"
            >
              <GitHubIcon />
            </a>
            <a
              href="https://norfolkaibi.com"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-charcoal transition-colors"
            >
              &larr; Norfolk <span className="text-norfolk-red">AI | BI</span>
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
            &copy; 2026 Norfolk <span className="text-norfolk-red">AI | BI</span> &middot; GraphRAG &middot; Neo4j &middot; LangChain &middot; LangSmith
          </footer>
        </div>
      </div>
      <Analytics />
      <SpeedInsights />
    </div>
  )
}

export default App
