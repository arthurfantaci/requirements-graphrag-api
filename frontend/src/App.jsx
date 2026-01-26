import { useSSEChat } from './hooks/useSSEChat'
import { MessageList, ChatInput, WelcomeScreen } from './components/chat'

function App() {
  const { messages, isLoading, sendMessage } = useSSEChat()

  return (
    <div className="min-h-screen bg-ivory flex flex-col">
      {/* Header */}
      <header className="bg-ivory-light border-b border-black/10 px-6 py-4 relative z-10">
        {/* Top bar: branding + nav */}
        <div className="flex items-center justify-between mb-2">
          <span className="font-heading text-2xl text-charcoal">
            Norfolk AI/BI
          </span>
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

      {/* Messages or Welcome Screen */}
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
  )
}

export default App
