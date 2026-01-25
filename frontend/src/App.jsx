import { useSSEChat } from './hooks/useSSEChat'
import { MessageList, ChatInput } from './components/chat'

function App() {
  const { messages, isLoading, sendMessage } = useSSEChat()

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 relative z-10">
        <h1 className="text-xl font-semibold text-gray-900">
          Requirements Management Assistant
        </h1>
        <p className="text-sm text-gray-500">
          Ask questions about Jama Software's{' '}
          <a
            href="https://www.jamasoftware.com/requirements-management-guide"
            target="_blank"
            rel="noopener noreferrer"
            className="font-semibold italic text-jama-orange hover:text-jama-orange-dark hover:underline"
          >
            The Essential Guide to Requirements Management and Traceability
          </a>
        </p>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto p-6 space-y-4">
        <MessageList messages={messages} />
      </main>

      {/* Input */}
      <ChatInput onSend={sendMessage} disabled={isLoading} />
    </div>
  )
}

export default App
