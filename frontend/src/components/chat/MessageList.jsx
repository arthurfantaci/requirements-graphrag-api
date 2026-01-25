import { AssistantMessage } from './AssistantMessage'

/**
 * User message component
 * Uses Jama brand orange for the message bubble
 */
function UserMessage({ content }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-2xl px-4 py-3 rounded-lg bg-jama-orange text-white">
        <p className="whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  )
}

/**
 * Empty state component shown when no messages
 */
function EmptyState() {
  return (
    <div className="text-center text-gray-400 mt-20">
      <p className="text-lg">Welcome! Ask me anything about requirements management.</p>
      <p className="text-sm mt-2">For example: &quot;What is requirements traceability?&quot;</p>
    </div>
  )
}

/**
 * Message list component that renders all messages
 */
export function MessageList({ messages }) {
  if (messages.length === 0) {
    return <EmptyState />
  }

  return (
    <>
      {messages.map((message) => (
        <div key={message.id}>
          {message.role === 'user' ? (
            <UserMessage content={message.content} />
          ) : (
            <AssistantMessage message={message} />
          )}
        </div>
      ))}
    </>
  )
}
