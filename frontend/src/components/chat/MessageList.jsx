import { AssistantMessage } from './AssistantMessage'

/**
 * User message component
 */
function UserMessage({ content }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-2xl px-4 py-3 rounded-lg bg-terracotta text-white">
        <p className="whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  )
}

/**
 * Message list component that renders all messages
 */
export function MessageList({ messages }) {
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
