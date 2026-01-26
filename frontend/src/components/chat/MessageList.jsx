import { AssistantMessage } from './AssistantMessage'

/**
 * User message component
 */
function UserMessage({ content }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-2xl px-5 py-4 rounded-2xl bg-ivory-medium text-charcoal">
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
