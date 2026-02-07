import { useState, useMemo, useCallback } from 'react'
import { API_URL, API_KEY } from '../utils/api'

/**
 * Generate a unique conversation ID (UUID v4)
 */
function generateConversationId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/**
 * Generate a unique message ID
 */
function generateMessageId() {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

/**
 * Create an initial assistant message placeholder
 */
function createAssistantMessage() {
  return {
    id: generateMessageId(),
    role: 'assistant',
    content: '',
    intent: null,
    // Explanatory metadata
    sources: [],
    entities: [],
    resources: { images: [], webinars: [], videos: [] },
    // Structured metadata
    cypher: null,
    results: null,
    rowCount: null,
    message: null,
    // Status
    status: 'streaming',
    error: null,
    // LangSmith correlation for feedback
    runId: null,
  }
}

/**
 * Custom hook for SSE chat functionality with enhanced metadata support
 *
 * Handles SSE events: routing, sources, token, cypher, results, done, error
 */
export function useSSEChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  // Generate a stable conversation ID for this session
  const conversationId = useMemo(() => generateConversationId(), [])

  /**
   * Update the last assistant message with new data
   */
  const updateLastMessage = useCallback((updater) => {
    setMessages((prev) => {
      const updated = [...prev]
      const lastIdx = updated.length - 1
      if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
        updated[lastIdx] =
          typeof updater === 'function'
            ? updater(updated[lastIdx])
            : { ...updated[lastIdx], ...updater }
      }
      return updated
    })
  }, [])

  /**
   * Process an SSE event and update message state
   */
  const processEvent = useCallback(
    (eventType, data) => {
      switch (eventType) {
        case 'routing':
          // Set the query intent (explanatory or structured)
          updateLastMessage({ intent: data.intent })
          break

        case 'sources':
          // Update sources, entities, and resources
          updateLastMessage({
            sources: data.sources || [],
            entities: data.entities || [],
            resources: data.resources || { images: [], webinars: [], videos: [] },
          })
          break

        case 'token':
          // Append token to content
          updateLastMessage((msg) => ({
            ...msg,
            content: msg.content + (data.token || ''),
          }))
          break

        case 'cypher':
          // Set the Cypher query for structured responses
          updateLastMessage({ cypher: data.query })
          break

        case 'results':
          // Set the results for structured responses
          updateLastMessage({
            results: data.results || [],
            rowCount: data.row_count,
          })
          break

        case 'done':
          // Mark as complete, use full answer if provided, capture run_id for feedback
          updateLastMessage((msg) => ({
            ...msg,
            content: data.full_answer || msg.content,
            status: 'complete',
            rowCount: data.row_count ?? msg.rowCount,
            message: data.message || msg.message,
            runId: data.run_id || null,
          }))
          break

        case 'error':
          // Set error state
          updateLastMessage({
            status: 'error',
            error: data.error || 'An unknown error occurred',
          })
          break

        default:
          // Ignore unknown event types
          break
      }
    },
    [updateLastMessage]
  )

  /**
   * Parse SSE line to extract event type and data
   *
   * Event detection order matters! The 'done' event for structured queries
   * has {query, row_count, run_id} which could match 'cypher' if not careful.
   * We check for 'done' indicators (run_id, full_answer, row_count) first.
   */
  const parseSSELine = useCallback((line) => {
    if (!line.startsWith('data: ')) return null

    try {
      const data = JSON.parse(line.slice(6))

      // Determine event type from data shape
      // IMPORTANT: Check 'done' patterns BEFORE 'cypher' to avoid misidentification
      if (data.intent !== undefined) return { type: 'routing', data }
      if (data.sources !== undefined) return { type: 'sources', data }
      if (data.token !== undefined) return { type: 'token', data }
      if (data.results !== undefined) return { type: 'results', data }
      // Done event: has run_id, or full_answer (explanatory), or row_count with query (structured)
      if (data.run_id !== undefined || data.full_answer !== undefined ||
          (data.row_count !== undefined && data.query !== undefined))
        return { type: 'done', data }
      // Cypher event: has query but NOT row_count (done event has both)
      if (data.query !== undefined && data.row_count === undefined) return { type: 'cypher', data }
      if (data.error !== undefined) return { type: 'error', data }

      return null
    } catch {
      return null
    }
  }, [])

  /**
   * Send a message and handle SSE response
   */
  const sendMessage = useCallback(
    async (input) => {
      if (!input.trim() || isLoading) return

      const userMessage = { id: generateMessageId(), role: 'user', content: input }
      setMessages((prev) => [...prev, userMessage])
      setIsLoading(true)

      // Add placeholder for streaming response
      const assistantMessage = createAssistantMessage()
      setMessages((prev) => [...prev, assistantMessage])

      try {
        // Build conversation history from completed messages
        const conversationHistory = messages
          .filter((m) => m.content && m.content.trim() && m.status !== 'streaming')
          .map((m) => ({ role: m.role, content: m.content }))

        const response = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(API_KEY && { 'X-API-Key': API_KEY }),
          },
          body: JSON.stringify({
            message: input,
            conversation_id: conversationId,
            conversation_history: conversationHistory.length > 0 ? conversationHistory : null,
          }),
        })

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status}`)
        }

        // Handle SSE streaming response
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            const event = parseSSELine(line)
            if (event) {
              processEvent(event.type, event.data)
            }
          }
        }

        // Process any remaining buffer
        if (buffer.trim()) {
          const event = parseSSELine(buffer)
          if (event) {
            processEvent(event.type, event.data)
          }
        }

        // Ensure message is marked as complete
        updateLastMessage((msg) => ({
          ...msg,
          status: msg.status === 'streaming' ? 'complete' : msg.status,
        }))
      } catch (error) {
        updateLastMessage({
          status: 'error',
          error: error.message || 'Failed to get response',
          content: 'Sorry, there was an error processing your request.',
        })
      } finally {
        setIsLoading(false)
      }
    },
    [messages, isLoading, conversationId, parseSSELine, processEvent, updateLastMessage]
  )

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    isLoading,
    conversationId,
    sendMessage,
    clearMessages,
  }
}
