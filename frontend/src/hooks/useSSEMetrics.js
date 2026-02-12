import * as Sentry from '@sentry/react'
import { useRef, useCallback } from 'react'

/**
 * Custom hook for SSE streaming performance metrics.
 *
 * Tracks TTFT, tokens/second, connection time, total duration,
 * and event sequence. Reports to Sentry as a custom span + breadcrumb.
 *
 * Usage:
 *   const { startStream, recordEvent, finishStream } = useSSEMetrics()
 *   startStream()           // before fetch()
 *   recordEvent('token')    // for each SSE event
 *   const m = finishStream() // on completion or error
 */
export function useSSEMetrics() {
  const metricsRef = useRef(null)
  const spanRef = useRef(null)

  const startStream = useCallback(() => {
    metricsRef.current = {
      sendTimestamp: performance.now(),
      firstEventTimestamp: null,
      firstTokenTimestamp: null,
      lastTokenTimestamp: null,
      tokenCount: 0,
      eventSequence: [],
      errors: [],
    }
    spanRef.current = Sentry.startInactiveSpan({
      name: 'sse.stream',
      op: 'http.stream',
    })
  }, [])

  const recordEvent = useCallback((eventType, data) => {
    const m = metricsRef.current
    if (!m) return
    const now = performance.now()

    if (!m.firstEventTimestamp) m.firstEventTimestamp = now
    m.eventSequence.push(eventType)

    if (eventType === 'token') {
      if (!m.firstTokenTimestamp) m.firstTokenTimestamp = now
      m.lastTokenTimestamp = now
      m.tokenCount++
    }
    if (eventType === 'error') {
      m.errors.push(data?.error || 'unknown')
    }
  }, [])

  const finishStream = useCallback(() => {
    const m = metricsRef.current
    if (!m) return null
    const now = performance.now()

    const streamingDuration =
      m.lastTokenTimestamp && m.firstTokenTimestamp
        ? m.lastTokenTimestamp - m.firstTokenTimestamp
        : null

    const metrics = {
      ttftMs: m.firstTokenTimestamp
        ? Math.round(m.firstTokenTimestamp - m.sendTimestamp)
        : null,
      connectionTimeMs: m.firstEventTimestamp
        ? Math.round(m.firstEventTimestamp - m.sendTimestamp)
        : null,
      totalDurationMs: Math.round(now - m.sendTimestamp),
      streamingDurationMs: streamingDuration ? Math.round(streamingDuration) : null,
      tokenCount: m.tokenCount,
      tokensPerSecond:
        streamingDuration && m.tokenCount > 1
          ? Math.round((m.tokenCount / (streamingDuration / 1000)) * 10) / 10
          : null,
      errorCount: m.errors.length,
      eventSequence: m.eventSequence,
      success: m.errors.length === 0,
    }

    // Report to Sentry span
    const span = spanRef.current
    if (span) {
      if (metrics.ttftMs != null) span.setAttribute('sse.ttft_ms', metrics.ttftMs)
      if (metrics.tokensPerSecond != null)
        span.setAttribute('sse.tokens_per_second', metrics.tokensPerSecond)
      span.setAttribute('sse.token_count', metrics.tokenCount)
      span.setAttribute('sse.total_duration_ms', metrics.totalDurationMs)
      span.setAttribute('sse.success', metrics.success)
      if (!metrics.success) span.setStatus({ code: 2, message: 'stream_error' })
      span.end()
      spanRef.current = null
    }

    // Add breadcrumb for error context
    Sentry.addBreadcrumb({
      category: 'sse',
      message: `Stream ${metrics.success ? 'completed' : 'failed'}: ${metrics.tokenCount} tokens in ${metrics.totalDurationMs}ms`,
      level: metrics.success ? 'info' : 'error',
      data: {
        ttft_ms: metrics.ttftMs,
        tokens_per_second: metrics.tokensPerSecond,
        token_count: metrics.tokenCount,
        total_duration_ms: metrics.totalDurationMs,
      },
    })

    metricsRef.current = null
    return metrics
  }, [])

  return { startStream, recordEvent, finishStream }
}
