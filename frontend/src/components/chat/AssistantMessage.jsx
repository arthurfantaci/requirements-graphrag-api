import ReactMarkdown from 'react-markdown'
import { SourceCitationRenderer } from './SourceCitationRenderer'
import { QueryIntentBadge } from '../metadata/QueryIntentBadge'
import { EntityBadges } from '../metadata/EntityBadges'
import { SourcesPanel } from '../metadata/SourcesPanel'
import { ResourceGallery } from '../metadata/ResourceGallery'
import { WebinarsGallery } from '../metadata/WebinarsGallery'
import { CypherDisplay } from '../metadata/CypherDisplay'
import { ResultsTable } from '../metadata/ResultsTable'
import { ResponseActions } from '../feedback'

/**
 * Assistant message component with rich metadata display
 *
 * Handles both explanatory (RAG) and structured (Cypher) response types
 */
export function AssistantMessage({ message }) {
  const {
    id,
    content,
    intent,
    sources,
    entities,
    resources,
    cypher,
    results,
    rowCount,
    status,
    error,
    runId,
  } = message

  const isStructured = intent === 'structured'
  const isExplanatory = intent === 'explanatory'
  const hasContent = content && content.trim()
  const hasSources = sources && sources.length > 0
  const hasEntities = entities && entities.length > 0
  const hasImages = resources?.images?.length > 0
  const hasWebinars = resources?.webinars?.length > 0
  const hasCypher = cypher && cypher.trim()
  const hasResults = results && results.length > 0

  return (
    <div className="max-w-2xl w-full space-y-3">
      {/* Header with intent badge */}
      {intent && (
        <div className="flex justify-start">
          <QueryIntentBadge intent={intent} />
        </div>
      )}

      {/* Structured: Cypher query display */}
      {isStructured && hasCypher && (
        <CypherDisplay query={cypher} />
      )}

      {/* Structured: Results table */}
      {isStructured && hasResults && (
        <ResultsTable results={results} rowCount={rowCount} />
      )}

      {/* Explanatory: Main content with Markdown rendering and clickable citations */}
      {isExplanatory && hasContent && (
        <div className="prose prose-sm prose-neutral max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-headings:mt-4 prose-headings:mb-2">
          {hasSources ? (
            <SourceCitationRenderer content={content} sources={sources} />
          ) : (
            <ReactMarkdown>{content}</ReactMarkdown>
          )}
        </div>
      )}

      {/* Fallback: Show content when no intent yet (streaming) */}
      {!intent && hasContent && (
        <div className="prose prose-sm prose-neutral max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-headings:mt-4 prose-headings:mb-2">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      )}

      {/* Streaming indicator */}
      {status === 'streaming' && !hasContent && (
        <div className="py-1">
          <div className="flex space-x-2">
            <div className="w-2 h-2 bg-charcoal-muted rounded-full animate-bounce" />
            <div
              className="w-2 h-2 bg-charcoal-muted rounded-full animate-bounce"
              style={{ animationDelay: '0.1s' }}
            />
            <div
              className="w-2 h-2 bg-charcoal-muted rounded-full animate-bounce"
              style={{ animationDelay: '0.2s' }}
            />
          </div>
        </div>
      )}

      {/* Error display */}
      {status === 'error' && (
        <p className="text-red-600">{error || 'An error occurred'}</p>
      )}

      {/* Sources panel */}
      {hasSources && (
        <SourcesPanel sources={sources} />
      )}

      {/* Terms (Entity badges) */}
      {hasEntities && (
        <EntityBadges entities={entities} />
      )}

      {/* Images gallery */}
      {hasImages && (
        <ResourceGallery resources={resources} />
      )}

      {/* Webinars gallery */}
      {hasWebinars && (
        <WebinarsGallery resources={resources} />
      )}

      {/* Response actions - show when complete */}
      {status === 'complete' && (
        <ResponseActions content={content} runId={runId} messageId={id} />
      )}
    </div>
  )
}
