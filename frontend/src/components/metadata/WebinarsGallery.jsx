import { useState, useRef } from 'react'
import { Tooltip, TooltipPortal, TooltipContent } from '../ui/Tooltip'

/**
 * Info icon for tooltip trigger
 */
function InfoIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  )
}

/**
 * Play icon for webinar thumbnails
 */
function PlayIcon() {
  return (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M8 5v14l11-7z" />
    </svg>
  )
}

/**
 * External link icon
 */
function ExternalLinkIcon() {
  return (
    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
      />
    </svg>
  )
}

/**
 * Webinar thumbnail card component with styled tooltip
 * Uses thumbnail_url from Neo4j, with icon placeholder fallback
 */
function WebinarThumbnail({ webinar }) {
  const [showTooltip, setShowTooltip] = useState(false)
  const thumbnailRef = useRef(null)
  const { url, title, thumbnail_url } = webinar

  const displayTitle = title || 'Webinar'
  const description = 'Click to watch this webinar from Jama Software.'

  return (
    <>
      <a
        ref={thumbnailRef}
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="group relative block w-24 h-16 rounded-lg overflow-hidden bg-gradient-to-br from-rose-50 to-rose-100 border border-rose-200 hover:border-rose-400 transition-colors"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {thumbnail_url ? (
          <>
            <img
              src={thumbnail_url}
              alt={title || 'Webinar thumbnail'}
              className="w-full h-full object-cover"
              onError={(e) => {
                e.target.style.display = 'none'
                e.target.nextSibling.style.display = 'flex'
              }}
            />
            {/* Fallback placeholder on image load error */}
            <div className="hidden w-full h-full items-center justify-center bg-gradient-to-br from-rose-50 to-rose-100 text-rose-400">
              <PlayIcon />
            </div>
          </>
        ) : (
          /* Icon placeholder when no thumbnail available */
          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-rose-50 to-rose-100 text-rose-400">
            <PlayIcon />
          </div>
        )}

        {/* Play overlay on hover */}
        <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/30 transition-colors">
          <div className="opacity-0 group-hover:opacity-100 text-white transition-opacity">
            <PlayIcon />
          </div>
        </div>

        {/* External link indicator */}
        <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 text-white transition-opacity">
          <ExternalLinkIcon />
        </div>
      </a>

      <TooltipPortal targetRef={thumbnailRef} show={showTooltip} position="top">
        <TooltipContent
          title={displayTitle}
          description={description}
          position="top"
          color="rose"
        />
      </TooltipPortal>
    </>
  )
}

/**
 * Webinars gallery component
 *
 * Displays webinars as thumbnail cards with "WEBINARS" section header
 */
export function WebinarsGallery({ resources }) {
  if (!resources) return null

  const { webinars = [] } = resources
  const hasWebinars = webinars.length > 0

  if (!hasWebinars) return null

  return (
    <div className="space-y-2">
      {/* Section header */}
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-medium text-rose-600 uppercase tracking-wide">
          Webinars
        </span>
        <Tooltip
          title="Video Content"
          description="Related video content from Jama Software. Click any thumbnail to watch the webinar."
          color="rose"
          position="top"
        >
          <span className="text-gray-400 hover:text-rose-600 transition-colors">
            <InfoIcon />
          </span>
        </Tooltip>
      </div>

      {/* Webinars row */}
      <div className="flex flex-wrap gap-2">
        {webinars.map((webinar, index) => (
          <WebinarThumbnail key={`webinar-${index}`} webinar={webinar} />
        ))}
      </div>
    </div>
  )
}
