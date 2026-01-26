import { useState, useRef } from 'react'
import { Tooltip, TooltipPortal, TooltipContent } from '../ui/Tooltip'

// Number of images to show in a single row (based on w-16 thumbnails in max-w-2xl container)
const MAX_VISIBLE_ROW = 8

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
 * Chevron icon component
 */
function ChevronIcon({ isOpen }) {
  return (
    <svg
      className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-90' : ''}`}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
  )
}

/**
 * Image icon component (fallback)
 */
function ImageIcon() {
  return (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  )
}

/**
 * Image thumbnail component with styled tooltip
 */
function ImageThumbnail({ image }) {
  const [showTooltip, setShowTooltip] = useState(false)
  const thumbnailRef = useRef(null)
  const { url, title, alt, source_title } = image

  const displayTitle = title || alt || 'Image'
  const description = source_title ? `From: ${source_title}. Click to view full size.` : 'Click to view full size.'

  return (
    <>
      <a
        ref={thumbnailRef}
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="group relative block w-16 h-16 rounded-lg overflow-hidden bg-amber-50 border border-amber-200 hover:border-amber-400 transition-colors"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        <img
          src={url}
          alt={alt || title || 'Resource image'}
          className="w-full h-full object-cover"
          onError={(e) => {
            e.target.style.display = 'none'
            e.target.nextSibling.style.display = 'flex'
          }}
        />
        <div className="hidden w-full h-full items-center justify-center text-amber-400">
          <ImageIcon />
        </div>
      </a>

      <TooltipPortal targetRef={thumbnailRef} show={showTooltip} position="top">
        <TooltipContent
          title={displayTitle}
          description={description}
          position="top"
          color="amber"
        />
      </TooltipPortal>
    </>
  )
}

/**
 * Collapsible images gallery component
 *
 * Displays images as clickable thumbnails inside an amber collapsible container.
 * Shows one row by default with "+N more" expand control.
 */
export function ResourceGallery({ resources }) {
  const [isOpen, setIsOpen] = useState(false)
  const [showAll, setShowAll] = useState(false)

  if (!resources) return null

  const { images = [] } = resources
  const hasImages = images.length > 0

  if (!hasImages) return null

  const visibleImages = showAll ? images : images.slice(0, MAX_VISIBLE_ROW)
  const hiddenCount = images.length - MAX_VISIBLE_ROW
  const hasOverflow = hiddenCount > 0 && !showAll

  return (
    <div className="border border-amber-200 rounded-lg overflow-hidden bg-amber-50/30">
      {/* Header - always visible */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-amber-700 hover:bg-amber-50 transition-colors"
      >
        <ChevronIcon isOpen={isOpen} />
        <span className="font-medium">Images ({images.length})</span>
        <Tooltip
          title="Visual Resources"
          description="Diagrams and illustrations from the source articles. Click any image to view full size."
          color="amber"
          position="top"
        >
          <span className="text-charcoal-muted hover:text-amber-600 transition-colors">
            <InfoIcon />
          </span>
        </Tooltip>
      </button>

      {/* Expandable content */}
      {isOpen && (
        <div className="px-3 pb-2 border-t border-amber-100 bg-ivory-light">
          <div className="flex flex-wrap gap-2 items-center pt-2">
            {visibleImages.map((image, index) => (
              <ImageThumbnail key={`img-${index}`} image={image} />
            ))}
            {hasOverflow && (
              <button
                onClick={() => setShowAll(true)}
                className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-ivory-medium text-charcoal-muted hover:bg-ivory transition-colors h-16"
              >
                +{hiddenCount} more
              </button>
            )}
            {showAll && images.length > MAX_VISIBLE_ROW && (
              <button
                onClick={() => setShowAll(false)}
                className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-ivory-medium text-charcoal-muted hover:bg-ivory transition-colors"
              >
                Show less
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
