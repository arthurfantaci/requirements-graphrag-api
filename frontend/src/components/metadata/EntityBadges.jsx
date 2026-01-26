import { useState, useRef } from 'react'
import { Tooltip, TooltipPortal, TooltipContent } from '../ui/Tooltip'

const MAX_VISIBLE = 8

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
 * Color configuration for Neo4j node labels
 *
 * Each node label gets distinct colors for visual differentiation.
 * Colors are chosen to be accessible and semantically meaningful:
 * - Purple: Concepts (abstract ideas)
 * - Blue: Definitions (formal explanations)
 * - Teal: Entities (general extracted terms)
 * - Amber: Challenges (problems/issues)
 * - Green: Best Practices (recommendations)
 * - Slate: Fallback for unknown labels
 */
const LABEL_COLORS = {
  Concept: {
    bg: 'bg-purple-50',
    text: 'text-purple-700',
    border: 'border-purple-200',
    hoverBg: 'hover:bg-purple-100',
    hoverBorder: 'hover:border-purple-300',
    indicator: 'text-purple-400',
    tooltipColor: 'purple',
  },
  Definition: {
    bg: 'bg-blue-50',
    text: 'text-blue-700',
    border: 'border-blue-200',
    hoverBg: 'hover:bg-blue-100',
    hoverBorder: 'hover:border-blue-300',
    indicator: 'text-blue-400',
    tooltipColor: 'blue',
  },
  Entity: {
    bg: 'bg-teal-50',
    text: 'text-teal-700',
    border: 'border-teal-200',
    hoverBg: 'hover:bg-teal-100',
    hoverBorder: 'hover:border-teal-300',
    indicator: 'text-teal-400',
    tooltipColor: 'gray',
  },
  Challenge: {
    bg: 'bg-amber-50',
    text: 'text-amber-700',
    border: 'border-amber-200',
    hoverBg: 'hover:bg-amber-100',
    hoverBorder: 'hover:border-amber-300',
    indicator: 'text-amber-400',
    tooltipColor: 'amber',
  },
  Bestpractice: {
    bg: 'bg-green-50',
    text: 'text-green-700',
    border: 'border-green-200',
    hoverBg: 'hover:bg-green-100',
    hoverBorder: 'hover:border-green-300',
    indicator: 'text-green-400',
    tooltipColor: 'emerald',
  },
  Standard: {
    bg: 'bg-indigo-50',
    text: 'text-indigo-700',
    border: 'border-indigo-200',
    hoverBg: 'hover:bg-indigo-100',
    hoverBorder: 'hover:border-indigo-300',
    indicator: 'text-indigo-400',
    tooltipColor: 'purple',
  },
  // Default fallback for unknown labels
  default: {
    bg: 'bg-slate-50',
    text: 'text-slate-700',
    border: 'border-slate-200',
    hoverBg: 'hover:bg-slate-100',
    hoverBorder: 'hover:border-slate-300',
    indicator: 'text-slate-400',
    tooltipColor: 'gray',
  },
}

/**
 * Get color configuration for a node label
 */
function getLabelColors(label) {
  return LABEL_COLORS[label] || LABEL_COLORS.default
}

/**
 * Single concept badge with optional definition popover
 *
 * Color-coded by Neo4j node label for visual distinction.
 * Desktop: hover to show definition
 * Mobile: tap to toggle definition
 */
function ConceptBadge({ entity }) {
  const [showTooltip, setShowTooltip] = useState(false)
  const buttonRef = useRef(null)

  // Handle both string entities (legacy) and object entities {name, definition, label}
  const name = typeof entity === 'string' ? entity : entity.name
  const definition = typeof entity === 'string' ? null : entity.definition
  const label = typeof entity === 'string' ? 'Entity' : (entity.label || 'Entity')

  const colors = getLabelColors(label)

  // No popover needed if no definition
  if (!definition) {
    return (
      <span
        className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium ${colors.bg} ${colors.text} border ${colors.border}`}
        title={`${label} node`}
      >
        {name}
      </span>
    )
  }

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium ${colors.bg} ${colors.text} border ${colors.border} ${colors.hoverBg} ${colors.hoverBorder} transition-colors cursor-help`}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={() => setShowTooltip(!showTooltip)}
        aria-describedby={showTooltip ? `tooltip-${name}` : undefined}
        title={`${label} node - click for definition`}
      >
        {name}
        {/* Small indicator that definition is available */}
        <span className={`ml-1 ${colors.indicator} text-[10px]`}>?</span>
      </button>

      {/* Popover tooltip rendered via portal */}
      <TooltipPortal targetRef={buttonRef} show={showTooltip}>
        <TooltipContent
          title={`${name} (${label})`}
          description={definition}
          position="top"
          color={colors.tooltipColor}
        />
      </TooltipPortal>
    </>
  )
}

/**
 * Collapsible concepts component displaying extracted knowledge graph concepts
 *
 * Shows a purple collapsible container with concept count in header.
 * Concepts are color-coded by their Neo4j node label.
 * Terms with definitions show a popover on hover/tap.
 */
export function EntityBadges({ entities }) {
  const [isOpen, setIsOpen] = useState(true)
  const [showAll, setShowAll] = useState(false)

  if (!entities || entities.length === 0) return null

  const visibleEntities = showAll ? entities : entities.slice(0, MAX_VISIBLE)
  const hiddenCount = entities.length - MAX_VISIBLE
  const hasOverflow = hiddenCount > 0 && !showAll

  // Get name for key - handles both string and object entities
  const getEntityKey = (entity, index) => {
    const name = typeof entity === 'string' ? entity : entity.name
    return `${name}-${index}`
  }

  return (
    <div className="border border-purple-200 rounded-lg overflow-hidden bg-purple-50/30">
      {/* Header - always visible */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-purple-700 hover:bg-purple-50 transition-colors"
      >
        <ChevronIcon isOpen={isOpen} />
        <span className="font-medium">Concepts ({entities.length})</span>
        <Tooltip
          title="Knowledge Graph Concepts"
          description="Extracted concepts from the knowledge graph. Colors indicate node labels: purple for Concepts, blue for Definitions, teal for Entities. Items with a ? have definitions on hover."
          color="purple"
          position="top"
        >
          <span className="text-charcoal-muted hover:text-purple-600 transition-colors">
            <InfoIcon />
          </span>
        </Tooltip>
      </button>

      {/* Expandable content */}
      {isOpen && (
        <div className="px-3 pb-2 border-t border-purple-100 bg-ivory-light">
          <div className="flex flex-wrap gap-2 items-center pt-2">
            {visibleEntities.map((entity, index) => (
              <ConceptBadge key={getEntityKey(entity, index)} entity={entity} />
            ))}
            {hasOverflow && (
              <button
                onClick={() => setShowAll(true)}
                className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-ivory-medium text-charcoal-muted hover:bg-ivory transition-colors"
              >
                +{hiddenCount} more
              </button>
            )}
            {showAll && entities.length > MAX_VISIBLE && (
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
