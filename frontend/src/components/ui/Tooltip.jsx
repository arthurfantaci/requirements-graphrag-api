import React, { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'

/**
 * Tooltip portal component that renders outside the DOM hierarchy
 * to avoid clipping by overflow-hidden containers.
 *
 * Uses fixed positioning calculated from getBoundingClientRect().
 * Automatically adjusts position to stay within viewport bounds.
 */
function TooltipPortal({ children, targetRef, show, position = 'top' }) {
  const [coords, setCoords] = useState({ top: 0, left: 0 })
  const [actualPosition, setActualPosition] = useState(position)
  const tooltipRef = useRef(null)

  useEffect(() => {
    if (show && targetRef.current) {
      const rect = targetRef.current.getBoundingClientRect()
      const tooltipWidth = 256 // w-64 = 16rem = 256px
      const tooltipHeight = 80 // Approximate height
      const padding = 16 // Minimum distance from viewport edge

      // Calculate horizontal position, keeping within viewport
      let left = rect.left + rect.width / 2
      const minLeft = tooltipWidth / 2 + padding
      const maxLeft = window.innerWidth - tooltipWidth / 2 - padding
      left = Math.max(minLeft, Math.min(maxLeft, left))

      // Determine vertical position based on available space
      let finalPosition = position
      let top

      if (position === 'top') {
        // Check if there's enough space above
        if (rect.top < tooltipHeight + padding) {
          // Not enough space above, flip to bottom
          finalPosition = 'bottom'
          top = rect.bottom + 8
        } else {
          top = rect.top - 8
        }
      } else {
        // Check if there's enough space below
        if (rect.bottom + tooltipHeight + padding > window.innerHeight) {
          // Not enough space below, flip to top
          finalPosition = 'top'
          top = rect.top - 8
        } else {
          top = rect.bottom + 8
        }
      }

      setCoords({ top, left })
      setActualPosition(finalPosition)
    }
  }, [show, targetRef, position])

  if (!show) return null

  const transformClass =
    actualPosition === 'top'
      ? '-translate-x-1/2 -translate-y-full'
      : '-translate-x-1/2'

  return createPortal(
    <div
      ref={tooltipRef}
      className={`fixed z-[9999] ${transformClass} pointer-events-none`}
      style={{ top: coords.top, left: coords.left }}
    >
      {React.cloneElement(children, { position: actualPosition })}
    </div>,
    document.body
  )
}

/**
 * Styled tooltip content with dark background and arrow
 */
function TooltipContent({ title, description, position = 'top', color = 'gray' }) {
  // Color variants for the tooltip background
  // Aligned with Neo4j node label color scheme
  const colorStyles = {
    gray: 'bg-gray-900',
    blue: 'bg-blue-900',
    emerald: 'bg-emerald-900',
    amber: 'bg-amber-900',
    rose: 'bg-rose-900',
    purple: 'bg-purple-900',
    teal: 'bg-teal-900',
    indigo: 'bg-indigo-900',
    green: 'bg-green-900',
  }

  const arrowColors = {
    gray: 'border-t-gray-900',
    blue: 'border-t-blue-900',
    emerald: 'border-t-emerald-900',
    amber: 'border-t-amber-900',
    rose: 'border-t-rose-900',
    purple: 'border-t-purple-900',
    teal: 'border-t-teal-900',
    indigo: 'border-t-indigo-900',
    green: 'border-t-green-900',
  }

  const arrowColorsBottom = {
    gray: 'border-b-gray-900',
    blue: 'border-b-blue-900',
    emerald: 'border-b-emerald-900',
    amber: 'border-b-amber-900',
    rose: 'border-b-rose-900',
    purple: 'border-b-purple-900',
    teal: 'border-b-teal-900',
    indigo: 'border-b-indigo-900',
    green: 'border-b-green-900',
  }

  const bgClass = colorStyles[color] || colorStyles.gray
  const arrowClass = position === 'top'
    ? (arrowColors[color] || arrowColors.gray)
    : (arrowColorsBottom[color] || arrowColorsBottom.gray)

  return (
    <div role="tooltip" className="w-64 max-w-xs">
      {/* Arrow pointing up (for bottom position) */}
      {position === 'bottom' && (
        <div className="flex justify-center">
          <div className={`w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-b-[6px] ${arrowClass.replace('border-t-', 'border-b-')}`} />
        </div>
      )}

      <div className={`${bgClass} text-white text-xs rounded-lg px-3 py-2 shadow-lg`}>
        {title && <p className="font-medium mb-1">{title}</p>}
        <p className="text-gray-300 leading-relaxed">{description}</p>
      </div>

      {/* Arrow pointing down (for top position) */}
      {position === 'top' && (
        <div className="flex justify-center">
          <div className={`w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] ${arrowClass}`} />
        </div>
      )}
    </div>
  )
}

/**
 * Tooltip trigger wrapper component
 *
 * Wraps any element and shows a styled tooltip on hover/focus.
 *
 * @param {string} title - Bold header text (optional)
 * @param {string} description - Tooltip body text
 * @param {string} position - 'top' or 'bottom'
 * @param {string} color - Color theme: 'gray', 'blue', 'emerald', 'amber', 'rose', 'purple'
 * @param {ReactNode} children - The trigger element
 */
export function Tooltip({ title, description, position = 'top', color = 'gray', children }) {
  const [show, setShow] = useState(false)
  const triggerRef = useRef(null)

  return (
    <>
      <span
        ref={triggerRef}
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onClick={() => setShow(!show)}
        className="cursor-help"
      >
        {children}
      </span>

      <TooltipPortal targetRef={triggerRef} show={show} position={position}>
        <TooltipContent
          title={title}
          description={description}
          position={position}
          color={color}
        />
      </TooltipPortal>
    </>
  )
}

export { TooltipPortal, TooltipContent }
