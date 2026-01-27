const MAX_ROWS = 50

/**
 * Format a cell value for display
 */
function formatCellValue(value) {
  if (value === null || value === undefined) {
    return <span className="text-charcoal-muted italic">null</span>
  }
  if (Array.isArray(value)) {
    return value.join(', ')
  }
  if (typeof value === 'object') {
    return JSON.stringify(value)
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }
  // Check if it's a URL
  if (typeof value === 'string' && (value.startsWith('http://') || value.startsWith('https://'))) {
    return (
      <a
        href={value}
        target="_blank"
        rel="noopener noreferrer"
        className="text-terracotta hover:underline truncate block max-w-xs"
        title={value}
      >
        {value}
      </a>
    )
  }
  return String(value)
}

/**
 * Results table component for structured query responses
 *
 * Displays query results in a clean data table format
 * Limits display to MAX_ROWS with total count indicator
 */
export function ResultsTable({ results, rowCount }) {
  if (!results || results.length === 0) {
    return (
      <div className="border border-black/10 rounded-lg p-4 text-center text-charcoal-muted text-sm">
        No results found
      </div>
    )
  }

  // Get column headers from first result
  const columns = Object.keys(results[0])
  const displayResults = results.slice(0, MAX_ROWS)
  const totalRows = rowCount ?? results.length
  const hasMore = totalRows > MAX_ROWS

  return (
    <div className="border border-black/10 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-ivory-medium border-b border-black/10">
        <span className="text-xs font-medium text-terracotta uppercase tracking-widest">
          Results
        </span>
        <span className="text-xs text-charcoal-muted">
          {hasMore ? `Showing ${MAX_ROWS} of ${totalRows} rows` : `${totalRows} row${totalRows !== 1 ? 's' : ''}`}
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-black/10">
          <thead className="bg-ivory-medium">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="px-3 py-2 text-left text-xs font-medium text-charcoal-muted uppercase tracking-wider"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-ivory-light divide-y divide-black/5">
            {displayResults.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-ivory-medium">
                {columns.map((column) => (
                  <td
                    key={`${rowIndex}-${column}`}
                    className="px-3 py-2 text-sm text-charcoal-light whitespace-nowrap"
                  >
                    {formatCellValue(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* More rows indicator */}
      {hasMore && (
        <div className="px-3 py-2 bg-ivory-medium border-t border-black/10 text-center text-xs text-charcoal-muted">
          {totalRows - MAX_ROWS} more row{totalRows - MAX_ROWS !== 1 ? 's' : ''} not shown
        </div>
      )}
    </div>
  )
}
