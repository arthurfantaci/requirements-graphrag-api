const MAX_ROWS = 50

/**
 * Format a cell value for display
 */
function formatCellValue(value) {
  if (value === null || value === undefined) {
    return <span className="text-gray-400 italic">null</span>
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
        className="text-blue-600 hover:underline truncate block max-w-xs"
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
      <div className="border border-gray-200 rounded-lg p-4 text-center text-gray-500 text-sm">
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
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 border-b border-gray-200">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          Results
        </span>
        <span className="text-xs text-gray-500">
          {hasMore ? `Showing ${MAX_ROWS} of ${totalRows} rows` : `${totalRows} row${totalRows !== 1 ? 's' : ''}`}
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-100">
            {displayResults.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-gray-50">
                {columns.map((column) => (
                  <td
                    key={`${rowIndex}-${column}`}
                    className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"
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
        <div className="px-3 py-2 bg-gray-50 border-t border-gray-200 text-center text-xs text-gray-500">
          {totalRows - MAX_ROWS} more row{totalRows - MAX_ROWS !== 1 ? 's' : ''} not shown
        </div>
      )}
    </div>
  )
}
