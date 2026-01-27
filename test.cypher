// =============================================================================
// Fix Issue #63: Remove orphaned webinar thumbnail Image nodes
// =============================================================================
// Phase 3 dedup (Issue #53) deleted duplicate Webinar nodes whose thumbnail_url
// values differed from the keeper nodes. The Image nodes referencing those old
// thumbnail URLs were left behind, causing them to leak into the IMAGES gallery.
//
// Run these queries in Neo4j Browser or cypher-shell.
// =============================================================================

// ---------------------------------------------------------------------------
// STEP 1: Audit — identify orphaned Image nodes from deleted duplicates
// ---------------------------------------------------------------------------
// These filename fragments come from the deleted duplicate Webinar nodes:
//   - Best-Practices-Requirements-Traceability.png  (Group 3 — confirmed leak)
//   - Group-1599.png                                (Group 3 — potential leak)
//   - Group-1603.png                                (Group 1 — potential leak)

MATCH (img:Image)
WHERE img.url CONTAINS 'Best-Practices-Requirements-Traceability'
   OR img.url CONTAINS 'Group-1599'
   OR img.url CONTAINS 'Group-1603'
RETURN img.url, elementId(img) AS id;

// ---------------------------------------------------------------------------
// STEP 2: Delete — remove the orphaned Image nodes identified above
// ---------------------------------------------------------------------------
// Only run this AFTER confirming the audit results in Step 1.

MATCH (img:Image)
WHERE img.url CONTAINS 'Best-Practices-Requirements-Traceability'
   OR img.url CONTAINS 'Group-1599'
   OR img.url CONTAINS 'Group-1603'
DETACH DELETE img;

// ---------------------------------------------------------------------------
// STEP 3: Verify — confirm no orphaned nodes remain
// ---------------------------------------------------------------------------

MATCH (img:Image)
WHERE img.url CONTAINS 'Best-Practices-Requirements-Traceability'
   OR img.url CONTAINS 'Group-1599'
   OR img.url CONTAINS 'Group-1603'
RETURN count(img) AS remaining;
