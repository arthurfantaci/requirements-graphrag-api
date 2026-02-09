"""Shared context formatting for LLM prompt assembly.

Provides a single code path for building the hybrid context string used
by both the production orchestrator and the evaluation pipeline.

The hybrid format places per-chunk entity names inline and appends a
deduplicated Knowledge Graph Context section (glossary, relationships,
industry standards) after the chunks.  Media resources are extracted to
a separate structure for SSE delivery but excluded from the context
string to keep token usage focused on textual knowledge.

Usage:
    from requirements_graphrag_api.core.context import (
        NormalizedDocument, format_context, format_entity_info_for_synthesis,
    )

    docs = [NormalizedDocument.from_retrieved_document(d) for d in ranked_results]
    formatted = format_context(docs)
    # formatted.context  -> hybrid context string for {context} prompt variable
    # formatted.resources -> media for SSE events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from requirements_graphrag_api.core.agentic.state import EntityInfo, RetrievedDocument
    from requirements_graphrag_api.core.definitions import Resource


# =============================================================================
# NORMALIZED DOCUMENT — adapter for the two data shapes
# =============================================================================


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """Normalized view of an enriched document.

    Adapts both ``RetrievedDocument`` (production) and raw ``dict``
    (evaluation pipeline) into a single shape consumed by
    :func:`format_context`.
    """

    content: str
    source: str
    score: float = 0.0
    entities: list[dict[str, Any]] = field(default_factory=list)
    glossary_definitions: list[dict[str, Any]] = field(default_factory=list)
    semantic_relationships: list[dict[str, Any]] = field(default_factory=list)
    industry_standards: list[dict[str, Any]] = field(default_factory=list)
    media: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    url: str = ""
    chunk_id: str = ""

    # ------------------------------------------------------------------
    # Factory: from RetrievedDocument (production path)
    # ------------------------------------------------------------------
    @classmethod
    def from_retrieved_document(cls, doc: RetrievedDocument) -> NormalizedDocument:
        """Create from a :class:`RetrievedDocument` instance.

        Reads enrichment data from ``doc.metadata``.
        """
        meta = doc.metadata or {}
        return cls(
            content=doc.content,
            source=doc.source,
            score=doc.score,
            entities=_coerce_entity_list(meta.get("entities")),
            glossary_definitions=_coerce_list(meta.get("glossary_definitions")),
            semantic_relationships=_normalize_relationships(meta.get("semantic_relationships")),
            industry_standards=_coerce_list(meta.get("industry_standards")),
            media=meta.get("media") or {},
            url=meta.get("url", ""),
            chunk_id=meta.get("chunk_id", ""),
        )

    # ------------------------------------------------------------------
    # Factory: from raw dict (evaluation / graph_enriched_search path)
    # ------------------------------------------------------------------
    @classmethod
    def from_raw_result(cls, result: dict[str, Any]) -> NormalizedDocument:
        """Create from a raw dict as returned by ``graph_enriched_search``.

        Enrichment keys live at the top level; source info in
        ``result["metadata"]["title"]``.
        """
        meta = result.get("metadata") or {}
        return cls(
            content=result.get("content", ""),
            source=meta.get("title", "Unknown"),
            score=result.get("score", 0.0),
            entities=_coerce_entity_list(result.get("entities")),
            glossary_definitions=_coerce_list(result.get("glossary_definitions")),
            semantic_relationships=_normalize_relationships(result.get("semantic_relationships")),
            industry_standards=_coerce_list(result.get("industry_standards")),
            media=result.get("media") or {},
            url=meta.get("url", ""),
            chunk_id=meta.get("chunk_id", ""),
        )


# =============================================================================
# FORMATTED CONTEXT — output of format_context()
# =============================================================================


@dataclass(frozen=True, slots=True)
class FormattedContext:
    """Result of :func:`format_context`.

    Attributes:
        context: Hybrid context string (chunks + KG section) for the
            ``{context}`` prompt variable.
        sources: Source metadata list for SSE ``sources`` event.
        entities_by_name: Deduplicated entities keyed by name.
        entities_str: Comma-joined entity names (for logging / light use).
        resources: Media grouped by type — NOT included in context string.
    """

    context: str
    sources: list[dict[str, Any]]
    entities_by_name: dict[str, dict[str, Any]]
    entities_str: str
    resources: dict[str, list[Any]] = field(default_factory=dict)


# =============================================================================
# PUBLIC API
# =============================================================================


def format_context(
    documents: list[NormalizedDocument],
    *,
    definitions: list[dict[str, Any]] | None = None,
    max_documents: int = 10,
    max_resources_per_type: int = 3,
) -> FormattedContext:
    """Build a hybrid context string from normalized documents.

    **Hybrid format** (what the LLM receives):

    1. Per-chunk sections with inline entity names.
    2. A deduplicated **Knowledge Graph Context** section containing
       glossary definitions, semantic relationships, and industry
       standards gathered across all chunks.

    Media resources are extracted into ``FormattedContext.resources``
    but are **not** embedded in the context string.

    Args:
        documents: Normalized documents (at most *max_documents* used).
        definitions: Optional glossary lookup results (from evaluation
            pipeline's ``search_terms``).  Prepended as definition
            blocks when present.
        max_documents: Cap on the number of chunk sections.
        max_resources_per_type: Max images/webinars/videos per source.

    Returns:
        A :class:`FormattedContext` with the assembled context string,
        source metadata, deduplicated entities, and media resources.
    """
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    all_entities: dict[str, dict[str, Any]] = {}

    # Deduplication sets for KG Context section
    seen_glossary: dict[str, str] = {}  # term -> definition
    all_relationships: list[str] = []
    seen_rel_keys: set[str] = set()
    all_standards: list[str] = []
    seen_std_keys: set[str] = set()

    # Media collection
    seen_urls: set[str] = set()
    all_images: list[Resource] = []
    all_webinars: list[Resource] = []
    all_videos: list[Resource] = []

    # Pre-collect webinar thumbnail URLs for image filtering
    webinar_thumbnail_urls: set[str] = set()
    for doc in documents[:max_documents]:
        for w in doc.media.get("webinars", []):
            thumb = w.get("thumbnail_url")
            if thumb:
                webinar_thumbnail_urls.add(thumb)

    # -----------------------------------------------------------------
    # Optional: definition blocks from glossary lookup
    # -----------------------------------------------------------------
    if definitions:
        for defn in definitions:
            if defn.get("score", 0) >= 0.5:
                term_display = defn["term"]
                if defn.get("acronym"):
                    term_display = f"{defn['term']} ({defn['acronym']})"
                defn_url = defn.get("url", "")
                context_parts.append(
                    f"[Definition: {term_display}]\n{defn['definition']}\nURL: {defn_url}\n"
                )
                sources.append(
                    {
                        "title": f"Definition: {term_display}",
                        "url": defn_url,
                        "chunk_id": None,
                        "relevance_score": defn.get("score", 0.5),
                    }
                )
                all_entities[defn["term"]] = {
                    "definition": defn.get("definition"),
                    "label": "Definition",
                }

    # -----------------------------------------------------------------
    # Per-chunk sections
    # -----------------------------------------------------------------
    for i, doc in enumerate(documents[:max_documents], 1):
        # Collect entity names for inline display
        entity_names: list[str] = []
        for entity in doc.entities:
            if isinstance(entity, dict) and entity.get("name"):
                name = entity["name"]
                entity_names.append(name)
                label = entity.get("type", "Entity")
                definition = entity.get("definition")
                if name not in all_entities:
                    all_entities[name] = {"definition": definition, "label": label}
                elif definition and not all_entities[name].get("definition"):
                    all_entities[name]["definition"] = definition
            elif isinstance(entity, str) and entity:
                entity_names.append(entity)
                if entity not in all_entities:
                    all_entities[entity] = {"definition": None, "label": "Entity"}

        # Glossary definitions → entities + KG section
        for defn in doc.glossary_definitions:
            if isinstance(defn, dict) and defn.get("term"):
                term = defn["term"]
                definition = defn.get("definition", "")
                if term not in all_entities:
                    all_entities[term] = {"definition": definition, "label": "Definition"}
                elif definition and not all_entities[term].get("definition"):
                    all_entities[term]["definition"] = definition
                    all_entities[term]["label"] = "Definition"
                # Collect for KG section
                if term not in seen_glossary and definition:
                    seen_glossary[term] = definition
            elif isinstance(defn, str) and defn:
                if defn not in all_entities:
                    all_entities[defn] = {"definition": None, "label": "Entity"}

        # Semantic relationships → KG section
        for rel in doc.semantic_relationships:
            if isinstance(rel, dict):
                from_e = rel.get("from_entity", "")
                rel_type = rel.get("relationship", "")
                to_e = rel.get("to_entity", "")
                if from_e and rel_type and to_e:
                    key = f"{from_e}|{rel_type}|{to_e}"
                    if key not in seen_rel_keys:
                        seen_rel_keys.add(key)
                        all_relationships.append(f"- {from_e} -> {rel_type} -> {to_e}")

        # Industry standards → KG section
        for std in doc.industry_standards:
            if isinstance(std, dict):
                standard = std.get("standard", "")
                org = std.get("organization", "")
                std_def = std.get("standard_definition", "")
                if standard:
                    key = standard
                    if key not in seen_std_keys:
                        seen_std_keys.add(key)
                        parts = [f"- {standard}"]
                        if std_def:
                            parts[0] += f": {std_def}"
                        if org:
                            parts[0] += f" (Organization: {org})"
                        all_standards.append(parts[0])

        # Build the chunk section
        section = f"[Source {i}: {doc.source}]"
        if entity_names:
            section += f"\n(Entities: {', '.join(entity_names)})"
        section += f"\n{doc.content}"
        context_parts.append(section)

        sources.append(
            {
                "title": doc.source,
                "content": doc.content,
                "url": doc.url,
                "chunk_id": doc.chunk_id,
                "relevance_score": doc.score,
            }
        )

        # ----- Media extraction (NOT in context string) -----
        _collect_media(
            doc,
            doc.source,
            seen_urls,
            webinar_thumbnail_urls,
            all_images,
            all_webinars,
            all_videos,
            max_resources_per_type,
        )

    # -----------------------------------------------------------------
    # Knowledge Graph Context section (appended after chunks)
    # -----------------------------------------------------------------
    kg_parts: list[str] = []
    if seen_glossary:
        kg_parts.append("### Glossary Definitions")
        for term, definition in seen_glossary.items():
            kg_parts.append(f"- **{term}**: {definition}")
    if all_relationships:
        kg_parts.append("### Semantic Relationships")
        kg_parts.extend(all_relationships)
    if all_standards:
        kg_parts.append("### Industry Standards")
        kg_parts.extend(all_standards)

    # -----------------------------------------------------------------
    # Assemble final context
    # -----------------------------------------------------------------
    if not context_parts and not kg_parts:
        context = "No relevant context found."
    else:
        context = "\n\n---\n\n".join(context_parts)
        if kg_parts:
            context += "\n\n---\n\n## Knowledge Graph Context\n\n" + "\n".join(kg_parts)

    sorted_names = sorted(all_entities.keys())[:20]
    entities_str = ", ".join(sorted_names) if sorted_names else "None identified"

    return FormattedContext(
        context=context,
        sources=sources,
        entities_by_name=all_entities,
        entities_str=entities_str,
        resources={
            "images": all_images,
            "webinars": all_webinars,
            "videos": all_videos,
        },
    )


def format_entity_info_for_synthesis(entity_contexts: list[EntityInfo]) -> str:
    """Format research subgraph ``EntityInfo`` objects for the ``{entities}`` prompt variable.

    This produces a structured text block distinct from the enrichment
    data in ``{context}``.  Research entities come from deep graph
    traversal in the research subgraph, not from per-chunk enrichment.

    Args:
        entity_contexts: Entity information from the research subgraph.

    Returns:
        Formatted string for the ``{entities}`` SYNTHESIS variable,
        or empty string if no entities.
    """
    if not entity_contexts:
        return ""

    parts: list[str] = []
    for entity in entity_contexts:
        info = f"**{entity.name}** ({entity.entity_type})"
        if entity.description:
            info += f": {entity.description}"
        if entity.related_entities:
            info += f"\n  Related: {', '.join(entity.related_entities[:5])}"
        if entity.mentioned_in:
            info += f"\n  Mentioned in: {', '.join(entity.mentioned_in[:3])}"
        parts.append(info)

    return "\n\n".join(parts)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _coerce_entity_list(value: Any) -> list[dict[str, Any]]:
    """Coerce entities to a list of dicts.

    Handles: None, list[dict], list[str], and other edge cases.
    """
    if not value:
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str) and item:
            result.append({"name": item, "type": "Entity"})
    return result


def _coerce_list(value: Any) -> list[dict[str, Any]]:
    """Return *value* as a list of dicts, or empty list."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_relationships(value: Any) -> list[dict[str, Any]]:
    """Handle ``semantic_relationships`` type inconsistency.

    The field can be a list of dicts **or** a single dict (observed in
    some graph traversal paths).  Normalizes to list[dict].
    """
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _collect_media(
    doc: NormalizedDocument,
    source_title: str,
    seen_urls: set[str],
    webinar_thumbnail_urls: set[str],
    all_images: list[Any],
    all_webinars: list[Any],
    all_videos: list[Any],
    max_per_type: int,
) -> None:
    """Extract media resources from a document (mutates collection lists)."""
    from requirements_graphrag_api.core.definitions import Resource

    if not doc.media:
        return

    count = 0
    for img in doc.media.get("images", []):
        img_url = img.get("url")
        if not img_url or img_url in seen_urls or img_url in webinar_thumbnail_urls:
            continue
        if count >= max_per_type:
            break
        seen_urls.add(img_url)
        count += 1
        alt_text = img.get("alt_text", "")
        all_images.append(
            Resource(
                title=alt_text or "Image",
                url=img_url,
                alt_text=alt_text,
                source_title=source_title,
            )
        )

    count = 0
    for webinar in doc.media.get("webinars", []):
        webinar_url = webinar.get("url")
        if not webinar_url or webinar_url in seen_urls:
            continue
        if count >= max_per_type:
            break
        seen_urls.add(webinar_url)
        count += 1
        all_webinars.append(
            Resource(
                title=webinar.get("title", "Webinar"),
                url=webinar_url,
                source_title=source_title,
                thumbnail_url=webinar.get("thumbnail_url", ""),
            )
        )

    count = 0
    for video in doc.media.get("videos", []):
        video_url = video.get("url")
        if not video_url or video_url in seen_urls:
            continue
        if count >= max_per_type:
            break
        seen_urls.add(video_url)
        count += 1
        all_videos.append(
            Resource(
                title=video.get("title", "Video"),
                url=video_url,
                source_title=source_title,
            )
        )


__all__ = [
    "FormattedContext",
    "NormalizedDocument",
    "format_context",
    "format_entity_info_for_synthesis",
]
