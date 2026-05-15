# Knowledge Graph Application Filtering

## Overview

Add application-based filtering to the knowledge graph page. Users can filter the 3D graph to show only entities originating from documents belonging to a selected application. This filter works alongside the existing persona/ACL filter, with intersection (AND) semantics when both are active.

## Architecture

Extend the existing `GET /api/knowledge-graph/filtered` endpoint with an optional `app_id` query parameter. The server handles intersection of both filter dimensions in a single call, returning only entities that pass both filters.

## Backend

### Extended endpoint: `GET /api/knowledge-graph/filtered`

**New query parameter:** `app_id: int = 0`

**Logic when `app_id > 0`:**
1. Query `DocumentRecord` rows where `application_id == app_id`
2. Collect filenames from matching documents
3. Use the existing LightRAG chunk-to-entity tracing pattern to resolve which entity names appear in those documents' chunks

**Intersection with persona filter:**
- If both `app_id` and `groups` are provided: compute `app_allowed_entities` and `acl_allowed_entities` independently, then intersect the two sets
- If only one is provided: filter by that dimension alone
- If neither is provided (or both are "all"): return the full graph

### New function: `_get_app_allowed_entities(app_id: int) -> set[str] | None`

Located in `src/knowledge/graph_rag.py`. Follows the same pattern as `_get_acl_allowed_entities()`:

1. Query `MetadataStore` for documents with `application_id == app_id`
2. Collect filenames from those documents
3. Load `data/lightrag/kv_store_text_chunks.json` to find chunks from those files
4. Parse GraphML to find entity names referenced in those chunks
5. Return the set of allowed entity names (or `None` if no filtering needed)

## Frontend

### Template: `knowledge_graph.html`

**New UI element:** An "Application" `<select>` dropdown in the filter row (between persona and type filter).

- Populated from `{{ applications | tojson }}` (already passed by the route)
- Default option: "All Applications" (value="")
- Options show `app.name` with `app.id` as the value

**Unified filter function:** Refactor `loadGraphForPersona()` into `loadFilteredGraph()` that:
1. Reads the persona dropdown value (`groups`)
2. Reads the application dropdown value (`app_id`)
3. Builds URL: `/admin/api/knowledge-graph/filtered?groups=X&app_id=Y`
4. Both dropdowns call `loadFilteredGraph()` on change

The rest of the JS (graph rebuild, type filter repopulation) remains unchanged.

## Data Flow

```
User selects App + Persona
  -> JS: GET /api/knowledge-graph/filtered?app_id=3&groups=finance,executives
  -> Server: load full graph from GraphML
  -> Server: _get_app_allowed_entities(3) -> set of entity names from app 3's docs
  -> Server: _get_acl_allowed_entities(["finance","executives"]) -> set of entity names from ACL
  -> Server: intersect both sets
  -> Server: filter entities/relationships to intersection
  -> Response: JSON {entities, relationships}
  -> JS: rebuild 3D Force Graph
```

## Error Handling

- Invalid `app_id` (no matching application or no documents): return empty entity/relationship lists
- `app_id=0` or absent: no application filtering applied (same as current behavior)

## Files Modified

1. `src/knowledge/graph_rag.py` — add `_get_app_allowed_entities(app_id)` function
2. `src/admin/routes.py` — extend `knowledge_graph_filtered()` with `app_id` param and intersection logic
3. `src/admin/templates/knowledge_graph.html` — add application dropdown, refactor JS filter function
