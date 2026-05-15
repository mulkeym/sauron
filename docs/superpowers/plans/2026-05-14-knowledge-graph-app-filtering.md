# Knowledge Graph Application Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add application-based filtering to the knowledge graph so users can view only entities from a specific application's documents, composable with the existing persona/ACL filter via intersection.

**Architecture:** Extend the existing `/api/knowledge-graph/filtered` endpoint with an `app_id` parameter. A new `_get_app_allowed_entities()` function in `graph_rag.py` traces Application → Documents → Chunks → Entities using the same pattern as the existing ACL filter. The template gets an application dropdown that feeds into a unified `loadFilteredGraph()` JS function.

**Tech Stack:** Python/FastAPI, SQLAlchemy, LightRAG GraphML, 3D Force Graph (JS)

---

### Task 1: Add `_get_app_allowed_entities()` to graph_rag.py

**Files:**
- Modify: `src/knowledge/graph_rag.py:120-187`
- Test: `tests/test_admin/test_routes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_admin/test_routes.py`:

```python
def test_knowledge_graph_filtered_by_app(client):
    """Filtering by app_id returns only entities from that app's documents."""
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.list_applications.return_value = []
        mock_get.return_value = store

        with patch("src.admin.routes._load_lightrag_graph") as mock_graph:
            mock_graph.return_value = (
                [{"name": "Acme Corp", "type": "organization"}, {"name": "Bob Smith", "type": "person"}],
                [{"source": "Acme Corp", "target": "Bob Smith", "label": "employs"}],
            )
            with patch("src.knowledge.graph_rag._get_app_allowed_entities") as mock_app_filter:
                mock_app_filter.return_value = {"Acme Corp"}
                resp = client.get("/admin/api/knowledge-graph/filtered?app_id=1")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "Acme Corp"
    assert len(data["relationships"]) == 0  # Bob filtered out, so edge is removed


def test_knowledge_graph_filtered_by_app_and_persona(client):
    """When both app_id and groups are set, result is the intersection."""
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        mock_get.return_value = store

        with patch("src.admin.routes._load_lightrag_graph") as mock_graph:
            mock_graph.return_value = (
                [
                    {"name": "Acme Corp", "type": "organization"},
                    {"name": "Bob Smith", "type": "person"},
                    {"name": "Secret Project", "type": "concept"},
                ],
                [],
            )
            with patch("src.knowledge.graph_rag._get_app_allowed_entities") as mock_app:
                mock_app.return_value = {"Acme Corp", "Bob Smith"}  # app has these two
                with patch("src.knowledge.graph_rag._get_acl_allowed_entities") as mock_acl:
                    mock_acl.return_value = {"Acme Corp", "Secret Project"}  # persona sees these two
                    resp = client.get("/admin/api/knowledge-graph/filtered?app_id=1&groups=finance")

    assert resp.status_code == 200
    data = resp.json()
    # Intersection: only "Acme Corp" is in both sets
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "Acme Corp"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_admin/test_routes.py::test_knowledge_graph_filtered_by_app tests/test_admin/test_routes.py::test_knowledge_graph_filtered_by_app_and_persona -v`

Expected: FAIL — `_get_app_allowed_entities` does not exist yet, and the endpoint doesn't accept `app_id`.

- [ ] **Step 3: Add `_get_app_allowed_entities()` function**

Add after `_get_acl_allowed_entities()` in `src/knowledge/graph_rag.py` (after line 187):

```python
def _get_app_allowed_entities(app_id: int) -> set[str] | None:
    """Get entity names from documents belonging to a specific application.
    Returns None if app_id is 0 (no filtering).
    """
    if not app_id:
        return None

    import asyncio
    import json as json_mod
    from pathlib import Path
    from src.db.metadata import MetadataStore

    # Get filenames for documents in this application
    async def _fetch():
        store = MetadataStore()
        await store.init()
        docs = await store.list_documents()
        return {d.filename for d in docs if d.application_id == app_id}

    try:
        allowed_files = asyncio.run(_fetch())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            allowed_files = pool.submit(asyncio.run, _fetch()).result()

    if not allowed_files:
        return set()

    # Load chunk-to-file mapping
    chunks_file = Path("data/lightrag/kv_store_text_chunks.json")
    if not chunks_file.exists():
        return set()

    chunk_data = json_mod.loads(chunks_file.read_text())
    allowed_chunks = {cid for cid, data in chunk_data.items()
                      if data.get("file_path", "") in allowed_files}

    # Load graph and find entities from allowed chunks
    import re
    graphml = Path("data/lightrag/graph_chunk_entity_relation.graphml")
    if not graphml.exists():
        return set()

    content = graphml.read_text()
    allowed_entities = set()

    for match in re.finditer(
        r'<node id="([^"]+)"[^>]*>(.*?)</node>', content, re.DOTALL
    ):
        name = match.group(1)
        source_match = re.search(r'<data key="d3">(.*?)</data>', match.group(2), re.DOTALL)
        if source_match:
            source_chunks = source_match.group(1).replace("&lt;SEP&gt;", "<SEP>").split("<SEP>")
            if any(c.strip() in allowed_chunks for c in source_chunks):
                allowed_entities.add(name)

    return allowed_entities
```

- [ ] **Step 4: Commit**

```bash
git add src/knowledge/graph_rag.py
git commit -m "feat: add _get_app_allowed_entities for application-based graph filtering"
```

---

### Task 2: Extend the `/filtered` endpoint with `app_id` parameter

**Files:**
- Modify: `src/admin/routes.py:1060-1085`

- [ ] **Step 1: Update the endpoint signature and logic**

Replace the `knowledge_graph_filtered` function in `src/admin/routes.py` (lines 1060-1085) with:

```python
@router.get("/api/knowledge-graph/filtered")
async def knowledge_graph_filtered(groups: str = "", app_id: int = 0):
    """Return filtered graph data by persona ACL and/or application."""
    from fastapi.responses import JSONResponse
    import asyncio

    user_groups = [g.strip() for g in groups.split(",") if g.strip()] if groups else ["ALL"]

    entities, relationships = await asyncio.to_thread(_load_lightrag_graph)

    no_persona_filter = "ALL" in user_groups or not groups
    no_app_filter = app_id == 0

    if no_persona_filter and no_app_filter:
        return JSONResponse({"entities": entities, "relationships": relationships})

    # Compute allowed entity sets for each active filter
    allowed = None

    if not no_persona_filter:
        from src.knowledge.graph_rag import _get_acl_allowed_entities
        acl_allowed = await asyncio.to_thread(_get_acl_allowed_entities, user_groups)
        if acl_allowed is not None:
            allowed = acl_allowed

    if not no_app_filter:
        from src.knowledge.graph_rag import _get_app_allowed_entities
        app_allowed = await asyncio.to_thread(_get_app_allowed_entities, app_id)
        if app_allowed is not None:
            if allowed is not None:
                allowed = allowed & app_allowed  # intersection
            else:
                allowed = app_allowed

    if allowed is None:
        return JSONResponse({"entities": entities, "relationships": relationships})

    filtered_entities = [e for e in entities if e["name"] in allowed]
    filtered_names = {e["name"] for e in filtered_entities}
    filtered_rels = [r for r in relationships
                     if r["source"] in filtered_names and r["target"] in filtered_names]

    return JSONResponse({"entities": filtered_entities, "relationships": filtered_rels})
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `pytest tests/test_admin/test_routes.py -v`

Expected: All tests PASS including the two new ones.

- [ ] **Step 3: Commit**

```bash
git add src/admin/routes.py
git commit -m "feat: extend /filtered endpoint with app_id parameter for application filtering"
```

---

### Task 3: Add application dropdown and unified filter function to template

**Files:**
- Modify: `src/admin/templates/knowledge_graph.html`

- [ ] **Step 1: Add the application dropdown**

In `knowledge_graph.html`, insert a new `<div class="form-group">` for the application filter between the persona selector (line 20 `</select></div>`) and the type filter (line 22 `<div class="form-group">`). The new block:

```html
    <div class="form-group">
        <label for="kg-app">Application</label>
        <select id="kg-app" onchange="loadFilteredGraph()">
            <option value="">All Applications</option>
            {% for app in applications %}
            <option value="{{ app.id }}">{{ app.name }}</option>
            {% endfor %}
        </select>
    </div>
```

Also change the persona selector's `onchange` from `loadGraphForPersona()` to `loadFilteredGraph()`:

```html
        <select id="kg-persona" onchange="loadFilteredGraph()">
```

- [ ] **Step 2: Replace `loadGraphForPersona()` with `loadFilteredGraph()`**

Replace the `loadGraphForPersona` function (lines 148-195) with:

```javascript
async function loadFilteredGraph() {
    const groups = document.getElementById('kg-persona').value;
    const appId = document.getElementById('kg-app').value;
    const params = new URLSearchParams();
    if (groups) params.set('groups', groups);
    if (appId) params.set('app_id', appId);
    const qs = params.toString();
    const url = '/admin/api/knowledge-graph/filtered' + (qs ? '?' + qs : '');
    const resp = await fetch(url);
    const data = await resp.json();

    // Rebuild nodes and links from filtered data
    nodes.length = 0;
    links.length = 0;
    nodeMap.clear();

    data.entities.forEach(e => {
        if (!nodeMap.has(e.name)) {
            const node = { id: e.name, type: e.type, color: TYPE_COLORS[e.type] || '#9ca3af', connections: 0 };
            nodeMap.set(e.name, node);
            nodes.push(node);
        }
    });

    data.relationships.forEach(r => {
        if (!nodeMap.has(r.source)) {
            const node = { id: r.source, type: 'unknown', color: '#9ca3af', connections: 0 };
            nodeMap.set(r.source, node);
            nodes.push(node);
        }
        if (!nodeMap.has(r.target)) {
            const node = { id: r.target, type: 'unknown', color: '#9ca3af', connections: 0 };
            nodeMap.set(r.target, node);
            nodes.push(node);
        }
        nodeMap.get(r.source).connections++;
        nodeMap.get(r.target).connections++;
        links.push({ source: r.source, target: r.target, label: r.label || '' });
    });

    graph.graphData({ nodes, links });

    // Update type filter
    const filterEl = document.getElementById('kg-filter');
    filterEl.innerHTML = '<option value="">All types</option>';
    const newTypes = [...new Set(data.entities.map(e => e.type))].sort();
    newTypes.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = `${t} (${data.entities.filter(e => e.type === t).length})`;
        filterEl.appendChild(opt);
    });
}
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_admin/test_routes.py -v`

Expected: All tests PASS.

- [ ] **Step 4: Manual verification**

Start the app and navigate to `/admin/knowledge-graph`. Verify:
1. The "Application" dropdown appears between "View as" and "Filter by type"
2. Selecting an application reloads the graph with only entities from that app's documents
3. Selecting both an application and a persona shows only the intersection
4. Resetting both to "All" shows the full graph

- [ ] **Step 5: Commit**

```bash
git add src/admin/templates/knowledge_graph.html
git commit -m "feat: add application dropdown to knowledge graph with unified filter function"
```
