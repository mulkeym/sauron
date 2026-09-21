"""Source-derived Mermaid component views for native Visio evidence."""
import re


def native_mermaid_answer(state, pack):
    """Offer a bounded component view without inventing network relationships.

    This intentionally does not reverse-engineer topology from icon positions or
    drawing groups. A group of shapes is not proof of network containment.
    """
    from src.agent.profiles import profile_for_state
    question = state.get('question', '')
    if not re.search(r'\bmermaid\b', question, re.I):
        return None
    # Explicit multi-page/connection requirements need normal synthesis/policy.
    if re.search(r'\b(?:compare|all|every|exact|connections?|flows?|arrows?|directions?)\b', question, re.I):
        return None
    profile = profile_for_state(state)
    if profile and profile.insufficient_evidence != 'partial':
        return None
    candidates = [c for c in pack.citations if c.figure_id and c.source_kind == 'document']
    if not re.search(r'\b(?:resources|fonts?|logos?|accessibility|colou?rs?)\b', question, re.I):
        candidates = [c for c in candidates if not (c.caption or '').lower().startswith('resources:')]
    if not candidates:
        return None
    # Follow retrieval ranking; name the chosen page rather than silently merge pages.
    first = max(candidates, key=lambda c: c.relevance)
    sources = [c for c in candidates if (c.doc_id, c.figure_id) == (first.doc_id, first.figure_id)]
    # Prefer the largest explicitly saved drawing group to omit page headings
    # and template instructions. This is graphical grouping, not network containment.
    grouped = {}
    for c in sources:
        for line in c.snippet.splitlines():
            m = re.fullmatch(r'Shape \d+ \(.*?\) in group (\d+): (.+)', line)
            if m and len(m[2]) <= 100:
                grouped.setdefault(m[1], set()).add(m[2])
    selected_group = max(grouped, key=lambda key: len(grouped[key])) if grouped else None
    if selected_group and len(grouped[selected_group]) < 4:
        selected_group = None
    nodes = {}
    seen_labels = set()
    used = []
    for c in sources:
        added = False
        for line in c.snippet.splitlines():
            match = re.fullmatch(r'Shape (\d+) \(.*?\)(?: in group (\d+))?: (.+)', line)
            if not match:
                continue
            sid, group, label = match.groups()
            if selected_group and group != selected_group:
                continue
            # Long prose is better left in the cited passage, not a diagram node.
            if (len(label) > 100 or len(nodes) >= 40 or sid in nodes
                    or label in seen_labels or label.lower().startswith('lorem ipsum')):
                continue
            nodes[sid] = label
            seen_labels.add(label)
            added = True
        if added:
            used.append(c)
    if not nodes:
        return None
    def literal(label):
        # Mermaid entity syntax keeps source labels from becoming diagram code.
        return ''.join(f'#{ord(ch)};' if not (ch.isalnum() or ch in ' -_.,:()/') else ch for ch in label)
    code = 'flowchart TB\n' + '\n'.join(f'    S{sid}["{literal(label)}"]' for sid, label in nodes.items())
    refs = ' '.join(f'[{c.evidence_id}]' for c in used)
    from src.citations import citation_label
    from html import escape
    label = re.sub(r'([\\`*_{}\[\]!|])', r'\\\1', escape(citation_label(first), quote=False))
    caption = re.sub(r'([\\`*_{}\[\]!|])', r'\\\1', escape(first.caption or first.figure_id, quote=False))
    answer = (f'Here is a simplified **component view** of **{caption}** from {label}. '
              f'It uses {len(nodes)} distinct labels from the saved native Visio page evidence. {refs}\n\n'
              f'```mermaid\n{code}\n```\n\n'
              'This is a partial view of one page, not an exact redraw. Lines, arrowheads and network containment '
              'are not reconstructed here; drawing groups and label proximity alone do not establish traffic flow. '
              'Use the original diagram for the full layout and connections.')
    return {'answer': answer, 'citations': used, 'warnings': pack.warnings + [
        'Mermaid component view: network connections and arrow direction are not reconstructed.'], 'response_kind': 'answer'}
