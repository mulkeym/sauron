(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  async function json(url, options) {
    const response = await fetch(url, {credentials: 'same-origin', ...options});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed.');
    return data;
  }
  async function inventory() {
    const docs = await json('/admin/api/figures/inventory');
    $('diagram-inventory').replaceChildren(...docs.map(doc => {
      const p = document.createElement('p');
      p.textContent = `${doc.filename}: ${doc.figures} stored figures, ${doc.analyzed} visually analyzed, ${doc.source_extracted || 0} with native source extraction, ${(doc.bytes / 1048576).toFixed(1)} MiB${doc.warnings.length ? ' — ' + doc.warnings.join(' ') : ''}`;
      return p;
    }));
    const pending = docs.filter(doc => doc.can_backfill);
    $('diagram-document').replaceChildren(...pending.map(doc => new Option(doc.filename, doc.doc_id)));
    $('diagram-backfill-button').disabled = !pending.length;
    if (!pending.length) $('diagram-backfill-status').textContent = 'No documents eligible for backfill. Documents without an original content hash need re-ingestion from the managed source.';
  }
  $('diagram-search').addEventListener('submit', async event => {
    event.preventDefault();
    const button = event.target.querySelector('button'); button.disabled = true;
    $('diagram-status').textContent = 'Searching…'; $('diagram-results').replaceChildren();
    try {
      const params = new URLSearchParams({query: $('diagram-query').value, groups: $('diagram-groups').value});
      const refs = await json('/admin/api/figures/search?' + params);
      $('diagram-results').replaceChildren(...refs.map(ref => {
        const figure = document.createElement('figure');
        const link = document.createElement('a');
        link.href = ref.content_url.replace('/api/v1/documents/', '/admin/api/figure-documents/');
        link.target = '_blank'; link.rel = 'noopener';
        const image = document.createElement('img'); image.src = link.href; image.alt = ref.caption || ref.figure_id;
        image.loading = 'lazy'; image.style.maxWidth = '100%'; image.style.maxHeight = '650px'; link.append(image);
        const caption = document.createElement('figcaption');
        caption.textContent = `${ref.filename} · ${ref.figure_id}${ref.page ? ' · page ' + ref.page : ''}${ref.slide ? ' · slide ' + ref.slide : ''} · ${ref.caption || ref.kind}`;
        const description = document.createElement('p'); description.textContent = ref.description;
        const notices = document.createElement('p'); notices.textContent = (ref.render_warnings || []).join(' ');
        figure.append(link, caption, description, notices); return figure;
      }));
      $('diagram-status').textContent = refs.length ? `${refs.length} candidate diagrams. Check the source and site/version before applying a topology.` : 'No stored diagrams matched in the permitted documents.';
    } catch (error) { $('diagram-status').textContent = error.message; }
    finally { button.disabled = false; }
  });
  $('diagram-backfill').addEventListener('submit', async event => {
    event.preventDefault(); $('diagram-backfill-button').disabled = true;
    $('diagram-backfill-status').textContent = 'Verifying source and extracting images…';
    try {
      const body = new FormData(); body.append('file', $('diagram-source').files[0]);
      const result = await json(`/admin/api/figure-documents/${encodeURIComponent($('diagram-document').value)}/backfill`, {method: 'POST', body});
      await inventory();
      $('diagram-backfill-status').textContent = `Stored ${result.stored} figures. ${(result.warnings || []).join(' ')}`;
    } catch (error) { $('diagram-backfill-status').textContent = error.message; $('diagram-backfill-button').disabled = false; }
  });
  inventory().catch(error => { $('diagram-backfill-status').textContent = error.message; });
})();
