(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const form = $('profile-editor');
  let book = JSON.parse($('profile-data').textContent);
  let selected = book.active.profile_id;
  let busy = false;
  let previewConfig = null;
  const bools = new Set(['graph_enrichment', 'structured_lookup', 'strategy_memory']);

  function config() {
    const values = {};
    for (const name of Object.keys(book.profiles[selected].draft)) {
      if (name !== 'clarification_fields') values[name] = form.elements.namedItem(name).value;
    }
    for (const name of bools) values[name] = values[name] === 'true';
    values.max_subtasks = Number(values.max_subtasks);
    values.clarification_fields = [...form.querySelectorAll('[name="clarification_fields"]:checked')].map(field => field.value);
    return values;
  }
  const signature = value => JSON.stringify(Object.keys(value).sort().map(k => [k, value[k]]));
  const dirty = () => signature(config()) !== signature(book.profiles[selected].draft);
  function updateControls() {
    for (const button of document.querySelectorAll('#profile-page button')) button.disabled = busy;
    for (const field of document.querySelectorAll('#profile-page input, #profile-page textarea, #profile-page select')) field.disabled = busy;
    $('profile-picker').disabled = busy;
    $('publish-profile').disabled = busy || dirty();
    $('activate-revision').disabled = busy || !$('revision-picker').value;
    $('draft-status').textContent = dirty() ? 'Unsaved edits. You can preview them before saving.' : 'Draft saved. Publishing makes it active for new questions.';
    if (previewConfig && previewConfig !== signature(config())) $('preview-status').textContent = 'The draft changed after this preview. Run it again to test the current edits.';
  }
  function activeRevision() {
    return book.profiles[book.active.profile_id].revisions.find(r => r.revision === book.active.revision);
  }
  function renderRevision() {
    const revision = book.profiles[selected].revisions.find(r => String(r.revision) === $('revision-picker').value);
    $('revision-settings').textContent = revision ? JSON.stringify(revision.config, null, 2) : 'This profile has no published revisions yet.';
    updateControls();
  }
  function render() {
    $('active-profile').textContent = `Active: ${activeRevision().config.name} · revision ${book.active.revision}`;
    $('profile-picker').replaceChildren(...Object.entries(book.profiles).map(([id, p]) => {
      const option = new Option(p.draft.name, id); option.selected = id === selected; return option;
    }));
    const draft = book.profiles[selected].draft;
    for (const [name, value] of Object.entries(draft)) {
      if (name === 'clarification_fields') {
        for (const field of form.querySelectorAll('[name="clarification_fields"]')) field.checked = value.includes(field.value);
      } else if (form.elements.namedItem(name)) form.elements.namedItem(name).value = String(value);
    }
    $('revision-picker').replaceChildren(...[...book.profiles[selected].revisions].reverse().map(r =>
      new Option(`Revision ${r.revision} · ${r.created_at} · ${r.author}`, r.revision)));
    if (book.active.profile_id === selected) $('revision-picker').value = String(book.active.revision);
    $('activation-history').replaceChildren(...[...book.activations].reverse().slice(0, 30).map(event => {
      const item = document.createElement('li');
      const revision = book.profiles[event.profile_id].revisions.find(r => r.revision === event.revision);
      item.textContent = `${revision.config.name} · r${event.revision} · ${event.action} · ${event.created_at}`;
      return item;
    }));
    if (!book.activations.length) {
      const item = document.createElement('li'); item.textContent = 'No publication changes yet.'; $('activation-history').append(item);
    }
    renderRevision();
  }
  function clearErrors() {
    $('profile-error').hidden = true;
    for (const field of form.querySelectorAll('[aria-invalid]')) {
      field.removeAttribute('aria-invalid');
      const ids = (field.getAttribute('aria-describedby') || '').split(' ').filter(id => id && id !== 'profile-error');
      if (ids.length) field.setAttribute('aria-describedby', ids.join(' ')); else field.removeAttribute('aria-describedby');
    }
  }
  async function request(path, body, method = 'POST') {
    const response = await fetch('/admin/api/answer-profiles' + path, {
      method, credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
      ...(body ? {body: JSON.stringify(body)} : {})
    });
    let data;
    try { data = await response.json(); } catch (_) { throw new Error('The server returned an unreadable response. Check the connection and try again.'); }
    if (!response.ok) {
      if (Array.isArray(data.detail)) {
        for (const error of data.detail) {
          const name = error.loc[error.loc.length - 1];
          const field = form.elements.namedItem(name);
          if (field && field.setAttribute) {
            field.setAttribute('aria-invalid', 'true');
            field.setAttribute('aria-describedby', `${field.getAttribute('aria-describedby') || ''} profile-error`.trim());
          }
        }
      }
      throw new Error(Array.isArray(data.detail) ? data.detail.map(e => `${e.loc.slice(1).join('.')}: ${e.msg}`).join(' ') : data.detail || `Request failed (${response.status}).`);
    }
    return data;
  }
  async function action(fn) {
    if (busy) return;
    clearErrors(); busy = true; updateControls();
    try { await fn(); } catch (error) {
      $('profile-error').textContent = error.message;
      $('profile-error').hidden = false; $('profile-error').focus();
      $('profile-status').textContent = 'Action did not complete in this page. Reload saved profiles to check the current state.';
    } finally { busy = false; updateControls(); }
  }
  async function showPrompt() {
    if (!form.reportValidity()) return;
    await action(async () => {
      const result = await request(`/${selected}/prompt`, {config: config()});
      $('resolved-prompt').textContent = result.system_prompt;
      $('routing-prompt').textContent = result.routing_prompt;
    });
  }
  form.addEventListener('input', () => { clearErrors(); updateControls(); });
  form.addEventListener('change', updateControls);
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    action(async () => {
      book = await request(`/${selected}/draft`, {expected_version: book.version, config: config()}, 'PUT');
      render(); $('profile-status').textContent = 'Draft saved. The published profile is unchanged.';
    });
  });
  $('profile-picker').addEventListener('change', event => {
    if (dirty() && !window.confirm('Discard unsaved edits and open another profile?')) { event.target.value = selected; return; }
    selected = event.target.value; previewConfig = null; $('preview-result').hidden = true; $('preview-status').textContent = 'No preview for this profile yet.';
    clearErrors(); render();
    if ($('prompt-disclosure').open) showPrompt();
  });
  $('new-profile').addEventListener('click', () => action(async () => {
    if (!form.reportValidity()) return;
    const copy = config(); copy.name = `Copy of ${copy.name}`.slice(0, 100);
    const result = await request('', {expected_version: book.version, config: copy});
    book = result.book; selected = result.profile_id; previewConfig = null; $('preview-result').hidden = true; $('preview-status').textContent = 'No preview for this profile yet.'; render();
    $('profile-name').focus(); $('profile-status').textContent = 'Profile copy created as a draft.';
  }));
  $('reload-profiles').addEventListener('click', () => {
    if (dirty() && !window.confirm('Discard unsaved edits and reload saved profiles?')) return;
    action(async () => { book = await request('', null, 'GET'); render(); $('profile-status').textContent = 'Saved profiles reloaded.'; });
  });
  $('publish-profile').addEventListener('click', () => action(async () => {
    book = await request(`/${selected}/publish`, {expected_version: book.version});
    render(); $('profile-status').textContent = `Published and activated revision ${book.active.revision}.`;
  }));
  $('revision-picker').addEventListener('change', renderRevision);
  $('activate-revision').addEventListener('click', () => action(async () => {
    book = await request(`/${selected}/revisions/${$('revision-picker').value}/activate`, {expected_version: book.version});
    render(); $('profile-status').textContent = `Activated revision ${book.active.revision}. The working draft is preserved.`;
  }));
  $('prompt-disclosure').addEventListener('toggle', () => { if ($('prompt-disclosure').open) showPrompt(); });
  $('refresh-prompt').addEventListener('click', showPrompt);
  $('profile-preview-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    action(async () => {
      const draft = config();
      $('preview-result').hidden = true;
      $('preview-status').textContent = 'Running draft preview. Broader searches can take several minutes…';
      $('preview-result').setAttribute('aria-busy', 'true');
      try {
        const result = await request(`/${selected}/preview`, {config: draft,
          question: $('preview-question').value,
          user_groups: [...new Set($('preview-groups').value.split(',').map(v => v.trim()).filter(Boolean))],
          dataset_id: Number($('preview-dataset').value)});
        previewConfig = signature(draft);
        $('preview-status').textContent = 'Preview complete. Nothing was published.';
        $('preview-meta').textContent = `${result.response_kind.replaceAll('_', ' ')} · ${result.query_type} · ${result.elapsed_seconds}s · ${result.chunks_retrieved} retrieved passages`;
        $('preview-answer').textContent = result.answer;
        $('preview-warnings').replaceChildren(...result.warnings.map(w => { const item = document.createElement('li'); item.textContent = w; return item; }));
        const cited = new Set(result.citations.map(c => c.evidence_id));
        $('preview-evidence').replaceChildren(...result.evidence.map(e => {
          const details = document.createElement('details');
          const summary = document.createElement('summary');
          summary.textContent = `[${e.evidence_id}] ${e.filename}${e.page == null ? '' : ` · page ${e.page}`}${cited.has(e.evidence_id) ? ' · cited' : ' · not cited'}`;
          const passage = document.createElement('pre'); passage.textContent = e.snippet;
          details.append(summary, passage);
          try {
            const url = new URL(e.source_url);
            if (['http:', 'https:'].includes(url.protocol)) {
              const link = document.createElement('a'); link.href = url.href; link.textContent = 'Open source'; link.target = '_blank'; link.rel = 'noopener noreferrer'; details.append(link);
            }
          } catch (_) { /* No external source link. */ }
          return details;
        }));
        if (!result.evidence.length) $('preview-evidence').textContent = 'No usable evidence was supplied.';
        $('resolved-prompt').textContent = result.system_prompt;
        $('preview-result').hidden = false;
      } catch (error) { $('preview-status').textContent = 'Preview failed. No settings were changed.'; throw error; }
      finally { $('preview-result').removeAttribute('aria-busy'); }
    });
  });
  window.addEventListener('beforeunload', event => { if (dirty()) { event.preventDefault(); event.returnValue = ''; } });
  render();
})();
