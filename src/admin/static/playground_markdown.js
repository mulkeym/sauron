/* Render untrusted answer Markdown; source images come only from authorized references. */
(() => {
    'use strict';
    const escape = text => String(text).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
    const parser = window.marked && new marked.Marked({
        gfm: true,
        breaks: false,
        renderer: {
            html: ({text}) => escape(text),
            // Never create a fetching element until the source reference is checked.
            image: ({href, text}) => `<span data-sauron-image="${escape(href)}">${escape(text || 'Source image')}</span>`,
        },
    });

    function imagePath(value) {
        try {
            const url = new URL(value, window.location.href);
            if (url.origin !== window.location.origin || url.username || url.password || url.hash) return null;
            const path = url.pathname.replace(/^\/api\/v1\/documents\//, '/admin/api/figure-documents/');
            if (!/^\/admin\/api\/figure-documents\/[^/]+\/figures\/[^/]+\/content$/.test(path)) return null;
            if (!/^\?variant=(preview|full)$/.test(url.search)) return null;
            return path + url.search;
        } catch (_) { return null; }
    }

    function render(element, markdown, {images = [], streaming = false} = {}) {
        element.classList.add('result-answer');
        if (!parser || !window.DOMPurify || !DOMPurify.isSupported) {
            element.textContent = markdown; // Fail closed if local dependencies cannot load.
            return;
        }
        const fragment = DOMPurify.sanitize(parser.parse(String(markdown || '')), {
            RETURN_DOM_FRAGMENT: true,
            ALLOWED_TAGS: ['p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'strong', 'em', 'del', 'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
                'table', 'thead', 'tbody', 'tr', 'th', 'td', 'a', 'span'],
            ALLOWED_ATTR: ['href', 'title', 'class', 'start', 'align', 'data-sauron-image'],
            ALLOW_DATA_ATTR: false,
            ALLOW_ARIA_ATTR: false,
        });
        for (const link of fragment.querySelectorAll('a')) {
            try {
                const url = new URL(link.getAttribute('href'), window.location.href);
                if (!link.hasAttribute('href') || !['http:', 'https:', 'mailto:'].includes(url.protocol)) {
                    link.removeAttribute('href');
                } else if (/^#citation-\d+$/.test(link.getAttribute('href'))) {
                    link.addEventListener('click', event => {
                        const card = element.closest('.result-card')?.querySelector(url.hash);
                        if (!card) return;
                        event.preventDefault();
                        const passage = card.querySelector('details');
                        if (passage) passage.open = true;
                        card.focus();
                        card.scrollIntoView({block: 'nearest'});
                    });
                } else if (url.origin !== window.location.origin) {
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                }
            } catch (_) { link.removeAttribute('href'); }
        }
        const allowed = new Set(images.map(imagePath).filter(Boolean));
        for (const placeholder of fragment.querySelectorAll('[data-sauron-image]')) {
            const path = imagePath(placeholder.dataset.sauronImage);
            if (path && allowed.has(path)) {
                const image = document.createElement('img');
                image.src = path;
                image.alt = placeholder.textContent;
                image.loading = 'lazy';
                placeholder.replaceWith(image);
            } else {
                // Remote/data URLs and guessed document IDs never trigger a request.
                placeholder.replaceWith(document.createTextNode(`[Image: ${placeholder.textContent}]`));
            }
        }
        for (const table of fragment.querySelectorAll('table')) {
            const scroll = document.createElement('div');
            scroll.className = 'markdown-table-scroll';
            scroll.tabIndex = 0;
            scroll.setAttribute('role', 'region');
            scroll.setAttribute('aria-label', 'Answer table');
            table.replaceWith(scroll);
            scroll.append(table);
        }
        element.replaceChildren(fragment);
        if (streaming) {
            const cursor = document.createElement('span');
            cursor.className = 'streaming-cursor';
            cursor.setAttribute('aria-hidden', 'true');
            cursor.textContent = '▌';
            element.append(cursor);
        }
    }

    function renderResult(container) {
        // These figure cards were selected by the backend for the current persona.
        const images = [...container.querySelectorAll('.result-card > figure img')]
            .map(image => image.getAttribute('src'));
        for (const answer of container.querySelectorAll('.result-answer')) {
            render(answer, answer.textContent, {images});
        }
    }
    window.SauronMarkdown = Object.freeze({render, renderResult});
})();
