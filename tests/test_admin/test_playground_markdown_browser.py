"""Optional real-browser coverage: pip install playwright; playwright install chromium."""
import html
import io
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw

playwright = pytest.importorskip('playwright.sync_api')
ROOT = Path(__file__).resolve().parents[2]
IMAGE = '/admin/api/figure-documents/doc-1/figures/visio-page-0/content?variant=preview'
API_IMAGE = IMAGE.replace('/admin/api/figure-documents/', '/api/v1/documents/')
ANSWER = '''## Branch deployment

Use **two routers** with *redundant* uplinks. [Vendor guide](https://example.com/guide).

1. Configure the interfaces.
2. Verify the tunnel.

- Primary path
- Backup path

| Site | Role |
| --- | --- |
| Branch A | Active |
| Branch B | Standby |

```text
show sdwan control connections
<configuration> stays literal
```

> Verify the device version first.

![Branch topology](''' + API_IMAGE + ''')

### Operational checks

Monitor path health after deployment.
'''


@pytest.fixture
def browser_page():
    with playwright.sync_playwright() as pw:
        executable = os.environ.get('SAURON_TEST_BROWSER')
        if not executable and not Path(pw.chromium.executable_path).exists():
            pytest.skip('Install Playwright Chromium or set SAURON_TEST_BROWSER')
        browser = pw.chromium.launch(executable_path=executable, headless=True)
        page = browser.new_page(viewport={'width': 1200, 'height': 1000})
        requests = []
        status = {'step': 'complete', 'result_html': ''}
        image = Image.new('RGB', (600, 140), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 35, 150, 105), fill='#5599ff')
        draw.rectangle((450, 35, 580, 105), fill='#5599ff')
        draw.line((150, 70, 450, 70), fill='black', width=4)
        draw.polygon([(435, 60), (450, 70), (435, 80)], fill='black')
        raw = io.BytesIO(); image.save(raw, format='PNG')
        env = Environment(loader=FileSystemLoader(ROOT/'src/admin/templates'), autoescape=select_autoescape())
        template = env.get_template('playground.html').render(personas=[{'name':'network','display_name':'Network','groups':['network']}], datasets=[], uncovered_groups=[])

        def route_request(route):
            request = route.request
            url = urlsplit(request.url)
            requests.append(request.url)
            if url.netloc != 'sauron.test':
                route.fulfill(status=200, body='', content_type='application/javascript')
            elif url.path.startswith('/admin/static/'):
                path = ROOT/'src/admin/static'/url.path.removeprefix('/admin/static/')
                route.fulfill(path=str(path))
            elif url.path == '/admin/playground':
                route.fulfill(body=template, content_type='text/html')
            elif url.path == '/admin/api/playground/start':
                route.fulfill(json={'query_id': 'q1'})
            elif url.path == '/admin/api/playground/status/q1':
                route.fulfill(json=status)
            elif url.path + '?' + url.query == IMAGE:
                assert 'sauron_session=test-session' in request.headers.get('cookie', '')
                route.fulfill(body=raw.getvalue(), content_type='image/png')
            else:
                route.fulfill(status=404, body='not found')
        page.route('**/*', route_request)
        page.context.add_cookies([{'name':'sauron_session','value':'test-session','url':'http://sauron.test'}])
        page.add_init_script('''window.EventSource = class {
            constructor() { window.testStream = this; }
            close() { this.closed = true; }
            emit(data) { this.onmessage({data: JSON.stringify(data)}); }
        };''')
        page.goto('http://sauron.test/admin/playground')
        yield page, requests, status
        browser.close()


def final_html(answer=ANSWER):
    return ('<div class="trace-panel">Completed</div><p class="status-err">Check source version</p>'
            '<div class="result-card"><div class="result-meta">Groups: network</div>'
            f'<div class="result-answer">{html.escape(answer)}</div>'
            '<h3>Citations (1)</h3><div class="citation-card">[E1] network.vsdx · page 2</div>'
            f'<figure><a href="{IMAGE}"><img src="{IMAGE}" alt="Topology"></a>'
            '<figcaption>Topology — converted preview; review labels</figcaption></figure></div>')


@pytest.mark.parametrize('streaming', [False, True])
def test_formatted_answer_and_authenticated_diagrams(browser_page, streaming):
    page, requests, status = browser_page
    status.update(step='streaming' if streaming else 'complete', result_html=final_html())
    page.fill('#play_question', 'Show deployment and topology')
    page.click('#ask-btn')
    if streaming:
        page.wait_for_function('Boolean(window.testStream)')
        page.evaluate("testStream.emit({token: '## Partial\\n\\n**Working**'})")
        assert page.locator('#stream-output h2').inner_text() == 'Partial'
        assert page.locator('#stream-output strong').inner_text() == 'Working'
        page.evaluate("testStream.emit({done: true, answer: 'Old streamed answer'})")
        status['step'] = 'complete'
    page.wait_for_selector('#play-results .citation-card')
    answer = page.locator('#play-results .result-answer')
    assert answer.locator('h2').inner_text() == 'Branch deployment'
    assert answer.locator('ol li').count() == 2
    assert answer.locator('ul li').count() == 2
    assert answer.locator('table tbody tr').count() == 2
    assert '<configuration> stays literal' in answer.locator('pre code').inner_text()
    assert answer.locator('blockquote').count() == 1
    link = answer.locator('a[href="https://example.com/guide"]')
    assert link.get_attribute('rel') == 'noopener noreferrer'
    assert answer.locator('img').get_attribute('src') == IMAGE
    assert page.locator('.result-card > figure').count() == 0
    assert answer.locator('figure').count() == 1
    assert answer.locator('figure + h3').inner_text() == 'Operational checks'
    answer.locator('figure img').scroll_into_view_if_needed()
    page.wait_for_function("[...document.querySelectorAll('#play-results img')].every(i => i.complete && i.naturalWidth > 0)")
    assert page.locator('#play-results .status-err').inner_text() == 'Check source version'
    assert 'review labels' in page.locator('#play-results figcaption').inner_text()
    assert any(IMAGE in url for url in requests)
    assert not any('/api/v1/documents/' in url for url in requests)
    assert not any('cdn.jsdelivr.net' in url for url in requests)
    if streaming:
        assert page.evaluate('testStream.closed')
    # Wide content scrolls inside the card in both themes.
    page.evaluate("document.documentElement.dataset.theme = 'dark'")
    page.set_viewport_size({'width': 520, 'height': 900})
    assert answer.evaluate('e => e.scrollWidth <= e.clientWidth + 1')
    if os.environ.get('SAURON_BROWSER_SCREENSHOTS'):
        destination = Path(os.environ['SAURON_BROWSER_SCREENSHOTS'])
        destination.mkdir(parents=True, exist_ok=True)
        page.set_viewport_size({'width':1200, 'height':1000})
        page.locator('#play-results').screenshot(path=str(destination/f'playground-{streaming}.png'))


def test_untrusted_markdown_and_image_urls(browser_page):
    page, requests, _ = browser_page
    attacks = '''# Safe heading

<script>window.pwned = true</script>
<img src="https://evil.test/html" onerror="window.pwned=true">
<svg onload="window.pwned=true"></svg>
<iframe src="https://evil.test/frame"></iframe>

[bad](javascript:alert(1)) [encoded](jav&#x61;script:alert(1))
[bad2](data:text/html,evil) [bad3](file:///etc/passwd)

![remote](https://evil.test/image.png)
![data](data:image/svg+xml;base64,PHN2Zy8+)
![guessed](/admin/api/figure-documents/secret/figures/f1/content?variant=preview)

```html
<img src=x onerror=alert(1)>
```
'''
    page.evaluate('text => { const e = document.querySelector("#play-results"); SauronMarkdown.render(e, text); }', attacks)
    assert page.locator('#play-results h1').inner_text() == 'Safe heading'
    assert page.locator('#play-results script, #play-results img, #play-results svg, #play-results iframe').count() == 0
    assert page.locator('#play-results a[href]').count() == 0
    assert page.evaluate('window.pwned') is None
    assert not any('evil.test' in url or '/secret/' in url for url in requests)
    assert '<img src=x onerror=alert(1)>' in page.locator('#play-results pre').inner_text()
    # Incomplete HTML while streaming is also escaped and sanitized on every update.
    for partial in ['<img src=x onerror="', '<img src=x onerror="window.pwned=true">']:
        page.evaluate('text => SauronMarkdown.render(document.querySelector("#play-results"), text, {streaming:true})', partial)
        assert page.locator('#play-results img').count() == 0
    # Missing sanitizer fails closed to plain text.
    page.evaluate('window.DOMPurify = undefined; SauronMarkdown.render(document.querySelector("#play-results"), "<img src=x onerror=alert(1)>")')
    assert page.locator('#play-results img').count() == 0
    assert '<img' in page.locator('#play-results').inner_text()


def test_readable_reference_opens_its_supporting_passage(browser_page):
    from src.admin.routes import _citation_html
    from src.citations import render_citations
    page, requests, status = browser_page
    citation = {'filename': 'sdwan-for-gov.pdf', 'page': 7, 'section_title': 'Security and Compliance',
                'evidence_id': 'E6c6a058d49b3', 'snippet': 'FIPS mode is enabled.'}
    answer = render_citations('FIPS mode is enabled [E6c6a058d49b3].', [citation], local_links=True)
    status['result_html'] = ('<div class="result-card"><div class="result-answer">' + html.escape(answer)
                             + '</div>' + _citation_html(citation, 1) + '</div>')
    page.fill('#play_question', 'What is special about government SD-WAN?')
    page.click('#ask-btn')
    link = page.locator('.result-answer a[href="#citation-1"]')
    assert link.inner_text() == 'sdwan-for-gov.pdf — page 7, Security and Compliance'
    link.click()
    assert page.locator('#citation-1 details').get_attribute('open') is not None
    assert page.locator('#citation-1 pre').is_visible()
    assert 'E6c6a058d49b3' not in page.locator('.result-card').inner_text()
