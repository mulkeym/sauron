# Inline diagrams with expiring links

Sauron can answer **Explain SD-WAN for government to me** with cited text and
relevant stored PNGs beside their mentions. SVG conversion, a new image model,
and changes to OpenWebUI's source are not required.

## Retrieval and placement

`ANSWER_IMAGES=auto` searches for figure evidence during ordinary questions,
without requiring words such as *diagram* or *topology*. Candidates remain within
the selected documents, dataset, revisions and access groups. Normal relevance
ranking and evidence validation apply. Only figures actually cited in the answer
are selected, bounded by `ANSWER_MAX_IMAGES` and the answer profile's limit.
The `requested` and `off` image policies retain their previous behavior.

At delivery, Sauron places each selected image after the first prose block citing
it. Code blocks and existing Markdown links do not act as placement anchors.
Repeated references to the same figure produce one image. Playground moves the
authorized source card, including caption and conversion warnings, beside the
mention instead of leaving a duplicate at the end. It uses the existing admin
session image route and works without public links. The separate Answer Profiles
preview retains its source-image gallery.

## Configuration

Configure **Settings → Answers & Evidence → Inline diagrams** (also available
in **All Settings → Inline diagrams**), or the corresponding
environment variables, to enable browser-facing links:

| Variable | Default | Meaning |
|---|---|---|
| `FIGURE_PUBLIC_BASE_URL` | Empty | Browser-facing Sauron HTTP(S) URL, including any reverse-proxy prefix |
| `FIGURE_LINK_TTL_SECONDS` | `900` | New link lifetime, from 30 seconds to 24 hours |
| `FIGURE_LINK_SIGNING_SECRET` | Empty | Dedicated random secret of at least 32 characters |

Both a base URL and a signing secret must be configured. Leave either empty to
retain authenticated content URLs and native MCP image attachments. These
settings apply live when saved in the admin UI. The signing secret is redacted
there; leaving its input blank preserves it, and the explicit clear action
disables links. For these three fields, nonempty process environment values (or `.env` values)
take precedence over the control panel. Unset or empty values use the saved
control-panel value, then the built-in default. The panel identifies environment
overrides while preserving an editable fallback, including a separately saved
secret. Clearing a saved secret does not clear an active environment override.
Other settings retain their existing precedence. Environment changes require
restarting/recreating the container. Docker Compose forwards these three variables
as blank when unset, so its defaults do not override the panel.

```dotenv
FIGURE_PUBLIC_BASE_URL=https://sauron.example.internal
FIGURE_LINK_TTL_SECONDS=900
FIGURE_LINK_SIGNING_SECRET=<dedicated-random-secret-at-least-32-characters>
ANSWER_IMAGES=auto
ANSWER_MAX_IMAGES=2
```

Generate a secret with:

```sh
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Keep the secret identical on all replicas. In Helm, use `config.extraEnv` for the
base URL and lifetime, and `secrets.extra` or an existing deployment secret for
the signing secret. The existing figure data volume must be accessible to all
replicas serving images, along with the same metadata database. Short diagram
links are resolved through the `figure_links` table in that database and survive
container restarts. Expired records are removed when new links are issued.

Public means **reachable by the user's browser**, and may be an internal hostname.
Use a URL that routes to Sauron, not an inaccessible container service name.
This can be the OpenWebUI HTTPS hostname with a dedicated proxy prefix.
For example, `https://example.internal/sauron` is valid. Route
`/sauron/api/v1/figure-links/...` through the proxy to Sauron. This exact route uses
its signed URL for authorization, so a proxy must not require a separate login or
API-key header for it. Keep all other Sauron routes protected. Use HTTPS in
production. The endpoint serves only stored PNGs with `image/png` and `nosniff`;
it does not accept a filesystem path or render arbitrary uploaded SVG/HTML.

## OpenWebUI and other clients

The OpenAI-compatible endpoint puts standard `![caption](https://.../TOKEN)`
directly in `choices[0].message.content`. REST and async query answers contain the
same Markdown. The `images` array (`sauron_images` on the OpenAI-compatible API)
also includes `inline_url` and Unix-seconds `expires_at`; `content_url` retains its
authenticated meaning. There is no dependency on a client-specific image extension.

For document questions, MCP discovery advertises
`tool_answer_from_documents(question="<original user request>")`. This retrieves
an answer; it does not prompt the user for clarification. The legacy `tool_ask`
alias remains callable but is hidden from discovery to avoid that ambiguity.
Malformed calls receive retry instructions and do not launch a search.

When enabled, native MCP returns inline Markdown in the answer and image metadata,
instead of duplicate base64 image blocks. Responses put `display_instructions`
and an exact-copy `diagram_markdown` catalog before the larger evidence body;
each image also includes its complete `markdown` line. Links use short opaque
tokens so the host model need not reproduce a long encoded document snapshot.
`tool_get_diagram` returns the same catalog for its selected image.
`tool_search_diagrams` also attaches each candidate's exact `markdown` and fresh
link when public links are enabled; displaying a search result does not require
the model to construct a URL or perform another lookup. Without public links,
search remains metadata-only and `tool_get_diagram` supplies native images.
A host model still composes its final
answer from the MCP result and can omit/rearrange material. Its system instructions
can reinforce the tool's display instructions:

> Include relevant source-diagram Markdown returned by Sauron beside the paragraph
> that discusses it, even without an explicit request for a visual. Preserve the
> complete image Markdown, including captions and closing parentheses. Only use
> returned images; never construct a URL or substitute a figure mentioned in text.
> Obtain a new authorized link with the diagram tools when another image is needed.
> Do not move images to a separate gallery or redraw stored diagrams.

Short links and clearer instructions reduce copying errors; Sauron cannot enforce
the final output of a separate host model. The OpenAI-compatible Sauron endpoint
delivers the already-rendered answer directly when deterministic presentation is
required. No OpenWebUI extension is installed by this feature.

## Expiry and authorization

New URLs contain a random 128-bit bearer capability (`s_` plus 22 URL-safe
characters). Sauron stores its SHA-256 lookup hash alongside an HMAC-SHA256 signed
snapshot in the metadata database. The browser token itself is not persisted.
Issuance rechecks the requesting user's document access. Each capability binds
one document, figure, PNG variant and digest, source revision, ACL snapshot,
issue time and expiry. API keys, user JWTs, group names and filesystem paths are
not put in the URL. Document and figure identifiers also stay out of new URLs.
Previously issued long signed URLs remain valid until their existing expiry.
A person possessing a valid URL can fetch that image until
expiry; it is not a new per-viewer login check.

Only GET/HEAD on the exact image-link route can bypass the application-key gate.
Sauron verifies the signature and lifetime before reading storage and checks
the current document, ACL, revision and PNG digest. Invalid, expired, modified,
replaced or deleted images return 404. Changing the document's ACL/revision,
rotating the signing secret, or clearing public-link configuration invalidates
existing links. Malformed tokens on that route also return 404 instead of a
misleading missing-API-key error. Changing a user's group membership alone does not revoke an
already-issued bearer link. Responses send `Cache-Control: private, no-store`.

Canonical answers and query caches contain no signed URLs. Fresh links are issued
when delivering a new answer, cache hit or async result poll. A changed stored
image is not silently substituted into an old answer. Existing chat messages are
not rewritten: after expiry, reopening one may show an unavailable image. Ask
again to receive fresh links.

Expiry stops future downloads. It cannot erase a downloaded PNG, screenshot, or
a copy stored by OpenWebUI or another integrating client. Proxy/access logs should
not retain full bearer-token paths longer than needed.

## Verification before rollout

1. Confirm a matching source PNG exists in **Diagrams** and that the chosen answer
   profile allows automatic images.
2. Configure the browser-facing base URL, secret and lifetime in a test deployment.
3. Ask the SD-WAN question through Playground and the production-style OpenWebUI
   integration. Check that each image follows the paragraph that cites it.
4. Open an issued image URL without Sauron credentials before expiry; it should
   return the selected PNG. Fetch it again after expiry; it should return 404.
5. Confirm ordinary image/document APIs still reject requests without credentials.

Tests cover signing, tampering, expiry, revocation, source replacement, access
checks, fresh cached-answer presentation, MCP output, REST/OpenAI delivery,
automatic retrieval policy, and browser placement with preserved captions.
