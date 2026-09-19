# Playground Markdown dependencies

Vendored browser builds so Markdown rendering works without CDN access.

- `marked.umd.js`: marked 18.0.13, SHA-256 `b147274a9ce27d17276587167e49483d719f6893eeca3a3667a59797661d3556`
- `MARKED-LICENSE`: marked 18.0.13, SHA-256 `8e3a3f82f59a60958f56ca08f445647c32a4733dc7ca6c2c46f6eb898471ab9c`
- `purify.min.js`: dompurify 3.4.15, SHA-256 `f263b05369e050fa175d4ecb9c9358eb4253602d510297adfb31df48b2f1c4d5`
- `DOMPURIFY-LICENSE`: dompurify 3.4.15, SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`

Sources: https://github.com/markedjs/marked and https://github.com/cure53/DOMPurify. Downloaded from their versioned npm packages; package SHA-512 integrity verified. Licenses are included. To update, select reviewed versions, verify package integrity, replace the browser builds/licenses, and run the Playground browser tests.
