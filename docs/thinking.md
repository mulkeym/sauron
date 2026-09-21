# Final answer generation

**Answer temperature** controls final answers, bounded repairs and previews. The accepted range is 0–2; the default remains 0.1 for existing behavior. It is also available in All Settings and as `LLM_ANSWER_TEMPERATURE`. It applies whether reasoning is enabled, disabled or left to the provider. Models already handled as fixed-sampling reasoning models omit temperature, as before.

For Gemma 4 thinking, 1.0 is a useful starting point: direct local tests completed at that temperature while the same prompt at 0.1 remained in reasoning past the test deadline. This is a latency finding, not an accuracy guarantee. Evaluate grounded answers before adopting it. The setting does not automatically switch when changing models.

In **Admin → Settings → Models → Final answer generation**, select a **Reasoning (thinking)** preference:

- **Provider default:** omit reasoning controls. This is the application/migration default, not a promise that a provider turns reasoning off.
- **Enabled:** request thinking for final answers, their bounded repairs and answer previews.
- **Disabled:** explicitly request no thinking when the model/provider supports disabling it.

Use **Check thinking support** for the selected endpoint/model. These settings are also available in All Settings and as `LLM_ANSWER_THINKING` / `LLM_REASONING_ADAPTER`. Saving applies the preference immediately. There is no profile override. Query cache scope already fingerprints all settings, so changing either answer control cannot reuse an answer generated under another preference.

This is an answer preference. Classification, document ingestion, vision descriptions and LightRAG enrichment keep their existing request settings and budgets. Wide-table SQL has its separate `sql_thinking_on_wide_table` preference; its best-effort hint now uses the appropriate supported adapter instead of sending a vLLM extension to every non-OpenAI endpoint.

## Provider handling

For `https://openrouter.ai/api/v1`, Sauron checks the exact selected model in `/models` for `supported_parameters` and reasoning metadata (cached for five minutes, keyed by model and endpoint). An explicit request sends:

```json
{
  "reasoning": {"enabled": true, "exclude": true},
  "provider": {"require_parameters": true}
}
```

Disabled uses `enabled: false`. A mandatory-reasoning model cannot be disabled. The provider restriction prevents routing to an endpoint that silently ignores a requested parameter; it can reduce eligible endpoints. Sauron does not guess effort levels or reasoning-token budgets from the model name. Unknown support, metadata failures, and rejected explicit controls fail with a provider/configuration error. There is no silent retry without the requested reasoning control.

For a documented vLLM-compatible server, explicitly select **vLLM template**. This sends `chat_template_kwargs.enable_thinking`. It is an operator-selected capability, not automatic verification: the server's model template and reasoning parser must support it. It cannot override server restrictions. Other endpoints, including the native Google API and direct OpenAI API, retain their existing behavior under Provider default; this feature does not yet implement their distinct reasoning controls. No prompt token or natural-language “think” instruction is injected.

## Output and budgets

Reasoning and the visible answer can share the output-token allowance. Answer generation retains `llm_max_output_tokens`; enabling thinking does not replace it with the smaller SQL budget. Increase that limit when necessary, within provider/model limits. Thinking may increase latency and cost and does not guarantee better answers. Buffered answer requests have a whole-response deadline (`vllm_request_timeout`, including a transport retry); incoming whitespace does not extend it. Playground also enforces `async_query_timeout_seconds` across the query. Deadline expiry closes the model connection and reports an error instead of leaving the final step spinning.

Only final `message.content` / streaming `delta.content` is accepted. Separate reasoning fields are never used as a fallback. Tagged `<think>` and Gemma thought channels are removed, including split delimiters and unclosed reasoning in streams. Empty final content is an error. `finish_reason: length` is a provider truncation error, never an insufficient-evidence response; any already-emitted streaming text is incomplete. The answer prompt explicitly keeps internal reasoning out of the visible final response. Existing citation/status validation still runs on final text.

## Verified model and sources

On 20 September 2026, the configured model `google/gemma-4-26b-a4b-it` advertised reasoning support with `default_enabled: false` and `mandatory: false`. “IT” means instruction-tuned and does not imply absence of reasoning. This metadata can change; use the capability check.

- [OpenRouter model](https://openrouter.ai/google/gemma-4-26b-a4b-it)
- [OpenRouter reasoning controls and output budget](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [OpenRouter parameter-aware routing](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters)
- [Google Gemma 4 sampling recommendations](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Google Gemma thinking and chat templates](https://ai.google.dev/gemma/docs/capabilities/thinking)
- [vLLM Gemma 4 request controls](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)

Regression tests cover actual outbound payloads, omission, enabling/disabling, metadata/unsupported errors, streaming splits, final-only JSON/vision parsing, token exhaustion, admin persistence and answer-repair propagation. Local live-test results are recorded separately; capability support alone is not an answer-quality benchmark.
