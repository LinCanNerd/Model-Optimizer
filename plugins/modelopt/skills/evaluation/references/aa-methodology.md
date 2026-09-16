# Artificial Analysis task policy

When reproducing an applicable AA benchmark/version, treat the
[AA intelligence benchmarking methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking)
(General Testing Parameters and benchmark-specific sections) as authoritative
task policy, ahead of generic model-card evaluation-provenance/default rules.
Confirm applicability and record the benchmark/version and sources; an AA-style
prompt or a generic GPQA run alone does not establish AA reproduction. Do not
migrate benchmark versions implicitly. Surface conflicting user requirements as
deviations from AA reproduction.

- **Temperature:** `0` for non-reasoning, `0.6` for reasoning, unless the model lab
  recommends another temperature for the applicable model/mode.
- **Output:** non-reasoning `16384` tokens, adjusted downward for a smaller context
  window or output cap. Reasoning uses the maximum output tokens allowed as
  disclosed by the model creators, resolved separately for each reasoning model.
  Do not substitute a generic `65536`, a context-window size, or universal `null`.
- **Deliberate provenance exception:** AA permits the lab-recommended temperature
  and requires the disclosed reasoning output maximum without a sentence tying
  those values to evaluation. Cite the exact model/mode disclosure, not a related
  model or arbitrary quickstart. Missing or ambiguous disclosures require
  clarification, not a guessed cap.
- **`top_p`:** no general AA setting; use the normal provenance/default policy
  unless an applicable benchmark-specific requirement supplies one.
- **Context:** AA-LCR v1.1 requires a minimum 128K context, not a universal exact
  `--max-model-len 131072`. Verify supported prompt + output capacity, including
  accumulated history; report infeasible budgets rather than silently shrinking
  required reasoning output.
- **API failures:** automatic retry up to **30 total attempts**. Verify the
  harness's retry semantics and nested retry layers: if `max_retries` counts
  retries after the initial attempt, 30 attempts means 29 retries, not 30.
  Do not change benchmark repeat counts to implement retries.

Apply overrides only to applicable tasks. Inspect resolved requests after
adapters/interceptors, keep baseline/candidate settings aligned, and update
export tags and provenance. Outside this scope, retain the normal
[generation-parameter policy](../SKILL.md#generation-parameters--provenance-and-precedence).
