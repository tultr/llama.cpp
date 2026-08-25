# Chunked prefill fairness implementation brief

This temporary file exists only to bootstrap the implementation PR. Delete it before marking the PR ready.

## Goal

Port the scheduling behavior from upstream draft PR `ggml-org/llama.cpp#10718` onto current `tultr/llama.cpp` so a large prompt prefill cannot monopolize scheduling while other slots are already generating.

Reference: https://github.com/ggml-org/llama.cpp/pull/10718

Desired behavior while generation is active:

`prefill chunk -> decode active slots -> prefill chunk -> decode active slots -> ...`

Do not reduce normal prompt throughput when no slots are generating.

## Current code location

The relevant current path is `tools/server/server-context.cpp`, in `server_context::update_slots()` around:

```cpp
iterate(generating, [&](server_slot & slot) {
    slot.handle_last_sampled_token(batch);
});

int32_t n_batch  = llama_n_batch(ctx_tgt);
int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

// next, batch any pending prompts without exceeding n_batch
```

Current prompt filling later uses:

```cpp
while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
```

## Requirements

1. Detect when the current scheduler iteration has active generation work.
2. While generation is active, cap the **aggregate prompt tokens** appended in that scheduler iteration.
3. Fairly share that prompt-token budget across multiple prompt-processing slots. Do not let the first prompt consume the whole allowance.
4. Preserve existing behavior for non-splittable prompts (`!slot.can_split()`), speculative decoding, checkpointing, multimodal chunks, LoRA/aLoRA, prompt caching, and continuous batching.
5. If no generation is active, retain the existing full `n_batch` prompt-processing behavior.
6. The limit applies to prompt tokens only; generation/speculative tokens already in `batch` must not consume the configured prompt allowance.

## Configuration

Add a llama-server option:

```text
--prefill-chunk-size N
```

Semantics:

- `N > 0`: maximum aggregate prompt tokens scheduled per iteration while at least one slot is generating.
- `0`: disable this special cap and retain current behavior.
- Default: `128` unless existing argument conventions/tests strongly suggest another conservative interactive default.
- Effective cap must never exceed `n_batch`.
- Follow normal argument/env conventions; use `LLAMA_ARG_PREFILL_CHUNK_SIZE` if consistent with neighboring server options.

Do not leave a magic `32` in `server-context.cpp` like the old upstream draft did.

## Fairness

For multiple prompt slots, divide the remaining aggregate prefill allowance so each eligible prompt gets a chance to progress in the same scheduler iteration when practical. Keep a hard aggregate cap. Avoid a minimum-per-slot rule that can exceed the configured aggregate budget.

## Edge cases

- `batch` can already contain sampled/speculative generation tokens before prompt tokens are appended.
- Non-splittable prompts must retain existing all-or-next-iteration semantics; do not partially add them.
- Prompt completion must continue through existing `SLOT_STATE_DONE_PROMPT` behavior.
- Checkpoint spacing and `n_ubatch` behavior must remain correct.
- `--prefill-chunk-size 0` must behave like current master.
- If configured chunk size is larger than `n_batch`, effective prompt budget is `n_batch`.

## Tests

Add focused coverage where practical for:

1. no active generation + long prompt => existing full-batch prefill unchanged;
2. active generation + long prompt => configured prompt cap enforced;
3. active generation + two long prompts => aggregate cap enforced and both make progress;
4. configured value `0` => uncapped current behavior;
5. configured value larger than `n_batch` => effective cap is `n_batch`.

Build affected server targets and run relevant existing server tests.

## Manual benchmark target

Representative validation:

- 8 server slots;
- 7 actively decoding;
- submit one ~100k-token prompt;
- compare active-slot decode latency/tok/s with feature disabled and enabled at 32/64/128/256.

This feature intentionally trades some peak prefill throughput for decode latency/fairness. Optimize for preventing decode starvation, not maximum aggregate prompt tok/s.

## Deliverable

Implement on this branch, update CLI help/docs as appropriate, add tests, reference upstream #10718 in the PR description, and delete this temporary brief file before completion.