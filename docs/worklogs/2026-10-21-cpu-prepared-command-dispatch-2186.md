# Prepared-execution dispatch cost on CPU (#2186)

## Decisions

- The issue's stated cause — `Runtime::run_prepared` leasing the executor and
  opening a backend session per scheduled node — no longer exists: sessions were
  removed with the caller-held engine (#2206). The measured 4T/1T regression is
  also gone (4T is now within 2% of 1T). The issue therefore needed re-basing,
  not the fix it proposed.
- Re-measured on a quiet host, the remaining cost is per **command**, not per
  matrix entry: a 25-command `grad_sum_qr_vjp` costs 2.84 µs per command and is
  flat in `n` (2x2/4x4/8x8). Instrumented split per command: engine operation
  ~0.8 µs (the same as the eager route), event-domain admission ~0.55 µs, slot
  staging and publication ~0.32 µs, output residency validation ~0.12 µs,
  last-use buffer reclaim ~0.15 µs; a fixed ~4.6 µs per run sat in the
  input-signature projection, output collection, extension-cache lending, and
  drain.
- Three changes, each independent, were chosen over a new execution model:
  1. `EventDomainRun::enable_single_domain_coalescing` now returns whether the
     run accepts the offer, instead of accepting silently: one call, one fact,
     and no state in which a run accepts without the scheduler knowing. A run
     that accepts is submitted through its own submission context with no
     dependency tokens and no recorded completion. The CPU immediate driver
     accepts (every launch retires before `enqueue` returns, so submission order
     already orders its nodes) and shares one completion token per run instead
     of allocating one per node. Rejected alternative: a second reporting method
     with a default, which made "accept without reporting" a reachable and
     meaningless state. CUDA keeps its own coalesced submission but answers
     `false`, so its scheduling is byte-for-byte the previous path: replacing
     the scheduler's per-node bookkeeping for it needs a device run this batch
     could not make.
  2. A single-location schedule (one engine, one storage class, one event
     domain, no transfer node) keeps each slot's value in the command array and
     skips per-command staging, publication, and location comparison. The
     per-location list stays for transfers and several engines.
  3. `Prepared` stores the input signature it was prepared for. A run presenting
     the same signature skips the specialization projection, which preparation
     already validated; any other signature takes the existing full check.
- Rejected: teaching the elementwise-region planner the CPU fusion floor
  (`ELEMENTWISE_FUSION_MIN_ELEMENTS`, 16K elements) so a tiny region is not
  planned at all. Declining earlier than the plan's broadcast-view validation
  turns a malformed plan's typed rejection into a silent decline
  (`elementwise_fusion_rejects_malformed_broadcast_views`), and the refused
  attempt costs well under 1 µs: the call stack across the engine boundary, not
  the size check, dominates it.
- Rejected: a specialized synchronous executor that bypasses event-domain
  submission entirely. The run's `enqueue` is also the provider's submission
  context (CUDA sets the stream and context there), so bypassing it would need
  a second provider-visible contract for no further CPU gain.

## Verification conclusions and constraints

- Measured on a quiet 24-vCPU host (one pinned core, load average 0.4, release),
  median of 11 batches of 50 runs, three independent repeats; the ranges below
  are the minima and maxima of those repeats.
- `grad_sum_qr_vjp` 2x2 (25 commands): 71.4–72.0 µs → **49.1–50.5 µs** at one
  thread and 73.8 → **50.4–51.0 µs** at four; 8x8: 75.7 → **53.2–54.9 µs**.
  Per command 2.84 → **1.96 µs**. The same operations executed eagerly cost
  0.74–0.82 µs per operation, so about 1.1 µs per command still belongs to the
  executor.
- A 200-instruction elementwise program (one planned region the CPU declines
  below its element floor, so every instruction executes on the per-instruction
  path): 546 µs → **275–296 µs** at one thread, 561 → **287–289 µs** at four,
  against 149 µs for the same chain executed eagerly.
- Correctness is covered by the existing suites (runtime 421, cpu 314, cpu-fused
  13, linalg and ad 116 + 377 + 141 + 194 + 155) plus two new
  `ScheduledEventDomains` tests: a run that reports coalescing launches every
  node with no dependency tokens and records no completion, and a run that
  accepts the offer without reporting it keeps one completion per node and
  resolves the second command's dependency.
- Not claimed: CUDA answers `false` here (no GPU in this session), so its
  schedule is byte-for-byte the previous path; answering `true` is a follow-up
  that needs a device run. The remaining ~1.1 µs per
  command is executor work this change did not remove; a codegen-style
  specialization of a prepared program is #1938 territory.
- Reproducer: `cargo bench -p tenferro-linalg --features autodiff --bench
  traced_small_ad_overhead` (see the bench module documentation for the
  environment and the protocol).
