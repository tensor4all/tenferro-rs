# Backend capability descriptor: declaration, tested, with a generated guide table

## Decisions

- The descriptor stays a **declaration**; enforcement at plan or dispatch time
  (option C of [#2477](https://github.com/tensor4all/tenferro-rs/issues/2477))
  is rejected. Every backend already refuses an unsupported dtype on its own,
  each in its own layer's spelling (`UnsupportedDType` on CPU, a kernel refusal
  wrapped as `RuntimeStateSource` on CUDA), so a table gate would be a second
  hand-kept copy of the same decision (the `dot_general` row that drifted in
  #2478 is the evidence),
  and most axes declare whether an internal copy happens rather than whether the
  operation works — refusing on those is the per-call policy #2552 withdrew.
  The durable statement is in
  [`docs/design/backend-capability.md`](../design/backend-capability.md).
- What replaces the gate is a test: one shared dispatcher over every table
  operation × the seven public dtypes, run against the CPU engine in normal CI
  and against the CUDA engine (with numerical parity against CPU) on the GPU
  lane. The dispatcher panics on an operation it has no fixture for, so a new
  table entry fails a test instead of passing unexamined.
- The conformance rule is the catalog's dtype policy plus the table: a pair the
  catalog does not admit is outside the descriptor (no table row, and an engine
  that serves it is an engine gap, not a missing row); an admitted pair either
  runs and returns the declared output dtype, or is not served at all. The rule
  deliberately stops at "not served": the two backends spell a refusal
  differently (`Unsupported`, `Validation(DTypeMismatch)`, and on CUDA a kernel
  refusal inside a runtime-state error), and demanding one kind is what the GPU
  lane caught as wrong.
- The CPU test enumerates only `cpu_capabilities()`, not the union with the CUDA
  table: `cuda_capabilities()` exists only behind `feature = "cuda"`, so a
  `tenferro-cpu` dev-dependency could not see it. The union is enumerated by the
  CUDA lane, and a default-build unit test in `tenferro-gpu` asserts both tables
  cover the same operations, which is the check that runs in normal CI.
- The CUDA table's data moved out of the `cuda` module into
  `tenferro-gpu/src/capability.rs` so it compiles without the feature, which is
  what lets a default-features test render the guide table and compare it. The
  engine-facing `TensorBackendCapability` impl and the public
  `cuda_capabilities()` stay behind the feature; no public item was added.
- The hand-written guide table was accurate: rendering it from the descriptor
  reproduced every cell. The one difference was row order (`dot_general` sat
  between `erf` and `reduce_sum`, the descriptor listed it last), resolved by
  putting the `dot_native` entries into catalog order in the table itself, so
  the regenerated block is byte-identical to the previous one.
- `TensorBackendCapability::require_capability` is removed (0.12 breaking
  window). It had no production caller, and with enforcement rejected it had no
  future one; `capability()` and `capability_entry()` read the declaration and
  the engine refuses on its own.

## Verification conclusions and constraints

- `cargo test -p tenferro-cpu --test integration capability_conformance` runs
  the whole CPU table in normal CI; the operations come from the table.
- `cargo test -p tenferro-gpu capability::` runs the guide-table check and the
  CPU/CUDA op-set check without a device.
- Damaging any of the three guards fails a test: flipping a served CPU entry to
  `Unsupported`, editing one cell of the guide block, and adding a table
  operation the dispatcher has no case for.
- The CUDA conformance and parity code is **not run** on this host (no CUDA
  device): it is compile-checked with `--features cuda`, and the paid GPU lane
  (`gpu_test_partition.py --kind cuda --lane gpu`) is the evidence that matters.
  It ran there and passed the whole `gpu` lane (371/371) at `c4e3ffbc` — paid run
  [`43613195709`](https://github.com/tensor4all/tenferro-rs/actions/runs/43613195709),
  job `132503756575`, which published the required check for tested ref
  `01cdf36…`. Later commits on this branch touch only documentation, so the tested
  binaries are the ones that run measured; a delivery whose tested ref already
  passed reuses that check instead of provisioning another pod. The run includes
  `every_claimed_entry_matches_what_the_engine_does` (every served pair, CUDA
  against the CPU numbers) and `the_descriptor_names_this_backend_and_its_dtype_pairs`.
  The tolerance split (8 ulps arithmetic, 64 ulps transcendentals) therefore
  holds over the whole table with these fixtures, which is weaker than a measured
  worst case per operation.
- What the lane found before it went green (paid runs `43604837947` and
  `43607680856`, one failed test each, both stopping at the same pair): `reduce_max`
  on `C32` was refused as
  `RuntimeStateSource { op: "reduce_max_read", source: UnsupportedDType }`, not as
  `ErrorKind::Unsupported`. The descriptor was right; the rule demanded one error
  kind. It now holds the contract the issue states — a pair the table does not
  claim must not be *served* — and leaves the spelling of a refusal to the
  backend that produces it.
- The other four axes stay declared-only; no test holds them to an engine.
- Independent reviews (read-only, `ollama-cloud/glm-5.3-flash`), all three in the
  reviewer's own session with no write access: the plan review
  returned `VERDICT: CHANGES` — the CPU lane cannot see the `cuda`-gated table,
  the renderer's row-order claim was wrong, and the op-set check had to run in a
  default build — all three are reflected above. The finished-diff review returned
  `VERDICT: CHANGES` with two blockers in the GPU-lane test only: its coverage
  counter skipped the pairs the catalog does not admit, and parity extracted
  values by input dtype although `abs` on complex and `compare` change it. Both
  are fixed (the counter now counts examined pairs and requires every one; parity
  compares by the declared output dtype and asserts it). The final review of the
  finished diff returned `VERDICT: APPROVE` with one important finding — the
  design doc and this worklog still promised the stricter "dtype refusal, not an
  unrelated validation error" rule that the code had already relaxed — and that
  wording is aligned here.
- Observed while testing, not changed here: the CPU engine serves `compare` on
  `C32`/`C64`, which the catalog's dtype policy (`CompareToBool`) excludes as an
  input dtype. The table stays truthful (no row), so the conformance test treats
  the pair as outside the descriptor; the engine behaviour is a separate issue.
- Also not changed here: `tenferro-gpu` maps a kernel's `UnsupportedDType` (and
  any other kernel refusal) into `RuntimeStateSource`, so a dtype refusal reads as
  a runtime-state failure at the public boundary. The descriptor does not depend
  on that mapping; fixing it belongs to the CUDA dispatch's error mapping.
