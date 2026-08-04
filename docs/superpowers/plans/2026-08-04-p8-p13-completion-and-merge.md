# P8–P13 Storage Ownership Completion And Merge Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the approved P8–P13 storage-ownership migration on one reviewable branch, prove every phase obligation, and merge the resulting PR.

**Architecture:** Keep one non-Clone owner per physical allocation span and one `AllocationGroup` per detached execution/AD retention bundle. Prepared access performs static validation once; provider bindings retain only opaque prepared state and lifetime leases. Detached submission consumes ownership and returns it only after pre-admission rejection or proven retirement; completion-unproven retains a private record with no owner recovery. WebGPU and Apple/Metal use the same root/group/prepared model, while provider namespaces, explicit transfers, and scoped raw interop preserve provider differences.

**Tech Stack:** Rust workspace, `tenferro-tensor` storage kernel, `tenferro-runtime` event-domain execution, `tenferro-ad` eager/traced retention, CubeCL CUDA/WebGPU/Metal providers, Python ledger/checker scripts, rustdoc/Quarto/tutorial checks, `gh` PR and CI commands.

## Global Constraints

- Follow authority order: parent Issue #1555, `docs/design/storage-ownership-contracts.md`, Issues #1564–#1569, then the approved P8–P13 specs.
- Preserve one move-only owner per physical allocation span; `Arc`, IDs, handles, and provider references retain lifetime or diagnostics only and never authorize writes.
- Validate static range/layout/storage/provider facts before prepared access; do not repeat those facts at bind, enqueue, or per element.
- Keep copies, transfers, reinterpretation, synchronization, and materialization explicit and reason-classified; never hide CPU/GPU transfers or fallback copies.
- Keep Rust soundness, mutable-alias prevention, asynchronous retirement, provider ordering, and numerical correctness; do not add security/attestation, quarantine/poison/retry/cancellation, speculative recovery, or redundant identity machinery.
- Do not discard existing user changes. Do not add public compatibility aliases or leave migration shims, TODO/TBD placeholders, dead paths, duplicated storage models, or undocumented behavior changes.
- Every production change follows TDD: write one focused failing test/contract, run it to observe the expected failure, implement the smallest fix, run the focused test and affected suite, then refactor only while green.
- Run `cargo fmt --all --check`, repository-local `check-pr-fast.sh`, storage contract tests, relevant workspace tests, clippy, rustdoc/docs checks, and the exact PR required checks before merge.

---

### Task 1: Establish the P9 RED gap inventory

**Files:**
- Modify: `crates/tenferro-runtime/src/runtime/execution.rs`
- Modify: `crates/tenferro-runtime/src/runtime/tests/execution.rs`
- Modify: `crates/tenferro-runtime/tests/integration/runtime_execution.rs`
- Modify: `crates/tenferro-tensor/tests/storage_compile_contract.rs`
- Create or modify: `crates/tenferro-runtime/src/runtime/tests/owner_bundle.rs`

**Interfaces:**
- Consume current `ExecutionInputs`, `SubmitError`, `ExecutionOutcome`, `InFlightSubmission`, and `AllocationGroup` APIs.
- Produce failing behavior tests for the approved P9 shapes: no `AllocationGroup::tensor_owners`, exact unchanged pre-admission recovery, no owner on completion-unproven, and alias-safe `ExecutionBundle` extraction.

- [ ] **Step 1: Write the failing tests.**

  Add tests with these exact behaviors:
  - constructing `ExecutionInputs` stores tensors in group allocations/descriptors and has no tensor-owner side table;
  - a preparation error and a worker-spawn error before admission return the same `ExecutionInputs` package through the typed rejection carrier;
  - a successful identity/repeated-output graph exposes borrowed output views and consuming extraction returns one owner without cloning or copying;
  - a completion-unproven test exposes diagnostic keys but no owner recovery API;
  - a borrowed scoped submission rejects CUDA/WebGPU/Metal before admission and does not let a borrowed input escape.

- [ ] **Step 2: Run the RED artifacts.**

  ```bash
  cargo test -p tenferro-runtime --lib runtime::tests::execution -- --nocapture
  cargo test -p tenferro-runtime --test runtime_execution -- --nocapture
  cargo test -p tenferro-tensor --test storage_compile_contract
  ```

  Confirm failures identify the existing tensor-owner side table, `Vec<Tensor>` result surface, missing completion-unproven state, or missing scoped API rather than test/setup errors.

- [ ] **Step 3: Commit the RED harness.**

  ```bash
  git add crates/tenferro-runtime crates/tenferro-tensor/tests/storage_compile_contract.rs
  git commit -m "test(runtime): define final group submission contracts"
  ```

### Task 2: Convert P9 detached submission to group-owned terminal outcomes

**Files:**
- Modify: `crates/tenferro-runtime/src/runtime/execution.rs`
- Modify: `crates/tenferro-runtime/src/runtime/snapshot.rs`
- Modify: `crates/tenferro-runtime/src/runtime/mod.rs`
- Modify: `crates/tenferro-runtime/src/error.rs`
- Modify: `crates/tenferro-tensor/src/storage/group.rs`
- Modify: `crates/tenferro-tensor/src/types.rs`
- Test: `crates/tenferro-runtime/src/runtime/tests/owner_bundle.rs`
- Test: `crates/tenferro-runtime/tests/integration/runtime_execution.rs`

**Interfaces:**
- `ExecutionInputs` owns one private `AllocationGroup` and local `DescriptorSlot` bindings.
- `SubmitError::PreAdmission` owns the exact unchanged package; no ownerless worker-spawn recovery variant is exposed.
- `ExecutionOutcome` contains `Completed(ExecutionBundle)`, `RetiredFailed { cause, inputs }`, and `CompletionUnproven { cause, diagnostic_keys }`.
- `ExecutionBundle` exposes borrowed `output()` and consuming alias-safe `into_output()` with unchanged-bundle recovery on extraction failure.

- [ ] **Step 1: Replace the side table and result vector in the failing implementation.**

  Remove `AllocationGroup::tensor_owners`, `tensor_refs()`, and every `Vec<Tensor>` result/retention path used by detached submission. Add descriptor-slot resolution and group borrowing so repeated bindings copy only local slot metadata. Keep all group fields private.

- [ ] **Step 2: Move the admission boundary before ownership transfer.**

  Keep ordered input metadata resolution, preparation, provider preflight, and worker creation before the first provider enqueue. On every pre-admission error, return the exact unchanged input package. Once an event-domain enqueue can occur, move the group, prepared bindings, event tokens, roots, and provider context into the private in-flight record; never reconstruct an owner after that point.

- [ ] **Step 3: Add retirement-proven and retirement-unproven states.**

  Publish `Completed` only after every event domain drains successfully. Publish `RetiredFailed` with the exact inputs only after retirement is proven. On worker/provider panic or failed retirement proof, publish `CompletionUnproven` with diagnostics and retain the private record permanently; do not expose retry, cancellation, quarantine, or owner recovery.

- [ ] **Step 4: Add alias-safe bundle access and extraction.**

  Represent identity, metadata, repeated outputs, and duplicate outputs as descriptor slots in one group. `output()` borrows the bundle. `into_output()` consumes the whole bundle and succeeds only when moving the selected owner cannot invalidate remaining descriptors; on failure return `(self, typed_error)` without mutation or copy.

- [ ] **Step 5: Run the focused GREEN suite.**

  ```bash
  cargo test -p tenferro-runtime --lib runtime::tests::execution owner_bundle -- --nocapture
  cargo test -p tenferro-runtime --test runtime_execution -- --nocapture
  cargo test -p tenferro-tensor --test storage_compile_contract
  cargo test -p tenferro-runtime --doc
  ```

- [ ] **Step 6: Commit the detached cutover.**

  ```bash
  git add crates/tenferro-runtime crates/tenferro-tensor
  git commit -m "feat(runtime): return group-owned submission outcomes"
  ```

### Task 3: Migrate P9 AD, checkpoints, and scoped borrowed execution

**Files:**
- Modify: `crates/tenferro-ad/src/eager.rs`
- Modify: `crates/tenferro-ad/src/eager_exec.rs`
- Modify: `crates/tenferro-ad/src/eager_ops.rs`
- Modify: `crates/tenferro-ad/src/eager_backend.rs`
- Modify: `crates/tenferro-ad/src/traced.rs`
- Modify: `crates/tenferro-runtime/src/exec.rs`
- Modify: `crates/tenferro-runtime/src/runtime/execution.rs`
- Modify: `crates/tenferro-ad/src/eager/tests.rs`
- Modify: `crates/tenferro-ad/src/eager_exec/tests.rs`
- Modify: `crates/tenferro-ad/tests/integration/checkpoint.rs`

**Interfaces:**
- Eager records retain `Arc` read-only metadata/record handles whose private container owns one group; they do not retain `Arc<Tensor>` owners or materialization caches.
- Value access is borrowed; explicit duplication returns a fresh owner; consuming extraction returns a typed unchanged-handle error on failure.
- Gradients/checkpoints retain descriptor records and extract a standalone owner only structurally.
- Scoped borrowed execution is synchronous to retirement, read-only, and rejected before admission for asynchronous CUDA/WebGPU/Metal providers.

- [ ] **Step 1: Add failing AD retention/copy-accounting tests.**

  Assert that cloning eager handles, recording checkpoints, retaining tape values, and creating real/complex aliases do not increment allocation or copy counters. Assert that `duplicate_value()` does increment explicit-copy counters and returns a fresh allocation identity. Add CPU forward/backward numerical checks and the asynchronous-provider rejection path.

- [ ] **Step 2: Replace `Arc<Tensor>` and lazy materialization cache surfaces.**

  Remove `TensorValue { Arc<TensorOwnerRecord> }`, `Arc<OnceLock<Arc<Tensor>>>`, `materialized() -> Arc<Tensor>`, `materialized_arc()`, `Completed(Vec<Tensor>)`, and `GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>`. Use direct group-backed records and borrowed value guards; classify operation outputs, checkpoint recomputation, explicit duplication, and transfers separately.

- [ ] **Step 3: Implement scoped read submission.**

  Add the lifetime-bounded read-only input bundle and synchronous outcome. CPU may execute it through retirement. CUDA/WebGPU/Metal return a typed pre-admission unsupported error with the original borrow still available. No scoped work, binding, or output survives the call or unwind.

- [ ] **Step 4: Run AD and runtime GREEN tests.**

  ```bash
  cargo test -p tenferro-ad --lib
  cargo test -p tenferro-ad --test checkpoint
  cargo test -p tenferro-ad --test traced_ad_explicit
  cargo test -p tenferro-runtime --lib runtime::tests::execution
  cargo test -p tenferro-runtime --test runtime_execution
  ```

- [ ] **Step 5: Commit P9 AD/scoped ownership.**

  ```bash
  git add crates/tenferro-ad crates/tenferro-runtime
  git commit -m "feat(ad): retain groups without tensor owner clones"
  ```

### Task 4: Complete the P8 WebGPU and Apple/Metal root/prepared migration

**Files:**
- Modify: `crates/tenferro-gpu/src/webgpu/mod.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/memory.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/apple.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/event_domain.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/exec_session.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/interop.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/structural.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/gemm.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/runtime_adapter.rs`
- Modify: `crates/tenferro-gpu/src/lib.rs`
- Test: `crates/tenferro-gpu/tests/storage_provider_webgpu_progress.rs`
- Test: `crates/tenferro-gpu/tests/storage_provider.rs`
- Test: `crates/tenferro-gpu/tests/integration/apple_context.rs`
- Test: `crates/tenferro-gpu/tests/integration/public_surface_contract.rs`

**Interfaces:**
- WebGPU/Apple storage enters `AllocationGroup` through scalar-independent root import and retains exact domain-qualified spans.
- Device prepared payloads retain checked layout and opaque provider state only; host guards appear only for explicitly host-visible allocations.
- Apple CPU↔Metal endpoint changes synchronize/map/unmap with zero tensor-byte transfer and one allocation identity.
- Raw WebGPU interop is hidden, unsafe, and callback/lifetime-scoped; it cannot return a raw handle or cloneable owner.

- [ ] **Step 1: Add failing provider contract tests.**

  Cover device-local host mapping rejection, map/device exclusion, read-after-device-write, immediate drop after enqueue, Apple CPU→Metal→CPU identity and zero transfer bytes, prepared-once resolution count, and raw-handle lifetime/owner-projection compile failures.

- [ ] **Step 2: Replace `WebGpuBuffer<T>`/optional Apple ownership with scalar-independent root imports.**

  Remove typed provider ownership as an independent path. Keep resource lifetime handles private and non-authoritative. Construct descriptors once with byte range, dtype, alignment, layout, provider, and write-injectivity proofs; carry the proof into `PreparedRead::Device`/`PreparedWrite::Device`.

- [ ] **Step 3: Narrow mapping and interop boundaries.**

  Keep host access behind span-scoped guards and provider synchronization. Replace flat raw-handle exports with `with_webgpu_prepared_bindings` and a private callback-only launch binding. Ensure kernels consume prepared bindings and do not repeat static identity/range/layout checks per launch or element.

- [ ] **Step 4: Preserve Apple shared allocation semantics.**

  Route CPU and Metal endpoints through one managed allocation domain/resource. Endpoint transitions may synchronize and map/unmap, but must not allocate, copy tensor payloads, upload, download, or change allocation identity.

- [ ] **Step 5: Run P8 provider GREEN tests.**

  ```bash
  cargo test -p tenferro-gpu --features webgpu --test storage_provider_webgpu_progress
  cargo test -p tenferro-gpu --features webgpu --test storage_provider
  cargo test -p tenferro-gpu --features webgpu --test integration -- webgpu apple
  cargo test -p tenferro-gpu --features webgpu --test public_surface_contract
  ```

- [ ] **Step 6: Commit P8.**

  ```bash
  git add crates/tenferro-gpu crates/tenferro-tensor
  git commit -m "feat(gpu): migrate WebGPU and Apple storage to root ownership"
  ```

### Task 5: Normalize P10 accelerator namespaces, transfer APIs, and hot paths

**Files:**
- Modify: `crates/tenferro-gpu/src/lib.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/mod.rs`
- Modify: `crates/tenferro-tensor/src/backend.rs`
- Modify: `crates/tenferro-tensor/src/types.rs`
- Modify: `crates/tenferro-cpu/src/*` only where canonical typed-view methods require forwarding
- Create/modify: `crates/tenferro-tensor/tests/storage_public_api.rs`
- Create: `scripts/check-storage-element-hot-path.py`
- Create: `scripts/check-storage-static-rank-codegen.py`
- Create: `docs/testing/storage-traversal-performance.md`
- Create: `docs/testing/storage-static-rank-codegen.md`

**Interfaces:**
- Provider namespaces expose deliberate constructors, discovery, `upload_tensor`, `download_tensor`, `duplicate`, and unsafe provider-specific interop only.
- Read-only methods are canonical on `TypedTensorView<T, R>`; owners and mutable views delegate through O(1) `as_view()`.
- `TypedTensorView::duplicate()` is explicit and returns a new owner; unsupported noncompact direct transfer returns a typed error rather than staging implicitly.
- Static-rank APIs preserve `R`; dynamic rank remains explicit.

- [ ] **Step 1: Add failing public-surface and source-contract tests.**

  Reject `Buffer<T>`, `BackendBuffer<T>`, no-op/default transfer methods, fixed engine IDs, safe raw handle access, provider namespace drift, and cloneable owner exports. Add API-parity tests for owner/view/view-mut read methods and static-rank return types.

- [ ] **Step 2: Converge public provider APIs.**

  Move CUDA, WebGPU, and Apple exports into deliberate namespaces; remove flat legacy aliases and transfer defaults. Ensure every transfer allocates a fresh identity and records `Transfer`, while view/reinterpret/synchronize/map operations record no transfer.

- [ ] **Step 3: Normalize typed and strided hot loops.**

  Use contiguous typed slice iteration and precomputed strided carry/offset metadata. Remove per-element provider/storage lookup, coordinate decoding, and repeated immutable validation. Add `// INVARIANT:` markers only where a validated boundary justifies an unchecked operation.

- [ ] **Step 4: Run API, structure, and performance evidence.**

  ```bash
  cargo test -p tenferro-tensor --test storage_public_api
  python3 scripts/check-storage-element-hot-path.py
  python3 scripts/check-storage-static-rank-codegen.py --report docs/testing/storage-static-rank-codegen.md
  cargo bench -p tenferro-tensor --bench element_access -- --noplot
  python3 scripts/verify-storage-element-access-baseline.py --report docs/testing/storage-traversal-performance.md
  ```

  Record inconclusive performance only with exact commit, environment, command, and reason; a skipped benchmark is not a pass for required hardware.

- [ ] **Step 5: Commit P10.**

  ```bash
  git add crates/tenferro-gpu crates/tenferro-tensor crates/tenferro-cpu scripts docs/testing
  git commit -m "refactor(storage): converge accelerator APIs and hot paths"
  ```

### Task 6: Prepare P12 documentation product and executable examples

**Files:**
- Modify: `docs/spec/tensor-semantics.md`
- Modify: `docs/getting-started/core-concepts.md`
- Modify: `docs/guides/_sidebar.md` or the repository's current guide navigation file
- Create: `docs/guides/views-and-slicing.md`
- Create/modify: `docs/storage-ownership.md`
- Modify: `README.md`
- Create/modify: `docs/tutorial-code/src/bin/storage_element_access.rs`
- Create: `scripts/check-storage-docs.py`
- Create: `scripts/check-storage-element-access-docs.py`

**Interfaces:**
- User docs teach owner/view/view-mut, static/dynamic rank, prepared access, host guards, explicit duplicate/upload/download, reinterpretation versus numeric cast, detached versus scoped submission, Apple shared endpoints, and completion-unproven diagnostics.
- The tutorial is the executable source of truth for the guide snippets and asserts values, aliasing, allocation-free view behavior, and explicit copy identity.

- [ ] **Step 1: Add failing stale-language and tutorial checks.**

  Make the docs checker reject rendered/source references to removed `Buffer<T>`, shallow tensor cloning, legacy map APIs, implicit canonicalization, hidden transfer/materialization, and deleted handoff files. Add the exact tutorial test target required by P12.

- [ ] **Step 2: Rewrite normative and onboarding docs.**

  Replace the old storage section in tensor semantics, introduce the capability triad in core concepts, add the navigable views guide, update ownership/API/transfer/runtime sections, and remove stale examples. State synchronization/mapping separately from transfer and state that validation completes before prepared access construction.

- [ ] **Step 3: Run executable documentation checks.**

  ```bash
  python3 scripts/check-storage-docs.py --include-rendered
  python3 scripts/check-storage-element-access-docs.py docs/guides/views-and-slicing.md
  cargo test -p tenferro-tutorial-code --release tutorial_binaries_run_successfully -- --exact
  cargo test --workspace --doc
  python3 scripts/check-docs-site.py
  python3 scripts/check-public-error-docs.py
  ```

- [ ] **Step 4: Commit pre-freeze documentation.**

  ```bash
  git add README.md docs scripts/check-storage-docs.py scripts/check-storage-element-access-docs.py
  git commit -m "docs(storage): publish the ownership and access model"
  ```

### Task 7: Execute P13-A cleanup and record the freeze candidate

**Files:**
- Delete: `HANDOFF-2026-07-25-tenferro-unification6-wip.md` and every inbound reference
- Modify/delete: all legacy storage, runtime, transfer, raw-handle, and migration paths found by the P13 inventory
- Modify: `scripts/storage-ownership-contracts.toml` only for rows proven complete
- Modify: `scripts/test-storage-ownership-contracts-v2.py` only for the matching active-ID cohort
- Create: `scripts/check-storage-contract-freeze.py`
- Create: `docs/design/storage-contract-freeze.md`
- Modify: `docs/worklogs/2026-08-04-p8-p13-completion.md`

**Interfaces:**
- One owner/view/view-mut surface, one root/group surface, one prepared-access path, and one detached/scoped submission path remain.
- Freeze report records one exact clean Git candidate and ordinary repository-relative evidence paths; no digest/nonce/attestation is added.

- [ ] **Step 1: Add the failing freeze checker.**

  The checker must parse the fixed owner/group/runtime/provider manifest with AST/token-aware checks, reject legacy declarations and forbidden machinery, verify no unapproved compatibility aliases remain, and reject any candidate with dirty tracked files or stale handoff references.

- [ ] **Step 2: Remove every inventoried legacy path.**

  Delete old typed/shallow owners, `Buffer<T>`/`BackendBuffer<T>`, materialization caches, safe raw-handle escapes, no-op transfer defaults, fixed IDs, duplicate dispatch/storage paths, temporary adapters, pair-only split paths, and old submission/AD values. Keep only the approved unsafe provider boundary and provider-private lifetime handles.

- [ ] **Step 3: Run the freeze prerequisites and checker.**

  ```bash
  cargo fmt --all --check
  cargo test --workspace --all-targets --release
  cargo clippy --workspace --all-targets -- -D warnings
  cargo doc --workspace --no-deps
  python3 scripts/check-storage-design-docs.py
  python3 scripts/check-storage-ownership-contracts.py
  python3 scripts/test-storage-ownership-contracts-v2.py
  python3 scripts/check-storage-contract-freeze.py --report docs/design/storage-contract-freeze.md
  ```

- [ ] **Step 4: Commit candidate C.**

  ```bash
  git add -A
  git commit -m "refactor(storage): freeze adapter-free ownership candidate"
  git status --short
  git rev-parse HEAD
  ```

  Record the exact candidate commit before any P11/P12 evidence. Any later semantic/API/docs/checker change creates a new candidate and invalidates affected evidence.

### Task 8: Produce P11 hardware matrix evidence on candidate C

**Files:**
- Create: `scripts/check-storage-hardware-matrix.py`
- Create: `docs/testing/storage-hardware-matrix.md`
- Modify: `docs/worklogs/2026-08-04-p8-p13-completion.md`

**Interfaces:**
- Report schema is `tenferro.storage-hardware-matrix.v1` with exact candidate commit, required lanes `cpu`, `cuda2`, `webgpu`, `metal`, `cuda-ad`, exact commands, concrete environment/device facts, test counts, observations, status, and ordinary evidence paths.
- Required-mode variables fail when required hardware/tests are absent; ordinary local runs report structured skips without mislabeling them as passes.

- [ ] **Step 1: Add the failing schema/checker tests.**

  Test wrong candidate format, missing required lane, missing concrete command/environment/device/evidence fields, unstructured skip, and mismatch between report candidate and freeze report.

- [ ] **Step 2: Run the matrix on candidate C.**

  ```bash
  python3 scripts/check-storage-hardware-matrix.py --report docs/testing/storage-hardware-matrix.md
  ```

  Run CPU reference and available CUDA/WebGPU/Metal/async-AD lanes. CUDA must exercise two visible devices when required mode is enabled; otherwise record the exact structured skip and owner. Do not mark unavailable hardware as pass.

- [ ] **Step 3: Commit evidence-only artifacts.**

  ```bash
  git add scripts/check-storage-hardware-matrix.py docs/testing/storage-hardware-matrix.md docs/worklogs/2026-08-04-p8-p13-completion.md
  git commit -m "test(storage): record frozen hardware matrix"
  ```

  If the evidence commit changes production/API/docs/checker semantics, return to Task 7 and create a new candidate instead of appending invalid evidence.

### Task 9: Audit P12 documentation on the same candidate

**Files:**
- Create: `docs/worklogs/storage-documentation-source-blind-audit.md`
- Modify: `docs/testing/storage-hardware-matrix.md` only for evidence cross-reference when no semantics change is involved
- Modify: `docs/worklogs/2026-08-04-p8-p13-completion.md`

**Interfaces:**
- The audit input is rendered Quarto/rustdoc/tutorial output without source links. It must reconstruct owner/view/mutable-disjoint/duplicate/reinterpret usage, CUDA/WebGPU transfer flow, Apple shared endpoint semantics, and detached versus borrowed outcomes.

- [ ] **Step 1: Build/render the docs from the frozen candidate.**

  ```bash
  bash scripts/build_docs_site.sh
  cargo doc --workspace --no-deps
  ```

- [ ] **Step 2: Run the source-blind usability audit and executable checks.**

  Use only rendered artifacts to write and compile a minimal CPU example against the public crates, then run:

  ```bash
  python3 scripts/check-storage-docs.py --include-rendered
  python3 scripts/check-storage-element-access-docs.py docs/guides/views-and-slicing.md
  cargo test -p tenferro-tutorial-code --release tutorial_binaries_run_successfully -- --exact
  python3 scripts/check-docs-site.py
  ```

- [ ] **Step 3: Record the audit.**

  Record concrete rendered paths, commands, reconstructed usage, findings, and zero Critical/Important usability gaps. If the audit finds a documentation product defect, fix it before freeze and repeat Tasks 7–9.

- [ ] **Step 4: Commit evidence-only audit.**

  ```bash
  git add docs/worklogs/storage-documentation-source-blind-audit.md docs/worklogs/2026-08-04-p8-p13-completion.md
  git commit -m "docs(storage): audit frozen ownership documentation"
  ```

### Task 10: Run P13-B independent closure audit

**Files:**
- Create: `scripts/check-storage-redesign-closure.py`
- Create: `docs/worklogs/storage-redesign-closure.md`
- Modify: `docs/worklogs/2026-08-04-p8-p13-completion.md`

**Interfaces:**
- Closure report audits architecture, Rust/resource lifecycle, performance, API/docs, CPU, GPU/multi-GPU, AD, and cross-lane integration independently of implementation edits.
- Critical/Important findings block closure. Performance inconclusive or required-hardware skip blocks closure unless the report identifies the exact required-mode evidence owner and command; no closure shortcut is allowed.

- [ ] **Step 1: Add checker RED tests.**

  Verify candidate mismatch, missing P11/P12 evidence, stale legacy inventory, missing required obligations, and Critical/Important finding rejection.

- [ ] **Step 2: Run the full closure verification.**

  ```bash
  python3 scripts/check-storage-redesign-closure.py --report docs/worklogs/storage-redesign-closure.md
  python3 scripts/run-storage-ownership-contracts.py --receipt /tmp/tenferro-storage-ownership-receipt-p13.json
  bash scripts/check-pr-fast.sh --base origin/main --no-fetch --coverage-reviewed --doc-snippets
  ```

- [ ] **Step 3: Resolve every finding before recording closure.**

  For each finding, map it to the owning phase and fix it with a failing regression test. Re-run the affected phase evidence and, if the candidate changed semantically, repeat freeze and all affected hardware/docs evidence.

- [ ] **Step 4: Commit closure evidence.**

  ```bash
  git add scripts/check-storage-redesign-closure.py docs/worklogs/storage-redesign-closure.md docs/worklogs/2026-08-04-p8-p13-completion.md
  git commit -m "docs(storage): record independent P13 closure audit"
  ```

### Task 11: Perform final PR review, CI, and merge

**Files:**
- Modify: PR body/metadata only through `gh`
- Final tracked changes: none after evidence is accepted

**Interfaces:**
- PR targets `main`, links Issues #1555 and #1564–#1569, approved specs, worklog, freeze report, hardware matrix, docs audit, and closure report.
- The PR is mergeable only with all required CI checks green and the final remote PR state `MERGED`.

- [ ] **Step 1: Run final local acceptance.**

  ```bash
  git diff --check origin/main...HEAD
  git status --short
  cargo fmt --all --check
  cargo test --workspace --release
  cargo llvm-cov --workspace --release --json --output-path coverage.json
  python3 scripts/check-coverage.py coverage.json
  cargo clippy --workspace --all-targets -- -D warnings
  cargo doc --workspace --no-deps
  python3 scripts/check-docs-site.py
  python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review-final.json
  bash scripts/check-pr-fast.sh --base origin/main --no-fetch --coverage-reviewed --doc-snippets
  ```

- [ ] **Step 2: Review the complete diff and task ledger.**

  Confirm all explicit goal requirements map to current files, commands, reports, and phase issue obligations. Confirm no user changes were discarded, no implementation plan/spec is stale, no unapproved shortcut remains, and worktree is clean.

- [ ] **Step 3: Create/update the PR and wait for required checks.**

  ```bash
  gh pr create --base main --head "$(git branch --show-current)" --title "Complete P8-P13 storage ownership migration" --body-file /tmp/p8-p13-pr-body.md
  gh pr checks --watch
  gh pr view --json number,state,mergeStateStatus,statusCheckRollup,url
  ```

  Enable the repository-required merge mode only after checks pass. For this ordinary PR use the repository rule:

  ```bash
  gh pr merge --auto --squash --delete-branch
  ```

- [ ] **Step 4: Verify actual merge.**

  ```bash
  gh pr view --json state,mergedAt,mergeCommit,url
  git fetch origin main
  git status --short --branch
  ```

  The goal is incomplete until the remote PR state is `MERGED`, required checks are green, the merged commit contains the final evidence, and the completion audit maps every requirement to fresh evidence.
