# P3+P9 Atomic Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cloneable host tensor/value ownership and borrow-and-clone runtime submission with one move-only owner/view/group model, activating P3 and P9 atomically.

**Architecture:** Host `TypedTensor` values own one P5 `AllocationGroup` and one local descriptor slot. Borrowed typed/dtype-erased views resolve through that group borrow and call P4 prepared access. `TensorValue` and detached runtime submission move groups; eager/AD handles share read-only records without cloning owners, and retirement owns admitted bundles until proven completion.

**Tech Stack:** Rust ownership/lifetimes, `tenferro-tensor` private storage kernel, existing runtime execution/retirement, trybuild, fake CPU provider counters, Python storage ledger.

---

### Task 1: Add the failing P3/P9 proof artifacts

**Files:**
- Create: `crates/tenferro-tensor/tests/storage_static_rank.rs`
- Create: `crates/tenferro-tensor/tests/storage_as_view_allocation.rs`
- Create: `crates/tenferro-tensor/tests/storage_auto_traits.rs`
- Modify: `crates/tenferro-tensor/tests/storage_compile_contract.rs`
- Modify: `crates/tenferro-tensor/tests/ui/storage/fail/typed_tensor_not_clone.rs`
- Modify: `crates/tenferro-tensor/tests/ui/storage/fail/typed_tensor_view_root_access.rs`

- [ ] **Step 1: Write tests that express the final surface.**

  The tests must use `TypedTensor::<f64, Rank<2>>`, `as_view`,
  `as_view_mut`, `duplicate`, and the group-backed typed views. Add compile-fail
  fixtures that attempt `tensor.clone()` and access the owner while a mutable
  view is live. The allocation test records the existing test counter snapshot
  before and after `as_view()`/`as_view_mut()` and asserts equality.

- [ ] **Step 2: Run the canonical artifacts and record the RED state.**

  Run:

  ```bash
  cargo test -p tenferro-tensor --test storage_static_rank
  cargo test -p tenferro-tensor --test storage_as_view_allocation
  cargo test -p tenferro-tensor --test storage_auto_traits
  ```

  Expected result: the new files do not compile because the current owner is
  cloneable and the final methods/counters are absent. Do not activate ledger
  rows while this RED harness is incomplete.

- [ ] **Step 3: Commit the RED harness.**

  ```bash
  git add crates/tenferro-tensor/tests
  git commit -m "test(storage): add p3 p9 cutover red harness"
  ```

### Task 2: Make host allocation a private root/provider owner

**Files:**
- Modify: `crates/tenferro-tensor/src/storage/root.rs`
- Modify: `crates/tenferro-tensor/src/storage/mod.rs`
- Test: `crates/tenferro-tensor/src/storage/tests/root_claims.rs`

- [ ] **Step 1: Add the host provider implementation.**

  Add a private `HostAllocation<T>` containing `Vec<T>` and one
  `AllocationKey`. Implement `BackendAllocation` with checked byte extent,
  `ProviderKind::Cpu`, host capability, and borrowed byte mappings. The only
  `unsafe` code is the byte-slice view of the typed `Vec`; document that
  `T: TensorScalar` is initialized, sized, and aligned for the exact extent.

- [ ] **Step 2: Add a checked host import helper.**

  Add `import_host_vec<T>(data: Vec<T>) -> Result<OwnedStorage, ...>` and a
  crate-private `OwnedStorage::root_span()` accessor. The helper consumes the
  vector and never clones or retains a second owner.

- [ ] **Step 3: Test drop identity and borrowed mapping.**

  Extend the root private tests with empty, scalar, and non-empty host imports,
  exact byte lengths, typed mapping, and one-drop behavior. Run:

  ```bash
  cargo test -p tenferro-tensor --lib storage::tests::root_claims
  cargo clippy -p tenferro-tensor --all-targets -- -D warnings
  ```

- [ ] **Step 4: Commit the owner boundary.**

  ```bash
  git add crates/tenferro-tensor/src/storage
  git commit -m "feat(storage): import host vectors as unique roots"
  ```

### Task 3: Expose typed group descriptors without exposing storage authority

**Files:**
- Modify: `crates/tenferro-tensor/src/storage/group.rs`
- Modify: `crates/tenferro-tensor/src/storage/prepared.rs`
- Modify: `crates/tenferro-tensor/src/storage/mod.rs`
- Test: `crates/tenferro-tensor/src/storage/tests/group.rs`

- [ ] **Step 1: Add a host-group constructor and descriptor accessors.**

  Add `AllocationGroup::from_host_vec<T, R>(shape, data)` returning the group
  and its `DescriptorSlot`. Retain the checked dynamic layout, root span, dtype,
  and injectivity proof in the descriptor. Add crate-private typed
  `prepare_read`/`prepare_write` methods on borrowed group children that call
  P4 exactly once.

- [ ] **Step 2: Add typed borrowed metadata methods.**

  Return shape, strides, offset, dtype, and placement from the retained
  descriptor. Do not return `OwnedStorage`, provider handles, raw pointers, or
  a cloneable buffer. Keep child lifetimes tied to `&AllocationGroup` or
  `&mut AllocationGroup`.

- [ ] **Step 3: Test validation-count invariants.**

  Test direct slot resolution, prepared mapping counters, missing injectivity,
  N-way split, and no map/enqueue during split. Run the existing 12 group tests
  plus the new focused tests.

- [ ] **Step 4: Commit the typed group boundary.**

  ```bash
  git add crates/tenferro-tensor/src/storage
  git commit -m "feat(storage): add typed group descriptor access"
  ```

### Task 4: Convert `TypedTensor` and typed views to move-only group owners

**Files:**
- Modify: `crates/tenferro-tensor/src/types.rs`
- Modify: `crates/tenferro-tensor/src/lib.rs`
- Modify: `crates/tenferro-tensor/src/tests/types_tests.rs`
- Modify: `crates/tenferro-internal-cpu-kernels/src/lib.rs`
- Modify: `crates/tenferro-internal-cpu-kernels/src/elementwise.rs`

- [ ] **Step 1: Replace cloneable storage fields.**

  Store one `AllocationGroup`, one `DescriptorSlot`, and scalar/rank phantom
  metadata in `TypedTensor<T, R>`. Remove `Clone` from `TypedTensor`, `Tensor`,
  and mutable views. Keep all constructors consuming `Vec<T>` and preserve
  checked shape/rank errors.

- [ ] **Step 2: Implement O(1) owner/view/reborrow operations.**

  `as_view()` returns a view borrowing the group. `as_view_mut()` returns a
  mutable child. Neither allocates, clones layout metadata, clones provider
  state, or increments an ownership/refcount counter. `duplicate()` explicitly
  materializes a new owner and is the only copy boundary.

- [ ] **Step 3: Port typed access to prepared access.**

  Replace direct `Buffer::Host`/`Buffer::Backend` pattern matching in CPU
  kernels with typed prepared read/write slices and iterators. The inner loop
  performs only typed access and the precomputed stride/carry increments.

- [ ] **Step 4: Remove public legacy storage names.**

  Make `Buffer`, `BackendBuffer`, `BufferHandle`, `TensorBufferRef`,
  `TensorBufferRefMut`, `TensorOwnedView`, and `TypedTensorViewMutPair` absent
  from the public module and update all in-repository imports to the canonical
  owner/view APIs. Do not add deprecated aliases.

- [ ] **Step 5: Run P3 compile and rank tests.**

  ```bash
  cargo test -p tenferro-tensor --test storage_compile_contract
  cargo test -p tenferro-tensor --test storage_static_rank
  cargo test -p tenferro-tensor --test storage_as_view_allocation
  cargo test -p tenferro-tensor --test storage_auto_traits
  ```

- [ ] **Step 6: Commit the public host owner migration.**

  ```bash
  git add crates/tenferro-tensor crates/tenferro-internal-cpu-kernels
  git commit -m "feat(storage): cut host tensors over to move-only owners"
  ```

### Task 5: Replace `TensorValue` and eager/AD retention with owner bundles

**Files:**
- Modify: `crates/tenferro-tensor/src/types.rs`
- Modify: `crates/tenferro-runtime/src/exec.rs`
- Modify: `crates/tenferro-ad/src/eager.rs`
- Modify: `crates/tenferro-ad/src/eager_ops.rs`
- Modify: `crates/tenferro-ad/src/eager_backend.rs`
- Modify: `crates/tenferro-ad/src/traced.rs`
- Test: `crates/tenferro-ad/src/eager/tests.rs`

- [ ] **Step 1: Define the non-clone `TensorValue` bundle.**

  Replace the `Arc<Tensor>`/lazy `TensorOwnedView` enum with one struct owning
  an `AllocationGroup`, descriptor slot, and output metadata. Read access
  borrows the bundle; consuming extraction uses local descriptor uniqueness.

- [ ] **Step 2: Make eager handles read-only record handles.**

  Keep an `Arc<EagerTensorRecord>` only as a read-only handle container. The
  record owns the bundle once; removing or retaining an eager handle never
  clones a tensor owner, provider state, or materialized cache. Mutable results
  are newly allocated bundles.

- [ ] **Step 3: Port AD/checkpoint retention.**

  Store direct bundle records/descriptor slots in eager gradients, tape values,
  and traced checkpoints. Add tests that compare allocation/refcount/provider
  counters before and after retention and that aliasing outputs retain one
  owner.

- [ ] **Step 4: Run AD/runtime unit tests.**

  ```bash
  cargo test -p tenferro-ad --lib
  cargo test -p tenferro-runtime --lib exec::tests
  ```

- [ ] **Step 5: Commit owner-bundle retention.**

  ```bash
  git add crates/tenferro-tensor crates/tenferro-ad crates/tenferro-runtime
  git commit -m "feat(ad): retain allocation groups without owner clones"
  ```

### Task 6: Replace runtime submission with consuming group ownership

**Files:**
- Modify: `crates/tenferro-runtime/src/runtime/execution.rs`
- Modify: `crates/tenferro-runtime/src/runtime/snapshot.rs`
- Modify: `crates/tenferro-runtime/src/runtime/mod.rs`
- Modify: `crates/tenferro-runtime/src/error.rs`
- Modify: `crates/tenferro-runtime/tests/integration/runtime_execution.rs`
- Modify: `crates/tenferro-tensor/tests/storage_compile_contract.rs`

- [ ] **Step 1: Add `ExecutionInputs` and owned result bundles.**

  `ExecutionInputs` owns one `AllocationGroup` plus descriptor bindings. The
  detached `submit` consumes it. `ExecutionHandle::wait` returns an owned
  result bundle only for `Completed`/`RetiredFailed`; pre-admission errors
  return the unchanged inputs and completion-unproven returns a typed error
  with no owner.

- [ ] **Step 2: Move admission before ownership transfer.**

  Resolve metadata, prepare provider access, and perform worker admission before
  constructing the in-flight owner. After the enqueue boundary, move the group
  into P4 retirement. No post-boundary owner recovery or retry state is added.

- [ ] **Step 3: Add synchronous borrowed submission eligibility.**

  Add a scoped borrowed operation that accepts only CPU/providers with a
  synchronous retirement witness. CUDA/WebGPU/Metal return typed unsupported
  before admission. The borrow cannot escape the call or unwind.

- [ ] **Step 4: Test detached/borrowed outcomes.**

  Cover unchanged pre-admission failure, handle detach, aliasing outputs,
  structural extraction, provider rejection, successful retirement, retired
  failure, and completion-unproven no-owner diagnostics. Add the compile-fail
  guard for a host view crossing consuming submission.

- [ ] **Step 5: Commit the submission cutover.**

  ```bash
  git add crates/tenferro-runtime crates/tenferro-tensor/tests
  git commit -m "feat(runtime): submit owned allocation groups"
  ```

### Task 7: Activate the atomic P3/P9 ledger and documentation

**Files:**
- Modify: `scripts/storage-ownership-contracts.toml`
- Modify: `scripts/test-storage-ownership-contracts-v2.py`
- Modify: `docs/design/storage-ownership-contracts.md`
- Create: `docs/worklogs/2026-08-04-issue-1559-1565-p3-p9-atomic-cutover.md`

- [ ] **Step 1: Promote exactly the four P3 rows and P9 row together.**

  Change only their tagged state to `{ kind = "active" }`; leave P6 and later
  deferred. Update the executable active-ID set and no other cohort.

- [ ] **Step 2: Document the final owner/value/submission contracts.**

  Update G3/G4/G7 and the current phase paragraph with the actual private
  implementation paths, explicit duplicate boundary, group-owned AD records,
  and pre/post-admission outcomes. Do not document unsupported hardware as
  passing evidence.

- [ ] **Step 3: Run exact atomic verification.**

  ```bash
  cargo fmt --all --check
  git diff --check
  cargo test --workspace --all-targets --quiet
  cargo clippy --workspace --all-targets -- -D warnings
  cargo llvm-cov -p tenferro-tensor --lib --summary-only
  python3 scripts/check-storage-ownership-contracts.py
  python3 scripts/check-storage-design-docs.py
  python3 scripts/test-storage-ownership-contracts-v2.py
  python3 scripts/repository-rules-review.py --dry-run --base origin/main --head "$(git rev-parse HEAD)" --output-json /tmp/p3-p9-rules-review.json
  ```

- [ ] **Step 4: Commit the atomic evidence.**

  ```bash
  git add scripts docs/design docs/worklogs
  git commit -m "docs(storage): activate atomic p3 p9 evidence"
  ```

P6 remains the next eligible phase only after this exact atomic candidate is
reviewed; no later phase is resumed implicitly.
