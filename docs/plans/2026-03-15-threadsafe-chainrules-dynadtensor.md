# Thread-Safe Chainrules and DynAdTensor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify `chainrules` reverse-mode onto one thread-safe graph engine and refactor `tenferro-dyadtensor` so `DynAdTensor` is the only public AD tensor object.

**Architecture:** Replace the duplicated `Tape/TrackedValue` and `Variable/AutogradContext` reverse engines with a shared `AutogradGraph<V>` backed by `Arc<Mutex<_>>`. Move downstream reverse-rule captured state to thread-safe ownership and remove `DynTape` from the dyadtensor public surface, with `DynAdTensor` wrapping primal/forward/reverse dynamic values.

**Tech Stack:** Rust, `Arc<Mutex<_>>`, `chainrules-core`, `chainrules`, `tenferro-einsum`, `tenferro-dyadtensor`

---

### Task 1: Rename and centralize the reverse graph core

**Files:**
- Modify: `extern/chainrules/src/engine/context.rs`
- Modify: `extern/chainrules/src/engine/mod.rs`
- Modify: `extern/chainrules/src/lib.rs`
- Test: `extern/chainrules/src/engine/tests/*.rs`

**Step 1: Write failing organization and Send/Sync tests**
Add tests asserting the shared graph core is the only reverse engine backing both `Tape` and `Variable`, and add compile-time `Send + Sync` assertions for the new graph wrapper types.

**Step 2: Run targeted tests to see them fail**
Run: `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p chainrules engine -- --nocapture`
Expected: failures because the old `Rc<RefCell<_>>` tape and duplicated graph naming still exist.

**Step 3: Rename `AutogradContext<V>` into `AutogradGraph<V>` and update exports/docs**
Keep behavior unchanged in this step; only establish the shared graph-core name and public/internal module structure.

**Step 4: Run targeted tests**
Run the same `cargo test -p chainrules engine` command.
Expected: naming/organization tests pass or progress to the next failures.

**Step 5: Commit**
`git add extern/chainrules/src/engine/context.rs extern/chainrules/src/engine/mod.rs extern/chainrules/src/lib.rs extern/chainrules/src/engine/tests`
`git commit -m "refactor: rename chainrules graph core"`

### Task 2: Make rule traits thread-safe

**Files:**
- Modify: `extern/chainrules-core/src/lib.rs`
- Modify: `extern/chainrules/src/ops/autograd.rs`
- Modify: `extern/chainrules/src/engine/context.rs`
- Test: `extern/chainrules-core/src/tests/*.rs`
- Test: `extern/chainrules/src/engine/tests/*.rs`

**Step 1: Write failing compile-time assertions**
Add tests that require `ReverseRule<V>` and `ForwardRule<V>` implementors used by chainrules helpers to be `Send + Sync` compatible.

**Step 2: Run targeted tests to verify failure**
Run: `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p chainrules-core -- --nocapture`
Expected: failures because the traits currently do not require thread-safe bounds.

**Step 3: Add `Send + Sync` bounds to `ReverseRule<V>` and `ForwardRule<V>`**
Update example code and internal helper signatures accordingly.

**Step 4: Run targeted tests**
Run both:
- `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p chainrules-core -- --nocapture`
- `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p chainrules engine -- --nocapture`
Expected: thread-safe trait changes compile and tests pass or expose downstream non-thread-safe users.

**Step 5: Commit**
`git add extern/chainrules-core/src/lib.rs extern/chainrules/src/ops/autograd.rs extern/chainrules/src/engine/context.rs extern/chainrules-core/src/tests extern/chainrules/src/engine/tests`
`git commit -m "refactor: require thread-safe autodiff rules"`

### Task 3: Rebuild `Tape<V>` on top of the shared graph core

**Files:**
- Modify: `extern/chainrules/src/engine/tape.rs`
- Modify: `extern/chainrules/src/engine/tracked.rs`
- Modify: `extern/chainrules/src/engine/results.rs`
- Test: `extern/chainrules/src/engine/tests/*.rs`

**Step 1: Write failing tests that compare `Tape` and `Variable` behavior**
Cover leaf creation, placeholder/recorded op behavior, pullback, pullback-with-seed, HVP, and `same_tape()` semantics.

**Step 2: Run targeted tests**
Run: `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p chainrules tape -- --nocapture`
Expected: failures because `Tape` still uses its own `Rc<RefCell<_>>` engine.

**Step 3: Refactor `Tape<V>` and `TrackedValue<V>` to share `AutogradGraph<V>`**
- replace `Rc<RefCell<_>>` with `Arc<Mutex<AutogradGraph<V>>>`
- delegate reverse traversal and accumulation to shared graph-core helpers
- keep the public `Tape` / `TrackedValue` API shape as much as possible

**Step 4: Run targeted tests**
Run:
- `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p chainrules tape -- --nocapture`
- `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p chainrules tracked -- --nocapture`
Expected: `Tape` now passes on the shared thread-safe engine.

**Step 5: Commit**
`git add extern/chainrules/src/engine/tape.rs extern/chainrules/src/engine/tracked.rs extern/chainrules/src/engine/results.rs extern/chainrules/src/engine/tests`
`git commit -m "refactor: unify tape with shared autograd graph"`

### Task 4: Remove thread-unsafe backend-context captures from einsum AD

**Files:**
- Modify: `tenferro-einsum/src/ad/tracked.rs`
- Modify: `tenferro-einsum/src/ad/reverse_rule.rs`
- Modify: `tenferro-einsum/src/ad/tests/*.rs`
- Test: `tenferro-einsum/tests/einsum_tests.rs`

**Step 1: Write failing tests / assertions for thread-safe tracked einsum AD**
Add compile-time or unit-level assertions that tracked reverse rules and captured contexts are thread-safe.

**Step 2: Run targeted tests**
Run: `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p tenferro-einsum tracked_einsum -- --nocapture`
Expected: failures or compile errors due to `Rc<RefCell<_>>` captures.

**Step 3: Replace `Rc<RefCell<BackendContext<...>>>` with `Arc<Mutex<BackendContext<...>>>` in tracked reverse paths**
Do not widen the change beyond AD-related paths.

**Step 4: Run targeted tests**
Run the same `tracked_einsum` tests, plus relevant `einsum_tests` slices.
Expected: tracked reverse-mode einsum compiles and passes under the new thread-safe bounds.

**Step 5: Commit**
`git add tenferro-einsum/src/ad/tracked.rs tenferro-einsum/src/ad/reverse_rule.rs tenferro-einsum/src/ad/tests tenferro-einsum/tests/einsum_tests.rs`
`git commit -m "refactor: make tracked einsum context thread-safe"`

### Task 5: Remove `DynTape` from dyadtensor public API

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_tape.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Modify: `extension/tenferro-dyadtensor/tests/public_surface_tests.rs`
- Test: `extension/tenferro-dyadtensor/tests/*.rs`

**Step 1: Write failing public-surface tests**
Add tests that use only `DynAdTensor` for reverse-mode graph creation and confirm `DynTape` is no longer part of the intended public surface.

**Step 2: Run targeted tests**
Run: `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p tenferro-dyadtensor public_surface -- --nocapture`
Expected: failures because current docs/tests still rely on `DynTape`.

**Step 3: Refactor `DynAdTensor` constructors and reverse-mode helpers to hide the graph handle**
`DynTape` may remain as an internal compatibility shim during this task, but it should disappear from public exports and examples.

**Step 4: Run targeted tests**
Run the same public-surface tests plus dynamic wrapper tests.
Expected: dynamic public API passes without `DynTape`.

**Step 5: Commit**
`git add extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_tape.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs extension/tenferro-dyadtensor/tests`
`git commit -m "refactor: hide dyadtensor graph internals"`

### Task 6: Rebuild `DynAdTensor` reverse mode on `Variable<DynTensor>`

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/value/tensor.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/*.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/common.rs`
- Modify: `extension/tenferro-dyadtensor/src/tape/registry.rs`
- Test: `extension/tenferro-dyadtensor/src/core/dynamic/tests/*.rs`
- Test: `extension/tenferro-dyadtensor/tests/*.rs`

**Step 1: Write failing tests for `DynAdTensor` reverse-mode behaviors under the new model**
Cover:
- `requires_grad_`
- `pullback_wrt`
- mixed-dtype reverse on one graph
- reverse-mode detach behavior
- `Send + Sync` assertions for `DynAdTensor`

**Step 2: Run targeted tests**
Run: `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p tenferro-dyadtensor dynamic_wrapper_coverage -- --nocapture`
Expected: failures because reverse state still centers on `TrackedValue<DynTensor>` and explicit tape plumbing.

**Step 3: Refactor reverse state to use `Variable<DynTensor>` as the canonical reverse payload**
- keep primal and forward dynamic paths coherent
- remove or collapse obsolete tape-registry helpers
- preserve `Diag` support

**Step 4: Run targeted tests**
Run the dynamic tests and public surface tests again.
Expected: `DynAdTensor` works without exposing tape internals and is thread-safe by construction.

**Step 5: Commit**
`git add extension/tenferro-dyadtensor/src/core/value/tensor.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor extension/tenferro-dyadtensor/src/ops/common.rs extension/tenferro-dyadtensor/src/tape/registry.rs extension/tenferro-dyadtensor/src/core/dynamic/tests extension/tenferro-dyadtensor/tests`
`git commit -m "refactor: rebuild dynadtensor on variable graph"`

### Task 7: Reconfirm dense-only linalg for structured inputs

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ops/linalg/**/*.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/eager_linalg.rs`
- Test: `extension/tenferro-dyadtensor/tests/*.rs`
- Docs: `docs/design/autodiff.md`, `docs/design/supported-ops.md`

**Step 1: Write failing tests for structured non-dense linalg rejection**
Use `Diag` or another non-dense structured layout and assert runtime errors for linalg entry points.

**Step 2: Run targeted tests**
Run: `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test -p tenferro-dyadtensor linalg -- --nocapture`
Expected: either missing coverage or inconsistent error behavior.

**Step 3: Add one shared dense-gate helper and use it in all linalg entry points**
Avoid ad hoc per-op checks.

**Step 4: Run targeted tests**
Re-run the linalg slice and confirm consistent runtime errors for non-dense structured inputs.

**Step 5: Commit**
`git add extension/tenferro-dyadtensor/src/ops/linalg extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/eager_linalg.rs docs/design/autodiff.md docs/design/supported-ops.md extension/tenferro-dyadtensor/tests`
`git commit -m "refactor: gate structured linalg behind dense-only checks"`

### Task 8: Full docs and verification sweep

**Files:**
- Modify: `extern/chainrules/src/lib.rs`
- Modify: `docs/api_index.md`
- Modify: `extension/tenferro-dyadtensor/README.upstream.md`
- Modify: public rustdoc touched by the refactor

**Step 1: Grep for stale API/model references**
Search for:
- `DynTape`
- `Rc<RefCell` in AD paths
- `AutogradContext` if renamed
- stale `TrackedTensor` / `DualTensor` names if still present

**Step 2: Fix docs and examples**
Update rustdoc examples and design docs to describe:
- thread-safe chainrules foundation
- `DynAdTensor` as the public dynamic AD tensor
- rank-0 scalar semantics
- dense-only linalg on structured inputs

**Step 3: Run full required verification**
Run:
- `cargo fmt --all --check`
- `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-target cargo test --workspace --release`
- `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-cov-target cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `env CARGO_TARGET_DIR=/tmp/tenferro-threadsafe-chainrules-doc-target cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-threadsafe-chainrules-doc-target/doc`

**Step 4: Re-read the diff for ad hoc regressions**
Inspect the changed files for duplicated traversal logic, leaked internal types, and stale non-thread-safe ownership.

**Step 5: Commit**
`git add extern/chainrules/src/lib.rs docs/api_index.md extension/tenferro-dyadtensor/README.upstream.md coverage.json`
`git commit -m "docs: finalize thread-safe chainrules redesign"`
