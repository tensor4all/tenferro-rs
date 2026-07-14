# CUDA Bug Remediation Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve #1353 and the CUDA correctness/compatibility issues #1356 through #1366 in one reviewable remediation branch and one non-squash PR.

**Architecture:** Preserve equal-shape primitive launches and explicit broadcast semantics, adding narrowly scoped CUDA kernels or dispatch helpers where CPU/CUDA parity requires them. Keep floating-point exceptional values as IEEE values, validate structural/index inputs before launch, and expose only behavior already promised by the tensor contracts. Each issue cluster lands as a coherent commit with focused CPU/source-contract/CUDA tests before the final repository-wide verification.

**Tech Stack:** Rust, tenferro tensor/backend traits, CubeCL CUDA kernels, `gh`, Cargo tests, rustdoc, `cargo llvm-cov`.

---

## File Map

- `docs/spec/tensor-semantics.md`: authoritative floating-point domain and explicit-cast contracts.
- `docs/spec/primitive-catalog.md`: short pointer from elementwise primitive semantics to tensor numeric semantics.
- `docs/design/gpu-backend-design.md`: CUDA dtype and validation capability notes.
- `docs/guides/devices-and-gpu.md`: user-visible CUDA limitations after parity changes.
- `crates/tenferro-gpu/src/kernels/elementwise.rs`: IEEE edge behavior, complex magnitude, and broadcast-aware binary kernels.
- `crates/tenferro-gpu/src/kernels/structural.rs`: Bool-compatible movement and cast kernels.
- `crates/tenferro-gpu/src/kernels/indexing.rs`: device-side index validation support.
- `crates/tenferro-gpu/src/cubecl/mod.rs`: CUDA dispatch, validation, reduction metadata, cast matrix, and backend hooks.
- `crates/tenferro-gpu/src/cubecl/fusion/classify.rs`: graceful fusion refusal.
- `crates/tenferro-gpu/src/cubecl/tests/*.rs`: ignored hardware-backed CPU/CUDA parity tests.
- `crates/tenferro-gpu/tests/cubecl_launch_contract.rs`: source contracts usable without a GPU.
- `docs/worklogs/2026-07-11-cuda-bug-remediation.md`: classification ledger, decisions, verification, and residual risks.

### Task 1: Establish the remediation ledger

**Files:**
- Create: `docs/worklogs/2026-07-11-cuda-bug-remediation.md`

- [ ] **Step 1: Record the exact issue classification**

Create a table with one row for each issue and these initial states:

```markdown
| Issue | Classification | Current evidence | Close condition |
| --- | --- | --- | --- |
| #1353 | Auto Fix | Float `% 0` already yields `NaN`; policy missing | spec + CPU/CUDA tests |
| #1356 | Auto Fix | generic CubeCL literals warn | warning-free CUDA build |
| #1357 | Auto Fix | direct div/rem launch rejects rank-0 | scalar lhs/rhs parity |
| #1358 | Auto Fix | direct pow launch rejects rank-0 | scalar lhs/rhs parity |
| #1359 | Auto Fix | reduction workaround leaks `[1]` | public shape is `[]` |
| #1360 | Verify First | dynamic_slice and scatter accept `1.5` | narrowed family rejects invalid float indices |
| #1361 | Auto Fix | abs/sign comparisons mishandle -0/NaN | CPU/CUDA IEEE parity |
| #1362 | Verify First | CUDA cast matrix is narrower than CPU | documented cast matrix parity |
| #1363 | Auto Fix | Bool movement ops dispatch to unsupported | listed structural paths work |
| #1364 | Auto Fix | complex abs is unsupported | C32→F32 and C64→F64 parity |
| #1365 | Auto Fix | unsupported fusion shape returns Err | returns Ok(None) |
| #1366 | Auto Fix | real scalar + complex tensor rejected | promoted scalar parity |
```

- [ ] **Step 2: Reconcile remote state before editing**

Run:

```bash
gh issue list --repo tensor4all/tenferro-rs --state open --limit 200 \
  --json number,title,updatedAt,labels
gh pr list --repo tensor4all/tenferro-rs --state open --limit 100 \
  --json number,title,body,headRefName
```

Expected: all twelve issues remain open and no open PR claims them. Update the ledger if remote state differs.

- [ ] **Step 3: Commit the ledger**

```bash
git add docs/worklogs/2026-07-11-cuda-bug-remediation.md
git commit -m "docs: start CUDA bug remediation ledger"
```

### Task 2: Freeze IEEE floating-point domain semantics (#1353)

**Files:**
- Modify: `docs/spec/tensor-semantics.md`
- Modify: `docs/spec/primitive-catalog.md`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`
- Test: `crates/tenferro-cpu/src/elementwise/tests.rs` or the existing module-local elementwise test file

- [ ] **Step 1: Add failing CPU special-value tests**

Cover `F32` and `F64` with assertions shaped like:

```rust
let quotient = div(&scalar(1.0), &scalar(0.0))?;
assert!(quotient.as_slice::<f64>()?[0].is_infinite());
let zero_over_zero = div(&scalar(0.0), &scalar(0.0))?;
assert!(zero_over_zero.as_slice::<f64>()?[0].is_nan());
let remainder = rem(&scalar(1.0), &scalar(0.0))?;
assert!(remainder.as_slice::<f64>()?[0].is_nan());
```

- [ ] **Step 2: Run the focused CPU test**

```bash
cargo test -p tenferro-cpu float_div_rem_preserve_ieee_special_values -- --exact
```

Expected: PASS if current CPU behavior already matches the contract; a failure is an implementation discrepancy to fix before documentation.

- [ ] **Step 3: Add the active specification**

Add a `Floating-point domain behavior` subsection stating that `F32/F64` elementwise domain edges produce IEEE `NaN`, infinity, and signed zero; integer domain errors remain typed; shape/dtype/index/config errors remain typed; no float preflight scan is required. Add one sentence to the elementwise section of `primitive-catalog.md` linking to that subsection.

- [ ] **Step 4: Add an ignored CUDA parity test**

Compare result classification and sign bits, not approximate equality:

```rust
assert_eq!(actual.is_nan(), expected.is_nan());
assert_eq!(actual.is_infinite(), expected.is_infinite());
if actual == 0.0 && expected == 0.0 {
    assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
}
```

- [ ] **Step 5: Run focused checks and commit**

```bash
cargo test -p tenferro-cpu float_div_rem_preserve_ieee_special_values
cargo test -p tenferro-gpu --features cuda --no-run
git add docs/spec/tensor-semantics.md docs/spec/primitive-catalog.md \
  crates/tenferro-cpu crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs
git commit -m "docs: define IEEE floating-point domain behavior (#1353)"
```

### Task 3: Fix IEEE unary kernels and future-incompatible literals (#1356, #1361)

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/elementwise.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`

- [ ] **Step 1: Add failing CUDA tests for negative zero and NaN**

Test both `F32` and `F64`:

```rust
assert_eq!(gpu_abs_neg_zero.to_bits(), cpu_abs_neg_zero.to_bits());
assert!(gpu_sign_nan.is_nan());
```

- [ ] **Step 2: Replace comparison-based abs/sign semantics**

Use CubeCL float primitives that preserve CPU `abs()`/`signum()` semantics. If CubeCL lacks a matching primitive, implement explicit NaN and zero branches and use typed literals:

```rust
let zero = F::new(0.0_f32);
let one = F::new(1.0_f32);
if value != value { value } else if value == zero { zero } else if value > zero { one } else { -one }
```

For abs, ensure both `+0.0` and `-0.0` return `+0.0` while NaN remains NaN.

- [ ] **Step 3: Verify warnings and values**

```bash
cargo test -p tenferro-gpu --features cuda --no-run
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  cubecl::tests::elementwise_tests::test_float_special_values_match_cpu -- --ignored
```

Expected: no float-literal fallback warning; CUDA test PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/tenferro-gpu/src/kernels/elementwise.rs \
  crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs
git commit -m "fix: align CUDA float unary edge semantics (#1356 #1361)"
```

### Task 4: Add CUDA complex magnitude (#1364)

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/elementwise.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`

- [ ] **Step 1: Add failing C32/C64 magnitude tests**

Use `(3,4) -> 5` and `(5,12) -> 13`, and assert output dtype is real.

- [ ] **Step 2: Add a mixed-input/output unary launch**

Add a kernel equivalent to:

```rust
#[cube(launch_unchecked)]
pub fn abs_complex<C: Complex, F: Float>(out: &mut Array<F>, input: &Array<C>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].magnitude();
    }
}
```

Use the pinned CubeCL complex `magnitude()` operation shown above so scaling remains stable; do not replace it with `sqrt(re*re + im*im)`.

- [ ] **Step 3: Dispatch C32→F32 and C64→F64**

Replace the unsupported complex branches in `CudaBackend::abs` with `launch_unary` calls whose output type is the matching real dtype.

- [ ] **Step 4: Run and commit**

```bash
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  test_cubecl_complex_abs_matches_cpu -- --ignored
git add crates/tenferro-gpu/src/kernels/elementwise.rs \
  crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs
git commit -m "fix: implement CUDA complex magnitude (#1364)"
```

### Task 5: Make unsupported fusion defuse cleanly (#1365)

**Files:**
- Modify: `crates/tenferro-gpu/src/cubecl/fusion/classify.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/fusion_tests.rs`

- [ ] **Step 1: Add the scalar-shape mismatch regression test**

Construct identity-view inputs with shapes `[3]` and `[]` and assert:

```rust
let result = backend.execute_elementwise_fusion(&[&vector, &scalar], &plan)?;
assert!(result.is_none());
```

- [ ] **Step 2: Verify the test fails with `ShapeMismatch`**

```bash
cargo test -p tenferro-gpu --features cuda fusion_shape_mismatch_defuses -- --ignored
```

- [ ] **Step 3: Return unsupported instead of invalid-input error**

In `classify`, change only the cross-input shape mismatch after descriptor validation:

```rust
if input.shape() != first.shape() {
    return Ok(None);
}
```

Keep dtype/descriptor corruption as errors.

- [ ] **Step 4: Run and commit**

```bash
cargo test -p tenferro-gpu --features cuda fusion_shape_mismatch_defuses -- --ignored
git add crates/tenferro-gpu/src/cubecl/fusion/classify.rs \
  crates/tenferro-gpu/src/cubecl/tests/fusion_tests.rs
git commit -m "fix: defuse unsupported CUDA fusion shapes (#1365)"
```

### Task 6: Add scalar-aware div/rem/pow dispatch (#1357, #1358)

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/elementwise.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`
- Modify: `crates/tenferro-gpu/tests/cubecl_launch_contract.rs`

- [ ] **Step 1: Add scalar lhs/rhs tests for float and integer operations**

Cover `[N] op []` and `[] op [N]` for float `div/rem/pow`; cover integer zero-divisor and negative-exponent checks after broadcast.

- [ ] **Step 2: Add a broadcast-aware binary launch helper**

Generalize the index-mapped kernel introduced for broadcast multiply so the operation is selected at compile time:

```rust
enum BroadcastBinaryKind { Divide, Remainder, Power }
```

The kernel must map each output index to lhs/rhs source indices and directly compute the operation. It must not call `broadcast_typed` or allocate a full-size operand temporary.

- [ ] **Step 3: Preserve integer domain reporting**

For integer broadcast kernels, retain the device error flag and map it to `DivisionByZero` or `NegativeExponent` after synchronization. Check the mapped rhs element before arithmetic.

- [ ] **Step 4: Add a source contract**

Assert that the scalar paths do not contain `broadcast_typed` and that `div`, `rem`, and `pow` route to the broadcast-aware launcher when exactly one input is rank-0.

- [ ] **Step 5: Run and commit**

```bash
cargo test -p tenferro-gpu --features cuda --test cubecl_launch_contract
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  test_scalar_div_rem_pow_match_cpu -- --ignored
git add crates/tenferro-gpu/src/kernels/elementwise.rs \
  crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs \
  crates/tenferro-gpu/tests/cubecl_launch_contract.rs
git commit -m "fix: support CUDA scalar div rem and pow (#1357 #1358)"
```

### Task 7: Add real-scalar/complex-tensor promotion (#1366)

**Files:**
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/kernels/elementwise.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`

- [ ] **Step 1: Add all symmetric promotion tests**

Cover `F32[]` with `C32[N]` and `F64[]` with `C64[N]` for add, sub, mul, and div, with the real scalar in every supported operand position.

- [ ] **Step 2: Reuse scalar-aware index mapping without host transfer**

Add mixed real/complex kernel variants that lift the scalar to complex in registers:

```rust
let promoted = Complex::new(real_scalar, F::new(0.0_f32));
```

Do not download the scalar, create a host tensor, or materialize a complex tensor of output size.

- [ ] **Step 3: Match the public promotion lattice exactly**

Accept only `F32↔C32` and `F64↔C64` rank-0 real operands already supported on CPU. Continue returning `DTypeMismatch` for non-scalar mixed tensors and cross-precision pairs.

- [ ] **Step 4: Run and commit**

```bash
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  test_real_scalar_complex_tensor_ops_match_cpu -- --ignored
git add crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-gpu/src/kernels/elementwise.rs \
  crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs
git commit -m "fix: promote CUDA real scalars in complex elementwise ops (#1366)"
```

### Task 8: Preserve rank-0 reduction output metadata (#1359)

**Files:**
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/reduction_tests.rs`

- [ ] **Step 1: Add full-axis shape tests**

For sum, product, min, and max where supported, compare both value and `shape() == []` against CPU.

- [ ] **Step 2: Separate launch metadata shape from public tensor shape**

Keep the public output shape returned by reduction shape inference unchanged (`[]`). If CubeCL needs one metadata dimension, introduce a private launch-only shape `[1]` while allocating/returning `TypedTensor` with rank zero.

- [ ] **Step 3: Run and commit**

```bash
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  test_full_axis_reductions_return_scalar_shape -- --ignored
git add crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-gpu/src/cubecl/tests/reduction_tests.rs
git commit -m "fix: preserve scalar CUDA reduction shapes (#1359)"
```

### Task 9: Enable Bool structural data movement (#1363)

**Files:**
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/kernels/structural.rs`
- Modify: `crates/tenferro-gpu/src/kernels/indexing.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/structural_tests.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/indexing_tests.rs`

- [ ] **Step 1: Add a Bool operation matrix test**

Cover transpose, broadcast, diagonal extraction/embedding, triangular masks, slice, dynamic slice with integer starts, pad, concatenate, reverse, gather, and scatter where CPU supports the same configuration.

- [ ] **Step 2: Route Bool through byte-storage-compatible kernels**

Use the existing Bool buffer binding helpers and add Bool kernel entry points for pure copying/indexing. Do not route Bool through numeric arithmetic traits. For triangular masks and pad fill, encode false as zero byte and preserve true as one byte.

- [ ] **Step 3: Audit every explicit Bool rejection in `CudaBackend`**

Remove a rejection only when the corresponding test and kernel path exist. Leave arithmetic/linalg Bool rejection unchanged.

- [ ] **Step 4: Run and commit**

```bash
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  test_bool_structural_ops_match_cpu -- --ignored
git add crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-gpu/src/kernels/structural.rs \
  crates/tenferro-gpu/src/kernels/indexing.rs \
  crates/tenferro-gpu/src/cubecl/tests
git commit -m "fix: support Bool CUDA structural operations (#1363)"
```

### Task 10: Validate floating-point index tensors on device (#1360)

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/indexing.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/indexing_tests.rs`

- [ ] **Step 1: Narrow the affected family with tests**

Test finite integral, fractional, NaN, positive/negative infinity, and values outside exact integer representation for dynamic slice, gather, and scatter. Record in the ledger which paths currently fail.

- [ ] **Step 2: Add a reusable device validation kernel**

For F32/F64 index tensors, set a one-element device error flag when:

```rust
value != value || value.is_infinite() || value.trunc() != value || value < I64_MIN || value > I64_MAX
```

Express these checks with CubeCL `is_nan`, `is_infinite`, `floor`, and comparison operations, converting the validated value only after the flag remains clear. Integer index tensors bypass this validation.

- [ ] **Step 3: Validate before the indexing launch**

Run the validation kernel, synchronize/read only the one-element error flag, and return the same typed validation category used by CPU. Never download the complete index tensor.

- [ ] **Step 4: Run and commit**

```bash
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  test_float_index_validation_matches_cpu -- --ignored
git add crates/tenferro-gpu/src/kernels/indexing.rs \
  crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-gpu/src/cubecl/tests/indexing_tests.rs
git commit -m "fix: validate CUDA float index tensors (#1360)"
```

### Task 11: Complete explicit CUDA cast parity (#1362)

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/structural.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/tests/structural_tests.rs`
- Modify: `docs/design/gpu-backend-design.md`
- Modify: `docs/guides/devices-and-gpu.md`

- [ ] **Step 1: Generate the CPU-supported cast matrix in one test**

Cover every source/target pair over F32, F64, I32, I64, Bool, C32, and C64. Assert values for representative finite inputs and use explicit cases for NaN/infinity/overflow rather than assuming Rust `as` behavior.

- [ ] **Step 2: Compare the CUDA matrix and update the ledger**

Classify each missing pair as numeric conversion, complex projection/injection, or Bool truthiness. The accepted behavior must match `docs/spec/tensor-semantics.md` and CPU.

- [ ] **Step 3: Add generic conversion kernels by family**

Implement shared kernels for real/integer conversion, Bool truthiness (`x != 0`, with the documented NaN rule), complex injection, and complex real projection. Reuse them across dtype pairs rather than copying one kernel per pair.

- [ ] **Step 4: Fill in `CudaBackend::cast` dispatch**

Replace each explicit unsupported branch only after its family kernel exists. Keep `convert` checked by the existing promotion lattice; this task changes explicit `cast`, not checked `convert`.

- [ ] **Step 5: Run and commit**

```bash
CUBECL_DEBUG_LOG=0 cargo test -p tenferro-gpu --features cuda \
  test_cuda_explicit_cast_matrix_matches_cpu -- --ignored
git add crates/tenferro-gpu/src/kernels/structural.rs \
  crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-gpu/src/cubecl/tests/structural_tests.rs \
  docs/design/gpu-backend-design.md docs/guides/devices-and-gpu.md
git commit -m "fix: complete explicit CUDA cast parity (#1362)"
```

### Task 12: Neighborhood scan, docs, and final verification

**Files:**
- Modify: `docs/worklogs/2026-07-11-cuda-bug-remediation.md`
- Modify: `docs/design/gpu-backend-design.md`
- Modify: `docs/guides/devices-and-gpu.md`

- [ ] **Step 1: Run the same-root-cause scan**

Search touched operation families for remaining scalar materialization, unsupported Bool movement, unchecked float-index conversion, shape `[1]` scalar workarounds, and incomplete cast branches. Add only same-contract fixes; record independent findings as follow-up issues.

- [ ] **Step 2: Run focused CUDA tests on hardware**

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-gpu --features cuda -- --ignored --test-threads=1
```

Expected: all ignored CUDA tests PASS. If the local CUDA root differs, record the actual path and command in the work log.

- [ ] **Step 3: Run the repository checklist**

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: every command exits zero and every modified source file meets coverage policy.

- [ ] **Step 4: Run the committed-head repository review**

Commit any final docs/work-log updates, then run:

```bash
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/cuda-bug-remediation-rules-review.json
```

Expected: pass with no unresolved findings.

- [ ] **Step 5: Prepare one final PR**

The PR body must list each issue and use `Closes #...` only for rows whose close condition is satisfied. Link the work log, include exact CUDA and workspace verification, and keep any narrowed/deferred row as `Refs #...` with residual risk.

- [ ] **Step 6: Merge without squash when authorized**

```bash
PR_NUMBER=$(gh pr view --json number --jq .number)
gh pr merge "$PR_NUMBER" --merge --delete-branch
```

Do not use squash merge; preserve the coherent remediation commits.
