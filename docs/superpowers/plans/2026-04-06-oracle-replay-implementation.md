# Oracle Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay the full `tensor-ad-oracles` database (171 ops, 9572 cases) as always-on integration tests in `tenferro`, validating forward evaluation, JVP, VJP, and HVP against PyTorch reference values.

**Architecture:** A single integration test binary `tenferro/tests/oracle_replay/` loads JSONL case files from `third_party/tensor-ad-oracles/cases/`, decodes row-major tensors into col-major tenferro `Tensor` values, dispatches each op to the corresponding tenferro API, and compares results using per-case tolerances. Unsupported ops and dtypes are tracked and reported in a summary.

**Tech Stack:** `serde`, `serde_json`, `tenferro` (TracedTensor, linalg free functions), integration tests

---

### Task 1: Add dev-dependencies

**Files:**
- Modify: `tenferro/Cargo.toml`
- Modify: `Cargo.toml` (workspace root, if needed)

- [ ] **Step 1: Add serde and serde_json to tenferro dev-dependencies**

In `tenferro/Cargo.toml`, add under `[dev-dependencies]`:

```toml
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p tenferro --tests`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tenferro/Cargo.toml
git commit -m "build: add serde dev-dependencies for oracle replay"
```

---

### Task 2: Decode module — JSONL parsing and tensor conversion

**Files:**
- Create: `tenferro/tests/oracle_replay/decode.rs`
- Create: `tenferro/tests/oracle_replay/main.rs` (minimal entrypoint)

- [ ] **Step 1: Create the test entrypoint stub**

Create `tenferro/tests/oracle_replay/main.rs`:

```rust
mod decode;
```

- [ ] **Step 2: Write decode module with serde structs and tensor conversion**

Create `tenferro/tests/oracle_replay/decode.rs` with:

1. Serde structs matching the oracle JSONL schema:
   - `CaseRecord` (top-level): `case_id`, `op`, `dtype`, `family`, `expected_behavior`, `comparison`, `inputs`, `observable`, `probes`, `op_kwargs`
   - `Comparison` with `first_order` and `second_order` tolerance fields
   - `Tolerance`: `kind`, `rtol`, `atol`
   - `TensorData`: `dtype`, `shape`, `order`, `data` (as `Vec<f64>`)
   - `Probe`: `probe_id`, `direction`, `cotangent`, `pytorch_ref`, `fd_ref`
   - `DerivativeRefs`: `jvp`, `vjp`, `hvp` (all `Option<HashMap<String, TensorData>>`)
   - `Observable`: `kind`

2. `decode_tensor(td: &TensorData) -> Option<Tensor>`:
   - Skip non-f64 dtypes (return `None`)
   - Convert row-major flat `data` to col-major `Tensor::F64(TypedTensor::from_vec(shape, col_major_data))`
   - Row-major to col-major conversion: for shape `[d0, d1, ..., dn]`, reindex `col[col_idx] = row[row_idx]` using standard stride arithmetic

3. `parse_case_record(line: &str) -> serde_json::Result<CaseRecord>`

- [ ] **Step 3: Write unit test for row-major to col-major conversion**

Add at the bottom of `decode.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_to_col_major_2x3() {
        // Row-major [2,3]: [[1,2,3],[4,5,6]]
        // data = [1,2,3,4,5,6]
        // Col-major: [1,4,2,5,3,6]
        let td = TensorData {
            dtype: "float64".to_string(),
            shape: vec![2, 3],
            order: "row_major".to_string(),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let tensor = decode_tensor(&td).unwrap();
        let data = match &tensor {
            tenferro::Tensor::F64(t) => t.host_data(),
            _ => panic!("expected f64"),
        };
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn skip_non_f64_dtype() {
        let td = TensorData {
            dtype: "float32".to_string(),
            shape: vec![2],
            order: "row_major".to_string(),
            data: vec![1.0, 2.0],
        };
        assert!(decode_tensor(&td).is_none());
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tenferro --test oracle_replay --release -- decode`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/tests/oracle_replay/
git commit -m "feat: oracle replay decode module with tensor conversion"
```

---

### Task 3: DB module — case file discovery and loading

**Files:**
- Create: `tenferro/tests/oracle_replay/db.rs`
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Write db module**

Create `tenferro/tests/oracle_replay/db.rs`:

1. `fn oracle_cases_dir() -> PathBuf`:
   - Locate `third_party/tensor-ad-oracles/cases/` relative to `CARGO_MANIFEST_DIR`

2. `fn discover_case_files(root: &Path) -> Vec<(String, PathBuf)>`:
   - Walk `root/*/` subdirectories
   - For each `*.jsonl` file, yield `(op_name, path)`
   - Sort by op name for deterministic ordering

3. `fn load_cases(path: &Path) -> Vec<CaseRecord>`:
   - Read file line-by-line
   - Parse each line with `parse_case_record`
   - Collect into Vec

- [ ] **Step 2: Add `mod db;` to main.rs**

```rust
mod db;
mod decode;
```

- [ ] **Step 3: Write discovery test**

In `db.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_oracle_cases() {
        let root = oracle_cases_dir();
        let files = discover_case_files(&root);
        // Should find at least qr, svd, sin, cos, exp, add, mul
        let ops: Vec<&str> = files.iter().map(|(op, _)| op.as_str()).collect();
        assert!(ops.contains(&"qr"), "missing qr");
        assert!(ops.contains(&"svd"), "missing svd");
        assert!(ops.contains(&"sin"), "missing sin");
        assert!(files.len() > 100, "expected 100+ case files, got {}", files.len());
    }

    #[test]
    fn loads_qr_cases() {
        let root = oracle_cases_dir();
        let qr_path = root.join("qr").join("identity.jsonl");
        let cases = load_cases(&qr_path);
        assert!(!cases.is_empty());
        assert_eq!(cases[0].op, "qr");
        assert_eq!(cases[0].dtype, "float64");
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tenferro --test oracle_replay --release -- db`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/tests/oracle_replay/
git commit -m "feat: oracle replay db module with case discovery"
```

---

### Task 4: Compare module — tolerance-aware tensor comparison

**Files:**
- Create: `tenferro/tests/oracle_replay/compare.rs`
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Write compare module**

Create `tenferro/tests/oracle_replay/compare.rs`:

1. `fn allclose(actual: &[f64], expected: &[f64], rtol: f64, atol: f64) -> Result<(), String>`:
   - For each pair `(a, e)`: check `|a - e| <= atol + rtol * |e|`
   - On failure, return error with index, actual, expected, and diff

2. `fn compare_tensor(actual: &Tensor, expected: &TensorData, tol: &Tolerance) -> Result<(), String>`:
   - Extract f64 data from actual tensor
   - Decode expected via `decode_tensor`
   - Call `allclose` with the tolerance
   - Include shape checks

- [ ] **Step 2: Add `mod compare;` to main.rs**

- [ ] **Step 3: Write unit tests for allclose**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allclose_exact_match() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(allclose(&a, &a, 1e-5, 1e-8).is_ok());
    }

    #[test]
    fn allclose_within_tolerance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7];
        assert!(allclose(&a, &b, 1e-5, 1e-6).is_ok());
    }

    #[test]
    fn allclose_fails_outside_tolerance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.1, 3.0];
        assert!(allclose(&a, &b, 1e-5, 1e-6).is_err());
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tenferro --test oracle_replay --release -- compare`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/tests/oracle_replay/
git commit -m "feat: oracle replay compare module with allclose"
```

---

### Task 5: Dispatch module — op name to tenferro API mapping

**Files:**
- Create: `tenferro/tests/oracle_replay/dispatch.rs`
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Write dispatch module**

Create `tenferro/tests/oracle_replay/dispatch.rs`:

1. `enum DispatchResult`:
   - `Supported(Vec<TracedTensor>)` — op outputs as traced tensors
   - `Skipped(SkipReason)` — op not supported

2. `enum SkipReason`:
   - `UnimplementedOp`
   - `UnsupportedDtype`
   - `UnsupportedKwargs`
   - `ExpectedError`

3. `fn dispatch_op(record: &CaseRecord) -> DispatchResult`:
   - Check `expected_behavior == "error"` → `Skipped(ExpectedError)`
   - Check `dtype != "float64"` → `Skipped(UnsupportedDtype)`
   - Decode input tensors via `decode_tensor`; skip if any fail
   - Match `record.op.as_str()` to tenferro API:

**Supported unary (TracedTensor methods):**
```rust
"abs" => vec![a.abs()],
"neg" => vec![a.neg()],
"exp" => vec![a.exp()],
"expm1" => vec![a.expm1()],
"log" => vec![a.log()],
"log1p" => vec![a.log1p()],
"sin" => vec![a.sin()],
"cos" => vec![a.cos()],
"tanh" => vec![a.tanh()],
"sqrt" => vec![a.sqrt()],
"rsqrt" => vec![a.rsqrt()],
"sign" | "sgn" => vec![a.sign()],
"conj" | "conj_physical" => vec![a.conj()],
```

**Supported binary (TracedTensor methods):**
```rust
"add" | "__radd__" => vec![a.add(&b)],
"mul" | "__rmul__" => vec![a.mul(&b)],
"sub" | "__rsub__" => vec![/* handle rsub: b - a */],
"div_no_rounding_mode" | "true_divide" | "__rdiv__" => vec![a.div(&b)],
"pow" | "__rpow__" | "float_power" => vec![a.pow(&b)],
```

**Supported reduction:**
```rust
"sum" => vec![a.reduce_sum(&all_axes)],
```

**Supported linalg (free functions):**
```rust
"svd" => { let (u, s, vt) = tenferro::svd(&a); vec![u, s, vt] },
"qr" => { let (q, r) = tenferro::qr(&a); vec![q, r] },
"eigh" => { let (vals, vecs) = tenferro::eigh(&a); vec![vals, vecs] },
"cholesky" => vec![tenferro::cholesky(&a)],
"solve" | "solve_ex" => vec![tenferro::solve(&a, &b)],
"solve_triangular" => {
    // Map op_kwargs to TriangularSolve params
    vec![tenferro::triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)]
},
```

**Everything else:** `Skipped(UnimplementedOp)`

4. `fn dispatch_jvp(record: &CaseRecord, outputs: &[TracedTensor], input_tt: &TracedTensor, direction: &Tensor) -> Vec<TracedTensor>`:
   - For each output, call `output.jvp(input_tt, &direction_tt)`
   - Return tangent outputs

5. `fn dispatch_vjp(record: &CaseRecord, scalar_output: &TracedTensor, input_tt: &TracedTensor) -> TracedTensor`:
   - Call `scalar_output.grad(input_tt)`

6. `fn dispatch_hvp(record: &CaseRecord, scalar_output: &TracedTensor, input_tt: &TracedTensor, direction: &Tensor) -> TracedTensor`:
   - `let g = scalar_output.grad(input_tt)?;`
   - `g.jvp(input_tt, &direction_tt)` (Forward-over-Reverse)

- [ ] **Step 2: Add `mod dispatch;` to main.rs**

- [ ] **Step 3: Write a smoke test for dispatch**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{load_cases, oracle_cases_dir};

    #[test]
    fn dispatch_sin_case() {
        let root = oracle_cases_dir();
        let cases = load_cases(&root.join("sin").join("identity.jsonl"));
        let f64_case = cases.iter().find(|c| c.dtype == "float64").unwrap();
        match dispatch_op(f64_case) {
            DispatchResult::Supported(outputs) => {
                assert_eq!(outputs.len(), 1);
            }
            other => panic!("expected Supported, got skip"),
        }
    }

    #[test]
    fn dispatch_unknown_op_is_skipped() {
        let root = oracle_cases_dir();
        let cases = load_cases(&root.join("det").join("identity.jsonl"));
        let case = &cases[0];
        match dispatch_op(case) {
            DispatchResult::Skipped(SkipReason::UnimplementedOp) => {}
            other => panic!("expected UnimplementedOp skip"),
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tenferro --test oracle_replay --release -- dispatch`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/tests/oracle_replay/
git commit -m "feat: oracle replay dispatch module with op mapping"
```

---

### Task 6: Observable module — spectral observable handling

**Files:**
- Create: `tenferro/tests/oracle_replay/observable.rs`
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Write observable module**

Create `tenferro/tests/oracle_replay/observable.rs`:

1. `fn apply_observable(kind: &str, outputs: &mut [TracedTensor]) -> Vec<TracedTensor>`:
   - `"identity"` → return outputs as-is
   - `"svd_s"` → return `[outputs[1]]` (singular values only)
   - `"svd_u_abs"` → return `[outputs[0].abs()]`
   - `"svd_vh_abs"` → return `[outputs[2].abs()]`
   - `"svd_uvh_product"` → reconstruct `einsum("ij,j,jk->ik", [U, S, Vt])`
   - `"eigh_values_vectors_abs"` → return `[outputs[0], outputs[1].abs()]`
   - Unknown → skip

2. `fn map_cotangent_to_outputs(kind: &str, cotangent: &HashMap<String, TensorData>) -> Vec<Option<Tensor>>`:
   - Map `"output_0"`, `"output_1"`, etc. to tensors in order
   - Decode each; None for missing outputs

- [ ] **Step 2: Add `mod observable;` to main.rs**

- [ ] **Step 3: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_observable_passes_through() {
        // Construct minimal TracedTensors and verify pass-through
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tenferro --test oracle_replay --release -- observable`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/tests/oracle_replay/
git commit -m "feat: oracle replay observable module for spectral ops"
```

---

### Task 7: Main replay test with summary reporting

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Write the replay driver and summary test**

Update `tenferro/tests/oracle_replay/main.rs`:

```rust
mod compare;
mod db;
mod decode;
mod dispatch;
mod observable;

use db::{discover_case_files, load_cases, oracle_cases_dir};
use dispatch::{DispatchResult, SkipReason};

struct ReplaySummary {
    passed: usize,
    failed: Vec<String>,          // case_id + error message
    skipped_dtype: usize,
    skipped_unimplemented: std::collections::BTreeSet<String>,
    skipped_kwargs: usize,
    expected_error: usize,
}

#[test]
fn oracle_replay_all() {
    let root = oracle_cases_dir();
    let files = discover_case_files(&root);
    let mut summary = ReplaySummary::new();

    for (op, path) in &files {
        let cases = load_cases(path);
        for record in &cases {
            replay_case(record, &mut summary);
        }
    }

    summary.print();
    assert!(
        summary.failed.is_empty(),
        "{} oracle cases failed:\n{}",
        summary.failed.len(),
        summary.failed.join("\n")
    );
}
```

The `replay_case` function:

1. Call `dispatch_op(record)` → handle `Skipped` variants by updating summary counters
2. For `Supported(outputs)`:
   - Evaluate outputs via `Engine::new(CpuBackend::new())`
   - Apply observable transformation
   - Compare forward results against expected (from oracle or via evaluation)
   - For each probe:
     - **JVP check**: dispatch JVP, eval, compare against `pytorch_ref.jvp`
     - **VJP check**: construct scalar observable from cotangent, dispatch VJP, compare against `pytorch_ref.vjp`
     - **HVP check** (if `pytorch_ref.hvp` present): construct scalar observable, dispatch HVP (FoR), compare against `pytorch_ref.hvp`
   - On comparison failure: push `"{case_id}: {error}"` into `summary.failed`
   - On success: increment `summary.passed`

- [ ] **Step 2: Run the full replay**

Run: `cargo test -p tenferro --test oracle_replay oracle_replay_all --release -- --nocapture`
Expected: PASS with summary showing passed, skipped, and 0 failures

- [ ] **Step 3: Commit**

```bash
git add tenferro/tests/oracle_replay/
git commit -m "feat: oracle replay main test with full database validation"
```

---

### Task 8: Verify CI integration

**Files:**
- No files to modify (oracle replay already included in `cargo nextest run --workspace --release`)

- [ ] **Step 1: Run the full pre-push checklist**

```bash
cargo fmt --all --check
cargo test --workspace --release
```

Expected: all PASS, including oracle replay

- [ ] **Step 2: Verify oracle replay appears in test output**

The `oracle_replay_all` test should appear in the workspace test run and show the summary.

- [ ] **Step 3: Commit any final adjustments**

If any formatting or minor fixes are needed, commit them:

```bash
cargo fmt --all
git add -u
git commit -m "chore: fmt and final oracle replay adjustments"
```
