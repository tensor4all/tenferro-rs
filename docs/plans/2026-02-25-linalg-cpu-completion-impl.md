# tenferro-linalg CPU Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete tenferro-linalg CPU path: implement eig/matrix_exp, add systematic FD-based AD verification for all operations, raise coverage to >=95%, align docs.

**Architecture:** Add `eig_general` backend method via faer's `Eigendecomposition`, implement `matrix_exp` via scaling-and-squaring Padé (PyTorch port). Build shared FD utilities, then systematically validate all 14 rrule + 14 frule functions. Reference PyTorch `torch.linalg` backward implementations exclusively — no custom math derivations.

**Tech Stack:** Rust, faer (CPU linear algebra), num-complex, chainrules-core AD traits, cargo test, cargo llvm-cov

---

### Task 1: Module cleanup — document CPU/GPU boundary (#217)

**Files:**
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/lib.rs` (crate-level doc)

**Step 1: Add module docs clarifying CPU-only status**

In `tenferro-linalg/src/backend/mod.rs`, update module doc:

```rust
//! Backend abstraction for linear algebra operations.
//!
//! Currently only the CPU backend ([`FaerBackend`]) is available.
//! GPU backends (cuSOLVER, hipSOLVER) are planned but not yet implemented.
//!
//! To add a GPU backend, implement [`LinalgBackend<T>`] for your backend type
//! and gate it behind a cargo feature (e.g., `#[cfg(feature = "cuda")]`).
```

In `tenferro-linalg/src/lib.rs` crate doc, replace any "API skeleton" or "POC" language with:

```rust
//! CPU decompositions and solvers are fully implemented via the
//! [`faer`](https://crates.io/crates/faer) backend. GPU backends are planned.
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-linalg`
Expected: PASS (no behavior change)

**Step 3: Commit**

```bash
git add tenferro-linalg/src/backend/mod.rs tenferro-linalg/src/lib.rs
git commit -m "docs(linalg): clarify CPU-only status and GPU extension points (#217)"
```

---

### Task 2: Add `eig_general` to LinalgBackend and implement in faer (#218 part 1)

**Files:**
- Modify: `tenferro-linalg/src/backend/mod.rs` (add trait method)
- Modify: `tenferro-linalg/src/backend/faer_backend.rs` (implement)

**Step 1: Add trait method to `LinalgBackend`**

Add to `LinalgBackend<T>` trait in `backend/mod.rs`:

```rust
    /// General (non-symmetric) eigendecomposition: `A V = V diag(lambda)`.
    ///
    /// Eigenvalues and eigenvectors are always complex-valued.
    /// Output slices hold interleaved real/imaginary pairs: `[re0, im0, re1, im1, ...]`.
    /// For real input `T`, each eigenvalue uses 2 floats.
    ///
    /// `values_ri`: length `2*n` (interleaved re/im pairs)
    /// `vectors_ri`: length `2*n*n` (interleaved re/im pairs, column-major)
    fn eig_general(
        &mut self,
        a: &[T],
        n: usize,
        values_ri: &mut [T],
        vectors_ri: &mut [T],
    ) -> Result<()>;
```

**Note on design**: We use interleaved real/imaginary output rather than complex types in the trait signature because `T` is the real scalar type (f64/f32). The caller (`eig()` in lib.rs) will convert to `Complex64`/`Complex32`. For complex input `T = Complex64`, the interleaving is identity (each Complex64 is already re+im).

**Step 2: Implement in faer backend for real types**

In the `impl_linalg_backend!` macro in `faer_backend.rs`, add:

```rust
fn eig_general(
    &mut self,
    a: &[$ty],
    n: usize,
    values_ri: &mut [$ty],
    vectors_ri: &mut [$ty],
) -> Result<()> {
    use faer::complex_native::c64;

    if a.len() < n * n {
        return Err(Error::InvalidArgument(format!(
            "eig_general: input slice length {} < n*n = {}", a.len(), n * n
        )));
    }
    if values_ri.len() < 2 * n {
        return Err(Error::InvalidArgument(format!(
            "eig_general: values_ri slice length {} < 2*n = {}", values_ri.len(), 2 * n
        )));
    }
    if vectors_ri.len() < 2 * n * n {
        return Err(Error::InvalidArgument(format!(
            "eig_general: vectors_ri slice length {} < 2*n*n = {}", vectors_ri.len(), 2 * n * n
        )));
    }

    // Convert real input to complex for faer eigendecomposition
    let a_complex: Vec<c64> = a.iter().map(|&v| c64::new(v as f64, 0.0)).collect();
    let mat = faer::mat::from_column_major_slice(&a_complex, n, n);
    let eig = mat.eigendecomposition::<c64>(faer::Side::Right);

    let s = eig.s();
    let u = eig.u();

    // Write eigenvalues as interleaved [re, im, re, im, ...]
    for i in 0..n {
        let val = s.read(i, i);
        values_ri[2 * i] = val.re as $ty;
        values_ri[2 * i + 1] = val.im as $ty;
    }

    // Write eigenvectors as interleaved column-major
    for j in 0..n {
        for i in 0..n {
            let val = u.read(i, j);
            vectors_ri[2 * (i + j * n)] = val.re as $ty;
            vectors_ri[2 * (i + j * n) + 1] = val.im as $ty;
        }
    }

    Ok(())
}
```

**Step 3: Implement in faer backend for complex types**

In the `impl_complex_linalg_backend!` macro, add the complex variant. For complex input, eigendecomposition output is already complex, so interleaving is straightforward.

**Step 4: Run tests**

Run: `cargo test -p tenferro-linalg`
Expected: PASS (new trait method has no callers yet)

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/mod.rs tenferro-linalg/src/backend/faer_backend.rs
git commit -m "feat(linalg): add eig_general to LinalgBackend trait (#218)"
```

---

### Task 3: Implement `eig()` forward and tests (#218 part 2)

**Files:**
- Modify: `tenferro-linalg/src/lib.rs` (replace eig stub)
- Modify: `tenferro-linalg/tests/linalg_tests.rs` (add tests)

**Step 1: Write failing tests**

```rust
#[test]
fn eig_2x2_real_eigenvalues() {
    // Diagonal matrix: eigenvalues are diagonal entries
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let result = eig(&mut backend, &a).unwrap();
    // eig returns Complex64 eigenvalues
    let vals = result.values.buffer().as_slice().unwrap();
    // Should contain 2.0 and 3.0 (as complex, sorted by real part)
    let mut reals: Vec<f64> = vals.iter().map(|c| c.re).collect();
    reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((reals[0] - 2.0).abs() < 1e-10);
    assert!((reals[1] - 3.0).abs() < 1e-10);
}

#[test]
fn eig_2x2_complex_eigenvalues() {
    // Rotation matrix: eigenvalues are complex conjugates
    let mut backend = FaerBackend::new();
    // [[0, -1], [1, 0]] has eigenvalues +i, -i
    let a = make_tensor(vec![0.0, 1.0, -1.0, 0.0], &[2, 2]);
    let result = eig(&mut backend, &a).unwrap();
    let vals = result.values.buffer().as_slice().unwrap();
    let mut imags: Vec<f64> = vals.iter().map(|c| c.im).collect();
    imags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((imags[0] - (-1.0)).abs() < 1e-10);
    assert!((imags[1] - 1.0).abs() < 1e-10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg eig_2x2`
Expected: FAIL (eig returns error)

**Step 3: Implement `eig()`**

Replace the stub in `lib.rs`. The function must:
1. Validate input is square (>=2D)
2. Call `backend.eig_general()` per batch
3. Convert interleaved output to `Tensor<Complex64>` / `Tensor<Complex32>`
4. Return `EigenResult` with complex eigenvalues and eigenvectors

The return type changes: `eig()` always returns complex types. Update the signature:

```rust
pub fn eig<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T::Real>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<EigenResult<num_complex::Complex<T::Real>, T::Real>>
```

**Important**: This changes the return type. The `EigenResult.values` will be `Tensor<T::Real>` (eigenvalues are real for symmetric; for general eig we need complex). Actually, per the design decision, eig() always returns complex. We need a new result type or adjust the existing one.

Define a new result type:

```rust
/// Result of general eigendecomposition (always complex-valued).
pub struct EigResult<R: Scalar> {
    /// Complex eigenvalues. Shape: `(n, *)`.
    pub values: Tensor<num_complex::Complex<R>>,
    /// Complex right eigenvectors (columns). Shape: `(n, n, *)`.
    pub vectors: Tensor<num_complex::Complex<R>>,
}
```

Update `eig()` to return `Result<EigResult<T::Real>>`.

**Step 4: Run tests**

Run: `cargo test -p tenferro-linalg eig_2x2`
Expected: PASS

**Step 5: Update existing error test**

Change `eig_returns_error` test to verify eig now succeeds (or remove it and rely on new tests).

**Step 6: Commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "feat(linalg): implement general eigendecomposition eig() (#218)"
```

---

### Task 4: Implement `eig_rrule()` and `eig_frule()` (#218 part 3)

**Files:**
- Modify: `tenferro-linalg/src/lib.rs` (replace AD stubs)
- Modify: `tenferro-linalg/tests/linalg_tests.rs` (add FD tests)

**Reference:** PyTorch `torch.linalg.eig` backward in
`torch/csrc/autograd/FunctionsLinearAlgebra.cpp` (`eig_backward`).

The PyTorch backward for `eig` follows Mike Giles' formula:
```
dA = V^{-H} (dLambda_diag + F * (V^H dV)) V^H
```
where `F[i,j] = 1/(lambda_j - lambda_i)` for `i != j`, `F[i,i] = 0`.

**Step 1: Implement `eig_rrule()`**

Replace the stub. Takes the primal eigendecomposition result and cotangents, returns gradient w.r.t. input matrix. Must handle complex arithmetic.

**Step 2: Implement `eig_frule()`**

Forward-mode: given tangent dA, compute d(lambda) and dV.

**Step 3: Add FD check tests**

```rust
#[test]
fn eig_rrule_fd_check() {
    // Finite-difference validation of eig_rrule
    // Use a well-conditioned real matrix with distinct eigenvalues
}

#[test]
fn eig_frule_fd_check() {
    // Finite-difference validation of eig_frule
}
```

**Step 4: Run tests**

Run: `cargo test -p tenferro-linalg eig_`
Expected: PASS

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "feat(linalg): implement eig AD rules following PyTorch (#218)"
```

---

### Task 5: Implement `matrix_exp()` forward (#218 part 4)

**Files:**
- Modify: `tenferro-linalg/src/lib.rs` (replace stub)
- Modify: `tenferro-linalg/tests/linalg_tests.rs` (add tests)

**Reference:** PyTorch `torch.matrix_exp` in `torch/csrc/autograd/FunctionsLinearAlgebra.cpp`
(`matrix_exp_forward`). Uses scaling-and-squaring with Padé approximation
(Al-Mohy & Higham, 2010).

**Step 1: Write failing tests**

```rust
#[test]
fn matrix_exp_identity() {
    // exp(0) = I
    let mut backend = FaerBackend::new();
    let zeros = make_tensor(vec![0.0; 9], &[3, 3]);
    let result = matrix_exp(&mut backend, &zeros).unwrap();
    let data = tensor_data(&result);
    // Should be identity matrix
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((data[i + j * 3] - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn matrix_exp_diagonal() {
    // exp(diag(a,b)) = diag(exp(a), exp(b))
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let result = matrix_exp(&mut backend, &a).unwrap();
    let data = tensor_data(&result);
    assert!((data[0] - 1.0_f64.exp()).abs() < 1e-10);
    assert!((data[3] - 2.0_f64.exp()).abs() < 1e-10);
    assert!(data[1].abs() < 1e-10);
    assert!(data[2].abs() < 1e-10);
}
```

**Step 2: Implement `matrix_exp()`**

Port PyTorch's scaling-and-squaring implementation. The algorithm:
1. Compute `norm_1 = ||A||_1`
2. Choose scaling factor `s` and Padé order based on `norm_1`
3. Scale: `A_scaled = A / 2^s`
4. Compute Padé approximant `P(A_scaled)/Q(A_scaled)`
5. Square `s` times: `result = result^{2^s}`

The implementation uses `backend.mat_mul()` for matrix products and `backend.solve()` for the Padé rational approximation.

**Step 3: Run tests**

Run: `cargo test -p tenferro-linalg matrix_exp_`
Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "feat(linalg): implement matrix_exp via scaling-and-squaring (#218)"
```

---

### Task 6: Implement `matrix_exp_rrule()` and `matrix_exp_frule()` (#218 part 5)

**Files:**
- Modify: `tenferro-linalg/src/lib.rs` (replace AD stubs)
- Modify: `tenferro-linalg/tests/linalg_tests.rs` (add FD tests)

**Reference:** PyTorch `matrix_exp_backward` in `FunctionsLinearAlgebra.cpp`.
Uses the auxiliary matrix trick: to compute `d(exp(A))`, form the 2n×2n matrix
`[[A, dA], [0, A]]`, compute its exponential, and extract the upper-right block.

**Step 1: Implement `matrix_exp_rrule()`**

The reverse-mode rule for matrix exponential:
```
dA = matrix_exp_backward(A, cotangent)
```
Following PyTorch: build auxiliary matrix, call `matrix_exp()` on it, extract block.

**Step 2: Implement `matrix_exp_frule()`**

Forward-mode: same auxiliary matrix approach but tangent goes in the upper-right block.

**Step 3: Add FD check tests**

**Step 4: Run tests, commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "feat(linalg): implement matrix_exp AD rules following PyTorch (#218)"
```

---

### Task 7: Expand JSON test case database (#222)

**Files:**
- Modify: `tenferro-linalg/tests/data/linalg_cases.json`
- Modify: `tenferro-linalg/tests/linalg_tests.rs` (add JSON loader)

**Step 1: Define JSON schema and expand cases**

Update `linalg_cases.json` with actual matrix data (not just shapes). Each entry:

```json
{
  "operation": "svd",
  "dtype": "f64",
  "input": {"shape": [3, 3], "data": [1.0, 2.0, ...]},
  "expected": {
    "u_shape": [3, 3],
    "s": [5.477, 1.0, 0.0],
    "reconstruction_tol": 1e-10
  }
}
```

Add cases for: svd, qr, lu, cholesky, eigen, eig, solve, lstsq, inv, det, slogdet, pinv, matrix_exp, norm.

Include edge cases: near-singular, ill-conditioned, triangular, diagonal.

**Step 2: Add JSON loading utility in test file**

```rust
fn load_test_cases() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/linalg_cases.json");
    let data = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&data).unwrap()
}
```

**Step 3: Add JSON-driven forward tests**

One test function per operation that iterates over JSON cases.

**Step 4: Run tests, commit**

```bash
git add tenferro-linalg/tests/data/linalg_cases.json tenferro-linalg/tests/linalg_tests.rs
git commit -m "test(linalg): expand JSON test case database (#222)"
```

---

### Task 8: Build shared FD verification utilities (#219 part 1)

**Files:**
- Modify: `tenferro-linalg/tests/linalg_tests.rs` (add FD helpers)

**Step 1: Implement FD Jacobian helper**

```rust
/// Compute numerical Jacobian column for a matrix→matrix function.
/// Perturbs input at position (i,j) by ±eps, computes (f(A+eps) - f(A-eps)) / (2*eps).
fn fd_gradient_matrix(
    f: impl Fn(&Tensor<f64>) -> Tensor<f64>,
    a: &Tensor<f64>,
    eps: f64,
) -> Vec<Tensor<f64>> {
    let n = a.dims().iter().product::<usize>();
    let a_data = tensor_data(a);
    let mut grads = Vec::with_capacity(n);
    for idx in 0..n {
        let mut plus = a_data.clone();
        let mut minus = a_data.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let f_plus = f(&make_tensor(plus, a.dims()));
        let f_minus = f(&make_tensor(minus, a.dims()));
        let fp = tensor_data(&f_plus);
        let fm = tensor_data(&f_minus);
        let grad: Vec<f64> = fp.iter().zip(&fm).map(|(p, m)| (p - m) / (2.0 * eps)).collect();
        grads.push(make_tensor(grad, f_plus.dims()));
    }
    grads
}

/// Check that analytic rrule gradient matches FD gradient.
fn check_rrule_fd<F, G>(
    forward: F,
    rrule: G,
    a: &Tensor<f64>,
    eps: f64,
    atol: f64,
) where
    F: Fn(&Tensor<f64>) -> Tensor<f64>,
    G: Fn(&Tensor<f64>, &Tensor<f64>) -> Tensor<f64>,
{
    let output = forward(a);
    let out_size: usize = output.dims().iter().product();
    // Use random cotangent (fixed seed)
    let cotangent_data: Vec<f64> = (0..out_size).map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0).collect();
    let cotangent = make_tensor(cotangent_data.clone(), output.dims());

    let analytic_grad = rrule(a, &cotangent);
    let analytic = tensor_data(&analytic_grad);

    // FD: sum_k cotangent[k] * df_k/da[ij]
    let fd_jac = fd_gradient_matrix(&forward, a, eps);
    let a_size: usize = a.dims().iter().product();
    let mut fd_grad = vec![0.0; a_size];
    for ij in 0..a_size {
        let jac_col = tensor_data(&fd_jac[ij]);
        for k in 0..out_size {
            fd_grad[ij] += cotangent_data[k] * jac_col[k];
        }
    }

    let max_err = analytic.iter().zip(&fd_grad)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < atol, "rrule FD check failed: max_err={max_err} > atol={atol}");
}
```

**Step 2: Implement FD JVP helper for frule checks**

```rust
/// Check that analytic frule tangent matches FD directional derivative.
fn check_frule_fd<F, G>(
    forward: F,
    frule: G,
    a: &Tensor<f64>,
    eps: f64,
    atol: f64,
) where
    F: Fn(&Tensor<f64>) -> Tensor<f64>,
    G: Fn(&Tensor<f64>, &Tensor<f64>) -> Tensor<f64>,
{
    let a_size: usize = a.dims().iter().product();
    // Random tangent direction (fixed seed)
    let tangent_data: Vec<f64> = (0..a_size).map(|i| ((i * 13 + 5) % 17) as f64 / 8.0 - 1.0).collect();
    let tangent = make_tensor(tangent_data.clone(), a.dims());

    let analytic_out = frule(a, &tangent);
    let analytic = tensor_data(&analytic_out);

    // FD: (f(A + eps*dA) - f(A - eps*dA)) / (2*eps)
    let a_data = tensor_data(a);
    let plus: Vec<f64> = a_data.iter().zip(&tangent_data).map(|(a, da)| a + eps * da).collect();
    let minus: Vec<f64> = a_data.iter().zip(&tangent_data).map(|(a, da)| a - eps * da).collect();
    let f_plus = forward(&make_tensor(plus, a.dims()));
    let f_minus = forward(&make_tensor(minus, a.dims()));
    let fp = tensor_data(&f_plus);
    let fm = tensor_data(&f_minus);
    let fd: Vec<f64> = fp.iter().zip(&fm).map(|(p, m)| (p - m) / (2.0 * eps)).collect();

    let max_err = analytic.iter().zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < atol, "frule FD check failed: max_err={max_err} > atol={atol}");
}
```

**Step 3: Run tests, commit**

```bash
git add tenferro-linalg/tests/linalg_tests.rs
git commit -m "test(linalg): add shared FD verification utilities (#219)"
```

---

### Task 9: Add FD checks for all rrule operations (#219 part 2)

**Files:**
- Modify: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Add rrule FD tests**

One test per operation using `check_rrule_fd`. Use well-conditioned test matrices.

Operations to cover:
- `svd_rrule` — cotangent through S only (simplest), then through U/Vt
- `qr_rrule`
- `lu_rrule`
- `eigen_rrule` (symmetric)
- `eig_rrule` (general)
- `cholesky_rrule`
- `solve_rrule` (grad w.r.t. both A and B)
- `inv_rrule`
- `det_rrule`
- `slogdet_rrule`
- `lstsq_rrule`
- `pinv_rrule`
- `matrix_exp_rrule`
- `norm_rrule` (Frobenius)

For each test:
1. Create a well-conditioned test matrix (avoid near-singular)
2. Run `check_rrule_fd` with `eps=1e-6`, `atol=1e-4`
3. For operations with structured output (SVD → U,S,Vt), test each cotangent component

**Step 2: Run tests**

Run: `cargo test -p tenferro-linalg fd_`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-linalg/tests/linalg_tests.rs
git commit -m "test(linalg): add FD checks for all rrule operations (#219)"
```

---

### Task 10: Add FD checks for all frule operations (#219 part 3)

**Files:**
- Modify: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Add frule FD tests**

Same pattern as Task 9 but using `check_frule_fd`:
- `svd_frule`, `qr_frule`, `lu_frule`, `eigen_frule`, `eig_frule`
- `cholesky_frule`, `solve_frule`, `inv_frule`
- `det_frule`, `slogdet_frule`, `lstsq_frule`
- `pinv_frule`, `matrix_exp_frule`, `norm_frule`

**Step 2: Run tests, commit**

```bash
git add tenferro-linalg/tests/linalg_tests.rs
git commit -m "test(linalg): add FD checks for all frule operations (#219)"
```

---

### Task 11: Coverage gap analysis and fill (#220)

**Files:**
- Modify: `tenferro-linalg/tests/linalg_tests.rs` (add targeted tests)
- Modify: `coverage-thresholds.json` (raise thresholds)

**Step 1: Run coverage and analyze gaps**

```bash
cargo llvm-cov -p tenferro-linalg --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json --report-only
```

Identify uncovered lines in `lib.rs` and `faer_backend.rs`.

**Step 2: Add tests targeting uncovered paths**

Likely gaps:
- Error paths (invalid shapes, non-square inputs, etc.)
- Batch dimension handling
- Complex scalar types (Complex64, Complex32)
- f32 variants
- Edge cases: 1×1 matrices, empty batch dims
- `solve_triangular` with upper=true and upper=false
- `norm` with different `NormKind` variants
- `lstsq` happy path (currently only error test exists)

**Step 3: Raise thresholds**

Update `coverage-thresholds.json`:
```json
{
    "tenferro-linalg/src/lib.rs": 95,
    "tenferro-linalg/src/backend/faer_backend.rs": 95
}
```

**Step 4: Verify coverage passes**

```bash
cargo llvm-cov -p tenferro-linalg --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/linalg_tests.rs coverage-thresholds.json
git commit -m "test(linalg): raise coverage to >=95% (#220)"
```

---

### Task 12: Align documentation (#221)

**Files:**
- Modify: `docs/design/linalg.md`
- Modify: `docs/design/testing.md` (if exists)

**Step 1: Update linalg design doc**

- Remove any "API skeleton only" or "POC phase" language
- Document complete list of implemented operations with status
- Document CPU/GPU boundary (CPU: complete, GPU: planned)
- Add section on AD testing strategy (FD verification for all operations)
- Reference PyTorch as source of AD formulas

**Step 2: Update testing doc**

- Document FD verification approach
- Document tolerance policy (eps=1e-6, atol=1e-4)
- Document JSON test case database

**Step 3: Run full workspace tests**

```bash
cargo test --workspace
cargo fmt --all --check
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

**Step 4: Commit**

```bash
git add docs/design/linalg.md docs/design/testing.md
git commit -m "docs(linalg): align docs with CPU completion (#221)"
```

---

## Execution Notes

- **PyTorch reference files** for AD implementations:
  - `torch/csrc/autograd/FunctionsLinearAlgebra.cpp`
  - `torch/_decomp/decompositions.py` (for higher-level decomps)
  - `torch/linalg/__init__.py` (for API conventions)
- **faer eigendecomposition**: Use `mat.eigendecomposition::<c64>(faer::Side::Right)` for general eigendecomposition. This computes right eigenvectors.
- **Tolerance policy**: FD step `eps=1e-6`, comparison `atol=1e-4` for well-conditioned cases. For ill-conditioned cases (near-singular, clustered eigenvalues), use `atol=1e-2` or skip.
- **Determinism**: All test matrices use fixed data (no random generation) or fixed seeds.
