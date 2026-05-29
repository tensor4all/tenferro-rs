# Tropical Einsum Shared Planning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore tropical einsum support for issue #212 while sharing ordinary einsum's semiring-neutral planning and adding an optimized value-plus-argmax CPU path inspired by `omeinsum-rs`.

**Architecture:** `tenferro-einsum` exposes read-only lowering plans derived from its existing `ContractionTree`; `tenferro-ext-tropical` consumes those plans and owns tropical semantics, argmax capture, and AD. Ordinary einsum remains `sum(mul(...))`; tropical einsum reuses only parsing, contraction order, shape planning, and GEMM lowering.

**Tech Stack:** Rust workspace crates, `tenferro-einsum`, `tenferro-runtime`, `tenferro-tensor`, `tenferro-cpu`, extension runtime APIs, optional `tropical-gemm`, Criterion benchmarks, TDD.

---

### Task 1: Expose Read-Only Einsum Lowering Plans

**Files:**
- Create: `tenferro-einsum/src/lowering.rs`
- Modify: `tenferro-einsum/src/lib.rs`
- Modify: `tenferro-einsum/src/planning/tree.rs`
- Test: `tenferro-einsum/src/planning/tree/tests.rs`

**Step 1: Write the failing test**

Add a test that uses only public-facing `ContractionTree` methods to inspect the GEMM lowering for `"ij,jk->ik"` and an output-permuted case.

```rust
#[test]
fn public_lowering_step_plan_exposes_gemm_layout() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();

    let step = tree.step_plan(0).expect("one pairwise step");
    let gemm = step.gemm();

    assert_eq!(gemm.left_only_modes(), &[0]);
    assert_eq!(gemm.right_only_modes(), &[2]);
    assert_eq!(gemm.contracted_modes(), &[1]);
    assert_eq!(gemm.batch_modes(), &[]);
    assert_eq!(gemm.m(), 2);
    assert_eq!(gemm.k(), 3);
    assert_eq!(gemm.n(), 4);
    assert_eq!(gemm.lhs_gemm_shape(), &[2, 3]);
    assert_eq!(gemm.rhs_gemm_shape(), &[3, 4]);
    assert_eq!(gemm.output_gemm_shape(), &[2, 4]);
    assert!(!gemm.needs_final_permute());
}

#[test]
fn public_lowering_step_plan_reports_final_permutation() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();

    let gemm = tree.step_plan(0).unwrap().gemm();

    assert_eq!(gemm.canonical_output_modes(), &[0, 2]);
    assert!(gemm.needs_final_permute());
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-einsum public_lowering_step_plan --lib
```

Expected: compile failure because `ContractionTree::step_plan` and the lowering wrapper types do not exist.

**Step 3: Add `tenferro_einsum::lowering` wrappers**

Create `tenferro-einsum/src/lowering.rs` with private-inner public wrappers around the existing internal plan structs.

```rust
use crate::planning::plan::{
    DiagPlan as InnerDiagPlan, DiagStage as InnerDiagStage, GemmPlan as InnerGemmPlan,
    ReducePlan as InnerReducePlan, StepPlan as InnerStepPlan,
};

/// Read-only lowering data for one pairwise einsum contraction step.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
/// let step = tree.step_plan(0).unwrap();
///
/// assert_eq!(step.gemm().contracted_modes(), &[1]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PairwiseStepPlan<'a> {
    pub(crate) inner: &'a InnerStepPlan,
}

impl<'a> PairwiseStepPlan<'a> {
    pub(crate) fn new(inner: &'a InnerStepPlan) -> Self {
        Self { inner }
    }

    /// Return the left operand diagonal extraction plan, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    ///
    /// assert!(tree.step_plan(0).unwrap().lhs_diag().is_none());
    /// ```
    pub fn lhs_diag(&self) -> Option<DiagPlan<'a>> {
        self.inner.diag_a.as_ref().map(DiagPlan::new)
    }

    pub fn rhs_diag(&self) -> Option<DiagPlan<'a>> {
        self.inner.diag_b.as_ref().map(DiagPlan::new)
    }

    pub fn lhs_reduce(&self) -> Option<ReducePlan<'a>> {
        self.inner.gemm.reduce_a.as_ref().map(ReducePlan::new)
    }

    pub fn rhs_reduce(&self) -> Option<ReducePlan<'a>> {
        self.inner.gemm.reduce_b.as_ref().map(ReducePlan::new)
    }

    pub fn gemm(&self) -> GemmPlan<'a> {
        GemmPlan::new(&self.inner.gemm)
    }
}
```

Then add wrappers for `DiagPlan`, `DiagStage`, `ReducePlan`, and `GemmPlan`. Each public type and public method must have a compact `# Examples` doctest. Accessors should return slices or scalars:

```rust
pub fn left_only_modes(&self) -> &'a [u32];
pub fn right_only_modes(&self) -> &'a [u32];
pub fn contracted_modes(&self) -> &'a [u32];
pub fn batch_modes(&self) -> &'a [u32];
pub fn lhs_target_modes(&self) -> &'a [u32];
pub fn rhs_target_modes(&self) -> &'a [u32];
pub fn canonical_output_modes(&self) -> &'a [u32];
pub fn m(&self) -> usize;
pub fn n(&self) -> usize;
pub fn k(&self) -> usize;
pub fn lhs_gemm_shape(&self) -> &'a [usize];
pub fn rhs_gemm_shape(&self) -> &'a [usize];
pub fn output_gemm_shape(&self) -> &'a [usize];
pub fn expanded_output_shape(&self) -> &'a [usize];
pub fn needs_final_permute(&self) -> bool;
```

Expose the module in `tenferro-einsum/src/lib.rs`:

```rust
pub mod lowering;
```

Add accessors to `ContractionTree` in `planning/tree.rs`:

```rust
/// Return the precomputed lowering plan for one pairwise contraction step.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
///
/// assert_eq!(tree.step_plan(0).unwrap().gemm().m(), 2);
/// ```
#[must_use]
pub fn step_plan(&self, step_idx: usize) -> Option<crate::lowering::PairwiseStepPlan<'_>> {
    self.step_plans
        .get(step_idx)
        .map(crate::lowering::PairwiseStepPlan::new)
}
```

**Step 4: Run tests to verify green**

Run:

```bash
cargo test -p tenferro-einsum public_lowering_step_plan --lib
cargo test -p tenferro-einsum --doc
cargo test -p tenferro-einsum --lib
```

Expected: all pass.

**Step 5: Commit**

```bash
git add tenferro-einsum/src/lowering.rs tenferro-einsum/src/lib.rs tenferro-einsum/src/planning/tree.rs tenferro-einsum/src/planning/tree/tests.rs
git commit -m "feat(einsum): expose shared lowering plans"
```

### Task 2: Restore `tenferro-ext-tropical` Skeleton Against Current Crates

**Files:**
- Create: `ext/tropical/Cargo.toml`
- Create: `ext/tropical/src/lib.rs`
- Create: `ext/tropical/src/newtype.rs`
- Create: `ext/tropical/src/traced.rs`
- Create: `ext/tropical/tests/tropical_ad.rs`
- Test: `ext/tropical/tests/smoke.rs`

**Step 1: Write the failing smoke test**

Create `ext/tropical/tests/smoke.rs`:

```rust
use tenferro_ext_tropical::{MaxPlus, MinPlus, TropicalKind};

#[test]
fn tropical_crate_exports_core_types() {
    assert_eq!(TropicalKind::MaxPlus, TropicalKind::MaxPlus);
    assert_eq!(MaxPlus(2.0_f64).value(), 2.0);
    assert_eq!(MinPlus(3.0_f64).value(), 3.0);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test smoke
```

Expected: fail because the crate does not exist or exported types are missing.

**Step 3: Restore and modernize the crate skeleton**

Restore the previous crate as source material, then update imports and docs for the no-facade crate split:

```bash
git restore --source=0e3ffa76^ -- ext/tropical
```

Update `ext/tropical/Cargo.toml` to depend on current crates:

```toml
[dependencies]
tenferro-cpu = { path = "../../tenferro-cpu" }
tenferro-einsum = { path = "../../tenferro-einsum" }
tenferro-internal-extension-macros = { path = "../../tenferro-internal-extension-macros" }
tenferro-internal-ops = { path = "../../tenferro-internal-ops" }
tenferro-runtime = { path = "../../tenferro-runtime" }
tenferro-tensor = { path = "../../tenferro-tensor" }
chainrules-core = { git = "https://github.com/tensor4all/chainrules-rs.git", rev = "2d7662140d92f0dd73b2402aefc2272feafc9270", package = "chainrules" }
computegraph = { git = "https://github.com/tensor4all/computegraph-rs.git", rev = "c20e8912560ef11166418bcea089126ef50443cc" }
num-traits = "0.2"

[dev-dependencies]
criterion = "0.5"
```

Keep `[workspace]` in `ext/tropical/Cargo.toml` so the extension remains a standalone nested workspace.

Update examples from old `tenferro::{...}` imports to explicit current crates such as:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::TracedTensor;
```

**Step 4: Run tests to verify green**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test smoke
cargo test --manifest-path ext/tropical/Cargo.toml --lib
```

Expected: smoke test passes; lib tests compile or reveal current API drift to fix in the restored skeleton.

**Step 5: Commit**

```bash
git add ext/tropical
git commit -m "feat(tropical): restore extension crate skeleton"
```

### Task 3: Add Generic Tropical GEMM With Argmax

**Files:**
- Create: `ext/tropical/src/cpu.rs`
- Modify: `ext/tropical/src/lib.rs`
- Test: `ext/tropical/tests/tropical_argmax.rs`

**Step 1: Write failing value-plus-argmax tests**

Create `ext/tropical/tests/tropical_argmax.rs`:

```rust
use tenferro_ext_tropical::cpu::{tropical_gemm_with_argmax, TropicalGemmKind};

#[test]
fn maxplus_gemm_returns_values_and_first_winner_indices() {
    let a = vec![10.0, 0.0, 1.0, 5.0]; // shape [2, 2], column-major
    let b = vec![1.0, 10.0, 0.0, 1.0]; // shape [2, 2], column-major

    let out = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 2, 2, &b, 2).unwrap();

    assert_eq!(out.values, vec![11.0, 15.0, 10.0, 6.0]);
    assert_eq!(out.argmax, vec![0, 1, 0, 1]);
}

#[test]
fn minplus_gemm_returns_values_and_first_winner_indices() {
    let a = vec![1.0, 4.0, 3.0, 2.0];
    let b = vec![5.0, 6.0, 7.0, 1.0];

    let out = tropical_gemm_with_argmax(TropicalGemmKind::MinPlus, &a, 2, 2, &b, 2).unwrap();

    assert_eq!(out.values, vec![6.0, 8.0, 4.0, 3.0]);
    assert_eq!(out.argmax, vec![0, 0, 1, 1]);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test tropical_argmax
```

Expected: compile failure because `cpu::tropical_gemm_with_argmax` does not exist.

**Step 3: Implement generic CPU fallback**

Implement a column-major fallback in `ext/tropical/src/cpu.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TropicalGemmKind {
    MaxPlus,
    MinPlus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TropicalGemmArgmax<T> {
    pub values: Vec<T>,
    pub argmax: Vec<u32>,
}

pub fn tropical_gemm_with_argmax<T>(
    kind: TropicalGemmKind,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> tenferro_tensor::Result<TropicalGemmArgmax<T>>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T>,
{
    if a.len() != m * k || b.len() != k * n {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "tropical_gemm_with_argmax",
            message: "input length does not match matrix dimensions".into(),
        });
    }
    if k == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "tropical_gemm_with_argmax",
            message: "contracted dimension must be nonzero".into(),
        });
    }

    let mut values = Vec::with_capacity(m * n);
    values.resize_with(m * n, || a[0] + b[0]);
    let mut argmax = vec![0_u32; m * n];

    for j in 0..n {
        for i in 0..m {
            let mut best = a[i] + b[j * k];
            let mut best_k = 0_u32;
            for kk in 1..k {
                let candidate = a[i + kk * m] + b[kk + j * k];
                let better = match kind {
                    TropicalGemmKind::MaxPlus => candidate > best,
                    TropicalGemmKind::MinPlus => candidate < best,
                };
                if better {
                    best = candidate;
                    best_k = kk as u32;
                }
            }
            values[i + j * m] = best;
            argmax[i + j * m] = best_k;
        }
    }

    Ok(TropicalGemmArgmax { values, argmax })
}
```

Document that ties keep the first winning `k`.

**Step 4: Run tests to verify green**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test tropical_argmax
cargo test --manifest-path ext/tropical/Cargo.toml --lib
```

Expected: all pass.

**Step 5: Commit**

```bash
git add ext/tropical/src/cpu.rs ext/tropical/src/lib.rs ext/tropical/tests/tropical_argmax.rs
git commit -m "feat(tropical): add value-plus-argmax GEMM fallback"
```

### Task 4: Execute Tropical Pairwise Contractions Through Shared Lowering

**Files:**
- Create: `ext/tropical/src/einsum.rs`
- Modify: `ext/tropical/src/lib.rs`
- Test: `ext/tropical/tests/tropical_einsum.rs`

**Step 1: Write failing einsum tests**

Create `ext/tropical/tests/tropical_einsum.rs`:

```rust
use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
use tenferro_tensor::{Tensor, TypedTensor};

#[test]
fn maxplus_matmul_uses_shared_einsum_lowering() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![10.0, 0.0, 1.0, 5.0]));
    let b = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 10.0, 0.0, 1.0]));

    let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")
        .unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(result.output.as_f64_slice().unwrap(), &[11.0, 15.0, 10.0, 6.0]);
    assert_eq!(result.argmax[0].indices(), &[0, 1, 0, 1]);
}

#[test]
fn output_permutation_matches_subscripts() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![10.0, 0.0, 1.0, 5.0]));
    let b = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 10.0, 0.0, 1.0]));

    let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ki")
        .unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(result.output.as_f64_slice().unwrap(), &[11.0, 10.0, 15.0, 6.0]);
}
```

Adjust the exact tensor slice helper names to current `tenferro_tensor` APIs while implementing.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test tropical_einsum
```

Expected: compile failure because `einsum::tropical_einsum_with_argmax` does not exist.

**Step 3: Implement the first lowering-backed executor**

Implement `ext/tropical/src/einsum.rs`:

- parse notation with `tenferro_einsum::Subscripts::parse`;
- build `ContractionTree::optimize`;
- for the first implementation, support `F32` and `F64` dense host tensors;
- loop through `tree.step_count()`;
- use `tree.step_pair(step_idx)`, `tree.step_subscripts(step_idx)`, and `tree.step_plan(step_idx)`;
- apply the GEMM plan for simple non-diagonal binary cases first;
- call `cpu::tropical_gemm_with_argmax`;
- reshape and transpose to match `gemm.needs_final_permute()`;
- keep per-step argmax trackers in the result.

Return explicit unsupported errors for diagonal/pre-reduction/batched cases until their tests are added.

**Step 4: Run tests to verify green**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test tropical_einsum
```

Expected: simple matmul and output permutation pass.

**Step 5: Commit**

```bash
git add ext/tropical/src/einsum.rs ext/tropical/src/lib.rs ext/tropical/tests/tropical_einsum.rs
git commit -m "feat(tropical): execute einsum through shared lowering"
```

### Task 5: Cover Batched, Fallback, and Unsupported Cases

**Files:**
- Modify: `ext/tropical/src/einsum.rs`
- Modify: `ext/tropical/src/cpu.rs`
- Test: `ext/tropical/tests/tropical_einsum.rs`

**Step 1: Write failing tests**

Add tests for:

- batched `"bij,bjk->bik"`;
- generic fallback comparison for a shape not routed to the optimized kernel;
- explicit unsupported errors for repeated labels if not yet implemented;
- first-winner tie behavior.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test tropical_einsum
```

Expected: failures identify missing batched or fallback support.

**Step 3: Implement missing executor paths**

Add batched looping over trailing batch dimensions from the shared GEMM plan.
Keep each batch slice column-major and call the same value-plus-argmax kernel.

For unsupported optimized cases, route to a generic index-loop fallback that
uses the same subscript labels and records a flat contracted-mode winner index.

**Step 4: Run tests to verify green**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test tropical_einsum
cargo test --manifest-path ext/tropical/Cargo.toml
```

Expected: all tropical crate tests pass.

**Step 5: Commit**

```bash
git add ext/tropical/src/einsum.rs ext/tropical/src/cpu.rs ext/tropical/tests/tropical_einsum.rs
git commit -m "test(tropical): cover batched and fallback einsum paths"
```

### Task 6: Add Optional `tropical-gemm` Fast Path

**Files:**
- Modify: `ext/tropical/Cargo.toml`
- Modify: `ext/tropical/src/cpu.rs`
- Test: `ext/tropical/tests/tropical_argmax.rs`

**Step 1: Write failing dispatch test**

Add a test that compares the fast-path result to the generic fallback behind a feature gate:

```rust
#[cfg(feature = "tropical-gemm")]
#[test]
fn tropical_gemm_feature_matches_generic_fallback() {
    let a = vec![10.0_f64, 0.0, 1.0, 5.0];
    let b = vec![1.0_f64, 10.0, 0.0, 1.0];

    let fast = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 2, 2, &b, 2).unwrap();
    let generic = tropical_gemm_with_argmax_generic(TropicalGemmKind::MaxPlus, &a, 2, 2, &b, 2)
        .unwrap();

    assert_eq!(fast, generic);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --features tropical-gemm --test tropical_argmax
```

Expected: compile failure until the optional dependency and dispatch wrapper exist.

**Step 3: Implement optional dispatch**

Add the optional dependency in `ext/tropical/Cargo.toml`:

```toml
[features]
default = []
tropical-gemm = ["dep:tropical-gemm"]

[dependencies]
tropical-gemm = { version = "0.2", optional = true }
```

Implement `try_tropical_gemm_with_argmax_fast` in `cpu.rs`, following the
`omeinsum-rs/src/backend/cpu/mod.rs` structure but without public `TypeId`
dispatch. Prefer explicit `f32`/`f64` wrapper functions or sealed traits so the
unsafe cast boundary stays private and minimal.

Document the fast-path eligibility conditions in the module docs.

**Step 4: Run tests to verify green**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --features tropical-gemm --test tropical_argmax
cargo test --manifest-path ext/tropical/Cargo.toml --features tropical-gemm
```

Expected: all pass.

**Step 5: Commit**

```bash
git add ext/tropical/Cargo.toml ext/tropical/src/cpu.rs ext/tropical/tests/tropical_argmax.rs
git commit -m "perf(tropical): add optional tropical-gemm argmax path"
```

### Task 7: Wire Tropical AD To Argmax-Capable Forward

**Files:**
- Modify: `ext/tropical/src/fused.rs`
- Modify: `ext/tropical/src/einsum.rs`
- Test: `ext/tropical/tests/fused_tropical_ad.rs`
- Test: `ext/tropical/tests/tropical_symbolic_ad.rs`

**Step 1: Write failing AD routing tests**

Add or restore tests covering:

- unique-winner max-plus matmul cotangent routing;
- mixed winners;
- first-winner tie behavior;
- fallback to indicator-mask AD where argmax scatter cannot be represented.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test fused_tropical_ad
```

Expected: failures until fused AD consumes the new argmax-capable forward path.

**Step 3: Implement AD integration**

Use the restored `fused.rs` indicator-mask implementation as the correctness
fallback. Add the argmax-capable path where runtime execution has concrete
inputs and can preserve the winner indices. Keep AD rule registration
idempotent.

If the graph vocabulary cannot represent scatter efficiently for a case, do
not fake it. Use the indicator-mask path and document the fallback.

**Step 4: Run tests to verify green**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --test fused_tropical_ad
cargo test --manifest-path ext/tropical/Cargo.toml --test tropical_symbolic_ad
cargo test --manifest-path ext/tropical/Cargo.toml
```

Expected: all tropical tests pass.

**Step 5: Commit**

```bash
git add ext/tropical/src/fused.rs ext/tropical/src/einsum.rs ext/tropical/tests/fused_tropical_ad.rs ext/tropical/tests/tropical_symbolic_ad.rs
git commit -m "feat(tropical-ad): route forward through argmax-capable path"
```

### Task 8: Add Benchmarks And Documentation

**Files:**
- Modify: `ext/tropical/benches/tropical_matmul.rs`
- Modify: `ext/tropical/BENCHMARK_RESULTS.md`
- Modify: `docs/spec/extension-op.md` or `docs/design/algebra.md` if architecture text is stale

**Step 1: Add benchmark cases**

Update the benchmark to compare:

- generic tropical contraction;
- optimized value-only path if available;
- optimized value-plus-argmax path.

Use small and medium sizes such as `16`, `64`, `128`, and `256`.

**Step 2: Run benchmark smoke**

Run:

```bash
cargo bench --manifest-path ext/tropical/Cargo.toml --bench tropical_matmul -- --sample-size 10 --measurement-time 3
```

Expected: benchmark completes and reports a measurable optimized-path improvement for at least medium shapes.

**Step 3: Update docs**

Document:

- optimized-path eligibility;
- first-winner tie policy;
- fallback behavior;
- public einsum lowering API purpose.

**Step 4: Run verification**

Run:

```bash
cargo fmt --all --check
cargo fmt --manifest-path ext/tropical/Cargo.toml --all --check
cargo test -p tenferro-einsum --lib
cargo test -p tenferro-einsum --doc
cargo test --manifest-path ext/tropical/Cargo.toml
cargo doc --workspace --no-deps
cargo doc --manifest-path ext/tropical/Cargo.toml --no-deps
```

Expected: all pass. If `cargo fmt --all --check` fails, run `cargo fmt --all` and repeat.

**Step 5: Commit**

```bash
git add ext/tropical/benches/tropical_matmul.rs ext/tropical/BENCHMARK_RESULTS.md docs/spec/extension-op.md docs/design/algebra.md
git commit -m "docs(tropical): document optimized argmax path"
```

### Task 9: Final Repository Verification

**Files:**
- No planned edits.

**Step 1: Re-read repository rules**

Run:

```bash
sed -n '1,240p' REPOSITORY_RULES.md
```

Check the local diff for hidden materialization, column-major violations,
public doctest coverage, and extension boundary mistakes.

**Step 2: Run required checks as far as the environment permits**

Run:

```bash
cargo fmt --all --check
cargo fmt --manifest-path ext/tropical/Cargo.toml --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
cargo test --manifest-path ext/tropical/Cargo.toml
cargo doc --manifest-path ext/tropical/Cargo.toml --no-deps
```

Expected: all pass. If runtime or coverage cost is too high for the current environment, record the exact command and failure or timeout.

**Step 3: Summarize residual risks**

Before creating a PR, summarize:

- whether `tropical-gemm` is optional or default;
- which tropical kinds and dtypes are optimized;
- unsupported GPU behavior;
- AD fallback cases;
- benchmark result location.

**Step 4: Commit any final fixes**

```bash
git status --short
git add <fixed-files>
git commit -m "fix(tropical): address final verification findings"
```

Skip this commit if there are no final fixes.
