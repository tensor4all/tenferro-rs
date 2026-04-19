# DotGeneralConfig Rank Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the structural fix for issue #664 by removing the redundant `lhs_rank` and `rhs_rank` fields from `DotGeneralConfig` so the dim-numbering config can no longer drift from the actual tensor ranks.

**Architecture:** `DotGeneralConfig` becomes purely a dim-numbering record (`lhs_contracting_dims`, `rhs_contracting_dims`, `lhs_batch_dims`, `rhs_batch_dims`). Rank info moves up to the enclosing op: `StdTensorOp::DotGeneral` becomes `{ config, lhs_rank, rhs_rank }` (parallel to how `ReduceSum` carries `input_shape`). `ExecOp::DotGeneral` drops rank entirely — the `ExecInstruction::output_shapes` table delivered by #732 is authoritative at the exec level. Compiler pass `transpose_folding` derives rank from `perm.len()` of the producer Transpose instead of reading a stored field. Runtime validation in `cpu/gemm` and `cpu/backend` uses the actual tensor `shape.len()` and is verified against the config dim ranges via an updated `validate_dims(lhs_rank, rhs_rank)` helper.

**Tech Stack:** Rust, `cargo`, `thiserror`, `computegraph`, `DimExpr` (symbolic shapes), existing `shape_infer` module.

---

## Background

### What the bug is

`DotGeneralConfig.lhs_rank` / `rhs_rank` duplicate the actual operand rank. Compiler passes that modify the config (or upstream code that constructs one from slightly-wrong metadata) can leave the rank fields inconsistent with the real tensor ranks, producing cascading `RankMismatch` panics or out-of-bounds indexing downstream. Mitigation from #707 (`validate_ranks()` at trace and runtime) surfaces the drift as a clean error, but the redundant fields are still there.

### What unblocked the fix

Before #732 the exec-level IR had no shape table, so removing the stored ranks would have left several sites with no source of truth. #732 added `ExecInstruction::output_shapes` and the `shape_infer` module. At the exec level, shapes are now authoritative and rank is derivable. At the higher `StdTensorOp` level, there is no shape table, but each op variant can carry only the metadata it genuinely needs (the `ReduceSum { axes, input_shape }` pattern). Applying that pattern to `DotGeneral` ends the "two sources of truth sharing one struct" problem without adding a shape table at the StdTensorOp level.

### Non-goals

- **Fixing `transpose_folding` free-dim reorder handling.** Already fixed on `main`; `tenferro/tests/compiler_passes.rs::test_transpose_folding_rejects_free_dim_reorder` (line 177) asserts the correct behavior. No changes needed.
- **Removing `validate_ranks()` entirely.** It's used as a defense-in-depth check at trace time and in runtime GEMM; it becomes a free-function style helper (or stays on the config taking explicit ranks) but does not disappear.
- **Changing `DimExpr` or shape inference semantics.** Out of scope.
- **Adding an `input_shapes: Vec<Vec<DimExpr>>` field to `ExecProgram`.** Not needed for this refactor; every call site can derive what it needs from either `ExecInstruction::output_shapes` or local `perm.len()`.

---

## File Structure

### Files to create

None. This is a refactor.

### Files to modify

| Path | Change |
|------|--------|
| `tenferro-tensor/src/config.rs` | Remove `lhs_rank`/`rhs_rank` from `DotGeneralConfig`; rewrite `validate_ranks` as `validate_against_shapes(lhs_shape, rhs_shape)`; change `validate_dims` to take explicit `lhs_rank, rhs_rank` params |
| `tenferro-ops/src/std_tensor_op.rs` | Change `DotGeneral(DotGeneralConfig)` variant to `DotGeneral { config: DotGeneralConfig, lhs_rank: usize, rhs_rank: usize }` |
| `tenferro-ops/src/ad/contraction.rs` | Read `lhs_rank`/`rhs_rank` from the op pattern match instead of `config.lhs_rank/rhs_rank`; change `transpose_plan_for_lhs`/`transpose_plan_for_rhs` to take explicit `lhs_rank, rhs_rank` and return `(DotGeneralConfig, new_lhs_rank, new_rhs_rank, Vec<usize>)` |
| `tenferro-ops/src/ad/linalg.rs` | `matrix_multiply_config(rank)` returns a rank-free config; callers wrap in `StdTensorOp::DotGeneral { config, lhs_rank: rank, rhs_rank: rank }` |
| `tenferro/src/traced.rs` | `dot_general(&self, other, config)` constructs `StdTensorOp::DotGeneral { config, lhs_rank: self.rank, rhs_rank: other.rank }`; uses `self.rank`/`other.rank` for the free-dim calc |
| `tenferro/src/compiler.rs` | `std_to_exec_op` drops the rank fields on the way to `ExecOp`; `is_transpose_foldable` uses `perm.len()` in place of `config.lhs_rank`/`rhs_rank`; `fold_transpose_into_dot` unchanged (it never touched the rank fields) |
| `tenferro/src/shape_infer.rs` | `dot_general_shape` uses `lhs_shape.len()` / `rhs_shape.len()` as rank directly (drops the now-removed `config.lhs_rank` / `rhs_rank` asserts) |
| `tenferro/src/lib.rs` | `matmul` helper: construct `StdTensorOp::DotGeneral { config, lhs_rank, rhs_rank }` via the new `dot_general` path (no direct change needed if it already goes through `traced.dot_general`) — otherwise drop the `lhs_rank/rhs_rank` in the config literal |
| `tenferro/src/linalg_api.rs` | `matmul_preserve_trailing_batch`: drop `lhs_rank`/`rhs_rank` from the `DotGeneralConfig` literal (goes through `traced.dot_general` which now supplies them on the op) |
| `tenferro-einsum/src/builder.rs` | Drop `lhs_rank`/`rhs_rank` from config literal; callers of `add_op(Op::dot_general(config), ...)` need updated signature — the builder needs to emit the new `StdTensorOp::DotGeneral { config, lhs_rank, rhs_rank }` variant |
| `tenferro-einsum/src/eager.rs` | Drop `lhs_rank`/`rhs_rank` from config literal; pass ranks explicitly to `backend.dot_general(lhs, rhs, config, lhs_rank, rhs_rank)` OR adjust backend trait to not need them (see Task 7) |
| `tenferro-tensor/src/types.rs` | `TypedTensor::matmul` — drop `lhs_rank`/`rhs_rank` from config literal |
| `tenferro-tensor/src/cpu/gemm/mod.rs` | `validate_config(lhs, rhs, config)` uses `lhs.shape.len()`/`rhs.shape.len()` directly (no `config.lhs_rank` read); `canonical_gemm_layout(config, lhs_rank, rhs_rank)` signature unchanged but internal config construction drops the fields |
| `tenferro-tensor/src/cpu/backend.rs` | `matmul_preserve_trailing_batch` — drop fields; semiring dispatch uses `lhs.shape.len()`/`rhs.shape.len()` |
| `tenferro-tensor/src/cubecl/linalg.rs` | `matmul_preserve_trailing_batch` — drop fields |
| `tenferro-tensor/src/cpu/backend.rs` dot_general entry — validate against `lhs.shape.len()`/`rhs.shape.len()` via the rewritten `validate_dims` |
| `tenferro-tensor/src/tests/cpu_tests.rs` | Remove `lhs_rank`/`rhs_rank` from every `DotGeneralConfig` literal (~20 sites) |
| `tenferro-tensor/src/tests/cpu_semiring_tests.rs` | Same as above |
| `tenferro-tensor/src/cpu/gemm/tests.rs` | Same as above |
| `tenferro/src/eager_ops.rs` | Update the doc example for `dot_general` |
| `tenferro-ops/tests/**/*.rs` (if any construct `DotGeneralConfig` literally) | Same as above |
| `tenferro/tests/compiler_passes.rs` | Update any `StdTensorOp::DotGeneral` literals to the new struct-variant shape |

### Source-of-truth map (after the refactor)

| Layer | Rank source |
|-------|-------------|
| `TracedTensor::dot_general` | `self.rank` / `other.rank` |
| `StdTensorOp::DotGeneral { config, lhs_rank, rhs_rank }` | Carried on the op variant |
| AD rule for DotGeneral | Op-variant fields (`lhs_rank`, `rhs_rank`) |
| `shape_infer::dot_general_shape` | `lhs_shape.len()` / `rhs_shape.len()` (function args) |
| `ExecOp::DotGeneral(config)` | `ExecInstruction::output_shapes` (producer) or local `perm.len()` |
| `transpose_folding::is_transpose_foldable` | `perm.len()` of the producer Transpose |
| `cpu::gemm::validate_config` | `lhs.shape.len()` / `rhs.shape.len()` at runtime |

---

## Task Breakdown

Tasks are ordered to keep the tree buildable after each commit. Early tasks introduce helpers; later tasks migrate call sites; the field-removal happens last.

### Task 0: Set up worktree

**Files:** none (prep)

- [ ] **Step 1: Create a worktree from the latest `main`**

```bash
git fetch origin
git worktree add -b fix/664-remove-rank-fields ../.worktrees/664-rank-removal origin/main
cd ../.worktrees/664-rank-removal
```

- [ ] **Step 2: Verify baseline test suite passes**

```bash
cargo build --workspace
cargo test --workspace --release
```

Expected: clean build, all tests pass. If not, stop — fix or rebase before continuing.

---

### Task 1: Expand `validate_dims` to take explicit ranks

**Files:**
- Modify: `tenferro-tensor/src/config.rs`
- Test: `tenferro-tensor/src/tests/config_tests.rs` (create if missing, else add to existing `tests/`)

**Why first:** every downstream caller of `validate_dims` currently relies on `config.lhs_rank`/`rhs_rank`. Before removing the fields we need an API that takes ranks explicitly, so call sites can migrate one at a time.

- [ ] **Step 1: Add the test for the new signature**

In `tenferro-tensor/src/tests/config_tests.rs` (create the file and register it via `mod config_tests;` in `src/tests/mod.rs` if the directory already uses that pattern — otherwise add a `#[cfg(test)] mod tests;` in `config.rs` with the same content):

```rust
use crate::DotGeneralConfig;

#[test]
fn validate_dims_with_explicit_ranks_rejects_out_of_range_contract() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    let err = config
        .validate_dims_with_ranks(2, 2)
        .expect_err("dim index 2 is out of range for rank 2");
    assert!(err.contains("out of bounds"));
}

#[test]
fn validate_dims_with_explicit_ranks_accepts_valid_config() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    config.validate_dims_with_ranks(2, 2).unwrap();
}
```

(These literals still carry `lhs_rank`/`rhs_rank` because the field removal comes in Task 10. After Task 10 we revisit this file and drop them.)

- [ ] **Step 2: Run the test and verify it fails because the method does not exist**

```bash
cargo test -p tenferro-tensor config_tests::validate_dims_with_explicit_ranks -- --nocapture
```

Expected: compile error, no such method `validate_dims_with_ranks`.

- [ ] **Step 3: Add `validate_dims_with_ranks(lhs_rank, rhs_rank)` to `DotGeneralConfig`**

In `tenferro-tensor/src/config.rs`, add (keeping the existing `validate_dims(&self)` method for now as a thin wrapper):

```rust
impl DotGeneralConfig {
    // ... existing methods above ...

    /// Validate dim indices against an explicit rank pair, independent of the
    /// (soon-to-be-removed) `lhs_rank`/`rhs_rank` fields.
    pub fn validate_dims_with_ranks(
        &self,
        lhs_rank: usize,
        rhs_rank: usize,
    ) -> Result<(), String> {
        for &d in &self.lhs_contracting_dims {
            if d >= lhs_rank {
                return Err(format!(
                    "lhs_contracting_dim {} out of bounds for lhs_rank {}",
                    d, lhs_rank
                ));
            }
        }
        for &d in &self.rhs_contracting_dims {
            if d >= rhs_rank {
                return Err(format!(
                    "rhs_contracting_dim {} out of bounds for rhs_rank {}",
                    d, rhs_rank
                ));
            }
        }
        for &d in &self.lhs_batch_dims {
            if d >= lhs_rank {
                return Err(format!(
                    "lhs_batch_dim {} out of bounds for lhs_rank {}",
                    d, lhs_rank
                ));
            }
        }
        for &d in &self.rhs_batch_dims {
            if d >= rhs_rank {
                return Err(format!(
                    "rhs_batch_dim {} out of bounds for rhs_rank {}",
                    d, rhs_rank
                ));
            }
        }
        Self::check_no_duplicates(&self.lhs_contracting_dims, "lhs_contracting_dims")?;
        Self::check_no_duplicates(&self.rhs_contracting_dims, "rhs_contracting_dims")?;
        Self::check_no_duplicates(&self.lhs_batch_dims, "lhs_batch_dims")?;
        Self::check_no_duplicates(&self.rhs_batch_dims, "rhs_batch_dims")?;
        for &d in &self.lhs_contracting_dims {
            if self.lhs_batch_dims.contains(&d) {
                return Err(format!(
                    "lhs dim {} appears in both contracting and batch dims",
                    d
                ));
            }
        }
        for &d in &self.rhs_contracting_dims {
            if self.rhs_batch_dims.contains(&d) {
                return Err(format!(
                    "rhs dim {} appears in both contracting and batch dims",
                    d
                ));
            }
        }
        if self.lhs_contracting_dims.len() != self.rhs_contracting_dims.len() {
            return Err(format!(
                "lhs/rhs contracting dim counts differ ({} vs {})",
                self.lhs_contracting_dims.len(),
                self.rhs_contracting_dims.len()
            ));
        }
        if self.lhs_batch_dims.len() != self.rhs_batch_dims.len() {
            return Err(format!(
                "lhs/rhs batch dim counts differ ({} vs {})",
                self.lhs_batch_dims.len(),
                self.rhs_batch_dims.len()
            ));
        }
        Ok(())
    }
}
```

Change the existing `validate_dims(&self)` method to delegate: `self.validate_dims_with_ranks(self.lhs_rank, self.rhs_rank)`.

- [ ] **Step 4: Run tests and verify**

```bash
cargo test -p tenferro-tensor config_tests::validate_dims_with_explicit_ranks -- --nocapture
```

Expected: both tests pass.

- [ ] **Step 5: Run the full crate test suite**

```bash
cargo test -p tenferro-tensor --release
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tenferro-tensor/src/config.rs tenferro-tensor/src/tests/
git commit -m "refactor(config): add DotGeneralConfig::validate_dims_with_ranks for explicit-rank validation"
```

---

### Task 2: Change `StdTensorOp::DotGeneral` to struct-variant carrying ranks

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Test: existing tests in `tenferro-ops` and downstream crates (rely on the compile-time check — any site that pattern-matches the old tuple variant will break, which is the point).

- [ ] **Step 1: Update the variant shape**

In `tenferro-ops/src/std_tensor_op.rs`, find the `StdTensorOp` enum. Change:

```rust
DotGeneral(DotGeneralConfig),
```

to:

```rust
DotGeneral {
    config: DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
},
```

- [ ] **Step 2: Fix every pattern-match and constructor inside `tenferro-ops`**

Compile and fix. Expected sites (based on search before starting):

```bash
cargo build -p tenferro-ops 2>&1 | head -80
```

For every `StdTensorOp::DotGeneral(config)` match pattern, change to `StdTensorOp::DotGeneral { config, lhs_rank, rhs_rank }`. When the pattern ignores the ranks use `StdTensorOp::DotGeneral { config, .. }`.

Key sites:
- `tenferro-ops/src/ad/mod.rs:182-184` — AD dispatch pattern match. Change to destructure and pass `lhs_rank`, `rhs_rank` through to the rule.
- `tenferro-ops/src/ad/linalg.rs` — the call sites of `matrix_multiply_config(rank)` become:
  ```rust
  StdTensorOp::DotGeneral {
      config: matrix_multiply_config(rank),
      lhs_rank: rank,
      rhs_rank: rank,
  }
  ```

At this point the AD rule `contraction::transpose_dot_general` still takes `config: &DotGeneralConfig`. We update it in Task 3.

- [ ] **Step 3: Keep the tree buildable**

```bash
cargo build -p tenferro-ops
```

Expected: compiles. (Downstream crates may not yet; that's OK — we fix them in the following tasks.)

- [ ] **Step 4: Commit**

```bash
git add tenferro-ops/src/std_tensor_op.rs tenferro-ops/src/ad/mod.rs tenferro-ops/src/ad/linalg.rs
git commit -m "refactor(std_tensor_op): make DotGeneral a struct variant carrying lhs_rank/rhs_rank"
```

(Workspace `cargo build` will still be red after this commit because downstream crates haven't updated. The following tasks are the migration; we commit each one as we go. If you want a single atomic change instead of a series of red-green commits, squash at the end with `git rebase -i`.)

---

### Task 3: Update AD rule signature to take explicit ranks

**Files:**
- Modify: `tenferro-ops/src/ad/contraction.rs`
- Modify: `tenferro-ops/src/ad/mod.rs`
- Test: existing AD tests

- [ ] **Step 1: Change `transpose_dot_general` signature**

In `tenferro-ops/src/ad/contraction.rs:233`, change:

```rust
pub fn transpose_dot_general(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &DotGeneralConfig,
) -> Vec<Option<LocalValId>> {
```

to:

```rust
pub fn transpose_dot_general(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
) -> Vec<Option<LocalValId>> {
```

Inside the body, replace every `config.lhs_rank` with `lhs_rank` and every `config.rhs_rank` with `rhs_rank` (two reads each at lines ~250 and ~256). The `compute_free_dims(config.lhs_rank, ...)` and `compute_free_dims(config.rhs_rank, ...)` calls become `compute_free_dims(lhs_rank, ...)` / `compute_free_dims(rhs_rank, ...)`.

- [ ] **Step 2: Update `transpose_plan_for_lhs`/`transpose_plan_for_rhs`**

These helpers currently read `config.lhs_rank`/`config.rhs_rank` and write ranks into the new config they return. Change signatures to take explicit ranks and return the new ranks too:

```rust
fn transpose_plan_for_lhs(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> (DotGeneralConfig, /* new_lhs_rank */ usize, /* new_rhs_rank */ usize, Vec<usize>) {
    let n_batch = config.lhs_batch_dims.len();
    let output_rank = lhs_free.len() + rhs_free.len() + n_batch;
    let ct_rhs_free_positions: Vec<usize> =
        (lhs_free.len()..lhs_free.len() + rhs_free.len()).collect();

    let rhs_contracting_order =
        compute_free_dims(rhs_rank, rhs_free, &config.rhs_batch_dims);
    let mut result_order = Vec::with_capacity(lhs_rank);
    result_order.extend(lhs_free.iter().copied());
    for rhs_dim in rhs_contracting_order {
        let pair_idx = config
            .rhs_contracting_dims
            .iter()
            .position(|&dim| dim == rhs_dim)
            .expect("rhs contracting dimension must be paired");
        result_order.push(config.lhs_contracting_dims[pair_idx]);
    }
    result_order.extend(config.lhs_batch_dims.iter().copied());

    let new_config = DotGeneralConfig {
        lhs_contracting_dims: ct_rhs_free_positions,
        rhs_contracting_dims: rhs_free.to_vec(),
        lhs_batch_dims: (lhs_free.len() + rhs_free.len()..output_rank).collect(),
        rhs_batch_dims: config.rhs_batch_dims.clone(),
        // Still carries the soon-to-be-removed fields. Task 10 drops them.
        lhs_rank: output_rank,
        rhs_rank,
    };
    (
        new_config,
        output_rank,
        rhs_rank,
        permutation_to_original_order(lhs_rank, &result_order),
    )
}
```

Apply the symmetric transformation to `transpose_plan_for_rhs`.

- [ ] **Step 3: Update the callers in `transpose_dot_general`**

```rust
let (transpose_config, new_lhs_rank, new_rhs_rank, perm) =
    transpose_plan_for_lhs(config, lhs_rank, rhs_rank, &lhs_free, &rhs_free);
let out = emitter.add_op(
    StdTensorOp::DotGeneral {
        config: transpose_config,
        lhs_rank: new_lhs_rank,
        rhs_rank: new_rhs_rank,
    },
    vec![cotangent.clone(), ValRef::Local(rhs_conj)],
    OpMode::Linear {
        active_mask: vec![true, false],
    },
);
```

Same pattern for the rhs branch.

- [ ] **Step 4: Update the dispatcher**

In `tenferro-ops/src/ad/mod.rs:182-184`, change:

```rust
StdTensorOp::DotGeneral(config) => {
    contraction::transpose_dot_general(emitter, cotangent_out, inputs, mode, config)
}
```

to:

```rust
StdTensorOp::DotGeneral {
    config,
    lhs_rank,
    rhs_rank,
} => contraction::transpose_dot_general(
    emitter,
    cotangent_out,
    inputs,
    mode,
    config,
    *lhs_rank,
    *rhs_rank,
),
```

- [ ] **Step 5: Build the crate**

```bash
cargo build -p tenferro-ops
```

Expected: compiles.

- [ ] **Step 6: Run AD tests**

```bash
cargo test -p tenferro-ops --release
```

Expected: all AD tests pass (the rank values flowing through are the same as before, just via a different channel).

- [ ] **Step 7: Commit**

```bash
git add tenferro-ops/src/ad/contraction.rs tenferro-ops/src/ad/mod.rs
git commit -m "refactor(ad): thread explicit lhs_rank/rhs_rank into transpose_dot_general"
```

---

### Task 4: Migrate `tenferro::traced::dot_general`

**Files:**
- Modify: `tenferro/src/traced.rs`
- Test: existing tests in `tenferro` (many exercise `dot_general` end-to-end)

- [ ] **Step 1: Update `dot_general`**

In `tenferro/src/traced.rs:1159`, change the body to use `self.rank` / `other.rank` instead of `config.lhs_rank` / `config.rhs_rank`:

```rust
pub fn dot_general(&self, other: &TracedTensor, config: DotGeneralConfig) -> TracedTensor {
    config
        .validate_dims_with_ranks(self.rank, other.rank)
        .expect("DotGeneral config dimension validation failed");
    let lhs_free: Vec<usize> = (0..self.rank)
        .filter(|d| {
            !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d)
        })
        .collect();
    let rhs_free: Vec<usize> = (0..other.rank)
        .filter(|d| {
            !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d)
        })
        .collect();
    let out_rank = config.lhs_batch_dims.len() + lhs_free.len() + rhs_free.len();
    let out_shape_hint = match (&self.shape_hint, &other.shape_hint) {
        (Some(lhs_shape), Some(rhs_shape)) => {
            let mut out_shape = Vec::with_capacity(out_rank);
            for &d in &lhs_free {
                out_shape.push(lhs_shape[d].clone());
            }
            for &d in &rhs_free {
                out_shape.push(rhs_shape[d].clone());
            }
            for &d in &config.lhs_batch_dims {
                out_shape.push(lhs_shape[d].clone());
            }
            Some(out_shape)
        }
        _ => None,
    };

    apply_binary(
        StdTensorOp::DotGeneral {
            config,
            lhs_rank: self.rank,
            rhs_rank: other.rank,
        },
        self,
        other,
        out_rank,
        out_shape_hint,
    )
}
```

Note the deletion of the `validate_ranks` call — it was redundant with `validate_dims_with_ranks` once the latter takes explicit ranks, and the rank equality assertion now holds by construction (we pass `self.rank` / `other.rank` in).

- [ ] **Step 2: Build + test**

```bash
cargo build -p tenferro
cargo test -p tenferro --release
```

Expected: compiles and tests pass. Any remaining build errors come from call sites in other crates — those are fixed in the next tasks.

- [ ] **Step 3: Commit**

```bash
git add tenferro/src/traced.rs
git commit -m "refactor(traced): dot_general uses self.rank/other.rank and emits struct-variant op"
```

---

### Task 5: Migrate einsum call sites (builder + eager)

**Files:**
- Modify: `tenferro-einsum/src/builder.rs`
- Modify: `tenferro-einsum/src/eager.rs`
- Test: `cargo test -p tenferro-einsum`

- [ ] **Step 1: Update `builder.rs:277`**

Replace the `DotGeneralConfig { ..., lhs_rank: lhs.shape.len(), rhs_rank: rhs.shape.len() }` literal with a rank-free literal, and wrap the op emission in the new struct-variant:

```rust
let config = DotGeneralConfig {
    lhs_contracting_dims,
    rhs_contracting_dims,
    lhs_batch_dims,
    rhs_batch_dims,
    // Fields still present until Task 10 — keep writing them so the tree stays buildable.
    lhs_rank: lhs.shape.len(),
    rhs_rank: rhs.shape.len(),
};

// ... later ...

let outputs = builder.add_op(
    Op::dot_general(config, lhs.shape.len(), rhs.shape.len()),
    vec![lhs.val.clone(), rhs.val.clone()],
    OpMode::Primal,
);
```

The `Op::dot_general(config)` constructor (if it exists — check `tenferro-ops/src/std_tensor_op.rs`) needs to become `Op::dot_general(config, lhs_rank, rhs_rank)`. If there's no such constructor today, construct `StdTensorOp::DotGeneral { ... }` inline.

- [ ] **Step 2: Update `eager.rs:348`**

Similar change. The eager einsum path calls `exec.dot_general(lhs.tensor(), rhs.tensor(), &config)`; the backend method signature is unchanged (it takes `&DotGeneralConfig` and reads the actual tensor ranks from the tensors). The only change is dropping `lhs_rank`/`rhs_rank` from the literal **after** Task 10; for now, keep writing them.

At this task (Task 5) we primarily make sure both files still compile and tests pass. No behavior change yet.

- [ ] **Step 3: Build + test**

```bash
cargo build -p tenferro-einsum
cargo test -p tenferro-einsum --release
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tenferro-einsum/src/builder.rs tenferro-einsum/src/eager.rs
git commit -m "refactor(einsum): construct struct-variant DotGeneral op from builder and eager paths"
```

---

### Task 6: Migrate `compiler.rs` and `shape_infer.rs`

**Files:**
- Modify: `tenferro/src/compiler.rs`
- Modify: `tenferro/src/shape_infer.rs`
- Test: `tenferro/tests/compiler_passes.rs`

- [ ] **Step 1: Update `std_to_exec_op` to strip the ranks on lowering**

In `tenferro/src/compiler.rs:210`, change:

```rust
StdTensorOp::DotGeneral(config) => ExecOp::DotGeneral(config.clone()),
```

to:

```rust
StdTensorOp::DotGeneral { config, .. } => ExecOp::DotGeneral(config.clone()),
```

`ExecOp::DotGeneral` keeps its current shape (single-variant holding a `DotGeneralConfig`). Rank info at the exec layer lives in `ExecInstruction::output_shapes` + input-shape lookups; it does not need to travel inside the op.

- [ ] **Step 2: Update `is_transpose_foldable` to use `perm.len()`**

In `tenferro/src/compiler.rs:500`, change:

```rust
fn is_transpose_foldable(config: &DotGeneralConfig, operand_idx: usize, perm: &[usize]) -> bool {
    let (rank, contracting_dims, batch_dims) = if operand_idx == 0 {
        (
            config.lhs_rank,
            config.lhs_contracting_dims.as_slice(),
            config.lhs_batch_dims.as_slice(),
        )
    } else {
        (
            config.rhs_rank,
            config.rhs_contracting_dims.as_slice(),
            config.rhs_batch_dims.as_slice(),
        )
    };

    if perm.len() != rank || !is_valid_permutation(perm, rank) {
        return false;
    }
    // ... rest unchanged
}
```

to:

```rust
fn is_transpose_foldable(config: &DotGeneralConfig, operand_idx: usize, perm: &[usize]) -> bool {
    let (contracting_dims, batch_dims) = if operand_idx == 0 {
        (
            config.lhs_contracting_dims.as_slice(),
            config.lhs_batch_dims.as_slice(),
        )
    } else {
        (
            config.rhs_contracting_dims.as_slice(),
            config.rhs_batch_dims.as_slice(),
        )
    };
    let rank = perm.len();

    if !is_valid_permutation(perm, rank) {
        return false;
    }

    let Some(free_dims) = free_axes(rank, contracting_dims, batch_dims) else {
        return false;
    };

    is_role_group_order_preserved(&free_dims, perm)
        && is_role_group_order_preserved(contracting_dims, perm)
        && is_role_group_order_preserved(batch_dims, perm)
}
```

Reasoning: the `perm` passed in is the producer Transpose's permutation; Transpose preserves rank, so `perm.len() == operand rank`. The equality check (`perm.len() != rank`) becomes vacuous and is dropped.

- [ ] **Step 3: Update `shape_infer::dot_general_shape`**

In `tenferro/src/shape_infer.rs:272`, remove the `config.lhs_rank`/`rhs_rank` asserts and use the lengths of the input-shape slices directly:

```rust
fn dot_general_shape(
    lhs_shape: &[DimExpr],
    rhs_shape: &[DimExpr],
    config: &DotGeneralConfig,
) -> Vec<DimExpr> {
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();

    let lhs_free = (0..lhs_rank).filter(|axis| {
        !config.lhs_contracting_dims.contains(axis) && !config.lhs_batch_dims.contains(axis)
    });
    let rhs_free = (0..rhs_rank).filter(|axis| {
        !config.rhs_contracting_dims.contains(axis) && !config.rhs_batch_dims.contains(axis)
    });

    let mut output_shape = Vec::new();
    output_shape.extend(lhs_free.map(|axis| lhs_shape[axis].clone()));
    output_shape.extend(rhs_free.map(|axis| rhs_shape[axis].clone()));
    output_shape.extend(
        config
            .lhs_batch_dims
            .iter()
            .map(|&axis| lhs_shape[axis].clone()),
    );
    output_shape
}
```

- [ ] **Step 4: Build + test**

```bash
cargo build -p tenferro
cargo test -p tenferro --release
```

Expected: all pass, including `compiler_passes.rs` transpose-folding tests.

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/compiler.rs tenferro/src/shape_infer.rs
git commit -m "refactor(compiler): derive rank from perm.len() / shape.len() instead of config"
```

---

### Task 7: Migrate runtime backends (`cpu::gemm`, `cpu::backend`, `cubecl::linalg`)

**Files:**
- Modify: `tenferro-tensor/src/cpu/gemm/mod.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-tensor/src/cubecl/linalg.rs`
- Modify: `tenferro-tensor/src/types.rs`
- Test: `cargo test -p tenferro-tensor`

- [ ] **Step 1: Runtime validation in `cpu/gemm/mod.rs`**

In `validate_config` at `tenferro-tensor/src/cpu/gemm/mod.rs:80`, drop the `config.lhs_rank != lhs.shape.len()` / `config.rhs_rank != rhs.shape.len()` rank-equality checks (lines 96-111). The actual tensor shapes are already authoritative; the remaining `validate_axis_list` calls already use `lhs_rank`/`rhs_rank` derived from `lhs.shape.len()` / `rhs.shape.len()`.

After the removal, the function looks like:

```rust
pub(crate) fn validate_config<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<()> {
    const OP: &str = "dot_general";

    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs contracting dim counts differ".into(),
        });
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs batch dim counts differ".into(),
        });
    }

    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();

    validate_axis_list(OP, "lhs_contracting", &config.lhs_contracting_dims, lhs_rank)?;
    validate_axis_list(OP, "rhs_contracting", &config.rhs_contracting_dims, rhs_rank)?;
    validate_axis_list(OP, "lhs_batch", &config.lhs_batch_dims, lhs_rank)?;
    validate_axis_list(OP, "rhs_batch", &config.rhs_batch_dims, rhs_rank)?;
    validate_role_disjoint(
        OP,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        "lhs_batch",
        &config.lhs_batch_dims,
    )?;
    validate_role_disjoint(
        OP,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        "rhs_batch",
        &config.rhs_batch_dims,
    )?;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        if lhs.shape[lhs_axis] != rhs.shape[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: format!(
                    "contracting dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs.shape[lhs_axis], rhs.shape[rhs_axis]
                ),
            });
        }
    }
    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs.shape[lhs_axis] != rhs.shape[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: "batch dim size mismatch".into(),
            });
        }
    }
    Ok(())
}
```

Remove the `Error::RankMismatch` variant if (and only if) it is no longer referenced anywhere; if it's used elsewhere, leave it.

- [ ] **Step 2: `canonical_gemm_layout`**

`canonical_gemm_layout(config, lhs_rank, rhs_rank)` already takes ranks explicitly. Inside, the new `DotGeneralConfig` literal at line 237 still writes `lhs_rank`/`rhs_rank` fields — keep them until Task 10.

No code change in Task 7 for this function. (It becomes a one-line deletion in Task 10.)

- [ ] **Step 3: `matmul_preserve_trailing_batch` in `cpu/backend.rs` and `cubecl/linalg.rs`**

These literals still write `lhs_rank`/`rhs_rank` — keep for now, drop in Task 10.

- [ ] **Step 4: `TypedTensor::matmul` in `types.rs`**

Same — no change in Task 7, delete fields in Task 10.

- [ ] **Step 5: Build + test**

```bash
cargo build --workspace
cargo test -p tenferro-tensor --release
```

Expected: all tests pass. The tree should now build end-to-end.

- [ ] **Step 6: Commit**

```bash
git add tenferro-tensor/src/cpu/gemm/mod.rs
git commit -m "refactor(gemm): drop redundant rank-equality checks in validate_config"
```

---

### Task 8: Full workspace test run (checkpoint)

**Files:** none

- [ ] **Step 1: Verify the workspace is green before the field-removal**

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo doc --workspace --no-deps
```

Expected: all pass. If any failure surfaces here, fix it in a targeted commit before proceeding to Task 10. This is the last safe checkpoint before the API-breaking change.

---

### Task 9: Remove `validate_ranks` (now redundant)

**Files:**
- Modify: `tenferro-tensor/src/config.rs`
- Modify: every caller of `validate_ranks` (search)

- [ ] **Step 1: Find all callers**

```bash
rg -n "\.validate_ranks\(" --type rust
```

Expected sites:
- `tenferro/src/traced.rs:1161` — already removed in Task 4.
- `tenferro-tensor/src/cpu/gemm/mod.rs` — already dropped in Task 7 (the rank-equality check).

If other callers remain, replace them with `.validate_dims_with_ranks(actual_lhs_rank, actual_rhs_rank)` using the tensor shapes available at the call site.

- [ ] **Step 2: Delete the method**

In `tenferro-tensor/src/config.rs`, remove the `validate_ranks` method (lines 32-67 of the pre-refactor file).

- [ ] **Step 3: Build + test**

```bash
cargo build --workspace
cargo test --workspace --release
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tenferro-tensor/src/config.rs
git commit -m "refactor(config): remove validate_ranks (subsumed by validate_dims_with_ranks)"
```

---

### Task 10: Remove `lhs_rank` / `rhs_rank` fields from `DotGeneralConfig`

**Files:**
- Modify: `tenferro-tensor/src/config.rs` (field removal + doc update)
- Modify: every `DotGeneralConfig { ... }` literal (production + test)

This is the committing blow. Every site that previously wrote `lhs_rank: ..., rhs_rank: ...` drops those lines.

- [ ] **Step 1: Remove the fields and update docs**

In `tenferro-tensor/src/config.rs`, change:

```rust
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: Vec<usize>,
    pub rhs_contracting_dims: Vec<usize>,
    pub lhs_batch_dims: Vec<usize>,
    pub rhs_batch_dims: Vec<usize>,
    pub lhs_rank: usize,
    pub rhs_rank: usize,
}
```

to:

```rust
/// DotGeneral dimension configuration.
///
/// Records only the dim-numbering roles (contracting / batch / free is derived).
/// Rank info travels with the enclosing `StdTensorOp::DotGeneral` variant at the
/// trace/StdTensorOp layer, and with `ExecInstruction::output_shapes` at the
/// exec layer. This separation makes it structurally impossible for stored
/// ranks to drift from actual tensor ranks (issue #664).
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::DotGeneralConfig;
///
/// let config = DotGeneralConfig {
///     lhs_contracting_dims: vec![1],
///     rhs_contracting_dims: vec![0],
///     lhs_batch_dims: vec![],
///     rhs_batch_dims: vec![],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: Vec<usize>,
    pub rhs_contracting_dims: Vec<usize>,
    pub lhs_batch_dims: Vec<usize>,
    pub rhs_batch_dims: Vec<usize>,
}
```

Also remove the `lhs_rank`/`rhs_rank` lines from every example in the remaining doc comments in this file (e.g., in `validate_dims_with_ranks`).

- [ ] **Step 2: Fix every literal**

```bash
cargo build --workspace 2>&1 | grep "missing field\|no field" | head -40
```

For each reported site, delete the two `lhs_rank: ...,` / `rhs_rank: ...,` lines from the `DotGeneralConfig { ... }` literal.

Expected production sites:
- `tenferro/src/lib.rs:60`
- `tenferro/src/linalg_api.rs:774`
- `tenferro-einsum/src/builder.rs:277`
- `tenferro-einsum/src/eager.rs:348`
- `tenferro-tensor/src/types.rs:1094`
- `tenferro-tensor/src/cpu/backend.rs:716`
- `tenferro-tensor/src/cpu/gemm/mod.rs:237` (inside `canonical_gemm_layout`)
- `tenferro-tensor/src/cubecl/linalg.rs:1197`
- `tenferro-ops/src/ad/linalg.rs:1264` (inside `matrix_multiply_config`)
- `tenferro-ops/src/ad/contraction.rs:702, 738` (inside `transpose_plan_for_{lhs,rhs}`)
- `tenferro/src/compiler.rs:566` area (inside `fold_transpose_into_dot`; the `config.clone()` + mutate pattern drops nothing — the fields are already gone from the cloned config)

Expected test sites (see the pre-refactor search output):
- `tenferro-tensor/src/tests/cpu_semiring_tests.rs:61, 62, 89, 90`
- `tenferro-tensor/src/tests/cpu_tests.rs` (~20 sites, all containing `lhs_rank: N, rhs_rank: N,`)
- `tenferro-tensor/src/cpu/gemm/tests.rs:51, 52`

For each test file, run a scoped search-and-edit pass:

```bash
rg -n "lhs_rank|rhs_rank" tenferro-tensor/src/tests/cpu_tests.rs
```

Manually delete each `lhs_rank: N,` and `rhs_rank: N,` line.

- [ ] **Step 3: Update doc examples**

In `tenferro/src/eager_ops.rs:114` and any other doc comment that constructs `DotGeneralConfig`, remove the `lhs_rank` / `rhs_rank` lines from the example block.

- [ ] **Step 4: Build + test**

```bash
cargo fmt --all
cargo build --workspace
cargo test --workspace --release
cargo doc --workspace --no-deps
```

Expected: all pass.

- [ ] **Step 5: Coverage check**

```bash
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: 90%+ per file. Investigate and add tests for any regressed files.

- [ ] **Step 6: Docs site check**

```bash
python3 scripts/check-docs-site.py
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(config): remove redundant lhs_rank/rhs_rank from DotGeneralConfig (closes #664)"
```

---

### Task 11: PR creation

**Files:** none (PR ceremony)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin fix/664-remove-rank-fields
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "refactor: remove redundant lhs_rank/rhs_rank from DotGeneralConfig (#664)" --body "$(cat <<'EOF'
## Summary

- Removes the redundant `lhs_rank` / `rhs_rank` fields from `DotGeneralConfig`; they duplicated the actual operand ranks and could drift, causing the cascading `RankMismatch` panics originally reported in #664.
- Rank info moves up to the enclosing `StdTensorOp::DotGeneral { config, lhs_rank, rhs_rank }` variant (parallel to how `ReduceSum` carries `input_shape`). At the exec layer, rank is derived from `ExecInstruction::output_shapes` and from `Transpose::perm.len()` in compiler passes.
- The `validate_ranks` mitigation added by #707 is no longer needed — drift is structurally impossible now that the config carries dim numbering only.

Closes #664.

## Test plan

- [ ] `cargo test --workspace --release`
- [ ] `cargo fmt --all --check`
- [ ] `cargo doc --workspace --no-deps`
- [ ] Coverage check (`scripts/check-coverage.py`) still at 90%+ per file
- [ ] Manually re-run the einsum benchmark on a 84+ tensor instance to confirm no regressions

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Enable auto-merge**

```bash
gh pr merge --auto --squash --delete-branch
```

---

## Self-Review Notes

**Spec coverage:**

| Issue #664 ask | Covered by |
|----------------|-----------|
| Remove `lhs_rank`/`rhs_rank` from `DotGeneralConfig` | Task 10 |
| Derive rank from per-instruction shape table | Task 6 (compiler passes use `perm.len()`; `shape_infer` uses `input_shape.len()`); Task 7 (runtime uses actual tensor shape) |
| Add validation at rewrite boundaries (`perm.len() == producer_rank`, dims in range, roles disjoint) | Task 1 (`validate_dims_with_ranks`), Task 4 (trace-time validation), Task 7 (runtime validation) |
| AD rules need operand rank but only see `ValRef`s — carry rank on op | Task 2 (struct variant); Task 3 (thread through) |
| Compiler pass drift (transpose_folding reading stale ranks) | Task 6 (use `perm.len()` instead of stored field) |
| Fix `transpose_folding` free-dim reorder bug | Out of scope — already fixed on main (test `test_transpose_folding_rejects_free_dim_reorder` exists) |

**Risk spots:**

- Task 2 makes `StdTensorOp::DotGeneral` API-breaking. Every in-tree match must update; Task 3–5 are the migration. Downstream repos (tensor4all-capi, Tensor4all.jl) do not pattern-match this variant directly, so no cross-repo churn.
- `DotGeneralConfig: Hash + PartialEq + Eq` is used as a cache key (`NaryEinsum` contraction-order LRU, #722). Removing the rank fields from the hash changes the key identity for existing cached plans — the cache is process-local and will rebuild on the first call after the release, so this is fine in practice. Flag it in the PR description if users have pickled configs on disk (they don't today).
- Some test files have 20+ `DotGeneralConfig { ... }` literals. Task 10 is mechanical but verbose; keep an eye out for formatting drift and run `cargo fmt --all` before the final commit.
