# TreeSA-Only Planner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the hard-coded greedy contraction planner with a TreeSA-based public planner API whose default is greedy-initialized `TreeSA` with zero annealing iterations.

**Architecture:** Keep `ContractionTree::optimize` as the simple entrypoint, but route it through a new public planner-options type. The default options use `TreeSA` only, with `initializer=Greedy`, `ntrials=1`, and `niters=0`, while exposing TreeSA knobs for future downstream control. Upgrade `omeco` to `0.2.4` so the public score and TreeSA APIs match current docs.

**Tech Stack:** Rust, `tenferro-einsum`, `omeco 0.2.4`, `cargo nextest`, release-mode microbenchmarks.

---

### Task 1: Add failing planner-option tests

**Files:**
- Modify: `tenferro-einsum/src/planning/tree/tests.rs`

**Step 1: Write the failing tests**

Add tests that:
- prove `ContractionTree::optimize` matches `optimize_with_options(Default::default())`
- prove the default planner options build the same pair sequence as `TreeSA::new(vec![], 1, 0, Initializer::Greedy, ScoreFunction::default())`
- prove a time-optimized score can be passed through `optimize_with_options(...)` without API ambiguity

**Step 2: Run the focused test to verify RED**

Run:

```bash
cargo test -p tenferro-einsum --lib planning::tree::tests::default_optimize_ planning::tree::tests::optimize_with_options_
```

Expected:
- compile failures because planner options and TreeSA-only routing do not exist yet

**Step 3: Commit**

```bash
git add tenferro-einsum/src/planning/tree/tests.rs
git commit -m "test: add TreeSA planner option coverage"
```

### Task 2: Implement TreeSA-only planner API

**Files:**
- Modify: `Cargo.toml`
- Modify: `tenferro-einsum/src/planning/tree.rs`
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Upgrade omeco**

Change workspace dependency from `omeco = "0.2.1"` to `omeco = "0.2.4"`.

**Step 2: Add the public planner option type**

In `tenferro-einsum/src/planning/tree.rs`, add a public options struct that contains:
- `betas: Vec<f64>`
- `ntrials: usize`
- `niters: usize`
- `score: omeco::ScoreFunction`

and uses `Initializer::Greedy` internally.

**Step 3: Add a new planner entrypoint**

Add:

```rust
pub fn optimize_with_options(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
    options: &ContractionOptimizerOptions,
) -> Result<Self>
```

Route `optimize(...)` through `optimize_with_options(..., &Default::default())`.

**Step 4: Route omeco through TreeSA only**

Replace `GreedyMethod::default()` in the omeco bridge with:

```rust
TreeSA::new(
    options.betas.clone(),
    options.ntrials,
    options.niters,
    Initializer::Greedy,
    options.score.clone(),
)
```

Keep the existing self-greedy fallback only for `None` from omeco.

**Step 5: Re-export the public options type**

Expose the new options type from `tenferro-einsum/src/lib.rs`.

**Step 6: Run focused tests**

Run:

```bash
cargo test -p tenferro-einsum --lib planning::tree::tests::default_optimize_ planning::tree::tests::optimize_with_options_
```

Expected:
- new tests pass

**Step 7: Commit**

```bash
git add Cargo.toml tenferro-einsum/src/planning/tree.rs tenferro-einsum/src/lib.rs
git commit -m "feat: add TreeSA-based planner options"
```

### Task 3: Verify planner behavior and benchmark impact

**Files:**
- Modify if needed: `tenferro-einsum/tests/bench_issue_336_fit_shapes.rs`
- Modify if needed: `tenferro-einsum/tests/bench_nary_fit_breakdown.rs`

**Step 1: Run library verification**

Run:

```bash
cargo nextest run --release -p tenferro-einsum --lib
```

Expected:
- `tenferro-einsum` library tests pass

**Step 2: Run issue 336 related microbenchmarks**

Run:

```bash
cargo test --release -p tenferro-einsum --test bench_issue_336_fit_shapes bench_issue_336_fit_shapes -- --ignored --exact --nocapture --test-threads=1
cargo test --release -p tenferro-einsum --test bench_nary_fit_breakdown bench_nary_fit_breakdown -- --ignored --exact --nocapture --test-threads=1
```

Expected:
- planner output is valid
- planning overhead stays acceptable
- issue 336 hot contractions do not regress

**Step 3: If behavior shifts, compare score and path quality**

Use targeted assertions or temporary diagnostics to compare:
- pair sequence
- `nested_flop`
- `peak_memory`
- score under `ScoreFunction::default()` and `ScoreFunction::time_optimized()`

Do not keep ad-hoc debug output in committed code.

**Step 4: Commit**

```bash
git add tenferro-einsum/tests/bench_issue_336_fit_shapes.rs tenferro-einsum/tests/bench_nary_fit_breakdown.rs
git commit -m "test: verify TreeSA planner impact"
```
