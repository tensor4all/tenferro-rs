# Einsum Inner Optimizer and Runtime Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the runtime compiler optimizer into a reusable module, add low-risk layout optimizations for einsum-lowered graphs, and cache the mutable runtime state used by `EinsumExtensionOp` inner graph execution.

**Architecture:** `compile_std_to_exec()` remains the public default entry point, but it delegates lowering and pass orchestration to separate compiler modules. The optimizer has an explicit config and fingerprint, so extension runtime caches are invalidated when pass selection changes. Fixed-path einsum lowers directly into the outer `StdTensorOp` graph where possible; runtime-shape-dependent Auto planning stays as `EinsumExtensionOp`. `EinsumExtensionOp` keeps the existing contraction-tree and inner `ExecProgram` caches, then extends the inner runtime entry with backend `RuntimeCache` and, where lifetime-compatible, slot workspace reuse.

**Tech Stack:** Rust, `smallvec`, `tenferro-runtime`, `tenferro-einsum`, `tenferro-ad` compiler pass tests, `tenferro-benchmark` CPU einsum benchmark.

## Implementation Status

Implemented on 2026-06-05:

- `CompilerOptions` / `OptimizerConfig` with optimizer fingerprint.
- `compiler::optimizer::optimize_exec_program()` orchestration module.
- Default-on algebraic layout simplification for identity transpose, adjacent transpose composition, and op-local identity reshape.
- Default-on conservative layout-chain transpose folding using the existing safe `Transpose -> DotGeneral` fold boundary.
- `dot_decomposer` remains opt-in and is reachable through `CompilerOptions` / `GraphCompiler`.
- `GraphCompiler::{with_compiler_options, compiler_options, set_compiler_options}`.
- Fixed-path traced einsum graph expansion for concrete and symbolic fixed paths; symbolic `Auto` remains an extension.
- Eager `EagerTensor` N-ary einsum graph replay through standard `StdTensorOp` nodes, preserving eager AD.
- Einsum runtime inner `ExecProgram` cache now keeps backend `RuntimeCache` and includes optimizer fingerprint in the key.
- Short hot metadata vectors use `SmallVec` in einsum builder axes/perms, runtime input indices, and compiler axis scratch.
- Benchmark knob: `TENFERRO_OPT_DOT_DECOMPOSER=1` enables the opt-in dot decomposer in the traced benchmark compiler.

Latest verification:

- `cargo test -p tenferro-runtime --lib`
- `cargo test -p tenferro-ad --test compiler_passes`
- `cargo test -p tenferro-ad --test graph_compile`
- `cargo test -p tenferro-einsum --lib`
- `cargo test -p tenferro-einsum --features autodiff --lib`
- `cargo test --no-default-features --features system-accelerate` in `tenferro-benchmark`
- 1T CPU einsum benchmark: `TENFERRO_CPU_FEATURES=system-accelerate TENFERRO_CPU_BACKEND_KIND=blas TENFERRO_OPT_DOT_DECOMPOSER=0 ./scripts/run_all.sh 1`, output in `tenferro-benchmark/data/results/cpu/einsum/20260605_071706/`.

Deferred follow-up:

- Global contraction-tree layout optimization and extension participation in outer graph optimization remain a separate larger design item. The current implementation only applies local graph optimizer passes after fixed-path expansion.

---

## Scope

This plan covers five connected changes:

1. Refactor runtime compiler optimizer code out of `tenferro-runtime/src/compiler/mod.rs`.
2. Add recommended optimizer passes that are local, measurable, and low risk:
   - algebraic/layout simplification, default-on;
   - layout-chain transpose folding into `DotGeneral`, default-on when single-use;
   - `dot_decomposer` wiring behind an explicit config gate, not default-on until benchmarked.
3. Expand fixed-path einsum into ordinary graph ops instead of carrying `ExecOp::Extension(EinsumExtensionOp)` through the outer graph:
   - concrete traced/static-tree expansion first;
   - symbolic fixed-path expansion after the graph builder accepts `DimExpr`;
   - Auto with runtime-only concrete shapes remains an extension.
4. Standardize hot-path short index/axis vectors on `SmallVec<[usize; N]>` or `SmallVec<[u32; N]>` aliases to reduce per-call allocation in einsum lowering and compiler passes.
5. Extend `EinsumExtensionOp` runtime cache entries to retain mutable per-shape runtime state:
   - compiled inner `ExecProgram`;
   - input index mapping;
   - backend `B::RuntimeCache`;
   - optional owned-input slot workspace.

This plan intentionally does not implement global contraction-tree layout optimization. That is larger: it needs contraction-tree-wide cost modeling and should be tracked separately.

## Current State

Relevant files:

- `tenferro-runtime/src/compiler/mod.rs`
  - Public `compile_std_to_exec()`.
  - Lowering from `CompiledProgram<StdTensorOp>` to `ExecProgram`.
  - Passes currently in the same file: `conj_sinking`, `dot_dimension_sorter`, `transpose_folding`, `dot_conj_folding`, `eliminate_dead_code`, `populate_last_use`.
  - `dot_decomposer` is implemented in `compiler/dot_decomposer.rs` and exported, but is not called by `compile_std_to_exec()`.

- `tenferro-einsum/src/extension.rs`
  - `cached_runtime_tree()` caches `ContractionTree`.
  - `cached_runtime_exec_program()` caches `Arc<CachedRuntimeExecProgram>`.
  - `CachedRuntimeExecProgram` currently contains only `program: ExecProgram` and `input_indices: Vec<usize>`.
  - Runtime execution calls `eval_exec_ir_unsegmented()`, which creates fresh backend cache and slot workspace.

- `tenferro-einsum/src/builder.rs`
  - `build_einsum_graph()` already lowers a `ContractionTree` to ordinary `StdTensorOp` graph ops.
  - It currently takes concrete `input_shapes: &[Vec<usize>]`, even though `StdTensorOp::Reshape` and `StdTensorOp::BroadcastInDim` can carry `DimExpr`.

- `tenferro-einsum/src/traced.rs`
  - Binary exact dot has a direct traced fast path.
  - N-ary concrete traced inputs compute and cache `static_tree`, but still wrap it in `EinsumExtensionOp`.
  - Symbolic traced inputs produce `EinsumPlanSpec` and use runtime extension expansion.

- `tenferro-runtime/src/extension_runtime.rs`
  - `ExtensionExecutionContext` owns separate mutable references to backend and extension cache store.
  - It currently exposes `backend_mut()` and `caches_mut()` separately, but not a method for splitting both fields in one borrow.

- `tenferro-runtime/src/segment.rs`
  - `eval_exec_segmented_with_cache_and_workspace()` already accepts external slot workspace and `B::RuntimeCache`.
  - `GraphExecutor` uses this path for outer graph execution.

## Design Decisions

### Optimizer API

Use an explicit optimizer config instead of hardcoding pass order in `compile_std_to_exec()`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompilerOptions {
    pub optimizer: OptimizerConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OptimizerConfig {
    pub algebraic_layout_simplifier: bool,
    pub layout_chain_transpose_folding: bool,
    pub dot_decomposer: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            algebraic_layout_simplifier: true,
            layout_chain_transpose_folding: true,
            dot_decomposer: false,
        }
    }
}

impl OptimizerConfig {
    pub const VERSION: u64 = 1;

    pub fn fingerprint(self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        Self::VERSION.hash(&mut hasher);
        self.hash(&mut hasher);
        hasher.finish()
    }
}
```

Keep `compile_std_to_exec()` as the stable default:

```rust
pub fn compile_std_to_exec(
    prog: &CompiledProgram<StdTensorOp>,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
) -> ExecProgram {
    compile_std_to_exec_with_options(prog, input_dtypes, input_shapes, CompilerOptions::default())
}
```

Add an option-bearing variant used by tests and the extension runtime:

```rust
pub fn compile_std_to_exec_with_options(
    prog: &CompiledProgram<StdTensorOp>,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
    options: CompilerOptions,
) -> ExecProgram
```

### Pass Order

Default pass order:

1. `conj_sinking`
2. `dot_dimension_sorter`
3. `algebraic_layout_simplifier`
4. `transpose_folding`
5. `layout_chain_transpose_folding`
6. `dot_conj_folding`
7. optional `dot_decomposer`
8. `algebraic_layout_simplifier`
9. `eliminate_dead_code`
10. `populate_last_use`

Rationale:

- Simplify layout chains before transpose folding so direct fold opportunities are exposed.
- Run simplification again after `dot_decomposer`, because it can emit identity `Transpose` or `Reshape` in degenerate cases.
- Keep `dot_decomposer` disabled by default until network benchmarks show it helps. It can create physical layout ops that are worse than direct strided `dot_general` for some CPU cases.

### Runtime Cache Entry

Replace the immutable `Arc<CachedRuntimeExecProgram>` cache entry with a mutable backend-specific entry:

```rust
struct CachedRuntimeExecProgram<C> {
    program: ExecProgram,
    input_indices: Vec<usize>,
    optimizer_fingerprint: u64,
    backend_cache: C,
    owned_slot_workspace: Vec<Option<tenferro_runtime::exec::ExecSlot<'static>>>,
}
```

The exact shape may change if `ExecSlot` remains `pub(crate)` and cannot be named from `tenferro-einsum`. In that case, do not make `ExecSlot` public just for this. Instead:

- add a runtime helper in `tenferro-runtime` that owns the workspace internally, or
- cache only `B::RuntimeCache` in phase 1 and leave slot workspace reuse for a focused executor workspace refactor.

The must-have cache for this plan is `B::RuntimeCache`, because CPU GEMM analysis lives there and repeated inner `DotGeneral` calls should use stable cache slots.

### Graph Expansion Policy

Graph expansion means replacing:

```text
Outer graph: ExecOp::Extension(EinsumExtensionOp)
Runtime: build/compile inner graph, then execute DotGeneral/Reduce/Transpose/Mul
```

with:

```text
Outer graph: DotGeneral/Reduce/Transpose/BroadcastInDim/Mul directly
Runtime: normal graph compiler optimizer sees the whole program
```

Apply this only when the contraction order is already fixed:

- `EinsumOptimize::Tree`: fixed by the provided tree.
- `EinsumOptimize::Path`: fixed by the provided pair path after rank/subscript validation.
- concrete traced `Auto`: fixed after static-tree planning.
- concrete eager `Auto`: can use the concrete eager executor directly, but AD-recorded eager mode should prefer recording expanded standard ops once an eager expansion helper exists.

Do not expand symbolic `Auto`: runtime concrete shapes can affect the chosen path, so it must remain an extension unless the path is computed and cached at runtime.

Because `TracedTensor` owns crate-private metadata fields, `tenferro-einsum` should not construct expanded traced tensors directly. Add a small helper in `tenferro-runtime::extension` that builds a graph from caller-supplied `StdTensorOp` instructions while preserving the same metadata merge behavior as `extension::apply()`.

### SmallVec Policy

Use `SmallVec` for short-lived and cached metadata vectors that are usually rank-sized or input-count-sized:

```rust
use smallvec::SmallVec;

type AxisVec = SmallVec<[usize; 4]>;
type SlotVec = SmallVec<[usize; 4]>;
type LabelVec = SmallVec<[u32; 8]>;
type ShapeVec<T> = SmallVec<[T; 4]>;
```

Targets:

- einsum labels, axes, permutations, broadcast dims, contraction dims;
- runtime `input_indices`;
- optimizer pass temporary vectors such as free axes, mapped axes, use-count scratch where rank-sized;
- graph-expansion shape metadata after it moves to `DimExpr`.

Do not rewrite every public `Vec<usize>` API in the same change. Keep `Vec` at external/computegraph boundaries where existing structs require it, and convert with `.into_vec()` only at the boundary. Internally, prefer aliases so future changes are consistent and easy to audit.

### Borrowing Model

Do not add `Mutex` or `Arc<Mutex<_>>` around the hot cache entry unless the type-erased cache API forces it. The `ExtensionExecutor` is already owned mutably during execution, so a mutable cache entry is enough.

Add a split-borrow API:

```rust
impl<'a, B: TensorBackend> ExtensionExecutionContext<'a, B> {
    pub fn parts_mut(&mut self) -> (&mut B, &mut ExtensionCacheStore) {
        (self.backend, self.caches)
    }
}
```

Then the einsum extension can borrow the backend and cache store at the same time without cloning the cache entry out of the store.

## Task 1: Extract Compiler Options and Optimizer Orchestration

**Files:**

- Modify: `tenferro-runtime/src/compiler/mod.rs`
- Create: `tenferro-runtime/src/compiler/options.rs`
- Create: `tenferro-runtime/src/compiler/optimizer/mod.rs`
- Modify: `tenferro-runtime/src/lib.rs` only if public re-export is needed
- Test: `tenferro-ad/tests/compiler_passes.rs`

- [ ] **Step 1: Add compiler option types**

Create `tenferro-runtime/src/compiler/options.rs`:

```rust
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompilerOptions {
    pub optimizer: OptimizerConfig,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            optimizer: OptimizerConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OptimizerConfig {
    pub algebraic_layout_simplifier: bool,
    pub layout_chain_transpose_folding: bool,
    pub dot_decomposer: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            algebraic_layout_simplifier: true,
            layout_chain_transpose_folding: true,
            dot_decomposer: false,
        }
    }
}

impl OptimizerConfig {
    pub const VERSION: u64 = 1;

    pub fn fingerprint(self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        Self::VERSION.hash(&mut hasher);
        self.hash(&mut hasher);
        hasher.finish()
    }
}
```

- [ ] **Step 2: Add optimizer orchestration module**

Create `tenferro-runtime/src/compiler/optimizer/mod.rs`:

```rust
use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::DType;

use crate::exec::ExecProgram;

use super::options::OptimizerConfig;

pub fn optimize_exec_program(
    program: &mut ExecProgram,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
    config: OptimizerConfig,
) {
    super::conj_sinking(program, input_dtypes, input_shapes);
    super::dot_dimension_sorter(program);
    if config.algebraic_layout_simplifier {
        super::algebraic_layout_simplifier(program);
    }
    super::transpose_folding(program);
    if config.layout_chain_transpose_folding {
        super::layout_chain_transpose_folding(program);
    }
    super::dot_conj_folding(program);
    if config.dot_decomposer {
        super::dot_decomposer(program, input_shapes);
        if config.algebraic_layout_simplifier {
            super::algebraic_layout_simplifier(program);
        }
    }
    super::eliminate_dead_code(program);
    super::populate_last_use(program);
}
```

This step intentionally calls pass functions still living in `compiler/mod.rs`. Moving pass bodies to smaller files happens after behavior is locked.

- [ ] **Step 3: Wire `compile_std_to_exec_with_options()`**

In `tenferro-runtime/src/compiler/mod.rs`, add:

```rust
mod options;
pub mod optimizer;

pub use options::{CompilerOptions, OptimizerConfig};
```

Then replace the hardcoded pass sequence in `compile_std_to_exec()` with:

```rust
compile_std_to_exec_with_options(prog, input_dtypes, input_shapes, CompilerOptions::default())
```

Add the new function:

```rust
pub fn compile_std_to_exec_with_options(
    prog: &CompiledProgram<StdTensorOp>,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
    options: CompilerOptions,
) -> ExecProgram {
    // Existing lowering body from compile_std_to_exec().
    let mut program = lower_std_to_exec(prog, input_dtypes, input_shapes);
    optimizer::optimize_exec_program(
        &mut program,
        input_dtypes,
        input_shapes,
        options.optimizer,
    );
    program
}
```

If `lower_std_to_exec()` is not extracted in this task, keep the existing lowering body inline in `compile_std_to_exec_with_options()` and make `compile_std_to_exec()` a one-line wrapper.

- [ ] **Step 4: Add a no-behavior-change test**

In `tenferro-ad/tests/compiler_passes.rs`, add a test that default and explicit default options compile to the same instruction shape:

```rust
#[test]
fn compile_default_options_match_legacy_entrypoint() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let lhs = builder.add_input(TensorInputKey::User { id: 0 });
    let rhs = builder.add_input(TensorInputKey::User { id: 1 });
    let out = builder.add_operation(
        StdTensorOp::DotGeneral(DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        }),
        vec![ValueRef::Local(lhs), ValueRef::Local(rhs)],
        OperationRole::Primary,
    )[0];
    builder.set_outputs(vec![out]);
    let graph = builder.build();
    let compiled = compile(&graph);

    let dtypes = [DType::F64, DType::F64];
    let shapes = [dim_shape(&[4, 5]), dim_shape(&[5, 6])];
    let legacy = compile_std_to_exec(&compiled, &dtypes, &shapes);
    let explicit = compile_std_to_exec_with_options(
        &compiled,
        &dtypes,
        &shapes,
        CompilerOptions::default(),
    );

    assert_eq!(legacy.instructions.len(), explicit.instructions.len());
    assert_eq!(legacy.n_slots, explicit.n_slots);
}
```

- [ ] **Step 5: Verify**

Run:

```bash
cargo test -p tenferro-ad --test compiler_passes compile_default_options_match_legacy_entrypoint
```

Expected: PASS.

## Task 2: Add Algebraic Layout Simplifier

**Files:**

- Modify: `tenferro-runtime/src/compiler/mod.rs`
- Later move to: `tenferro-runtime/src/compiler/optimizer/algebraic_layout.rs`
- Test: `tenferro-ad/tests/compiler_passes.rs`

- [ ] **Step 1: Add identity transpose test**

Add:

```rust
#[test]
fn algebraic_layout_simplifier_removes_identity_transpose() {
    let mut program = make_exec_program(
        vec![
            ExecInstruction {
                op: ExecOp::Transpose { perm: vec![0, 1] },
                input_slots: vec![0],
                output_slots: vec![1],
                dtype: DType::F64,
                output_shapes: vec![dim_shape(&[2, 3])].into(),
                output_extents: vec![exact_extents(&dim_shape(&[2, 3]))].into(),
                last_use: Vec::new(),
            },
            ExecInstruction {
                op: ExecOp::Neg,
                input_slots: vec![1],
                output_slots: vec![2],
                dtype: DType::F64,
                output_shapes: vec![dim_shape(&[2, 3])].into(),
                output_extents: vec![exact_extents(&dim_shape(&[2, 3]))].into(),
                last_use: Vec::new(),
            },
        ],
        vec![0],
        vec![2],
        3,
    );

    algebraic_layout_simplifier(&mut program);
    eliminate_dead_code(&mut program);

    assert!(matches!(program.instructions[0].op, ExecOp::Transpose { .. }));
    assert_eq!(program.instructions[1].input_slots, vec![0]);
    eliminate_dead_code(&mut program);
    assert!(program.instructions.iter().all(|inst| !matches!(inst.op, ExecOp::Transpose { .. })));
}
```

If existing test helpers already expose `make_exec_program`, `dim_shape`, and `exact_extents`, reuse them. If the first DCE assertion is too brittle after implementation, assert only the final instruction list.

- [ ] **Step 2: Add transpose composition test**

Add:

```rust
#[test]
fn algebraic_layout_simplifier_composes_adjacent_transposes() {
    let shape = dim_shape(&[2, 3, 4]);
    let mut program = make_exec_program(
        vec![
            ExecInstruction {
                op: ExecOp::Transpose { perm: vec![1, 2, 0] },
                input_slots: vec![0],
                output_slots: vec![1],
                dtype: DType::F64,
                output_shapes: vec![dim_shape(&[3, 4, 2])].into(),
                output_extents: vec![exact_extents(&dim_shape(&[3, 4, 2]))].into(),
                last_use: Vec::new(),
            },
            ExecInstruction {
                op: ExecOp::Transpose { perm: vec![2, 0, 1] },
                input_slots: vec![1],
                output_slots: vec![2],
                dtype: DType::F64,
                output_shapes: vec![shape.clone()].into(),
                output_extents: vec![exact_extents(&shape)].into(),
                last_use: Vec::new(),
            },
        ],
        vec![0],
        vec![2],
        3,
    );

    algebraic_layout_simplifier(&mut program);
    eliminate_dead_code(&mut program);

    assert_eq!(program.output_slots, vec![0]);
    assert!(program.instructions.is_empty());
}
```

- [ ] **Step 3: Implement pass**

Add `pub fn algebraic_layout_simplifier(program: &mut ExecProgram)` near existing passes first.

Minimum behavior:

- Identity `Transpose { perm: [0, 1, ...] }`: replace all downstream uses and output slots with the source slot.
- Adjacent single-use `Transpose(Transpose(x))`: compose the permutations.
- Identity composed transpose: replace with source slot.
- Identity `Reshape` when input and output shapes are equal: replace uses with source slot.

Core helpers:

```rust
fn replace_slot_uses(program: &mut ExecProgram, from: usize, to: usize) {
    for instr in &mut program.instructions {
        for slot in &mut instr.input_slots {
            if *slot == from {
                *slot = to;
            }
        }
    }
    for slot in &mut program.output_slots {
        if *slot == from {
            *slot = to;
        }
    }
}

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(idx, &axis)| idx == axis)
}

fn compose_transpose_perms(first: &[usize], second: &[usize]) -> Option<Vec<usize>> {
    if first.len() != second.len() {
        return None;
    }
    Some(second.iter().map(|&axis| *first.get(axis)?).collect())
}
```

Only compose when the intermediate slot use count is one. This avoids changing semantics for shared transposes before we have CSE or clone semantics.

- [ ] **Step 4: Verify**

Run:

```bash
cargo test -p tenferro-ad --test compiler_passes algebraic_layout_simplifier
```

Expected: PASS.

## Task 3: Add Layout-Chain Transpose Folding Into DotGeneral

**Files:**

- Modify: `tenferro-runtime/src/compiler/mod.rs`
- Later move to: `tenferro-runtime/src/compiler/optimizer/transpose_folding.rs`
- Test: `tenferro-ad/tests/compiler_passes.rs`

- [ ] **Step 1: Add test for `Transpose -> Reshape -> DotGeneral`**

Add a test where `DotGeneral` consumes a single-use layout chain and the transpose is foldable after bypassing layout metadata.

The assertion should be:

- dot operand no longer consumes the transpose output slot;
- the remaining `Reshape` shape-input slots are rewritten if necessary;
- DCE removes the dead transpose.

Use a small rank-2 or rank-3 dot where the folded dimension numbers are easy to inspect.

- [ ] **Step 2: Implement `layout_chain_transpose_folding()`**

Implement by copying the traversal style from `find_dot_operand_conj_fold()`:

```rust
pub fn layout_chain_transpose_folding(program: &mut ExecProgram) {
    let producer_by_slot = producer_index_by_slot(program);
    let use_counts = slot_use_counts(program);
    // For each DotGeneral operand, walk backwards through single-use layout ops.
    // If a Transpose is found and fold_transpose_into_dot() accepts it, rewrite
    // either the dot input or the nearest layout op input so the transpose is bypassed.
}
```

Supported transparent ops in this task:

- `Reshape`
- identity `Transpose` already handled by Task 2

Do not include `BroadcastInDim`, `Slice`, `Pad`, `Gather`, or diagonal ops in this pass yet. Those are conjugation-transparent, but not necessarily layout-equivalent for dot dimension folding.

- [ ] **Step 3: Verify**

Run:

```bash
cargo test -p tenferro-ad --test compiler_passes transpose_folding
```

Expected: PASS.

## Task 4: Wire DotDecomposer Behind Config

**Files:**

- Modify: `tenferro-runtime/src/compiler/optimizer/mod.rs`
- Modify: `tenferro-runtime/src/compiler/dot_decomposer.rs` only if imports need adjustment
- Test: `tenferro-ad/tests/compiler_passes/dot_decomposer_tests.rs`
- Test: `tenferro-ad/tests/compiler_passes.rs`

- [ ] **Step 1: Add options test proving default-off**

Add a test with a non-canonical `DotGeneral`:

```rust
#[test]
fn dot_decomposer_is_disabled_by_default() {
    let exec = compile_std_to_exec(&compiled, &dtypes, &shapes);
    assert_eq!(
        exec.instructions
            .iter()
            .filter(|inst| matches!(inst.op, ExecOp::DotGeneral(_)))
            .count(),
        1
    );
    assert!(
        exec.instructions
            .iter()
            .all(|inst| !matches!(inst.op, ExecOp::Reshape { .. }))
    );
}
```

- [ ] **Step 2: Add options test proving opt-in**

Use:

```rust
let options = CompilerOptions {
    optimizer: OptimizerConfig {
        dot_decomposer: true,
        ..OptimizerConfig::default()
    },
};
let exec = compile_std_to_exec_with_options(&compiled, &dtypes, &shapes, options);
assert!(
    exec.instructions
        .iter()
        .any(|inst| matches!(inst.op, ExecOp::Reshape { .. }))
);
```

- [ ] **Step 3: Verify**

Run:

```bash
cargo test -p tenferro-ad --test compiler_passes dot_decomposer
```

Expected: PASS.

## Task 5: Add Runtime Helper for Expanded Traced Graphs

**Files:**

- Modify: `tenferro-runtime/src/extension.rs`
- Test: `tenferro-runtime/src/extension.rs`

- [ ] **Step 1: Add an expansion helper next to `extension::apply()`**

Add a helper that shares `extension::apply()` metadata merging, but lets an extension crate build ordinary `StdTensorOp` graph nodes instead of an `Extension` carrier:

```rust
pub fn apply_expanded_graph(
    inputs: &[&TracedTensor],
    output_metas: Vec<(DType, Vec<SymDim>)>,
    build: impl FnOnce(
        &mut GraphBuilder<StdTensorOp>,
        &[ValueRef<StdTensorOp>],
    ) -> Result<Vec<LocalValueId>>,
) -> Result<Vec<TracedTensor>>
```

Implementation requirements:

- add each input graph as a parent, as `apply()` does;
- pass `ValueRef::External(...)` input refs to `build`;
- set the returned local outputs as graph outputs;
- preserve `inputs_map`, `extra_roots`, `checkpoint_chain`, and `metadata_scopes` merge behavior from `apply()`;
- set `shape_hint` when every output `SymDim` is concrete or all inputs have concrete hints, matching the existing `apply()` invariant.

If `computegraph::types::LocalValueId` is not exported through the current imports, import it explicitly.

- [ ] **Step 2: Add a unit test**

Add a simple test that expands `x + y` with `StdTensorOp::Add`:

```rust
#[test]
fn apply_expanded_graph_builds_standard_op_without_extension() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
    let out = apply_expanded_graph(
        &[&x, &y],
        vec![(DType::F64, vec![SymDim::from(2)])],
        |builder, inputs| {
            let outputs = builder.add_operation(
                StdTensorOp::Add,
                inputs.to_vec(),
                OperationRole::Primary,
            );
            Ok(outputs)
        },
    )
    .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].rank, 1);
}
```

- [ ] **Step 3: Verify**

Run:

```bash
cargo test -p tenferro-runtime --lib apply_expanded_graph
```

Expected: PASS.

## Task 6: Make Einsum Graph Lowering Shape-Generic

**Files:**

- Modify: `tenferro-einsum/src/builder.rs`
- Modify: `tenferro-einsum/src/tests/builder_tests.rs`

- [ ] **Step 1: Convert builder metadata from `usize` to `DimExpr`**

Change:

```rust
shape: Vec<usize>
```

to:

```rust
shape: Vec<DimExpr>
```

in `LabeledVal` and helper metadata inside `builder.rs`.

Keep the existing concrete entry point as a wrapper:

```rust
pub(crate) fn build_einsum_graph(
    builder: &mut GraphBuilder<StdTensorOp>,
    tree: &ContractionTree,
    input_vals: &[ValueRef<StdTensorOp>],
    input_shapes: &[Vec<usize>],
) -> Result<ValueRef<StdTensorOp>> {
    let dim_shapes: Vec<Vec<DimExpr>> = input_shapes
        .iter()
        .map(|shape| DimExpr::from_concrete(shape))
        .collect();
    build_einsum_graph_dim_expr(builder, tree, input_vals, &dim_shapes)
}
```

Add the new core function:

```rust
pub(crate) fn build_einsum_graph_dim_expr(
    builder: &mut GraphBuilder<StdTensorOp>,
    tree: &ContractionTree,
    input_vals: &[ValueRef<StdTensorOp>],
    input_shapes: &[Vec<DimExpr>],
) -> Result<ValueRef<StdTensorOp>>
```

- [ ] **Step 2: Update shape helpers**

Replace label-size helpers with label-dimension helpers:

```rust
fn label_dim_map(labels: &[u32], shape: &[DimExpr]) -> Vec<(u32, DimExpr)> {
    labels.iter().copied().zip(shape.iter().cloned()).collect()
}

fn find_label_dim(label: u32, label_dims: &[&[(u32, DimExpr)]]) -> Result<DimExpr> {
    for dims in label_dims {
        for (candidate, dim) in *dims {
            if *candidate == label {
                return Ok(dim.clone());
            }
        }
    }
    Err(builder_invalid_argument(format!("missing dim for label {label}")))
}
```

Use `DimExpr` values directly when building:

- `StdTensorOp::BroadcastInDim { shape, dims }`
- `StdTensorOp::Reshape { to_shape }`

Do not try to prove arbitrary symbolic equality in this task. Validate rank and repeated-label structure; keep concrete equality checks only when both sides are `DimExpr::Const`.

- [ ] **Step 3: Add symbolic builder test**

Add a test that builds an outer product with symbolic dimensions:

```rust
#[test]
fn build_einsum_graph_accepts_symbolic_outer_product_shape() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(TensorInputKey::User { id: 0 });
    let b = builder.add_input(TensorInputKey::User { id: 1 });
    let tree = ContractionTree::from_path_for_testing(
        EinsumSubscripts::new(&[&[0], &[1]], &[0, 1]),
        vec![(0, 1)],
    );
    let result = build_einsum_graph_dim_expr(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b)],
        &[
            vec![DimExpr::InputDim { input_idx: 0, axis: 0 }],
            vec![DimExpr::InputDim { input_idx: 1, axis: 0 }],
        ],
    )
    .unwrap();

    assert!(matches!(result, ValueRef::Local(_)));
}
```

If `ContractionTree::from_path_for_testing` does not exist, add a small test-only helper in the existing builder test module using the same constructor pattern already used by the builder tests.

- [ ] **Step 4: Verify**

Run:

```bash
cargo test -p tenferro-einsum --lib builder
```

Expected: PASS.

## Task 7: Expand Fixed-Path Traced Einsum Into the Outer Graph

**Files:**

- Modify: `tenferro-einsum/src/traced.rs`
- Modify: `tenferro-einsum/src/builder.rs`
- Test: `tenferro-einsum/src/traced.rs` tests or a new traced einsum test module

- [ ] **Step 1: Add traced input shape conversion**

Add helper:

```rust
fn traced_dim_expr_shapes(inputs: &[&TracedTensor]) -> Result<Vec<Vec<DimExpr>>> {
    inputs
        .iter()
        .enumerate()
        .map(|(input_idx, tensor)| {
            Ok((0..tensor.rank)
                .map(|axis| {
                    tensor
                        .axis_sym_dim(axis)
                        .to_dim_expr(&[(tensor.id, input_idx)])
                        .map_err(Error::Internal)
                })
                .collect::<Result<Vec<_>>>()?)
        })
        .collect()
}
```

If a shape expression references multiple traced tensors, build a full tensor-id map for all inputs instead of a one-entry map.

- [ ] **Step 2: Add expansion helper in `traced.rs`**

Add:

```rust
fn expand_traced_einsum_graph(
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    tree: &ContractionTree,
    output_shape_hint: Vec<SymDim>,
) -> Result<TracedTensor>
```

It should call `tenferro_runtime::extension::apply_expanded_graph()`:

```rust
let input_shapes = traced_dim_expr_shapes(inputs)?;
let outputs = tenferro_runtime::extension::apply_expanded_graph(
    inputs,
    vec![(inputs[0].dtype, output_shape_hint)],
    |builder, input_refs| {
        let result = build_einsum_graph_dim_expr(builder, tree, input_refs, &input_shapes)
            .map_err(|err| Error::ContractionError(err.to_string()))?;
        let local = match result {
            ValueRef::Local(local) => local,
            ValueRef::External(_) => {
                return Err(Error::Internal(
                    "expanded einsum returned external value".to_string(),
                ))
            }
        };
        Ok(vec![local])
    },
)?;
```

Return the single output.

- [ ] **Step 3: Wire concrete static-tree expansion first**

In `einsum_subscripts_with()`:

- keep direct binary dot first;
- compute `plan_spec` and `static_tree` as today;
- if `static_tree.is_some()`, call `expand_traced_einsum_graph()` and return it;
- only build `EinsumExtensionOp` when no fixed tree is available.

This removes `ExecOp::Extension(EinsumExtensionOp)` for concrete traced N-ary Auto cases.

- [ ] **Step 4: Add concrete traced test**

Test:

```rust
#[test]
fn concrete_traced_nary_einsum_expands_to_standard_graph() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
    let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]);
    let mut compiler = GraphCompiler::new();
    let out = einsum(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
    let program = compiler.compile(&out).unwrap();

    assert!(
        program.exec.instructions.iter().all(|inst| {
            !matches!(inst.op, ExecOp::Extension(_))
        })
    );
    assert!(
        program.exec.instructions.iter().any(|inst| {
            matches!(inst.op, ExecOp::DotGeneral(_) | ExecOp::DotGeneralWithConj { .. })
        })
    );
}
```

Use public accessors if `program.exec` is private; otherwise place the test where crate-private fields are visible.

- [ ] **Step 5: Add symbolic Auto fallback test**

Build symbolic traced inputs with known ranks and call default Auto. Assert the compiled program still contains one `ExecOp::Extension(_)`.

This protects the rule that runtime-shape-dependent Auto remains an extension.

- [ ] **Step 6: Verify**

Run:

```bash
cargo test -p tenferro-einsum --lib traced
cargo test -p tenferro-runtime --lib apply_expanded_graph
```

Expected: PASS.

## Task 8: Expand Explicit Fixed-Path Symbolic Einsum

**Files:**

- Modify: `tenferro-einsum/src/traced.rs`
- Modify: `tenferro-einsum/src/planning/tree.rs` if path-to-tree construction currently requires concrete sizes
- Test: traced symbolic einsum tests

- [ ] **Step 1: Add path-to-tree resolution that does not need concrete extents**

Add a helper that builds a `ContractionTree` from:

- `EinsumSubscripts`
- input ranks
- explicit `EinsumOptimize::Path` pairs or `EinsumOptimize::Tree`

It must validate:

- pair indices are valid for the shrinking operand list;
- repeated labels are internally consistent by rank;
- output labels appear in at least one input;
- label dimensions are not proven inconsistent when both sides are concrete.

It must not run cost-based search.

- [ ] **Step 2: Wire symbolic fixed-path expansion**

In `einsum_subscripts_with()`:

- `EinsumOptimize::Path(_)`: resolve a fixed tree without concrete sizes and call `expand_traced_einsum_graph()`;
- `EinsumOptimize::Tree(_)`: use the supplied tree directly when ranks/subscripts match;
- `EinsumOptimize::Auto(_)`: remain unchanged for symbolic shapes.

- [ ] **Step 3: Add symbolic fixed-path test**

Test:

```rust
#[test]
fn symbolic_path_einsum_expands_to_standard_graph() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let c = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut compiler = GraphCompiler::new();
    let out = einsum_with(
        &mut compiler,
        &[&a, &b, &c],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
    )
    .unwrap();
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[
                (&a, DType::F64, &[2, 3]),
                (&b, DType::F64, &[3, 4]),
                (&c, DType::F64, &[4, 5]),
            ],
        )
        .unwrap();

    assert!(
        program.exec.instructions.iter().all(|inst| {
            !matches!(inst.op, ExecOp::Extension(_))
        })
    );
}
```

- [ ] **Step 4: Verify**

Run:

```bash
cargo test -p tenferro-einsum --lib symbolic_path_einsum_expands_to_standard_graph
```

Expected: PASS.

## Task 9: Expand Fixed-Path EagerTensor Recording

**Files:**

- Modify: `tenferro-einsum/src/eager_tensor.rs`
- Modify: `tenferro-ad/src/extension.rs` only if a reusable eager standard-op recording helper is needed
- Test: `tenferro-einsum/src/eager_tensor.rs` tests

- [ ] **Step 1: Keep concrete non-AD execution simple**

For non-AD eager tensors, it is acceptable to continue using the concrete eager executor or the existing binary direct dot path. The target of this task is AD recording: recorded eager graphs should not contain `StdTensorOp::Extension` when the path is fixed.

- [ ] **Step 2: Add eager standard-op expansion helper**

If `EagerTensor` lacks public methods needed by the einsum lowering, add a small helper in `tenferro-ad`:

```rust
pub fn apply_eager_standard_graph(
    inputs: &[&EagerTensor],
    build: impl FnOnce(&mut EagerGraphBuilder, &[EagerValueRef]) -> Result<Vec<EagerValueRef>>,
) -> Result<Vec<EagerTensor>>
```

If that is too large, defer this task and document that traced graph expansion is the first implementation target.

- [ ] **Step 3: Add eager fixed-path test**

For a requires-grad eager N-ary einsum with explicit path, assert the recorded graph does not contain `StdTensorOp::Extension`.

- [ ] **Step 4: Verify**

Run:

```bash
cargo test -p tenferro-einsum --lib eager_tensor
```

Expected: PASS if implemented; otherwise this task must remain unchecked with the deferral documented in the commit message.

## Task 10: Standardize Hot Metadata on SmallVec

**Files:**

- Modify: `tenferro-einsum/src/builder.rs`
- Modify: `tenferro-einsum/src/extension.rs`
- Modify: `tenferro-runtime/src/compiler/mod.rs`
- Modify: files moved under `tenferro-runtime/src/compiler/optimizer/**` if Task 11 has already run
- Test: existing compiler/einsum tests

- [ ] **Step 1: Add local aliases**

In `tenferro-einsum/src/builder.rs`:

```rust
use smallvec::SmallVec;

type AxisVec = SmallVec<[usize; 4]>;
type LabelVec = SmallVec<[u32; 8]>;
type ShapeExprVec = SmallVec<[DimExpr; 4]>;
```

In compiler pass modules:

```rust
use smallvec::SmallVec;

type AxisVec = SmallVec<[usize; 4]>;
```

In `tenferro-einsum/src/extension.rs`:

```rust
use smallvec::SmallVec;

type InputIndexVec = SmallVec<[usize; 8]>;
```

Pick inline capacities from the common einsum cases:

- rank <= 4 for most axes/permutations;
- input count <= 8 for most benchmark equations;
- labels <= 8 for common tensor-network binary steps.

- [ ] **Step 2: Convert builder hot vectors**

Change internal builder fields:

```rust
struct LabeledVal {
    val: ValueRef<StdTensorOp>,
    labels: LabelVec,
    shape: ShapeExprVec,
}
```

Use `AxisVec` for:

- `reduce_axes`
- `perm`
- `lhs_contracting_dims`
- `rhs_contracting_dims`
- `lhs_batch_dims`
- `rhs_batch_dims`
- broadcast `dims`

Convert to `Vec` only at `StdTensorOp` and `DotGeneralConfig` construction:

```rust
StdTensorOp::Transpose {
    perm: perm.into_vec(),
}

DotGeneralConfig {
    lhs_contracting_dims: lhs_contracting_dims.into_vec(),
    rhs_contracting_dims: rhs_contracting_dims.into_vec(),
    lhs_batch_dims: lhs_batch_dims.into_vec(),
    rhs_batch_dims: rhs_batch_dims.into_vec(),
}
```

- [ ] **Step 3: Convert cached runtime input indices**

Change:

```rust
input_indices: Vec<usize>
```

to:

```rust
input_indices: InputIndexVec
```

in `CachedRuntimeExecProgram`.

Update retained-byte accounting:

```rust
fn smallvec_retained_bytes<A: smallvec::Array>(values: &SmallVec<A>) -> usize {
    if values.spilled() {
        values.capacity() * std::mem::size_of::<A::Item>()
    } else {
        0
    }
}
```

Use it for `input_indices`; do not count inline storage as retained heap bytes.

- [ ] **Step 4: Convert optimizer temporary axis vectors**

Use `AxisVec` in:

- `free_axes`
- `map_axes`
- `apply_perm`
- layout-chain folding temporary axes
- dot decomposer merge/free/contracting axis groups where those are rank-sized

Keep function return types as `Vec<usize>` only when public tests or downstream callers depend on that exact type. Otherwise return `AxisVec`.

- [ ] **Step 5: Add allocation guard test where practical**

Do not add a global allocator test. It will be brittle. Instead, add focused unit assertions for inline storage:

```rust
#[test]
fn input_index_vec_stays_inline_for_common_input_count() {
    let mut indices: InputIndexVec = SmallVec::new();
    indices.extend(0..4);
    assert!(!indices.spilled());
}
```

Add similar tests only for aliases defined in test-visible modules.

- [ ] **Step 6: Verify**

Run:

```bash
cargo test -p tenferro-einsum --lib
cargo test -p tenferro-ad --test compiler_passes
cargo test -p tenferro-runtime --lib compiler
```

Expected: PASS.

## Task 11: Physically Split Optimizer Files

**Files:**

- Modify: `tenferro-runtime/src/compiler/mod.rs`
- Create: `tenferro-runtime/src/compiler/lowering.rs`
- Create: `tenferro-runtime/src/compiler/optimizer/algebraic_layout.rs`
- Create: `tenferro-runtime/src/compiler/optimizer/conj_sinking.rs`
- Create: `tenferro-runtime/src/compiler/optimizer/dot_dimension_sorter.rs`
- Create: `tenferro-runtime/src/compiler/optimizer/transpose_folding.rs`
- Move or keep: `tenferro-runtime/src/compiler/dot_decomposer.rs`
- Create: `tenferro-runtime/src/compiler/optimizer/dot_conj_folding.rs`
- Create: `tenferro-runtime/src/compiler/optimizer/dce.rs`
- Test: existing compiler pass tests

- [ ] **Step 1: Extract lowering**

Move the body that converts `CompiledProgram<StdTensorOp>` to `ExecProgram` into:

```rust
pub(super) fn lower_std_to_exec(
    prog: &CompiledProgram<StdTensorOp>,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
) -> ExecProgram
```

Keep shape helpers that are used by both lowering and `dot_decomposer` visible as `pub(super)`.

- [ ] **Step 2: Move pass implementations one file at a time**

Move in this order, running tests after each move:

1. `algebraic_layout_simplifier`
2. `transpose_folding` and `layout_chain_transpose_folding`
3. `dot_conj_folding`
4. `conj_sinking`
5. `dot_dimension_sorter`
6. `eliminate_dead_code` and `populate_last_use`
7. `dot_decomposer` under `optimizer/dot_decomposer.rs` if import churn is acceptable

Keep re-exports compatible:

```rust
pub use optimizer::{
    algebraic_layout_simplifier,
    dot_decomposer,
    eliminate_dead_code,
    transpose_folding,
};
```

This preserves existing tests that import pass functions from `tenferro_runtime::compiler`.

- [ ] **Step 3: Verify**

Run:

```bash
cargo test -p tenferro-ad --test compiler_passes
cargo test -p tenferro-runtime --lib compiler
```

Expected: PASS.

## Task 12: Add Extension Runtime Split Borrow API

**Files:**

- Modify: `tenferro-runtime/src/extension_runtime.rs`
- Test: `tenferro-runtime/src/extension_runtime.rs` doctest or unit test

- [ ] **Step 1: Add API**

Add:

```rust
pub fn parts_mut(&mut self) -> (&mut B, &mut ExtensionCacheStore) {
    (self.backend, self.caches)
}
```

- [ ] **Step 2: Verify API compiles**

Run:

```bash
cargo test -p tenferro-runtime --lib extension_runtime
```

Expected: PASS.

## Task 13: Cache Inner Backend Runtime State for EinsumExtensionOp

**Files:**

- Modify: `tenferro-einsum/src/extension.rs`
- Modify: `tenferro-runtime/src/segment.rs` only if a helper needs visibility
- Modify: `tenferro-runtime/src/exec.rs` only if a helper needs visibility
- Test: `tenferro-einsum/src/extension.rs` tests or `tenferro-einsum/src/eager/tests.rs`

- [ ] **Step 1: Replace immutable cache entry**

Change:

```rust
struct CachedRuntimeExecProgram {
    program: ExecProgram,
    input_indices: Vec<usize>,
}
```

to:

```rust
struct CachedRuntimeExecProgram<C> {
    program: ExecProgram,
    input_indices: Vec<usize>,
    optimizer_fingerprint: u64,
    backend_cache: C,
}
```

Do not add slot workspace in this step. `B::RuntimeCache` is the measurable cache and avoids the lifetime problem around borrowed `ExecSlot<'a>`.

- [ ] **Step 2: Include optimizer fingerprint in cache key**

In `cached_runtime_exec_program`, compute:

```rust
let compiler_options = tenferro_runtime::compiler::CompilerOptions::default();
let optimizer_fingerprint = compiler_options.optimizer.fingerprint();
let key_data = (
    op.subscripts().clone(),
    shapes.to_vec(),
    input_dtypes.clone(),
    plan_hasher.finish(),
    optimizer_fingerprint,
);
```

Then pass `compiler_options` and `optimizer_fingerprint` into `build_runtime_exec_program()`.

- [ ] **Step 3: Use mutable cache entry**

Change the cache lookup from `get::<Arc<CachedRuntimeExecProgram>>()` to `get_mut::<CachedRuntimeExecProgram<B::RuntimeCache>>()`.

Pseudo-flow:

```rust
let key = runtime_exec_program_key(...);
let (backend, caches) = ctx.parts_mut();
if !caches.contains_typed::<CachedRuntimeExecProgram<B::RuntimeCache>>(&key) {
    let entry = build_runtime_exec_program::<B>(tree, inputs, shapes, compiler_options)?;
    caches.put(key, entry, retained_bytes);
}
let entry = caches
    .get_mut::<CachedRuntimeExecProgram<B::RuntimeCache>>(&key)
    .expect("entry inserted above");
```

If `ExtensionCacheStore` does not have `contains_typed`, add a small helper or use `get_mut().is_none()` followed by `put()` and a second `get_mut()`.

- [ ] **Step 4: Execute with cached backend cache**

Use segmented execution with an external backend cache:

```rust
let program_inputs = runtime_program_inputs(inputs, entry.input_indices.as_slice())?;
let mut slot_workspace = Vec::new();
let mut outputs = tenferro_runtime::segment::eval_exec_segmented_with_cache_and_workspace(
    backend,
    &entry.program,
    program_inputs,
    &mut slot_workspace,
    &mut entry.backend_cache,
    None,
)?;
```

If `segment` helpers are not visible to `tenferro-einsum`, add a public runtime helper with a narrow name, for example:

```rust
pub fn eval_exec_ir_with_backend_cache<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    backend_cache: &mut B::RuntimeCache,
) -> Result<Vec<Tensor>>
```

Do not make broad internal dispatch APIs public.

- [ ] **Step 5: Add cache stats retained bytes**

Update `cached_runtime_exec_program_retained_bytes()` to include:

```rust
cached.backend_cache.stats().retained_bytes
```

Do not depend on exact retained bytes in tests; backend caches can evolve.

- [ ] **Step 6: Add cache reuse test**

Add a backend test that runs the same traced or eager extension twice and checks extension cache entries remain stable while backend cache entries become nonzero for a dot-containing N-ary einsum.

Expected properties:

- first run inserts runtime exec program cache entry;
- second run reuses the same entry;
- backend cache stats inside the entry are nonzero for CPU BLAS-backed dot workloads.

If direct inspection of inner backend cache is not exposed, test through public executor stats only after adding aggregation in Task 14.

- [ ] **Step 7: Verify**

Run:

```bash
cargo test -p tenferro-einsum --lib
cargo test -p tenferro-runtime --lib extension_runtime
```

Expected: PASS.

## Task 14: Expose Extension Cache Stats Enough for Benchmark Debugging

**Files:**

- Modify: `tenferro-runtime/src/extension_cache.rs`
- Modify: `tenferro-runtime/src/extension_runtime.rs`
- Modify: `tenferro-einsum/src/extension.rs` only for retained-byte calculations
- Test: `tenferro-runtime/src/extension_cache.rs`

- [ ] **Step 1: Keep public stats coarse**

Do not expose type-erased cache internals. Existing `ExtensionExecutor::cache_stats()` is enough for entry count and retained bytes.

If backend inner cache stats must be visible, add a family-specific debug method in `tenferro-einsum` test support rather than a public runtime API.

- [ ] **Step 2: Verify cache clear clears inner backend cache**

Add a test:

```rust
#[test]
fn extension_executor_clear_caches_drops_inner_backend_cache_entries() {
    let mut executor = ExtensionExecutor::<CpuBackend>::new();
    // Register einsum runtime, run a dot-containing extension twice.
    assert!(executor.cache_stats().entries > 0);
    executor.clear_caches();
    assert_eq!(executor.cache_stats().entries, 0);
}
```

Use existing helper constructors from `tenferro-einsum` tests; do not add public API solely for this test.

## Task 15: Benchmark and Decide DotDecomposer Default

**Files:**

- Modify: `tenferro-benchmark` only if an optimizer option flag is needed
- Read: `tenferro-benchmark/result/cpu/einsum.md`
- Test/benchmark: selected CPU einsum benchmark instances

- [ ] **Step 1: Add benchmark knob if needed**

If `tenferro-benchmark` cannot choose compiler optimizer options, add an env var:

```text
TENFERRO_OPT_DOT_DECOMPOSER=0|1
```

This must affect only traced/eager paths that compile inner graphs. It should not silently change backend provider or thread variables.

- [ ] **Step 2: Run targeted 1-thread benchmark**

Use fixed Accelerate and one thread:

```bash
BENCH_INSTANCE=str_nw_mera_open_26 \
TENFERRO_MODE=trace \
TENFERRO_CPU_BACKEND_KIND=blas \
RAYON_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
VECLIB_NUM_THREADS=1 \
cargo run --release --no-default-features --features system-accelerate --bin tenferro-einsum-benchmark
```

Repeat for:

- `str_nw_mera_open_26`
- `gm_queen5_5_3.wcsp`
- `bin_batched_matmul_b32_m128_n128_k128`
- `bin_matmul_1024`

Run each with `TENFERRO_OPT_DOT_DECOMPOSER=0` and `TENFERRO_OPT_DOT_DECOMPOSER=1`.

- [ ] **Step 3: Decision rule**

Keep `dot_decomposer` default-off unless:

- it improves at least two N-ary/network benchmarks by more than 5%;
- it does not regress `bin_matmul_1024` or batched matmul by more than 2%;
- compile time increase is below 5% for traced mode.

If the rule passes, bump `OptimizerConfig::default().dot_decomposer` to `true`, increment `OptimizerConfig::VERSION`, and rerun Task 4 tests.

## Task 16: Full Verification

**Files:**

- All modified files

- [ ] **Step 1: Rust tests**

Run:

```bash
cargo test -p tenferro-runtime --lib
cargo test -p tenferro-einsum --lib
cargo test -p tenferro-ad --test compiler_passes
```

Expected: PASS.

- [ ] **Step 2: Formatting and diff check**

Run:

```bash
cargo fmt --check
git diff --check
```

Expected: both pass with no output from `git diff --check`.

- [ ] **Step 3: Benchmark summary**

Run the targeted benchmark matrix from Task 15 and record:

- tenferro eager/trace time;
- compile time for trace;
- PyTorch/JAX reference only if comparing against previous result files;
- optimizer config fingerprint;
- tenferro-rs commit hash.

Do not include tenferro-benchmark commit hash in the result payload. Git already records it, and benchmark runs often happen from dirty benchmark worktrees.

## Expected Outcome

After this plan:

- The optimizer is a real module rather than a long tail inside `compiler/mod.rs`.
- Low-risk layout simplifications are default-on.
- `dot_decomposer` is available for controlled benchmarking without changing default semantics prematurely.
- Inner einsum extension execution reuses backend analysis cache across repeated runtime calls with the same subscripts, shapes, dtypes, plan, and optimizer fingerprint.
- Remaining N-ary performance work is easier to isolate because inner graph rebuild/compile/backend analysis overhead is separated from actual kernel/runtime layout cost.

## Follow-Up Issues

Track separately:

1. Expand fixed-path eager/traced einsum into the outer graph when the contraction tree is already known.
2. Let extensions participate in graph optimization so an `EinsumExtensionOp` with fixed path can expose `DotGeneral`, `Reduce`, `Transpose`, and `Mul` to the common optimizer.
3. Add contraction-tree-wide layout optimization instead of only local transpose folding.
4. Design a lifetime-neutral reusable workspace for borrowed `ExecSlot<'a>` execution. This is needed before clone-free inner extension execution can also reuse slot workspace cleanly.
