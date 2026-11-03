# TracedTensor as a typed operation layer over TraceContext

Status: proposed design for [#2417]. No implementation before maintainer review.

All counts, line numbers and sizes below were taken from the checkout at
`bdde3f2b6e82f0c9991f4170a642ad80dca7b707` with the commands quoted next to
them. Counts measured over `*.rs` restricted to `crates/**/src` are marked
"production"; the others include tests and rustdoc.

## 1. Scope

tenferro has two public ways to build a semantic graph:

| entry point | representation | op surface |
|---|---|---|
| `TracedTensor` | one `Arc<Graph<StdTensorOp>>` per tensor plus five merge/scope carriers | 75 inherent `pub fn` in `crates/tenferro-runtime/src/traced.rs`, 13 composites in `traced/composite_ops.rs`, the extension traits of §2.4, and `TracedTensorAdExt` |
| `TraceContext` | one `SemanticProgramBuilder` plus `ExtensionCacheStore` (`crates/tenferro-runtime/src/trace.rs:19-24`) | `input`, `input_with_default`, `bind_input`, `add_op`, `add_extension`, `value_metadata`, `finish` (`trace.rs:39,52,70,87,106,121,141`) plus `TraceContextEinsumExt` |

The two describe the same object. `TracedTensor` builds many small graphs and
asks `GraphCompiler` to merge them; `TraceContext` appends to one program.

This document specifies the target design in which `TracedTensor` is a typed
view over `TraceContext`, and records what that removes, what it breaks, and in
which order it can land.

## 2. Current design

### 2.1 TracedTensor fields and their users

`TracedTensor` (`crates/tenferro-runtime/src/traced.rs:54-75`):

```text
id, rank, dtype, val                  identity and cached result metadata
graph: Arc<Graph<StdTensorOp>>        the tensor's own op fragment
data: Option<Arc<RetainedValue>>      attached constant / checkpoint value
shape_hint: Option<Vec<SymDim>>       symbolic extents of this value
inputs_map: Arc<RetainedInputMap>     bound leaf data, carried per tensor
leaf_metas: Arc<HashMap<.., TensorMeta>>  construction-time leaf metadata
extra_roots: Vec<Arc<Graph<..>>>      additional graph roots to compile
checkpoint_chain: Option<Arc<CheckpointNode>>  checkpointed pre-images
metadata_scopes: MetadataScopeChain   registered metadata of live graphs
constraint_scopes: ConstraintScopeChain  extension shape constraints
```

Production occurrences of the five carriers, over the 366 production `*.rs`
files (`rg -c` summed per file, test modules excluded by path):

| field | production occurrences | definition / accessors |
|---|---|---|
| `extra_roots` | 51 | `traced.rs:71`, `ad_support.rs:304`, `traced.rs:3487` (`resolve_roots`) |
| `inputs_map` | 45 | `traced.rs:62`, `merge_traced_inputs_map` `traced.rs:90`, `ad_support.rs:294` |
| `leaf_metas` | 41 | `traced.rs:70`, `merge_traced_leaf_metas` `traced.rs:133`, `ad_support.rs:300` |
| `checkpoint_chain` | 47 | `traced.rs:72`, `checkpoint.rs:147` |
| `metadata_scopes` | 50 | `traced.rs:73`, `metadata.rs:25` |
| `constraint_scopes` | 48 | `traced.rs:74`, `ad_support.rs:51` |

`extra_roots` is never populated. Every occurrence either initialises it empty,
clones or extends it from another tensor's clone, clears it
(`ad_support.rs:528`), or drains it in `resolve_roots`. Searching for any
population site returns nothing:

```console
$ rg -n 'extra_roots' --glob '!docs/**' | rg 'push|= vec!|extend\(vec!|from\(\['   # exit 1
$ rg -n 'extra_roots' --glob '!docs/**' | wc -l
41
```

It was presumably the hook for a graph-carrying representation that no longer
uses it. It contributes 41 lines of code and no behaviour.

### 2.2 State that exists only for the per-tensor representation

**Merge and root resolution.** `GraphCompiler` has two entry points:
`compile_traced_graph` / `compile_frozen_program` (`compiler.rs:188,225`),
which lower an already-frozen program, and `compile` / `compile_many` /
`compile_ad_source_many`, which accept `&TracedTensor` and merge their graphs.
The merge path is:

| region | lines | role |
|---|---|---|
| `InputDescriptor`, `InputExtentIdentity` | 29-80 | input shapes derived from merged graph inputs |
| `compile_many_with_descriptors` | 500-670 | collect roots, `resolve`, `materialize_merge`, slot maps |
| `compile_materialized_semantic_program` | 946-1170 | re-add every materialized instruction to a builder |
| `collect_default_inputs` | 1180-1200 | bindings from per-tensor maps |
| `insert_checkpoint_alias`, `AliasTracker` | 1200-1377 | checkpoint aliasing |
| `materialize_merge_with_input_aliases` | 1377-1399 | merge with alias substitution |
| placeholder validation | 1399-1445 | `TracedTensor` arguments of `compile_with_input_specs` |
| `descriptor_for_input`, `default_input_extent_identity` | 1445-1532 | binding/default reconciliation |
| `default_tensors_equivalent` | 1599-1660 | conflict detection between per-tensor bindings |

Line spans sum to 803 of the file's 1992 lines (span arithmetic, not a diff
size). All of it exists because the graph being compiled is a merge of many
per-tensor fragments, and because bindings live on tensors instead of on one
program.

**Scope chains.** `metadata_scopes` and `constraint_scopes` record, per tensor,
the metadata and shape constraints registered for the graph fragments that
produced it. They are merged per op (`traced.rs:3084-3190, 3381-3458`), carried
through AD (`ad_support.rs:51` `ConstraintScopeTransfer`), and replayed into
the program by the compiler (`compiler.rs:509-570` for constraints). The
builder does the same work inline when the op is added:

* `SemanticProgramBuilder::add_extension` infers metadata *and* shape guards
  and stores the guards on the operation (`program/builder.rs:338`, inference
  at `builder.rs:432-478`).
* `SemanticProgramBuilder::add_op` infers metadata for core ops
  (`builder.rs:251-273`); core ops generate no shape constraints in either
  representation (`shape_infer::infer_output_shapes`, `shape_infer.rs:250`,
  returns shapes only).

So the scope chains and the analysis pass they exist for
(`ad_support.rs:372` `analyze_deferred_semantic_trace`, 60 lines) duplicate
work the builder already performs when an op is appended to one program.

**`inputs_map` / `leaf_metas`.** Both exist because bindings and leaf metadata
are per tensor. `ProgramBindings` already holds frozen bindings, and the
builder binds at creation time (`bind_input`, `input_with_default`,
`bind_input_retained` `program/builder.rs:63,78`). Bindings are excluded from
semantic identity (`compiler.rs:190-200`), which is exactly the split
`leaf_metas` reimplements by hand: symbolic extents in identity, concrete data
as bindings (`traced.rs:66-70`, `ad_support.rs:390-420`).

**`checkpoint_chain`.** `checkpoint_tensor` (`ad_support.rs:500-548`) replaces
a tensor's graph with a fresh bound leaf, keeps the old graph as an extra root,
and records `alias_key -> alias_target` (`checkpoint.rs:147-205`). The AD
compile then merges the old graphs back and substitutes the alias
(`compiler.rs:526-560`, `1399-1450`).

### 2.3 Eager semantic trace and the AD round trip

`EagerTensor::semantic_trace: Option<TracedTensor>` (`tenferro-ad/src/eager.rs:4387`,
also on `EagerTensorRecord` 4398 and `EagerTensorParts` 4409). Per op:

1. `record_semantic_eager_outputs` (`eager.rs:5528`, called from the record
   helpers at 5524/5631) promotes/casts/reshapes the
   semantic inputs to mirror eager execution, then appends **one raw graph
   fragment** with `append_raw_eager_outputs` (`runtime/extension.rs:205`). No
   metadata inference, no constraint inference: the fragment is built from the
   parent fragments, `merge_traced_inputs_map`, `merge_traced_leaf_metas`,
   `MetadataScopeChain::merge`, and an `extra_roots` union.
2. At the first AD request, `analyze_deferred_semantic_trace`
   (`ad_support.rs:372`) walks the whole parent chain post-order, registers
   metadata and constraints for every graph, seeds leaf metadata from
   `inputs_map` + `leaf_metas`, and returns a new `TracedTensor` carrying the
   analysis (`eager.rs:3940`).
3. `compile_ad_source_many` merges the analyzed trace plus the saved residual
   traces into one program (`eager.rs:3953`).
4. The program-level VJP runs (`AdContext::vjp_program`, `eager.rs:4015`), the
   derivative program is compiled, and each result is converted **back** into a
   `TracedTensor` by `derivative_trace_from_frozen_program`
   (`tenferro-ad/src/traced.rs:905`): rebuild a `Graph<StdTensorOp>` op by op
   (`traced.rs:979-1050`), re-register analysis, re-merge input maps, leaf
   metas, metadata and constraint scopes (`traced.rs:1080-1160`, leaf metas at
   1122), and construct `TracedTensorParts` (`ad_support.rs:202`).
5. The same round trip exists for the non-eager traced path:
   `TracedTensorAdExt::grad/jvp/vjp` compile the tensor
   (`tenferro-ad/src/traced.rs:613,706`), transform, and convert back with
   `derivative_tensors_from_program` (`traced.rs:885`).

Residuals for backward are `SavedValue { trace: TracedTensor, value: .. }`
(`tenferro-ad/src/eager/residuals.rs:26-30`, filled at 100-130);
`EagerTrace::collect` reads `value.trace.graph().values()[value.trace.val].key`
(`residuals.rs:52`), i.e. residual identity is a `ValueKey<StdTensorOp>` of a
graph fragment.

`EagerTensor: Send + Sync` is already asserted inside tenferro-ad:
`crates/tenferro-ad/tests/integration/eager_tensor.rs:1431-1435`. The downstream
assertion in tensor4all-rs (`crates/tensor4all-core/tests/send_sync.rs`) is the
same property, so the invariant already has a local guard; a live
`Rc<RefCell<_>>` handle in eager would break it immediately.

### 2.4 Extension surface per family

| family | traced surface | implementation |
|---|---|---|
| linalg | `TracedTensorLinalgExt`, 25 methods (`linalg/src/traced.rs:17-411`, impl at 412) | 19 `apply(` sites in the same file construct `LinalgExtensionOp` and call `tenferro_runtime::extension::apply`; `pinv`, `norm`, `slogdet` and friends are composites over `TracedTensor` standard ops (helpers at `traced.rs:1700-2147`) |
| fft | `TracedTensorFftExt`, 8 methods (`fft/src/lib.rs:638-930`) | one `apply` per method |
| einsum | `TraceContextEinsumExt`, 6 methods (`einsum/src/traced.rs:50-149`); `TracedTensorEinsumExt`, `tensordot` only (`traced.rs:313-340`) | the context trait is the real implementation; the traced one delegates |
| ext/sparse | `SparseCooTracedTensor` holds `TracedTensor` values (`ext/sparse/src/sparse.rs:135,227`) | `extension::apply` |
| ext/tropical | 7 free functions taking `&TracedTensor` (`ext/tropical/src/traced.rs:150,206,266,329,397,453,642`) | extension ops plus traced standard ops |
| core composites | 13 methods on `TracedTensor` in `traced/composite_ops.rs` (`sigmoid`, `silu`, `softplus`, `gelu`, `gelu_tanh`, `reduce_mean`, `softmax`, `log_softmax`, `masked_softmax`, `masked_log_softmax`, `layer_norm`, `rms_norm`, `take_along_axis`) | standard ops only |

What is tied to `TracedTensor` is therefore the *wrapper*: op construction and
composites. Op semantics (`ExtensionOp` payloads), AD rules
(`SemanticLinearizeRule`, `SemanticPrimalVjpRule`, written against
`SemanticProgramBuilder`) and kernels are independent of it.

The missing piece on the context side is a typed standard-op surface:
`tenferro-einsum` is the only extension with a `TraceContext` entry point, and
composites (`pinv`, `norm`, `softmax`, ...) cannot be written on `TraceContext`
today because it has no `matmul`, `reduce_max`, `transpose`, `compare`, ...

### 2.5 Downstream

**tensor4all-rs**, `origin/main` = `22621f25e6357ebf147a062060f252162743579c`:

| symbol | files |
|---|---|
| `EagerTensor` | 13 |
| `TraceContext` | 3 (`tensor4all-tensorbackend/src/{context.rs,tenferro_bridge.rs,context/tests/mod.rs}`) |
| `TracedGraph` | 1 |
| `TracedTensor`, `TraceValue`, `ProgramBindings` | 0 |

Graph construction there uses `TraceContext` + `TraceContextEinsumExt` +
`GraphCompiler::compile_traced_graph`; AD uses `EagerTensor` with
`backward()`/`vjp_from_output`-style calls; no data-attached traced leaves.

**yao-rs**, `4a398fe0a641b98792f8d3c28c56c2b79e84cedb`, pins `tenferro* = "=0.4.0"`:

| symbol | occurrences |
|---|---|
| `EagerTensor` | 81 |
| `TracedTensor` | 13 |
| `GraphCompiler` | 11 |
| `TraceContext` | 5 |
| `TraceValue` | 3 |
| `compile_traced_graph` | 2 |

Its contraction path is `TraceContext::new` → `einsum_subscripts_with` →
`trace.finish(&[out])` → `compile_traced_graph` (`src/tenferro.rs:128,143,236`).
`TracedTensor` appears only in `circuit_apply` (`src/tenferro_ad.rs:268-282`)
and its tests, with `circuit_apply_eager` (`:344`) offering the same operation
for `EagerTensor`; the AD rules are written against
`SemanticProgramBuilder`/`RuleSet`.

Consequence: everything downstream uses `TraceContext`, `EagerTensor` and the
frozen program. `TracedTensor` is consumed only by tenferro's own docs, tests,
samples and the extension crates.

## 3. Target design

### 3.1 One tracing handle

```rust
#[derive(Clone)]
pub struct TraceContext { inner: Rc<RefCell<TraceInner>> }   // !Send, !Sync

pub struct TracedTensor { ctx: TraceContext, value: TraceValue }  // !Send, cheap to clone
```

* `TraceInner` holds the `SemanticProgramBuilder`, the `ExtensionCacheStore`,
  the context-level scopes of §3.4 and the checkpoint alias table of §3.5.
* `TraceContext` methods take `&self`. A value carries its owner nonce, so a
  foreign value is already a typed `ProgramBuildError::ForeignValue`
  (`program/builder.rs:100-115`), and a foreign `TracedTensor` becomes a typed
  error where today it silently merges two graphs.
* Re-entrant use borrows with `try_borrow_mut` and maps a failure to a typed
  error; no public path may panic (REPOSITORY_RULES "Public Boundary Safety
  Audits").
* `finish(&self, outputs)` freezes a snapshot of the ordered closure of
  `outputs` and leaves the context usable. It is not a new representation: the
  snapshot is a `FrozenProgram` produced by the same builder code.

`TraceValue` is `Copy` (`trace.rs:155`), so `TracedTensor` stays a two-word
value plus a handle clone.

`TracedTensor` reads `rank`/`dtype`/symbolic shape from
`TraceContext::value_metadata` (`trace.rs:121`), which returns
`ProgramValueMetadata` (`program/metadata.rs:72-150`). `id`, `rank`, `dtype`,
`val` and `graph()` stop being public (#1422); `dtype()`/`rank`-style accessors
stay as methods.

### 3.2 Typed standard operation layer

`CoreSemanticOp` is a complete image of the non-extension `StdTensorOp`:
`TryFrom<&StdTensorOp> for CoreSemanticOp` (`program/op.rs:150-246`) covers all
54 core variants and rejects only `StdTensorOp::Extension(_)`. The typed layer
is therefore a mechanical mapping:

```rust
impl<T: TensorOpSpec> TraceContext {
    fn apply_std(&self, spec: T, inputs: &[TraceValue]) -> Result<..> // -> builder.add_op
}
```

Each `TracedTensor` method becomes: build the op payload from its arguments,
`add_op` (or `add_extension`), wrap the returned `TraceValue`. Argument
validation moves into the shared pre-checks that already exist in the compiler
and shape inference (`shape_infer`, `tenferro_ops`), not into per-op graph
assembly. The 75 inherent methods of `traced.rs` and the 13 composites keep
their names and signatures; only their bodies change. Composites are rewritten
once against `(&TraceContext, &[TraceValue])` and called by the
`TracedTensor` methods.

### 3.3 Extension layer

One implementation per family, on the context:

```rust
trait TraceContextLinalgExt { fn svd(&self, a: &TraceValue) -> Result<..>; .. }
trait TraceContextFftExt    { fn fft(&self, x: &TraceValue, ..) -> Result<..>; .. }
trait TraceContextEinsumExt { /* existing */ }
```

`TracedTensorLinalgExt` and `TracedTensorFftExt` remain as the method-chaining
surface and delegate in one line. That removes the duplicated implementation
(19 `apply` sites in linalg, 8 in fft, the sparse and tropical wrappers) while
keeping both entry points, which is what "written once against the context and
value pair" requires. `ext/sparse`'s `SparseCooTracedTensor` stores
`(TraceContext, TraceValue)` instead of `TracedTensor`.

### 3.4 Scopes and metadata

Delete the per-tensor scope chains, `MetadataScopeChain`
(`metadata.rs:25-155`), `ConstraintScopeChain`, `ConstraintScopeTransfer` and
`analyze_deferred_semantic_trace`. The builder infers metadata and extension
guards when the op is appended (§2.2), and the frozen program carries them.

### 3.5 Checkpoint

A checkpoint becomes context state instead of a per-tensor chain:

* `checkpoint` evaluates the value once (as today), appends a fresh program
  input bound to the resulting data, and records `input -> value` in a
  context-owned alias table;
* forward snapshots use the bound input, so the pre-checkpoint expression
  becomes dead and is not executed;
* AD snapshots substitute the alias, so the derivative sees the expression and
  the gradient reaches the original leaves.

`CheckpointNode`, `collect_graphs`, `collect_aliases`, `collect_inputs`,
`merge_chains` (`checkpoint.rs:147-227`) and the alias materialization in the
compiler (`compiler.rs:1200-1399`) disappear. Semantics are preserved by the
existing tests (`crates/tenferro-ad/tests/integration/checkpoint.rs`,
`checkpoint_truncate_integration.rs`).

### 3.6 Program snapshot, import and AD in one context

AD on a live context is: snapshot the closure of the differentiated outputs
(non-consuming, §3.1) → program-level VJP/JVP transform → **import** the
derivative program back into the same context so results compose with further
ops.

The current import path cannot do the last step and needs one contract
addition:

* `ProgramImport` (`program/import.rs:6-13`) has only `program`, `bindings`,
  `roots`; `ImportTransaction::prepare` (`program/builder.rs:720-905`) creates
  a **fresh destination input per needed source input** and resolves
  `InputDim` extents against bound concrete shapes (`builder.rs:815-850`).
* A derivative program keeps the source program's input prefix
  (`with_input_prefix_bindings_from`, `program/semantic.rs:672`), so without a
  remap every `grad` would append duplicate inputs for the primal leaves and
  would concretize their symbolic extents.

Target: `ProgramImport` gains `input_map: &[(ProgramValue, ProgramValue)]`
(source input → existing destination value). Mapped inputs reuse the
destination value and metadata unchanged; unmapped inputs are imported as
today. With that, `grad`/`jvp`/`vjp` are one snapshot + one transform + one
import, and `tenferro-ad/src/traced.rs`'s graph reconstruction
(`derivative_tensors_from_program`, 400+ lines including the shape/metadata
plumbing) disappears.

Cost: snapshot and import are O(program) per AD call, not per op. Today's path
is also O(program) per AD call (compile + transform + rebuild). Repeated
`grad` inside one context grows the context; users that loop should take one
snapshot and start a new context per iteration, as they start new tensors
today. This becomes a documented contract and is measured (§8).

### 3.7 GraphCompiler

After the change `GraphCompiler` has one lowering path,
`compile_frozen_program`, plus `compile(&TracedTensor)` implemented as
snapshot + `compile_frozen_program`. Everything in §2.2 disappears, including
`CompiledGraph::input_slots` reordering for merged graphs;
`compile_with_input_specs` is re-expressed against snapshot program inputs.

### 3.8 Eager: immutable journal, materialization per backward

`EagerTensor` must stay `Send + Sync` (§2.3), so it cannot hold a
`Rc<RefCell<_>>` handle. It also cannot freeze a `FrozenProgram` per op:
`finish` consumes the builder (`builder.rs:194`) and `import` copies the program
(`builder.rs:720-905`), so per-op freezing is O(history) per op, i.e. O(n²).

Target: eager keeps an **immutable, private operation journal** — not a graph
representation, but the record of "which op was applied to which recorded
values":

```rust
enum EagerSemantic {
    Op  { op: SemanticOpRef<'_>, inputs: Box<[Arc<EagerSemantic>]>, outputs: usize },
    Program { frozen: Arc<FrozenProgram>, output: usize, inputs: Box<[Arc<EagerSemantic>]> },
}
```

* one `Arc` node per eager op, O(inputs) to create, immutable and `Send + Sync`
  (`ExtensionOp: Debug + Send + Sync + 'static`,
  `tenferro-internal-ops/src/ext_op.rs`);
* the `Program` node carries a frozen program only where one already exists: a
  derivative result, so higher-order AD works without rebuilding graphs;
* metadata is not stored on the journal: outputs already have concrete
  dtypes/shapes from the eager tensors, and symbolic extents are reconstructed
  from rank at replay (`ProgramInputSpec` with `DimExpr::InputDim`), which is
  the split §2.2 says `leaf_metas` reimplements;
* backward replays the journal into a fresh `TraceContext` (one topological
  pass, dedupe by pointer), then runs the same snapshot/transform/import as
  §3.6. One context per backward graph, not per `EagerRuntime`.

This deletes `append_raw_eager_outputs`, `analyze_deferred_semantic_trace`,
the residual `TracedTensor` field, and the graph↔program conversions in eager.
Per-op cost is one allocation plus parent `Arc` clones, against today's graph
fragment plus three map merges.

### 3.9 Public API changes

Removed or changed (breaking):

* `TracedTensor::from_tensor_concrete_shape`, `from_tensor_symbolic_shape`,
  `from_tensor_value_symbolic_shape`, `from_shared_tensor_value_symbolic_shape`,
  `from_vec_col_major`, `input_concrete_shape`, `input_symbolic_shape`,
  `input_concrete_shape_declaring_scalar` — replaced by `TraceContext`
  constructors (`trace.input(dtype, shape)`, `trace.constant(data)`, ...).
  Occurrence counts at HEAD: `from_tensor_concrete_shape` 358,
  `from_vec_col_major` 546, `from_tensor_symbolic_shape` 12,
  `from_shared_tensor_value_symbolic_shape` 5.
* `TracedTensor::graph()`, and the public fields `id`, `rank`, `dtype`, `val`.
* `TraceContext::finish` keeps its name but no longer consumes the context.
* `tenferro_runtime::extension::apply(op, &[&TracedTensor])` becomes a context
  method (yao-rs `circuit_apply` and `ext/sparse` use it).
* `ProgramImport` gains `input_map` (§3.6).
* `ad_support`'s `TracedTensorParts`, `tensor_from_parts`, `inputs_map`,
  `leaf_metas`, `extra_roots`, `checkpoint_chain`, `metadata_scopes`,
  `resolve_roots`, `analyze_deferred_semantic_trace` are removed (hidden but
  `pub`).

Migration size inside tenferro: 258 files mention `TracedTensor`
(112 under `crates/`, 110 under `docs/` of which 102 are Markdown, 29 under
`ext/`, 7 under `samples/`), 3141 occurrences.

## 4. Answers to the open questions

**Q1 `TraceHandle` sharing.** `Rc<RefCell<TraceInner>>`, `TracedTensor: !Send`.
Tracing is single threaded in every caller (§2.5), eager no longer holds a
handle (§3.8), and no `Send`/`Sync` bound is required of `TracedTensor`
anywhere in tenferro or in either downstream project (searched: no
`assert_send`/`Arc<TracedTensor>`/`thread::spawn` use). `Arc<Mutex<_>>` would
buy nothing and would put a lock on the per-op path.

**Q2 one context per what.** Per backward graph. `EagerRuntime` keeps no
context; each backward materializes one and drops it (§3.8).

**Q3 `ProgramBindings` for data-attached leaves.** Keep them. Data leaves are
pervasive (occurrences at HEAD, calls plus rustdoc examples: 358 + 546 + 12 +
5, §3.9), bindings are
already excluded from semantic identity, and `input_with_default`/`bind_input`
already exist on the builder. What disappears is the *per-tensor* `inputs_map`
copy, not bindings.

**Q4 (raised by this design) eager's immutable representation.** §3.8's private
journal. Rejected: `Arc<SemanticProgram>` + value id per tensor (O(n²) freezing
with the current builder), persistent/chunked program core (new core machinery,
not needed for one consumer), `Arc<Mutex<Vec<..>>>` linear tape (retains every
intermediate for every tensor, lock on the per-op path).

**Q5 (raised by this design) import input remap.** §3.6. Without it AD in one
context duplicates primal inputs and loses symbolic extents.

## 5. Migration phases

Each phase is one PR into `main`, green on its own. Refs #2417; the last phase
closes it.

| phase | scope | removed | public API broken | guards |
|---|---|---|---|---|
| P1 eager journal | tenferro-ad only: `EagerSemantic` journal, record one node per op, replay at backward, residual identity by node, `Program` node for derivative results | `append_raw_eager_outputs` (`runtime/extension.rs:205-250`), residual `TracedTensor` (`eager/residuals.rs`), eager's use of `analyze_deferred_semantic_trace` and of `derivative_trace_from_frozen_program` | none | `eager_tensor.rs:1431` Send+Sync assertion (already present) plus a new one on the journal; existing eager AD tests unchanged; new counter test: one node per op, one materialization per backward |
| P2 context core | `TraceContext` handle + typed standard ops + composites; `TracedTensor{ctx,value}`; constructors; AD via snapshot/transform/import; checkpoint aliases on the context; `GraphCompiler` single lowering path | §2.2 in full: merge path (803-line span of `compiler.rs`), `traced.rs` graph assembly, `ad_support.rs` (641 lines) except thin helpers, `checkpoint.rs` (234), `metadata.rs` scope chain (655-line file, chain part), `metadata_scopes`/`constraint_scopes`, `extra_roots` (41 lines), `inputs_map`/`leaf_metas` carriers, `ProgramImport` extension | all traced constructors; `graph()`, pub fields; `extension::apply`; `finish` non-consuming; `ProgramImport` field | existing traced/AD/eager suites must pass with migrated call sites only; new property tests §7 |
| P3 extensions | per-family `TraceContext*Ext` for linalg, fft, einsum (merged), sparse, tropical; `TracedTensor` traits delegate; composites rewritten once on `(ctx, value)` | 19 `apply` sites in linalg `traced.rs`, 8 fft wrappers, sparse/tropical wrappers, duplicated composite bodies | linalg/fft/einsum/sparse/tropical extension traits keep names; `SparseCooTracedTensor` holds `(ctx, value)` | existing extension tests, oracle replay, `traced_ad_explicit.rs` |
| P4 downstream and cleanup | docs/samples/skills final pass, hidden `pub` items removed, `public_surface_contract` refreshed, tensor4all-rs and yao-rs build check at the new API (local patch only, no push) | remaining `pub` hooks in `ad_support`, `TracedTensorParts`, `tensor_from_parts` | closes the removal list | downstream build check recorded in the PR body |

Docs and samples move with the break (P2/P3 carry their share of the 110 `docs/`
files); P4 only fixes what the mechanical migration missed.

`GraphCompiler::compile(&TracedTensor)` stays in P2 so that downstream code
that compiles mid-trace keeps working; it becomes snapshot + frozen compile.

## 6. Test strategy

Existing tests are the semantic oracle and must pass with migrated call sites,
not rewritten expectations:

* 62 integration test files under `crates/*/tests` mention `TracedTensor` (109
  occurrences in `tenferro-runtime/tests/integration`, 391 in
  `tenferro-ad/tests/integration`);
* `crates/tenferro-ad/tests/integration/{checkpoint,checkpoint_truncate_integration,hvp,iterative_ad,symbolic_input,primitive_ops,semantic_transform}.rs`;
* `crates/tenferro-linalg/tests/integration/traced_*`, `oracle_replay`;
* `crates/tenferro-runtime/tests/integration/public_surface_contract.rs`.

New properties, one per mechanism that changes:

1. **Typed layer == raw layer.** For a representative op sequence, the typed
   path and hand-written `TraceContext::add_op`/`add_extension` produce
   programs with equal `semantic_fingerprint` and equal shape guards.
2. **Cross-context use is a typed error**, for every constructor and op path
   (property test over op kinds), never a panic.
3. **Snapshot independence.** `finish`/`compile` after further tracing yields
   the same program as freezing at that point; growing the context afterwards
   does not change a previously frozen program.
4. **AD in one context.** `grad`/`jvp` compose: `hvp` from a gradient value
   reuses the same leaves and does not duplicate program inputs (assert input
   count), with symbolic extents preserved.
5. **Checkpoint.** Gradient equals the un-checkpointed gradient; forward
   provider-call count is one per checkpoint (counted, not timed).
6. **Eager.** `EagerTensor: Send + Sync` (exists) plus: one journal node per
   op, one materialization per backward, and journal replay program fingerprint
   equal to the traced path's for the same op sequence, including symbolic
   inputs after shape change (replaces
   `eager_recording_retains_symbolic_semantic_trace_for_shape_churn`).
7. **Constraint parity.** The guard sets recorded by `add_extension` cover the
   cases in `graph/compiler/constraint_scope_tests.rs` (migrated).

No test may be relaxed; tolerances are unchanged.

## 7. Performance evidence

Trace and compile fixed cost, paired before/after on the same host, 1 thread,
pinned:

* `crates/tenferro-linalg/benches/traced_small_ad_overhead.rs`
* `crates/tenferro-ad/benches/{eager_leaf_construction,eager_dispatch_baseline,eager_backward_shape_churn,eager_ad_transform_cache}.rs`
* `crates/tenferro-runtime/benches/session_chain.rs`
* new: op-count scaling (1, 10, 100, 1000 chained ops) for record, compile and
  grad, to catch an accidental O(n²) in the journal or the snapshot.

Expected direction: per-op cost falls (one builder append instead of a graph
fragment plus three map merges and a global metadata registration), and compile
cost becomes one lowering instead of merge + re-lowering. If a measured cell
regresses, the cause is diagnosed before the phase lands; a regression is not
accepted as the price of the design.

## 8. Residual risks

* **Eager backward materializes the journal once per backward** (O(history)).
  Today's eager path also compiles O(history) per backward, so this is not a
  new asymptote, but the constant changes; measured in §7.
* **Context growth in a loop.** Each `grad` imports a derivative program into
  the live context. Documented contract: one snapshot, one context per
  iteration.
* **`RefCell` borrow errors** become typed errors; a mistaken nested borrow is
  a correctness bug that tests must catch, not a panic at runtime.
* **Docs migration** touches 102 Markdown files; mechanical, but it is the
  largest single review surface of P2.
