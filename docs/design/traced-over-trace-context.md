# TracedTensor as a typed operation layer over TraceContext

Status: **accepted** (maintainer review on #2418 at `38877fabe`). D1-D5 approved
with the decisions recorded in §5, plus two accepted additions: public multi-`wrt`
reverse mode in P2 (§3.6, §7) and an end-to-end `samples/kdv-pinn` check in §8.
No implementation before this design was approved; P1 may start after merge.

Every count, line number and size below was taken from the checkout at
`bdde3f2b6e82f0c9991f4170a642ad80dca7b707` with the command quoted next to it.
"Production" means `*.rs` under `crates/**/src` or `ext/**/src`; other counts
include tests and rustdoc and say so.

## 1. Scope

tenferro has two public ways to build a semantic graph:

| entry point | representation | op surface |
|---|---|---|
| `TracedTensor` | one `Arc<Graph<StdTensorOp>>` per tensor plus five merge/scope carriers | 75 inherent `pub fn` in `crates/tenferro-runtime/src/traced.rs`, 13 composites in `traced/composite_ops.rs`, the extension traits of §2.4, and `TracedTensorAdExt` |
| `TraceContext` | one `SemanticProgramBuilder` plus `ExtensionCacheStore` (`crates/tenferro-runtime/src/trace.rs:19-24`) | `input`, `input_with_default`, `bind_input`, `add_op`, `add_extension`, `value_metadata`, `extension_caches_mut`, `finish` (`trace.rs:39,52,70,87,106,121,129,141`) plus `TraceContextEinsumExt` |

The two describe the same object. `TracedTensor` builds many small graphs and
asks `GraphCompiler` to merge them; `TraceContext` appends to one program.

This document specifies the target design in which `TracedTensor` is a typed
view over `TraceContext`, records what that removes, what it breaks, and in
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

Carrier usage, counted as word occurrences with
`python3` over the 366 production `*.rs` files (`/src/` in the path, `/tests`
excluded), with the matching-line count over the same files in parentheses:

| field | production occurrences (lines) | definition / accessors |
|---|---|---|
| `extra_roots` | 51 (37) | `traced.rs:71`, `ad_support.rs:304`, `traced.rs:3487` (`resolve_roots`) |
| `inputs_map` | 45 (40) | `traced.rs:62`, `merge_traced_inputs_map` `traced.rs:90`, `ad_support.rs:294` |
| `leaf_metas` | 41 (35) | `traced.rs:70`, `merge_traced_leaf_metas` `traced.rs:133`, `ad_support.rs:300` |
| `checkpoint_chain` | 47 (40) | `traced.rs:72`, `checkpoint.rs:147` |
| `metadata_scopes` | 50 (41) | `traced.rs:73`, `metadata.rs:25` |
| `constraint_scopes` | 48 (44) | `traced.rs:74`, `ad_support.rs:51` |

`extra_roots` is never populated: every site initialises it empty, clones or
extends it from another tensor's clone, clears it (`ad_support.rs:528`), or
drains it in `resolve_roots`. Searching for a population site finds none:

```console
$ rg -n 'extra_roots' --glob '!docs/**' | rg 'push|= vec!|extend\(vec!|from\(\['   # exit 1
$ rg -n 'extra_roots' --glob '!docs/**' | wc -l
41          # matching lines repo-wide, excluding docs/
```

It is dead state: 37 production lines that carry nothing and no behaviour.

### 2.2 State that exists only for the per-tensor representation

**Merge and root resolution.** `GraphCompiler` has two lowering entry points,
`compile_traced_graph` / `compile_frozen_program` (`compiler.rs:188,225`), which
lower an already-frozen program, and `compile` / `compile_many` /
`compile_ad_source_many`, which accept `&TracedTensor` and merge their graphs.
The merge path occupies these regions of `compiler.rs` (1992 lines total):

| region | lines | role |
|---|---|---|
| `InputDescriptor`, `InputExtentIdentity` | 29-80 | input shapes derived from merged graph inputs |
| `compile_many_with_descriptors` | 500-670 | collect roots, `resolve`, `materialize_merge`, slot maps |
| `compile_materialized_semantic_program` | 946-1170 | re-add every materialized instruction to a builder |
| `collect_default_inputs` | 1180-1200 | bindings from per-tensor maps |
| `insert_checkpoint_alias`, `AliasTracker` | 1200-1377 | checkpoint aliasing and shape refinement |
| `materialize_merge_with_input_aliases` | 1377-1399 | merge with alias substitution |
| placeholder validation | 1399-1445 | `TracedTensor` arguments of `compile_with_input_specs` |
| `descriptor_for_input`, `default_input_extent_identity` | 1445-1532 | binding/default reconciliation |
| `default_tensors_equivalent` | 1599-1660 | conflict detection between per-tensor bindings |

The regions are disjoint; their spans sum to 867 lines of source (span
arithmetic from the `fn` line numbers, not a diff size). All of it exists
because the graph being compiled is a merge of many per-tensor fragments and
because bindings live on tensors instead of on one program.

**Scope chains.** `metadata_scopes` and `constraint_scopes` record, per tensor,
the metadata and shape constraints registered for the graph fragments that
produced it. They are merged per op (`traced.rs:3084-3190, 3381-3458`), carried
through AD (`ad_support.rs:51` `ConstraintScopeTransfer`), and replayed into the
program by the compiler (`compiler.rs:509-570` for constraints). The builder
does the same work inline when the op is added:

* `SemanticProgramBuilder::add_extension` infers metadata *and* shape guards and
  stores the guards on the operation (`program/builder.rs:338`, inference at
  `builder.rs:432-478`);
* `SemanticProgramBuilder::add_op` infers metadata for core ops
  (`builder.rs:251-273`), and core ops generate no shape constraints in either
  representation (`shape_infer::infer_output_shapes`, `shape_infer.rs:250`
  returns shapes only).

So the scope chains, and the analysis pass they exist for
(`ad_support.rs:372` `analyze_deferred_semantic_trace`, 60 lines), duplicate
work the builder already performs when an op is appended to one program.

**`inputs_map` / `leaf_metas`.** Both exist because bindings and leaf metadata
are held per tensor. `ProgramBindings` already holds frozen bindings, and the
builder binds at creation time (`bind_input` `program/builder.rs:63`,
`bind_input_retained` `:78`, `input_with_default` `trace.rs:52`). Bindings are
excluded from semantic identity (`compiler.rs:190-200`), which is exactly the
split `leaf_metas` reimplements by hand: symbolic extents in identity, concrete
data as bindings (`traced.rs:63-70`, `ad_support.rs:390-420`).

**`checkpoint_chain`.** `checkpoint_tensor` (`ad_support.rs:500-548`) replaces a
tensor's graph with a fresh bound leaf, keeps the old graph as an extra root, and
records `alias_key -> alias_target` plus the concrete shape of the bound data
(`checkpoint.rs:147-205`, shapes from `default_inputs` at `compiler.rs:530-545`).
The AD compile merges the old graphs back and substitutes the alias; because the
alias target is symbolically shaped while the checkpointed value is concrete
(a `dynamic_truncate` result), the substitution inserts a prefix
`Slice(0..shape)` (`compiler.rs:1282-1307`).

### 2.3 Eager semantic trace and the AD round trip

`EagerTensor::semantic_trace: Option<TracedTensor>`
(`tenferro-ad/src/eager.rs:4387`; also on `EagerTensorRecord` 4398 and
`EagerTensorParts` 4409). Per op:

1. `record_semantic_eager_outputs` (`eager.rs:5528`, entered from 5524/5631)
   promotes, casts and reshapes the semantic inputs to mirror eager execution,
   then appends **one raw graph fragment** with `append_raw_eager_outputs`
   (`runtime/extension.rs:205`). No metadata inference and no constraint
   inference: the fragment is built from the parent fragments,
   `merge_traced_inputs_map`, `merge_traced_leaf_metas`,
   `MetadataScopeChain::merge`, and an `extra_roots` union.
2. At the first AD request, `analyze_deferred_semantic_trace` (`ad_support.rs:372`)
   walks the whole parent chain post-order, registers metadata and constraints
   for every graph, seeds leaf metadata from `inputs_map` + `leaf_metas`, and
   returns a new `TracedTensor` carrying the analysis (`eager.rs:3940`).
3. `compile_ad_source_many` merges the analyzed trace plus the saved residual
   traces into one program (`eager.rs:3953`).
4. The program-level VJP runs (`AdContext::vjp_program`, `eager.rs:4015`), the
   derivative program is compiled, and each result is converted **back** into a
   `TracedTensor` by `derivative_trace_from_frozen_program`
   (`tenferro-ad/src/traced.rs:905`): rebuild a `Graph<StdTensorOp>` op by op
   (`traced.rs:979-1050`), re-register analysis and re-merge input maps, leaf
   metas, metadata and constraint scopes (`traced.rs:1080-1160`, leaf metas at
   1122), then construct `TracedTensorParts` (`ad_support.rs:202`).
5. The same round trip exists for the non-eager traced path:
   `TracedTensorAdExt::grad/jvp/vjp` compile the tensor
   (`tenferro-ad/src/traced.rs:613,706`), transform, and convert back with
   `derivative_tensors_from_program` (`traced.rs:885`).

Backward residuals are `SavedValue { trace: TracedTensor, value: .. }`
(`tenferro-ad/src/eager/residuals.rs:26-30`, filled at 100-130);
`EagerTrace::collect` identifies a residual by
`value.trace.graph().values()[value.trace.val].key` (`residuals.rs:52`), i.e. by
a `ValueKey` of a graph fragment.

`EagerTensor: Send + Sync` is already asserted inside tenferro-ad
(`crates/tenferro-ad/tests/integration/eager_tensor.rs:1431-1435`), the same
property tensor4all-rs asserts. A live `Rc<RefCell<_>>` handle in eager would
break it immediately.

### 2.4 Extension surface per family

| family | traced surface | implementation |
|---|---|---|
| linalg | `TracedTensorLinalgExt`, 25 methods (`linalg/src/traced.rs:17-411`, impl at 412) | 19 `apply(` sites in the same file build `LinalgExtensionOp` and call `tenferro_runtime::extension::apply`; `pinv`, `norm`, `slogdet` and friends are composites over `TracedTensor` standard ops (helpers at `traced.rs:1700-2147`) |
| fft | `TracedTensorFftExt`, 8 methods (`fft/src/lib.rs:638-930`) | one `apply` per method |
| einsum | `TraceContextEinsumExt`, 6 methods (`einsum/src/traced.rs:50-149`); `TracedTensorEinsumExt`, `tensordot` only (`:313-340`) | the context trait is the real implementation; the traced trait delegates |
| ext/sparse | `SparseCooTracedTensor` holds a `TracedTensor` (`ext/sparse/src/sparse.rs:135,227`) | `extension::apply` |
| ext/tropical | 7 free functions taking `&TracedTensor` (`ext/tropical/src/traced.rs:150,206,266,329,397,453,642`) | extension ops plus traced standard ops |
| core composites | 13 methods on `TracedTensor` in `traced/composite_ops.rs` (`sigmoid`, `silu`, `softplus`, `gelu`, `gelu_tanh`, `reduce_mean`, `softmax`, `log_softmax`, `masked_softmax`, `masked_log_softmax`, `layer_norm`, `rms_norm`, `take_along_axis`) | standard ops only |

What is tied to `TracedTensor` is the *wrapper*: op construction and composites.
Op semantics (`ExtensionOp` payloads), AD rules (`SemanticLinearizeRule`,
`SemanticPrimalVjpRule`, written against `SemanticProgramBuilder`) and kernels do
not depend on it.

The missing piece on the context side is a typed standard-op surface:
`tenferro-einsum` is the only extension with a `TraceContext` entry point
(`TraceContextEinsumExt` is the only such trait in the workspace), and
composites (`pinv`, `norm`, `softmax`, ...) cannot be written on `TraceContext`
today because it has no `matmul`, `reduce_max`, `transpose`, `compare`, ...

Note that the typed methods are not just "payload plus `add_op`": binary ops
insert broadcasts (`broadcast_binary`, `traced.rs:272-289`) and `Convert` ops on
dtype mismatch (`apply_binary`, `traced.rs:3194-3220`). That normalisation is
part of the typed layer, not of the builder.

### 2.5 Downstream

**tensor4all-rs**, `origin/main` = `22621f25e6357ebf147a062060f252162743579c`
(`git grep` at that ref):

| symbol | files |
|---|---|
| `EagerTensor` | 13 |
| `TraceContext` | 3 (`tensor4all-tensorbackend/src/{context.rs,tenferro_bridge.rs,context/tests/mod.rs}`) |
| `TracedGraph` | 1 |
| `TracedTensor`, `TraceValue`, `ProgramBindings` | 0 |

Graph construction there uses `TraceContext` + `TraceContextEinsumExt` +
`GraphCompiler::compile_traced_graph`; AD uses `EagerTensor`. No
data-attached traced leaves.

**yao-rs**, `4a398fe0a641b98792f8d3c28c56c2b79e84cedb`, pins
`tenferro* = "=0.4.0"`:

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
`TracedTensor` appears in `circuit_apply` (`src/tenferro_ad.rs:268-282`) and its
tests, while `circuit_apply_eager` (`:344`) offers the same operation for
`EagerTensor`; the AD rules are written against the semantic rule set. Both
downstream projects therefore need a traced-extension migration, not a rework
of their graph construction.

## 3. Target design

### 3.1 One tracing handle

```rust
#[derive(Clone)]
pub struct TraceContext { inner: Rc<RefCell<TraceInner>> }   // !Send, !Sync

pub struct TracedTensor { ctx: TraceContext, value: TraceValue }  // !Send, cheap to clone
```

* `TraceInner` holds the `SemanticProgramBuilder`, the `ExtensionCacheStore` and
  the checkpoint alias table of §3.5.
* `TraceContext` methods take `&self`. A value carries its owner nonce, so a
  foreign value is already a typed `ProgramBuildError::ForeignValue`
  (`program/builder.rs:100-115`), and a foreign `TracedTensor` becomes a typed
  error where today it silently merges two graphs.
* Borrows use `try_borrow_mut` and map failure to a typed error; no public path
  may panic (REPOSITORY_RULES "Public Boundary Safety Audits").
* Two accessors change shape because a borrow cannot escape the cell:
  `value_metadata` (`trace.rs:121`) returns an owned `ProgramValueMetadata`
  (already `Clone`, `program/metadata.rs:72`), and `extension_caches_mut`
  (`trace.rs:129`) becomes a closure-scoped `with_extension_caches(|c| ..)` or
  moves the store to a separate owner. Both are public breaks (§3.9).
* `finish(&self, outputs)` freezes a snapshot of the ordered closure of
  `outputs`, preserving effects (an effectful op stays in the program even when
  none of its outputs is a root, as `ImportTransaction` already enforces at
  `program/builder.rs:759-765`) and the declared input order. It leaves the
  context usable and exposes the source→snapshot value map so that callers can
  keep referring to the same values.

`TraceValue` is `Copy` (`trace.rs:155`), so `TracedTensor` is a handle clone plus
a two-word value. `rank`, `dtype` and symbolic shape come from
`TraceContext::value_metadata`; `id`, `rank`, `dtype`, `val` and `graph()` stop
being public (#1422).

Lifetime contract (changed, and to be documented): a `TracedTensor` keeps its
whole context alive, so one surviving tensor retains every branch and bound
tensor of that context. Today independent fragments die independently. Release
is per context, not per tensor; no cycle collector or per-value release is
introduced.

### 3.2 Typed standard operation layer

`CoreSemanticOp` has 52 variants and `TryFrom<&StdTensorOp> for CoreSemanticOp`
(`program/op.rs:150-246`) maps every non-extension `StdTensorOp` variant,
rejecting only `StdTensorOp::Extension(_)`. The typed layer is therefore a
mechanical mapping, plus the normalisation that belongs to the typed surface
(§2.4):

```rust
impl TraceContext {
    fn append_core(&self, op: CoreSemanticOp, inputs: &[TraceValue]) -> Result<Box<[TraceValue]>>; // -> add_op
    fn append_extension(&self, op: Arc<dyn ExtensionOp>, inputs: &[TraceValue]) -> Result<..>;     // -> add_extension
}
```

The shared helpers currently embedded in `TracedTensor` move into this layer:
broadcast planning (`broadcast_binary`/`broadcast_ternary`, `traced.rs:272-289`),
dtype promotion with `Convert` insertion (`apply_binary`, `traced.rs:3194-3220`),
output-dtype inference and rank/axis validation. Each of the 75 `TracedTensor`
methods and the 13 composites keeps its name and signature; its body becomes:
normalise inputs, build the payload, append, wrap the returned value.

### 3.3 Extension layer

One implementation per family on the context:

```rust
trait TraceContextLinalgExt { fn svd(&self, a: &TraceValue) -> Result<..>; .. }
trait TraceContextFftExt    { fn fft(&self, x: &TraceValue, ..) -> Result<..>; .. }
trait TraceContextEinsumExt { /* existing */ }
```

`TracedTensorLinalgExt`, `TracedTensorFftExt` and `TracedTensorEinsumExt` remain
as the method-chaining surface and delegate in one line, which removes the
duplicated implementation (19 `apply` sites in linalg, 8 fft wrappers, sparse and
tropical wrappers, composite bodies) while keeping both entry points.

Two traits per family is a deviation from the issue's wording ("one extension
trait per family"), and Rust cannot merge them: an inherent or trait method with
a `&self` receiver cannot be implemented for both `TraceContext` and
`TracedTensor` with one signature, because only the tensor carries the value.
The alternatives are (a) context-only traits and dropping the traced
method-chaining entry point, or (b) the two delegating traits above. This design
takes (b) and asks for approval; see §5.

`ext/sparse`'s `SparseCooTracedTensor` keeps its `TracedTensor` field (it already
is the context+value wrapper); only its construction moves to the context
constructors.

### 3.4 Metadata and scopes

Delete the per-tensor scope chains, `MetadataScopeChain` (`metadata.rs:25-155`),
`ConstraintScopeChain`, `ConstraintScopeTransfer` and
`analyze_deferred_semantic_trace`. The builder infers metadata and extension
guards when the op is appended (§2.2) and the frozen program carries them; there
is no other context-level scope state.

### 3.5 Checkpoint

A checkpoint becomes context state instead of a per-tensor chain:

* `checkpoint` evaluates the value once (as today), appends a fresh program input
  bound to the resulting data, and records `(input, value, bound_shape)` in a
  context-owned alias table;
* forward snapshots use the bound input, so the pre-checkpoint expression becomes
  dead and is not executed;
* AD snapshots substitute the alias, and when the target's extents are not
  exactly the recorded bound shape they insert the prefix `Slice(0..shape)` that
  `refine_alias_to_checkpoint_shape` (`compiler.rs:1282-1307`) inserts today.
  Without that refinement the substitution is not equivalent for
  runtime-sized `DynamicTruncate` values.

`CheckpointNode`, `collect_graphs`, `collect_aliases`, `collect_inputs`,
`merge_chains` (`checkpoint.rs:147-227`) and the graph materialisation in
`compiler.rs:1200-1399` disappear. Semantics are held by
`crates/tenferro-ad/tests/integration/checkpoint.rs` and
`checkpoint_truncate_integration.rs`, which must pass unchanged, including the
nested-truncate case.

### 3.6 Program snapshot, import and AD in one context

AD on a live context is: snapshot the closure of the differentiated outputs
(non-consuming, §3.1) → program-level VJP/JVP transform → **import** the
derivative program back into the same context, so results compose with further
ops.

The import path needs two contract changes; the first is inside the AD transform
itself.

1. **The transform must not concretise extents.** `import_source`
   (`tenferro-ad/src/semantic_transform.rs:418-434`) imports the source program
   *with its bindings*, and `ImportTransaction::prepare` resolves `InputDim`
   extents against bound concrete shapes (`program/builder.rs:815-850`), and
   resolves operation-output extents the same way (`:889-901`). The derivative
   program would therefore carry concrete extents before any remap.
   `ProgramImport.bindings` becomes optional (or gains an
   "extents stay symbolic" mode) and the AD transform imports structure only;
   bindings are re-attached afterwards by `with_input_prefix_bindings_from`
   (`program/builder.rs:672`), which exists for that purpose.
2. **The derivative must land on existing values.** `ProgramImport`
   (`program/import.rs:6-13`) has `program`, `bindings`, `roots` only, and
   `ImportTransaction::prepare` creates a fresh destination input per needed
   source input. Every `grad` would append duplicate inputs for the primal
   leaves. `ProgramImport` gains
   `input_map: &[(ProgramValue, ProgramValue)]` (source input → existing
   destination value); mapped inputs reuse the destination value and its
   metadata unchanged.

With both, `grad`/`jvp`/`vjp` are one snapshot + one transform + one import, and
the graph reconstruction in `tenferro-ad/src/traced.rs`
(`derivative_tensors_from_program`, 400+ lines with its metadata plumbing)
disappears.

As built, `ProgramImport::bindings` keeps its meaning (process-local tensor
defaults) but no longer drives extent resolution: import copies the source's
declared extents and translates their `InputDim`s into destination input
positions, so a transformed program never inherits the bound shapes. The bindings
are attached during the import itself rather than afterwards with
`with_input_prefix_bindings_from`, and a symbolic `Reshape`/`BroadcastInDim`
payload is translated from program input positions into the consuming
operation's own input coordinates by
`SemanticProgramBuilder::translate_input_shape_payload`, which the AD transform
and the einsum rule share. A derivative built from symbolic extents is
therefore shape-polymorphic, including for a symbolically declared input that
carries data.

Polymorphism holds for the *structure*, not for every decision a rule makes while
it builds that structure. Two exceptions are as-built and accepted:

1. **A literal payload cannot be polymorphic.** `Slice`, `Pad`, `DynamicSlice` and
   the concatenate transpose take `usize` offsets and sizes, so those rules read
   the lengths from the bindings through
   `SemanticProgramBuilder::resolved_extents` (one axis at a time through
   `resolved_extent`) and the resulting derivative is specialised to the binding
   present at differentiation time.
2. **A broadcast's reduction axes depend on the construction-time binding.** A
   broadcast VJP has to know whether the primal axis was a singleton, which is a
   fact about the binding, not about the expression: no operation reduces "only if
   the axis is 1". The reduce axes are therefore fixed when the derivative is
   built. The payload of every `Reshape`/`BroadcastInDim` it emits stays symbolic,
   and re-running the same derivative against a binding whose singleton structure
   differs is out of contract.

Everything derived from extents (`Reshape`, `BroadcastInDim`, `DynamicTruncate`)
stays symbolic, so a derivative is reusable across any binding for which the
literal payloads it was built with are still valid: the same singleton pattern
*and* the same lengths where a rule embedded one (a `Slice` inverse padding built
for length 5 is wrong for length 7, whatever its singleton pattern). A derivative
is not reusable across a binding that changes either.

As built, dead extension outputs are pruned at **snapshot** time, so the graph a
snapshot exposes (and therefore the primal program a caller compiles) already
carries the values-only form of a decomposition whose factor outputs are unused;
the full-factor residuals a derivative rule needs live in the AD context's own
record, not in the primal graph. A test that inspects the primal graph for
residuals therefore observes `SvdVals`, not `Svd`.

Cost: snapshot and import are O(program) per AD call, not per op. Today's path is
also O(program) per AD call (compile + transform + rebuild). Repeated `grad`
inside one context grows the context; the documented contract is one snapshot and
a new context per iteration, which is what callers do today by starting new
tensors.

**Public multi-`wrt` reverse mode.** Today `jvp_many`/`vjp_many` are
`pub(crate)` and `TracedTensorAdExt` exposes only single-`wrt` `grad`,
`grad_optional`, `jvp`, `vjp`, so a caller with several parameters builds one
program per parameter. `samples/kdv-pinn/src/main.rs:114-141` is the concrete
case: `net.parameters().map(|p| total_loss.grad(p))` followed by one
`compile_with_input_specs` per gradient, and one program run per parameter per
epoch (`:173-175`). tenferro-rs#2419 lists this as a prerequisite.

P2 exposes, on the same snapshot/transform/import path:

```rust
fn grad_many(&self, wrts: &[&TracedTensor]) -> Result<Vec<Option<TracedTensor>>>;
fn value_and_grad(&self, wrts: &[&TracedTensor]) -> Result<(TracedTensor, Vec<Option<TracedTensor>>)>;
```

Both run one snapshot, one program-level transform for the whole `wrt` set and
one import, so the backward is a single sweep. `jvp_many`/`vjp_many` become
public at the same time, since they are the same operation with an explicit
seed. The per-`wrt` methods stay as thin wrappers.

### 3.7 GraphCompiler

After the change `GraphCompiler` has one lowering path,
`compile_frozen_program`, plus `compile(&TracedTensor)` implemented as snapshot +
`compile_frozen_program`. Everything in §2.2 disappears, including
`CompiledGraph::input_slots` reordering for merged graphs;
`compile_with_input_specs` is re-expressed against snapshot program inputs.

`tenferro_runtime::extension::apply(op, &[&TracedTensor])` keeps its signature in
the core phase (implemented over the handle) so that the extension crates
continue to compile, and is removed in the extension phase, after every caller
uses the context methods.

As built, `compile_many(&outputs)` requires at least one output: the snapshot is
taken through the first output's context, so an empty output list has no context
to freeze and is a typed `InvalidArgument`. A zero-output program is built with
`SemanticProgramBuilder::finish(&[])` instead.

### 3.8 Eager: immutable journal, materialisation per backward

`EagerTensor` must stay `Send + Sync` (§2.3), so it cannot hold a
`Rc<RefCell<_>>` handle, and it cannot freeze a `FrozenProgram` per op: `finish`
consumes the builder (`program/builder.rs:194`) and `import` copies the program
(`:720-905`), so per-op freezing is O(history) per op, i.e. O(n²).

Target: eager keeps a private, immutable **record of op applications**. A
recorded value is `(Arc<Node>, output_index)`, because one op can produce several
values (SVD has three) and residual identity and edges must name the value, not
the operation.

```rust
enum Node {                                    // private to tenferro-ad
    Leaf   { key: TensorInputKey, meta: TensorMeta },              // bound constant / input
    Op     { op: StdTensorOp, inputs: Box<[ValueRef]> },           // owned carrier, 1..n outputs
    Program{ frozen: Arc<FrozenProgram>, output: usize, inputs: Box<[ValueRef]> },
}
type ValueRef = (Arc<Node>, u8);
```

* One `Arc` node per eager op, O(inputs) to create, immutable and `Send + Sync`
  (`StdTensorOp` is the owned op carrier the eager path already builds;
  `ExtensionOp: Debug + Send + Sync + 'static`,
  `tenferro-internal-ops/src/ext_op.rs:361`). No borrowed `SemanticOpRef`
  (`program/op.rs:366`) is stored.
* `Leaf` carries the binding key and metadata that `inputs_map`/`leaf_metas`
  carry today, and never retains an `EagerTensor` or the runtime.
* The `Program` node exists only where a frozen program already exists — a
  derivative result — so higher-order AD needs no graph rebuild.
* No metadata inference happens on the journal: eager outputs already have
  concrete dtype and shape, and at replay inputs are declared with symbolic
  extents (`ProgramInputSpec` with `DimExpr::InputDim`), which is the split
  §2.2 says `leaf_metas` reimplements.
* Backward replays the journal into a fresh `TraceContext` (one topological pass,
  dedupe by node pointer), then runs the same snapshot/transform/import as §3.6.
  One context per backward graph, not per `EagerRuntime`.

This deletes `append_raw_eager_outputs`, `analyze_deferred_semantic_trace`, the
residual `TracedTensor` field, and the graph↔program conversions in eager.
Per-op cost is one allocation plus parent `Arc` clones, against today's graph
fragment plus three map merges and a metadata registration.

The journal *is* a DAG of op applications, so this is a deliberate exception to
"one representation": no inference, no identity, no merge and no compile ever
reads it — it is replayed into the one canonical builder, which owns metadata,
shape guards, `SemanticProgram` identity and AD. Approval for that exception is
requested in §5; the alternative (a persistent chunked program core) is new core
machinery for a single consumer.

### 3.9 Public API changes

Removed or changed (breaking):

* `TracedTensor::from_tensor_concrete_shape`, `from_tensor_symbolic_shape`,
  `from_tensor_value_symbolic_shape`, `from_shared_tensor_value_symbolic_shape`,
  `from_vec_col_major`, `input_concrete_shape`,
  `input_concrete_shape_declaring_scalar`, `input_symbolic_shape` — replaced by
  context constructors (`input`, `constant`, ...). Occurrences at HEAD (calls and
  rustdoc examples): `from_vec_col_major` 546, `from_tensor_concrete_shape` 358,
  `from_tensor_symbolic_shape` 12, `from_shared_tensor_value_symbolic_shape` 5.
* `TracedTensor::graph()`, and the public fields `id`, `rank`, `dtype`, `val`.
  `crates/tenferro-runtime/tests/integration/public_surface_contract.rs:131-150`
  asserts the `graph()` accessor, so it is updated in the same change.
* `TraceContext::finish` no longer consumes the context;
  `value_metadata` returns an owned metadata value; `extension_caches_mut`
  becomes closure-scoped (§3.1).
* `tenferro_runtime::extension::apply` becomes a context method (removed in the
  extension phase, §3.7).
* `ProgramImport` gains `input_map`, and its `bindings` becomes optional (§3.6).
* `ad_support`'s `TracedTensorParts`, `tensor_from_parts`, `inputs_map`,
  `leaf_metas`, `extra_roots`, `checkpoint_chain`, `metadata_scopes`,
  `resolve_roots`, `analyze_deferred_semantic_trace` are removed (hidden but
  `pub`).

Migration size inside tenferro: 258 files mention `TracedTensor` (112 under
`crates/`, 110 under `docs/` of which 102 are Markdown, 29 under `ext/`, 7 under
`samples/`), 3141 occurrences.

### 3.10 Acceptance checklist (#2417 "Done when")

| requirement | where it is met | how it is checked |
|---|---|---|
| `TracedTensor` holds a context handle and a `TraceValue` | §3.1 | type definition; `graph()`/`val` gone from the public surface |
| `GraphCompiler` has no graph merge path | §3.2, §3.7 | merge regions of §2.2 deleted; only `compile_frozen_program` lowers |
| `TracedTensorAdExt`, linalg, fft, einsum run through program-level AD and one wrapper layer per family | §3.3, §3.6 | one implementation per family; AD is snapshot+transform+import; §5 deviation on trait count |
| context-free constructors removed, docs/samples/ext migrated | §3.9 | no `TracedTensor::` constructor remains; `docs/`, `samples/`, `ext/` build |
| tensor4all-rs and yao-rs build against the new API | phase 4 | local build of both with a local patch; summary in the PR body |
| `EagerTensor: Send + Sync` stays a checked property | §3.8 | `eager_tensor.rs:1431` plus a new assertion on the journal type |

## 4. Answers to the open questions

**Q1 `TraceHandle` sharing.** `Rc<RefCell<TraceInner>>`, `TracedTensor: !Send`.
Tracing is single threaded in every caller (§2.5); eager no longer holds a
handle (§3.8); and no `Send`/`Sync` bound is required of `TracedTensor` in
tenferro or in either downstream project (searched for `assert_send`,
`Arc<TracedTensor>` and `thread::spawn` uses). `Arc<Mutex<_>>` would buy nothing
and would put a lock on the per-op path.

**Q2 one context per what.** Per backward graph. `EagerRuntime` keeps no
context; each backward materialises one and drops it (§3.8).

**Q3 `ProgramBindings` for data-attached leaves.** Keep them. Data leaves are
pervasive (§3.9), bindings are already excluded from semantic identity, and
`input_with_default`/`bind_input` already exist on the builder. What disappears
is the per-tensor `inputs_map` copy, not bindings.

## 5. Decisions that need maintainer sign-off

All five were approved in the maintainer review of #2418 at `38877fabe`.

* **D1 approved — handle representation.** `Rc<RefCell<TraceInner>>`, `&self`
  methods, non-consuming `finish`, `TracedTensor: !Send` (§3.1). No public path
  may panic on a borrow.
* **D2 approved — eager recording.** A private immutable op-application journal
  replayed into a fresh context per backward, as the design's one exception to
  "one representation": it is never read by inference, identity, merge or
  compile, only replayed into the canonical builder (§3.8). It stays private to
  tenferro-ad, with `Send + Sync` assertions on both `EagerTensor` and the
  journal type.
* **D3 approved — import contract.** `ProgramImport` gains an input remap and
  optional bindings; the semantic AD transform imports structure only (§3.6).
* **D4 approved, option (b) — extension trait count.** Two traits per family, the
  traced trait delegating in one line; method chaining stays (§3.3).
* **D5 approved — checkpoint alias substitution.** Context alias table with the
  `Slice(0..shape)` refinement preserved; the nested-truncate tests are the gate
  (§3.5).

Accepted additions beyond the original open questions: public multi-`wrt` reverse
mode in P2 (§3.6, §7 test 4b) and the end-to-end `samples/kdv-pinn` epoch-time
check in §8.

## 6. Migration phases

Each phase is one PR into `main`, green on its own; the last one closes #2417.

| phase | scope | removed | public API broken | guards |
|---|---|---|---|---|
| P1 eager journal | tenferro-ad only: the `Node`/`ValueRef` journal (§3.8), one node per op, leaf nodes, replay at backward, residual identity by `(node, output)`, `Program` node for derivative results; replay uses the **existing** `SemanticProgramBuilder` API, so P1 does not depend on §3.1/§3.6 | eager's `append_raw_eager_outputs` use, residual `TracedTensor`, eager's deferred-analysis and graph-rebuild calls | none | `eager_tensor.rs:1431` Send+Sync (exists) plus a `Send + Sync` assertion on the journal; existing eager AD tests unchanged; counters: one node per op, one replay per backward; journal replay fingerprint equals the traced path's for the same op sequence, symbolic after shape change |
| P2 context core | `TraceContext` handle, typed standard ops + composites (§3.2), `TracedTensor{ctx,value}`, context constructors, AD via snapshot/transform/import, public multi-`wrt` `grad_many`/`value_and_grad` (§3.6), checkpoint aliases, `GraphCompiler` single lowering path | §2.2 in full: merge regions of `compiler.rs`, graph assembly in `traced.rs`, `ad_support.rs` (641 lines) except thin helpers, `checkpoint.rs` (234), metadata scope chain, `extra_roots`, `inputs_map`/`leaf_metas` carriers, AD graph rebuild (400+ lines of `tenferro-ad/src/traced.rs`) | §3.9, plus `extension::apply` kept working and `public_surface_contract.rs` updated in the same PR | existing traced/AD/eager suites pass with migrated call sites only; new properties §7; docs/samples/ext migrated in the same PR |
| P3 extensions | `TraceContextLinalgExt`, `TraceContextFftExt`, merged einsum, sparse, tropical; traced traits delegate; composites written once on `(ctx, value)` | 19 `apply` sites in linalg `traced.rs`, 8 fft wrappers, sparse/tropical wrappers, duplicated composite bodies, `extension::apply` | family traits keep their names; `SparseCooTracedTensor` construction moves to the context | extension tests, oracle replay, `traced_ad_explicit.rs`, `householder_qr_small_appends` bench |
| P4 downstream and cleanup | final docs/samples/skills pass, remaining hidden `pub` hooks removed, tensor4all-rs and yao-rs local build check against the new API (patch only, no push) | `TracedTensorParts`, `tensor_from_parts`, remaining `ad_support` accessors | closes the §3.9 removal list | `Closes #2417`; downstream build summary in the PR body |

P2 stays one PR (two live representations on `main` between phases is worse), cut
as a reviewable commit series so each commit builds:

1. context handle and typed core ops (§3.1, §3.2);
2. AD snapshot / transform / import, including public multi-`wrt` (§3.6);
3. checkpoint aliases (§3.5);
4. `GraphCompiler` single lowering path and the §2.2 deletions (§3.7);
5. call-site migration in `crates/`, `ext/`, `samples/`;
6. docs.

The PR body maps each commit to the section of this document it implements.

P2 is the large one: it rewrites `traced.rs` (3498 lines) and touches the 110
`docs/` files and 62 integration test files that name `TracedTensor`. Splitting
it further would leave two live representations on `main` between the phases,
which is what this design removes.

## 7. Test strategy

Existing tests are the semantic oracle and must pass with migrated call sites,
not rewritten expectations:

* 62 integration test files under `crates/*/tests` mention `TracedTensor` (109
  occurrences in `tenferro-runtime/tests/integration`, 391 in
  `tenferro-ad/tests/integration`);
* `crates/tenferro-ad/tests/integration/{checkpoint,checkpoint_truncate_integration,hvp,iterative_ad,symbolic_input,primitive_ops,semantic_transform}.rs`;
* `crates/tenferro-linalg/tests/integration/traced_*`, `oracle_replay`;
* `crates/tenferro-runtime/tests/integration/public_surface_contract.rs`.

New properties, one per mechanism that changes:

1. **Typed layer == raw layer.** For a representative op sequence (including
   broadcast and dtype-promotion cases), the typed path and hand-written
   `add_op`/`add_extension` sequence produce programs with equal
   `semantic_fingerprint` and equal shape guards.
2. **Cross-context use is a typed error** for every constructor and op path
   (property test over op kinds), never a panic.
3. **Snapshot independence.** `finish`/`compile` after further tracing yields the
   same program as freezing at that point; growing the context afterwards does
   not change a previously frozen program; effectful ops are retained; declared
   input order is preserved.
4. **AD in one context.** `hvp` from a gradient value reuses the same leaves and
   does not duplicate program inputs (assert input count); symbolic extents are
   preserved through the transform (program fingerprint with two different
   concrete input shapes is identical).
4b. **Multi-`wrt` (accepted addition).** `grad_many`/`value_and_grad` equal the
   per-`wrt` `grad` results elementwise, and the backward runs once: count the
   program-level transform calls (one, not one per `wrt`) and the program runs
   per epoch in the migrated `samples/kdv-pinn`, which drops from one program per
   parameter to one.
5. **Checkpoint.** Gradient equals the un-checkpointed gradient, including a
   nested `dynamic_truncate` checkpoint; the forward provider-call count is one
   per checkpoint (counted, not timed).
6. **Eager.** `EagerTensor: Send + Sync`, one journal node per op, one replay per
   backward, multi-output ops distinguish their outputs (SVD residuals), and a
   dropped-branch/dropped-binding test for the retention contract of §3.1.
7. **Constraint parity.** The guard sets recorded by `add_extension` cover the
   cases in `graph/compiler/constraint_scope_tests.rs` (migrated).

No test may be relaxed; tolerances are unchanged.

## 8. Performance evidence

Trace and compile fixed cost, paired before/after on the same host, one thread,
pinned; the change is not accepted with a measured regression:

* `crates/tenferro-linalg/benches/traced_small_ad_overhead.rs`
* `crates/tenferro-ad/benches/{eager_leaf_construction,eager_dispatch_baseline,eager_backward_shape_churn,eager_ad_transform_cache}.rs`
* `crates/tenferro-runtime/benches/session_chain.rs`
* new: op-count scaling (1, 10, 100, 1000 chained ops) for record, compile and
  grad, to catch an accidental O(n²) in the journal or the snapshot.

End-to-end check (accepted addition): the migrated `samples/kdv-pinn`, wall time
per epoch before and after on the same host, same thread and engine settings,
reported next to the micro benchmarks. Its epoch loop is the one place where the
multi-`wrt` change is visible end to end: at HEAD it compiles one gradient
program per parameter (`main.rs:114-141`) and runs one program per parameter per
epoch (`:173-175`), so the epoch time is the metric the multi-`wrt` API is
supposed to move.

Expected direction: per-op cost falls (one builder append instead of a graph
fragment plus three map merges and a metadata registration), and compile becomes
one lowering instead of merge + re-lowering.

## 9. Residual risks

* **Eager backward replays the journal once per backward** (O(history)). Today's
  eager path also compiles O(history) per backward, so the asymptote is the same
  and the constant changes; measured in §8.
* **Context growth in a loop.** Each `grad` imports a derivative program into the
  live context; the contract is one snapshot and a new context per iteration.
* **Retention granularity.** One tensor retains its whole context (§3.1).
* **`RefCell` borrow errors** become typed errors; a mistaken nested borrow is a
  test failure, not a runtime panic.
* **Docs migration** touches 102 Markdown files: mechanical, but the largest
  review surface of P2.
