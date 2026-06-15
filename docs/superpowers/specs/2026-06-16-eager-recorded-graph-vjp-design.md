# Eager Recorded Graph VJP Design

Date: 2026-06-16

## Context

Issues #1060 and #1061 identified a performance problem in tracked eager
einsum. Eager VJP is not a separate AD system: eager backward already uses
tidu primitive graph transformations. `tenferro-ad` calls into tidu with
`LinearizedGraph` and `PrimitiveGraph`, and `EagerPrimitiveBuilder` implements
tidu's primitive builder interface for `StdTensorOp`.

The problem is granularity. Trace mode differentiates a whole graph:

```text
graph -> linearize -> transpose -> compile -> whole-program execution
```

Tracked eager mode records each primitive execution separately:

```text
primitive op -> single-op graph -> linearize -> transpose -> per-node execution
```

For expanded eager einsum, the contraction tree becomes many binary primitive
contractions. Those primitives leak into the eager tape as many nodes, so
forward recording and backward transformation happen at primitive granularity.
The untracked whole-program eager prototype already shows that executing the
expanded graph as one program removes much of the forward overhead. The tracked
path should use the same graph-level granularity while preserving generic AD.

## Decision

Change tidu eager recording so the only recorded computation unit is a graph
invocation. A trace node no longer stores a single primitive operation.

`try_build_single_op_linear` and `Recorder::record(op, ...)` are removed. There
is no compatibility layer and no separate single-op recording path. A single
primitive eager operation is represented by a one-op `RecordedGraph`; a
composite operation such as einsum is represented by a multi-op `RecordedGraph`.
Backward always linearizes and transposes the recorded graph.

This makes the eager AD abstraction match Option C:

```text
forward:
  run graph G as one eager invocation
  record one tape node pointing to G

backward:
  linearize(G)
  transpose(linearize(G))
  run the transposed graph once
  return cotangents for G's external inputs
```

## Goals

- Make tidu eager tape record graph invocations, not primitive operations.
- Remove `try_build_single_op_linear` completely so no future code can reuse the
  old single-op AD path.
- Remove `Recorder::record(op, ...)`; all recording goes through `record_graph`.
- Reuse existing per-primitive JVP and transpose rules through tidu graph
  transforms.
- Make tracked eager einsum the first client by recording the binary contraction
  graph generated from the contraction tree as one eager node.
- Keep the mechanism generic for later linalg, FFT, and other composite eager
  operations.
- Allow downstream runtimes to execute forward, residual replay, and transposed
  graphs as whole programs.

## Non-Goals

- Do not add `StdTensorOp::Subgraph` or `StdTensorOp::Call`.
- Do not derive or maintain an einsum-specific VJP rule.
- Do not keep a legacy primitive recording API.
- Do not special-case einsum inside tidu.
- Do not require optimal residual liveness in the first implementation. Saving
  too many forward values is acceptable initially if the API allows tightening
  later.

## Tidu API

Add a recorded graph type to tidu eager:

```rust
pub struct RecordedGraph<Op: GraphOperation> {
    graph: Arc<Graph<Op>>,
    input_keys: Vec<Op::InputKey>,
    output_keys: Vec<ValueKey<Op>>,
}
```

`input_keys` is aligned with the eager input order. These keys are the `wrt`
inputs for `try_linearize`.

`output_keys` is aligned with the eager output slot order. These keys select
which graph outputs are active when only some eager outputs receive cotangents.

The public eager recorder exposes one recording method:

```rust
pub fn record_graph<Op>(
    &mut self,
    graph: RecordedGraph<Op>,
    inputs: &[EagerInput<Op>],
    outputs: &[Arc<Op::Operand>],
    retained_values: HashMap<ValueKey<Op>, Arc<Op::Operand>>,
) -> Vec<EagerOutput<Op>>
where
    Op: Primitive,
    Op::InputKey: ADKey,
    K: KeySource<Op>;
```

`retained_values` contains forward values from inside the recorded graph that
may be needed during backward. `record_graph` also records graph inputs and graph
outputs from `inputs` and `outputs`, so callers do not have to duplicate those
entries.

For ordinary eager primitives, the downstream frontend builds a one-op
`RecordedGraph` and passes it to `record_graph`. That one-op conversion is a
graph construction concern, not a separate recording or backward path.

## Trace Node Model

Replace the current `TraceNode` shape with a graph-only model:

```rust
pub(crate) struct TraceNode<Op: GraphOperation> {
    computation: RecordedGraph<Op>,
    primal_out_keys: Vec<ValueKey<Op>>,
    saved_data: HashMap<ValueKey<Op>, Arc<Op::Operand>>,
    input_edges: Vec<TraceEdge<Op>>,
}
```

There are two distinct output key domains:

- `RecordedGraph::output_keys` are graph-internal value keys used for
  linearization and residual lookup.
- `TraceNode::primal_out_keys` are eager tape output keys used for cotangent
  accumulation between eager nodes.

Keeping these separate avoids forcing graph output keys to become user-visible
eager value keys. A later eager node stores an input edge to the previous eager
output key, while its own recorded graph receives fresh graph input keys.

`TraceNode` no longer stores:

- `operation: Op`
- `primal_in_keys`

The graph's `input_keys` are the single source of truth for input order during
linearization and for aligning returned cotangents with `input_edges`.

## Backward Flow

`try_backward` remains responsible for walking the eager tape in reverse
topological order and accumulating cotangents on eager value keys. The per-node
work becomes graph-only:

```rust
let active_output_slots = node.active_output_slots(&cotangents);
let linear = node.computation.linearize(&active_output_slots, ctx)?;
let replay_graph = PrimitiveGraph::new(linear.as_graph());
let all_values = executor.execute_forward(replay_graph, node.saved_data());
let cotangent_in =
    executor.run_transposed_linear(&linear, &active_cotangent_out, &all_values, ctx)?;
```

`RecordedGraph::linearize` resolves its graph and calls tidu's existing
`try_linearize`:

```rust
let selected_outputs = output_slots
    .iter()
    .map(|slot| self.output_keys[*slot].clone())
    .collect::<Vec<_>>();

try_linearize(
    &resolve(vec![self.graph.clone()]),
    &selected_outputs,
    &self.input_keys,
    0,
    ctx,
    &HashMap::new(),
)
```

Because each recorded graph invocation owns fresh graph input keys, no alias map
is required for the normal eager path.

## Transpose Execution

The first API-preserving step can keep the current `BackwardExecutor` method
name, but the intended downstream implementation is whole-program execution.
tidu already has `try_linear_transpose`, which produces a transposed graph from a
`LinearizedGraph`. The tenferro eager backward executor should move away from
`try_linear_transpose_with_builder` for the composite path and instead:

1. Build the transposed graph once with `try_linear_transpose`.
2. Bind cotangent seeds as graph inputs.
3. Bind retained primal values as external data.
4. Run the transposed graph with the backend executor.
5. Return outputs aligned with `RecordedGraph::input_keys`.

This is what removes per-input recomputation for expanded einsum: shared
intermediate cotangents are represented once in the transposed DAG and executed
once by the backend.

## Residual Retention

The recorded graph forward path must retain enough primal values for backward.
The initial implementation should be conservative:

- Save graph external inputs.
- Save graph external outputs.
- Save all forward values produced by the recorded graph execution, or at least
  all live values exposed by the executor.

After correctness and graph-level execution are in place, retention can be
tightened by computing the values required by `linearize(G)` and
`transpose(linearize(G))`. The public API should already separate retained
values from outputs so this optimization does not change recording semantics.

## Tenferro Integration

`tenferro-ad` gets a graph recording helper beside the existing eager tensor
plumbing:

```rust
record_eager_graph_outputs(
    recorded_graph: RecordedGraph<StdTensorOp>,
    outputs: &[Arc<Tensor>],
    inputs: &[&EagerTensor],
    retained_values: HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
) -> RecordedEagerOutputs
```

The current primitive helper is replaced, not kept as an alternative AD path.
Primitive eager ops build a one-op `RecordedGraph<StdTensorOp>` and call
`record_eager_graph_outputs`.

`tenferro-einsum` uses the same helper for tracked n-ary eager einsum:

1. Build the contraction-tree graph with `build_einsum_graph`.
2. Execute that graph as one whole-program forward call.
3. Retain graph values needed for backward.
4. Record one eager tape node for the graph invocation.

The untracked whole-program path becomes the forward-only special case of the
same graph execution machinery. It does not need a separate expanded-eager
interpreter.

## Caching

A cached einsum plan or graph template is still useful, but recorded graph
invocations must have fresh graph input keys so eager trace keys do not collide.
The implementation should distinguish:

- reusable structural data: parsed equation, contraction tree, shapes, lowering
  plan, and backend program cache entries;
- per-invocation data: graph input keys, graph output keys, retained tensors, and
  eager tape output keys.

If an existing cache key includes concrete graph value keys, the first
implementation may instantiate and compile per invocation. That is acceptable
only as a stepping stone; the intended design is to cache structural lowering and
bind fresh values per invocation.

## Migration

This is a breaking tidu eager API change.

1. Add `RecordedGraph` and `Recorder::record_graph` to tidu.
2. Replace `TraceNode::operation` and `TraceNode::primal_in_keys` with
   `TraceNode::computation`.
3. Move backward linearization to `RecordedGraph::linearize`.
4. Delete `try_build_single_op_linear`.
5. Delete `Recorder::record`.
6. Update tidu eager tests, examples, and docs to build one-op graphs explicitly.
7. Update `tenferro-ad` to record primitive eager ops via one-op recorded graphs.
8. Add the tenferro graph-recording helper for composite eager ops.
9. Wire tracked eager einsum to whole-program forward plus one-node graph
   recording.
10. Move tenferro eager backward transpose execution to transposed graph
    execution for graph nodes.

## Testing

tidu tests:

- Recording a one-op graph produces the same gradients as the old primitive
  recorder tests.
- Recording a multi-op graph produces correct gradients and calls the backward
  executor once for the graph node.
- Multi-output recorded graphs linearize only active output slots.
- Shared intermediate cotangents are represented by the transposed graph, not by
  multiple independent per-input backward calls.

tenferro tests:

- Existing eager primitive gradient tests continue to pass after primitive
  recording moves to one-op graphs.
- Tracked eager n-ary einsum gradients match trace-mode or numerical reference
  values.
- Tracked eager n-ary einsum records one graph node for the expanded contraction
  graph.
- The untracked whole-program einsum path and tracked graph-recorded path share
  the same forward graph execution helper.

Repository checks:

- `rg "try_build_single_op_linear"` returns no matches.
- `rg "Recorder::record"` returns no production call sites.
- `cargo fmt --all --check`
- `cargo test --workspace --release`

## Risks

The main correctness risk is key confusion between graph-internal keys and eager
tape keys. The design avoids this by keeping `RecordedGraph::output_keys` and
`TraceNode::primal_out_keys` separate and documenting their roles.

The main performance risk is adding graph construction overhead to small eager
primitive ops. This is preferable to preserving a second AD path. If it becomes
measurable, optimize one-op graph construction or cache graph templates without
reintroducing primitive recording.

The main memory risk is retaining too many graph values during the first
implementation. This is acceptable for the initial correctness path, but the API
must keep retained values explicit so liveness-based retention can replace the
conservative strategy.

## Review Criteria

The implementation should be rejected if any of these are true:

- tidu eager backward has a branch that linearizes a single primitive directly.
- `TraceNode` stores `operation: Op`.
- `Recorder::record(op, ...)` remains as a public or internal recording path.
- tracked eager einsum differentiates an einsum extension op instead of the
  expanded primitive graph.
- tenferro adds an einsum-specific VJP rule for this issue.
