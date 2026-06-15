# Eager Recorded Graph VJP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement graph-invocation eager recording in tidu and use it from tenferro tracked eager einsum to collapse the backward path from per-input recomputation to one shared transposed DAG.

**Architecture:** First change tidu's eager tape so every trace node stores a `RecordedGraph`, with no `Recorder::record(op, ...)` or `try_build_single_op_linear` path. Then update tenferro to record ordinary eager primitives as one-op recorded graphs and tracked eager einsum as one multi-op recorded graph. Finally publish PRs in dependency order: tidu-rs first, tenferro-rs second with the tidu rev pinned to the merged or PR commit.

**Tech Stack:** Rust, tidu, computegraph, tenferro-ad, tenferro-einsum, GitHub PRs, cargo nextest/cargo test.

---

## File Map

tidu-rs:

- Modify `src/eager/record.rs`: add `RecordedGraph`, replace `Recorder::record` with `Recorder::record_graph`, and move one-op graph construction into `RecordedGraph::from_primitive`.
- Modify `src/eager/trace.rs`: store `RecordedGraph` on `TraceNode`, remove `operation` and `primal_in_keys`.
- Modify `src/eager/backward.rs`: replace `try_build_single_op_linear` with `TraceNode::computation().try_linearize(...)`.
- Modify `src/eager/mod.rs`: export `RecordedGraph`.
- Modify `tests/eager_record_tests.rs` and `tests/eager_backward_tests.rs`: update tests to use `RecordedGraph` and add multi-op graph tests.
- Modify `examples/eager_reverse_mode.rs`, `docs/guides/eager-integration.md`, `docs/tutorials/eager-reverse-mode.md`, and docs index pages that mention primitive recording.

tenferro-rs:

- Modify root `Cargo.toml` and `Cargo.lock`: pin tidu to the new commit.
- Modify `crates/tenferro-ad/src/eager.rs`: add one-op recorded graph construction and `record_eager_graph_outputs`.
- Modify `crates/tenferro-ad/src/eager_ops.rs` and `crates/tenferro-ad/src/extension.rs`: route primitive/extension eager recording through graph recording.
- Modify `crates/tenferro-ad/src/eager/backward.rs`: adapt to the tidu API and keep transposed graph execution as the follow-up optimization point.
- Modify `crates/tenferro-einsum/src/eager_tensor.rs`: add tracked graph-recorded eager einsum path using the existing contraction-tree graph.
- Modify `crates/tenferro-einsum/src/eager_tensor/tests.rs`: add tracked graph-recorded correctness and trace-node-count regression tests.
- Modify affected docs/worklogs as needed for PR review context.

## Task 1: tidu red tests for graph recording

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/tests/eager_record_tests.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/tests/eager_backward_tests.rs`

- [ ] **Step 1: Add a helper that records one-op graphs instead of primitives**

Add test helper code that calls the not-yet-existing API:

```rust
fn one_op_graph(
    op: RecorderOp,
    input_keys: Vec<Key>,
) -> tidu::eager::RecordedGraph<RecorderOp> {
    let mut builder = computegraph::graph::GraphBuilder::new();
    let graph_input_keys = input_keys.clone();
    let inputs = input_keys
        .into_iter()
        .map(|key| builder.add_input(key))
        .collect::<Vec<_>>();
    let outputs = builder.add_operation(
        op,
        inputs.iter().map(|id| ValueRef::Local(*id)).collect(),
        OperationRole::Primary,
    );
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let output_keys = outputs
        .iter()
        .map(|id| graph.values()[*id].key.clone())
        .collect();
    tidu::eager::RecordedGraph::new(graph, graph_input_keys, output_keys)
}
```

- [ ] **Step 2: Add a multi-op graph backward test**

Add a test equivalent to `z = (x * y) + y`, record it as one graph node, and assert:

```rust
assert_eq!(*cotangents.get(&ValueKey::Input(sk("x"))).unwrap().as_ref(), 3.0);
assert_eq!(*cotangents.get(&ValueKey::Input(sk("y"))).unwrap().as_ref(), 3.0);
```

Use `x = 2`, `y = 3`, seed `1`.

- [ ] **Step 3: Verify red**

Run:

```bash
cargo nextest run --release --test eager_record_tests --test eager_backward_tests
```

Expected: compile failure because `tidu::eager::RecordedGraph` and `Recorder::record_graph` do not exist.

## Task 2: tidu graph-only eager recorder

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/src/eager/record.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/src/eager/trace.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/src/eager/backward.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/src/eager/mod.rs`

- [ ] **Step 1: Add `RecordedGraph`**

Implement:

```rust
pub struct RecordedGraph<Op: GraphOperation> {
    graph: Arc<Graph<Op>>,
    input_keys: Vec<Op::InputKey>,
    output_keys: Vec<ValueKey<Op>>,
}
```

with `new`, `from_primitive`, `as_graph`, `input_keys`, `output_keys`, and `try_linearize(output_slots, ctx)`.

The concrete one-op constructor is:

```rust
pub fn from_primitive(op: Op, input_keys: Vec<Op::InputKey>) -> Self
where
    Op::InputKey: ADKey,
```

It builds a graph whose inputs use `input_keys`, adds `op` with `OperationRole::Primary`, sets all operation outputs as graph outputs, and stores the derived graph output keys.

- [ ] **Step 2: Replace `Recorder::record` with `record_graph`**

The new method validates input/output slot counts, creates eager `primal_out_keys`, saves graph inputs plus `retained_values`, builds `TraceEdge`s, and returns `EagerOutput`s.

Also add:

```rust
pub fn fresh_input_keys<Op>(&mut self, count: usize) -> Vec<Op::InputKey>
where
    Op: GraphOperation,
    K: KeySource<Op>;
```

Downstream frontends use this helper to allocate per-invocation graph input keys before constructing a `RecordedGraph`.

- [ ] **Step 3: Replace `TraceNode` fields**

`TraceNode` stores `computation: RecordedGraph<Op>`, `primal_out_keys`, `saved_data`, and `input_edges`. Remove `operation`, `primal_in_keys`, and their accessors.

- [ ] **Step 4: Remove `try_build_single_op_linear`**

`try_backward` calls:

```rust
let linear = node.computation().try_linearize(&active_output_slots, ctx)?;
```

and no helper named `try_build_single_op_linear` remains.

- [ ] **Step 5: Verify green**

Run:

```bash
cargo nextest run --release --test eager_record_tests --test eager_backward_tests
rg "try_build_single_op_linear|Recorder::record\\(" src tests examples docs
```

Expected: tests pass; search returns no production use of the removed APIs.

## Task 3: tidu docs and public examples

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/examples/eager_reverse_mode.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/docs/guides/eager-integration.md`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/docs/tutorials/eager-reverse-mode.md`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/docs/getting-started/terminology.md`
- Modify: `/Users/hiroshi/projects/tensor4all/tidu-rs/docs/internals/index.md`

- [ ] **Step 1: Update examples to build one-op recorded graphs**

Replace direct `recorder.record(ScalarOp::Mul, ...)` calls with:

```rust
let input_keys = recorder.fresh_input_keys::<ScalarOp>(inputs.len());
let graph = RecordedGraph::from_primitive(ScalarOp::Mul, input_keys);
let outputs = recorder.record_graph(graph, &inputs, &[arc(9.0)], HashMap::new());
```

- [ ] **Step 2: Update user-facing docs**

Explain that eager frontends record graph invocations. A single primitive is a one-op recorded graph; composite eager operations can record multi-op graphs.

- [ ] **Step 3: Verify docs locally**

Run:

```bash
cargo test --doc --release --workspace
```

Expected: doctests pass.

## Task 4: tidu verification and PR

**Files:**
- Commit all tidu-rs changes.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt --all
```

- [ ] **Step 2: Full local checks**

Run:

```bash
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
```

If `cargo nextest` is unavailable, use `cargo test --workspace --release -- --nocapture`.

- [ ] **Step 3: Commit and push**

Commit:

```bash
git add src tests examples docs Cargo.toml Cargo.lock
git commit -m "feat: record eager graphs in tidu"
git push -u origin codex/eager-recorded-graph-vjp
```

- [ ] **Step 4: Open PR and enable auto-merge**

Create a draft PR against `main` with a body that explains the breaking eager API change, the O(N^2) -> O(N) motivation, and the checks run. Enable auto-merge after checks are expected to pass.

## Task 5: tenferro red tests for graph-recorded eager recording

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/crates/tenferro-einsum/src/eager_tensor/tests.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/crates/tenferro-ad/src/eager.rs`

- [ ] **Step 1: Add a test-only trace-node count helper**

Under `#[cfg(test)]`, expose a crate-private helper that counts distinct eager trace nodes reachable from an `EagerTensor`.

- [ ] **Step 2: Add tracked eager einsum one-node test**

Create a tracked 3-input einsum that currently expands into more than one primitive, call backward, assert gradients are numerically correct, and assert the output trace has one recorded graph node for the einsum path.

- [ ] **Step 3: Verify red**

Run:

```bash
cargo test -p tenferro-einsum --release eager_tensor::tests::tracked_whole_program_einsum_records_one_graph_node
```

Expected: compile failure until tidu is updated, or assertion failure because tracked einsum still records per primitive.

## Task 6: tenferro tidu update and primitive graph recording

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/Cargo.toml`
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/Cargo.lock`
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/crates/tenferro-ad/src/eager.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/crates/tenferro-ad/src/eager_ops.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/crates/tenferro-ad/src/extension.rs`

- [ ] **Step 1: Pin tidu**

Temporarily pin to the tidu PR branch or commit while the tidu PR is pending; after merge, repin to the merged commit.

- [ ] **Step 2: Add one-op graph recording helper**

Replace `record_eager_outputs` internals so primitive eager ops construct `RecordedGraph::from_primitive` and call `record_graph`.

- [ ] **Step 3: Verify primitive eager tests**

Run:

```bash
cargo test -p tenferro-ad --release eager
```

Expected: existing eager gradients pass.

## Task 7: tenferro tracked eager einsum recorded graph path

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/crates/tenferro-einsum/src/eager_tensor.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/tenferro-rs/crates/tenferro-einsum/src/eager_tensor/tests.rs`

- [ ] **Step 1: Build one recorded graph for tracked n-ary einsum**

Use `build_einsum_graph` to build a graph with fresh graph inputs, execute that graph forward, collect conservative O(N) retained values, and call `record_eager_graph_outputs`.

- [ ] **Step 2: Keep untracked path as forward-only special case**

Share graph construction and forward execution helpers where practical, but do not block the O(N) backward fix on executor cache refactors.

- [ ] **Step 3: Verify green**

Run:

```bash
cargo test -p tenferro-einsum --release eager_tensor
```

Expected: tracked one-node test and existing eager tests pass.

## Task 8: tenferro verification and PR

**Files:**
- Commit all tenferro-rs changes.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt --all
```

- [ ] **Step 2: Local checks**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
```

Run additional docs/coverage checks if time allows before PR:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

- [ ] **Step 3: Commit and push**

Commit:

```bash
git add Cargo.toml Cargo.lock crates docs
git commit -m "feat: record tracked eager einsum graphs"
git push origin issue-1060-eager-einsum-profile
```

- [ ] **Step 4: Open PR and monitor**

Open a draft PR against `main`, link issue #1060 and the tidu PR, enable auto-merge when ready, and monitor checks every 30 seconds. If a check fails, inspect logs, fix, push, and resume monitoring.

## Self-Review

- Spec coverage: graph-only tidu recording, `try_build_single_op_linear` removal, tracked einsum one-node recording, conservative O(N) residual, docs update, and PR monitoring are covered.
- Placeholder scan: no task depends on an unspecified file; constructor signatures may be adjusted only where the implementation task defines them.
- Type consistency: `RecordedGraph`, `record_graph`, `EagerInput`, `EagerOutput`, and `BackwardExecutor` are used consistently across tidu and tenferro tasks.
