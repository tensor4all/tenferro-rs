# Checkpoint + DynamicTruncate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add checkpoint mechanism and DynamicTruncate/PadToMatch/ShapeOf ops for memory-efficient iterative TN AD with adaptive truncated SVD.

**Architecture:** Checkpoint uses Arc-linked-list (`CheckpointNode`) to separate eval roots from AD roots. New ops (`DynamicTruncate`, `PadToMatch`, `ShapeOf`) form an adjoint trio. `tidu::differentiate()` gains an alias map parameter for tracing gradients through checkpoint boundaries.

**Tech Stack:** Rust, tidu-rs (AD engine), computegraph-rs, tenferro-ops, tenferro-tensor, tenferro

---

## Task 1: tidu-rs — Add alias map to `differentiate()`

**Files:**
- Modify: `.worktrees/tidu-rs/src/differentiate.rs`
- Modify: `.worktrees/tidu-rs/src/lib.rs`
- Create: `.worktrees/tidu-rs/tests/checkpoint_alias_tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tests/checkpoint_alias_tests.rs
use std::collections::HashMap;
use std::sync::Arc;

use chainrules::{ADKey, PrimitiveOp};
use computegraph::fragment::FragmentBuilder;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::LocalValId;
use tidu::differentiate;

// Use tidu's existing test op type (from tests/common/mod.rs)
mod common;
use common::TestOp;

/// Checkpoint alias: differentiate through an aliased leaf back to the real input.
///
/// Graph:  x (leaf) -> Mul(x, x) -> y (checkpointed)
///         y_alias (leaf) -> Mul(y_alias, 2) -> z
///
/// Alias: y_alias_key -> y_derived_key
/// wrt: x
///
/// Expected: dz/dx = d(2*x^2)/dx = 4x
#[test]
fn differentiate_through_alias() {
    // Build primal fragment: x -> x*x -> y
    let mut builder = FragmentBuilder::<TestOp>::new();
    let x_key = "x".to_string();
    let x = builder.add_input(x_key.clone());
    let y_outputs = builder.add_op(
        TestOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    builder.set_outputs(y_outputs.clone());
    let primal_frag = Arc::new(builder.build());
    let y_key = primal_frag.vals()[y_outputs[0]].key.clone();

    // Build post-checkpoint fragment: y_alias -> y_alias * 2 -> z
    let mut builder2 = FragmentBuilder::<TestOp>::new();
    let y_alias_key = "y_alias".to_string();
    let y_alias = builder2.add_input(y_alias_key.clone());
    let two = builder2.add_op(TestOp::Constant(2.0), vec![], OpMode::Primal);
    let z_outputs = builder2.add_op(
        TestOp::Mul,
        vec![ValRef::Local(y_alias), ValRef::Local(two[0])],
        OpMode::Primal,
    );
    builder2.set_outputs(z_outputs.clone());
    let post_frag = Arc::new(builder2.build());
    let z_key = post_frag.vals()[z_outputs[0]].key.clone();

    // Resolve both fragments
    let view = resolve(vec![post_frag, primal_frag]);

    // Alias map: y_alias_key -> y_key (connects post-checkpoint to pre-checkpoint)
    let mut aliases: HashMap<String, GlobalValKey<TestOp>> = HashMap::new();
    aliases.insert(y_alias_key, y_key);

    // Differentiate z w.r.t. x through the alias
    let linear = differentiate(&view, &[z_key], &[x_key], 1, &mut (), &aliases);

    // Should have an active tangent output (dz/dx is non-zero)
    assert!(
        linear.tangent_outputs[0].is_some(),
        "gradient through alias should be active"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/tidu-rs && cargo test checkpoint_alias -- --nocapture`
Expected: Compile error (differentiate doesn't accept aliases param yet)

- [ ] **Step 3: Modify `differentiate()` signature to accept aliases**

In `.worktrees/tidu-rs/src/differentiate.rs`, change the signature:

```rust
pub fn differentiate<Op: PrimitiveOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
    ctx: &mut Op::ADContext,
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,  // NEW
) -> LinearFragment<Op>
where
    Op::InputKey: ADKey,
{
```

- [ ] **Step 4: Modify `topological_order` to follow aliases**

```rust
fn topological_order<Op: GraphOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
) -> Vec<GlobalValKey<Op>> {
    fn visit<Op: GraphOp>(
        key: &GlobalValKey<Op>,
        view: &ResolvedView<Op>,
        aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
        visited: &mut HashSet<GlobalValKey<Op>>,
        order: &mut Vec<GlobalValKey<Op>>,
    ) {
        if !visited.insert(key.clone()) {
            return;
        }

        match view.resolve_val(key) {
            Some(ValDef::Produced { input_keys, .. }) => {
                for input_key in input_keys {
                    visit(&input_key, view, aliases, visited, order);
                }
            }
            Some(ValDef::Input { key: input_key }) => {
                // Follow alias if present
                if let Some(aliased_key) = aliases.get(&input_key) {
                    visit(aliased_key, view, aliases, visited, order);
                }
            }
            None => {}
        }

        order.push(key.clone());
    }

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    for output_key in outputs {
        visit(output_key, view, aliases, &mut visited, &mut order);
    }
    order
}
```

- [ ] **Step 5: Modify main differentiation loop to propagate tangents through aliases**

In the main loop, replace the `ValDef::Input` arm:

```rust
ValDef::Input { key: input_key } => {
    // Check if this input is aliased to a derived value
    if let Some(aliased_key) = aliases.get(&input_key) {
        // Use the tangent from the aliased value
        let aliased_tangent = tangent_env.get(aliased_key).copied().flatten();
        tangent_env.insert(key, aliased_tangent);
    } else {
        tangent_env.insert(key, None);
    }
}
```

- [ ] **Step 6: Update lib.rs re-export**

No change needed — `differentiate` is already re-exported.

- [ ] **Step 7: Run test to verify it passes**

Run: `cd .worktrees/tidu-rs && cargo test checkpoint_alias -- --nocapture`
Expected: PASS

- [ ] **Step 8: Add a test for HVP through alias (2nd-order)**

```rust
#[test]
fn differentiate_through_alias_twice() {
    // Same setup but differentiate twice (Forward-over-Forward through alias)
    // This validates that aliases work in nested differentiation
    // ... (build graph as above, differentiate once, then differentiate the linear fragment)
    // The key check: aliases at the tangent level still connect properly
}
```

- [ ] **Step 9: Run all tidu tests**

Run: `cd .worktrees/tidu-rs && cargo test`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
cd .worktrees/tidu-rs && git add -A && git commit -m "feat: add alias map to differentiate() for checkpoint support"
```

---

## Task 2: Add `[patch]` override for local tidu-rs development

**Files:**
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add patch section**

Add to end of workspace `Cargo.toml`:

```toml
[patch."https://github.com/tensor4all/tidu-rs.git"]
tidu = { path = ".worktrees/tidu-rs" }
```

- [ ] **Step 2: Verify workspace builds**

Run: `cargo build -p tenferro 2>&1 | tail -5`
Expected: Build succeeds (may have warnings)

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock && git commit -m "chore: add local tidu-rs patch for checkpoint development"
```

---

## Task 3: Update tenferro `try_vjp()` to pass empty aliases (backward compat)

**Files:**
- Modify: `tenferro/src/traced.rs`

- [ ] **Step 1: Update `try_vjp()` call to pass empty aliases**

At line ~412, update the `differentiate` call:

```rust
let linear = differentiate(
    &view,
    std::slice::from_ref(&output_key),
    std::slice::from_ref(&wrt_input_key),
    next_pass_id(),
    &mut ad_ctx,
    &HashMap::new(),  // no aliases yet
);
```

Add `use std::collections::HashMap;` import if not already present.

- [ ] **Step 2: Verify all existing tests pass**

Run: `cargo test --workspace --release 2>&1 | tail -10`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tenferro/src/traced.rs && git commit -m "refactor: pass empty aliases to differentiate (backward compat)"
```

---

## Task 4: Add `CheckpointNode` struct and `checkpoint_chain` field

**Files:**
- Create: `tenferro/src/checkpoint.rs`
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/lib.rs` (if mod declaration needed)

- [ ] **Step 1: Create checkpoint module**

```rust
// tenferro/src/checkpoint.rs
use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::Fragment;
use computegraph::types::GlobalValKey;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::Tensor;

/// A node in the persistent checkpoint linked list.
///
/// Each checkpoint captures the old computation fragment, the alias mapping
/// from the new leaf key to the old derived key, and the concrete input data
/// needed for re-evaluation of the old fragment during AD.
#[derive(Clone, Debug)]
pub(crate) struct CheckpointNode {
    /// The old computation fragment (pre-checkpoint graph).
    pub fragment: Arc<Fragment<StdTensorOp>>,
    /// Maps new leaf key → old output GlobalValKey for AD continuation.
    pub alias_key: TensorInputKey,
    pub alias_target: GlobalValKey<StdTensorOp>,
    /// Concrete input data from the old fragment's inputs_map.
    pub old_inputs: HashMap<TensorInputKey, Tensor>,
    /// Link to previous checkpoint (persistent linked list).
    pub prev: Option<Arc<CheckpointNode>>,
}

impl CheckpointNode {
    /// Collect all alias mappings from this node to the tail.
    pub fn collect_aliases(&self) -> HashMap<TensorInputKey, GlobalValKey<StdTensorOp>> {
        let mut aliases = HashMap::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            aliases.insert(node.alias_key.clone(), node.alias_target.clone());
            current = node.prev.as_deref();
        }
        aliases
    }

    /// Collect all old fragments from this node to the tail.
    pub fn collect_fragments(&self) -> Vec<Arc<Fragment<StdTensorOp>>> {
        let mut fragments = Vec::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            fragments.push(node.fragment.clone());
            current = node.prev.as_deref();
        }
        fragments
    }

    /// Collect all old input data from this node to the tail.
    pub fn collect_inputs(&self) -> HashMap<TensorInputKey, Tensor> {
        let mut inputs = HashMap::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            inputs.extend(node.old_inputs.iter().map(|(k, v)| (k.clone(), v.clone())));
            current = node.prev.as_deref();
        }
        inputs
    }
}
```

- [ ] **Step 2: Add `checkpoint_chain` field to `TracedTensor`**

In `tenferro/src/traced.rs`, add field to the struct (after `extra_roots`):

```rust
pub(crate) checkpoint_chain: Option<Arc<CheckpointNode>>,
```

Add import: `use crate::checkpoint::CheckpointNode;`

- [ ] **Step 3: Add `mod checkpoint;` to tenferro/src/lib.rs**

- [ ] **Step 4: Initialize `checkpoint_chain: None` in all TracedTensor constructors**

Update `from_tensor()`, `apply_unary_with_dtype()`, `apply_nullary()`,
`apply_binary()`, `apply_multi_output()`, `try_vjp()`, and `try_jvp()` to
include `checkpoint_chain: None`.

- [ ] **Step 5: Propagate `checkpoint_chain` in `apply_unary_with_dtype`**

```rust
checkpoint_chain: input.checkpoint_chain.clone(),
```

- [ ] **Step 6: Propagate `checkpoint_chain` in `apply_binary`**

```rust
// Merge checkpoint chains: prefer the longer one (both typically share)
checkpoint_chain: lhs.checkpoint_chain.clone().or(rhs.checkpoint_chain.clone()),
```

- [ ] **Step 7: Propagate `checkpoint_chain` in `apply_multi_output`**

```rust
checkpoint_chain: input.checkpoint_chain.clone(),
```

- [ ] **Step 8: Verify build and all tests pass**

Run: `cargo test --workspace --release 2>&1 | tail -10`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add tenferro/src/checkpoint.rs tenferro/src/traced.rs tenferro/src/lib.rs
git commit -m "feat: add CheckpointNode struct and checkpoint_chain field"
```

---

## Task 5: Implement `checkpoint()` method

**Files:**
- Modify: `tenferro/src/traced.rs`
- Create: `tenferro/tests/checkpoint.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tenferro/tests/checkpoint.rs
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::{CpuBackend, Tensor, TypedTensor};

fn f64_scalar(val: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(vec![], vec![val]))
}

fn get_f64_scalar(t: &Tensor) -> f64 {
    match t {
        Tensor::F64(inner) => inner.host_data()[0],
        _ => panic!("expected F64"),
    }
}

#[test]
fn checkpoint_preserves_eval_value() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));
    let mut y = &x * &x; // y = 9.0

    y.checkpoint(&mut engine).unwrap();

    let val = get_f64_scalar(y.data.as_ref().unwrap());
    assert!((val - 9.0).abs() < 1e-12);
}

#[test]
fn checkpoint_downstream_eval_uses_leaf() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));
    let mut y = &x * &x; // y = 9.0

    y.checkpoint(&mut engine).unwrap();

    // z = y + 1.0 should only need to eval from y (leaf), not from x
    let one = TracedTensor::from_tensor_concrete_shape(f64_scalar(1.0));
    let mut z = &y + &one;
    let z_val = get_f64_scalar(z.eval(&mut engine).unwrap());
    assert!((z_val - 10.0).abs() < 1e-12);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro --test checkpoint -- --nocapture`
Expected: Compile error (checkpoint method doesn't exist)

- [ ] **Step 3: Implement `checkpoint()` method**

In `tenferro/src/traced.rs`, add the method to the `impl TracedTensor` block:

```rust
/// Evaluate and promote this tensor to a leaf for eval efficiency.
///
/// After checkpoint:
/// - Forward eval: downstream ops start from this leaf (O(K) per step)
/// - Backward AD: gradients flow through the old graph via checkpoint_chain
///
/// # Examples
///
/// ```ignore
/// let mut y = &x * &x;
/// y.checkpoint(&mut engine)?;
/// // y is now a leaf; subsequent ops build small graphs
/// ```
pub fn checkpoint<B: TensorBackend>(&mut self, engine: &mut Engine<B>) -> Result<()> {
    // 1. Evaluate to get concrete data
    self.eval(engine)?;
    let data = self.data.clone().expect("eval populates data");

    // 2. Capture old state for AD
    let old_fragment = self.fragment.clone();
    let old_output_key = old_fragment.vals()[self.val].key.clone();
    let old_inputs = (*self.inputs_map).clone();

    // 3. Create new leaf key and fragment
    let new_key = next_input_key();
    let mut builder = FragmentBuilder::new();
    let leaf_val = builder.add_input(new_key.clone());
    builder.set_outputs(vec![leaf_val]);
    let new_fragment = Arc::new(builder.build());

    // 4. Build checkpoint node (prepend to chain)
    let node = CheckpointNode {
        fragment: old_fragment,
        alias_key: new_key.clone(),
        alias_target: old_output_key,
        old_inputs,
        prev: self.checkpoint_chain.take(),
    };

    // 5. Update self to be a leaf
    self.fragment = new_fragment;
    self.val = leaf_val;
    // data stays as-is (already computed)
    self.extra_roots = vec![];
    self.checkpoint_chain = Some(Arc::new(node));

    // 6. Rebuild inputs_map with new key + preserve old inputs for AD
    let mut merged = HashMap::new();
    if let Some(ref chain) = self.checkpoint_chain {
        merged.extend(chain.collect_inputs());
    }
    merged.insert(new_key, data);
    self.inputs_map = Arc::new(merged);

    Ok(())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro --test checkpoint -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/traced.rs tenferro/tests/checkpoint.rs
git commit -m "feat: implement checkpoint() method on TracedTensor"
```

---

## Task 6: Wire checkpoint_chain into `try_vjp()` for AD through checkpoints

**Files:**
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/tests/checkpoint.rs`

- [ ] **Step 1: Write the failing test**

```rust
// Add to tenferro/tests/checkpoint.rs
const TOL: f64 = 1e-6;
const FD_H: f64 = 1e-6;

#[test]
fn checkpoint_grad_correct() {
    // f(x) = (x^2)^2 = x^4, with checkpoint after x^2
    // df/dx = 4x^3
    let x_val = 2.0_f64;
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor_concrete_shape(f64_scalar(x_val));
    let mut y = &x * &x; // x^2
    y.checkpoint(&mut engine).unwrap();

    let z = &y * &y; // (x^2)^2 = x^4
    let grad = z.grad(&x).unwrap();
    let mut grad_t = grad;
    let grad_val = get_f64_scalar(grad_t.eval(&mut engine).unwrap());

    // Finite difference
    let f = |v: f64| v.powi(4);
    let fd = (f(x_val + FD_H) - f(x_val - FD_H)) / (2.0 * FD_H);

    assert!(
        (grad_val - fd).abs() < TOL,
        "grad={grad_val}, fd={fd}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro --test checkpoint checkpoint_grad -- --nocapture`
Expected: FAIL (grad returns None or incorrect value because alias not wired)

- [ ] **Step 3: Update `try_vjp()` to use checkpoint_chain aliases**

In `try_vjp()`, before calling `differentiate()`:

```rust
fn try_vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> Option<TracedTensor> {
    let wrt_input_key = leaf_input_key(wrt);
    let output_key = self.fragment.vals()[self.val].key.clone();

    // Collect checkpoint info for AD
    let aliases = self
        .checkpoint_chain
        .as_ref()
        .map(|c| c.collect_aliases())
        .unwrap_or_default();
    let checkpoint_fragments = self
        .checkpoint_chain
        .as_ref()
        .map(|c| c.collect_fragments())
        .unwrap_or_default();

    // Include checkpoint fragments in resolve
    let mut roots = self.resolve_roots();
    roots.extend(checkpoint_fragments);

    let view = resolve(roots);
    let mut ad_ctx = ShapeGuardContext::default();
    let linear = differentiate(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&wrt_input_key),
        next_pass_id(),
        &mut ad_ctx,
        &aliases,
    );
    // ... rest unchanged
```

Also update the `inputs_map` construction to include checkpoint old_inputs:

```rust
    let mut inputs_map = (*self.inputs_map).clone();
    if let Some(ref chain) = self.checkpoint_chain {
        inputs_map.extend(chain.collect_inputs());
    }
    // ... existing cotangent and zero_tangent inserts ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro --test checkpoint checkpoint_grad -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/traced.rs tenferro/tests/checkpoint.rs
git commit -m "feat: wire checkpoint_chain into try_vjp for AD through checkpoints"
```

---

## Task 7: Checkpoint HVP test

**Files:**
- Modify: `tenferro/tests/checkpoint.rs`

- [ ] **Step 1: Write HVP test**

```rust
#[test]
fn checkpoint_hvp_correct() {
    // f(x) = (x^2)^2 = x^4, checkpoint after x^2
    // f'(x) = 4x^3
    // f''(x) = 12x^2
    // HVP = f''(x) * v
    let x_val = 2.0_f64;
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor_concrete_shape(f64_scalar(x_val));
    let mut y = &x * &x;
    y.checkpoint(&mut engine).unwrap();
    let z = &y * &y;

    // Forward-over-Reverse HVP
    let grad = z.grad(&x).unwrap(); // f'(x) = 4x^3
    let v = TracedTensor::from_tensor_concrete_shape(f64_scalar(1.0));
    let hv = grad.jvp(&x, &v); // f''(x) * v

    let hv_val = get_f64_scalar(eval_tensor(hv.clone()));
    let expected = 12.0 * x_val * x_val; // 48.0

    assert!(
        (hv_val - expected).abs() < TOL,
        "HVP: actual={hv_val}, expected={expected}"
    );

    // Also verify against finite difference of gradient
    let fd_grad = |v: f64| {
        let x2 = TracedTensor::from_tensor_concrete_shape(f64_scalar(v));
        let mut y2 = &x2 * &x2;
        y2.checkpoint(&mut Engine::new(CpuBackend::new())).unwrap();
        let z2 = &y2 * &y2;
        let g = z2.grad(&x2).unwrap();
        get_f64_scalar(&eval_tensor(g))
    };
    let fd_hv = (fd_grad(x_val + FD_H) - fd_grad(x_val - FD_H)) / (2.0 * FD_H);
    assert!(
        (hv_val - fd_hv).abs() < TOL,
        "HVP vs FD: actual={hv_val}, fd={fd_hv}"
    );
}

fn eval_tensor(traced: TracedTensor) -> Tensor {
    let mut engine = Engine::new(CpuBackend::new());
    let mut t = traced;
    t.eval(&mut engine).unwrap().clone()
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p tenferro --test checkpoint checkpoint_hvp -- --nocapture`
Expected: PASS

- [ ] **Step 3: Add multi-step checkpoint loop test**

```rust
#[test]
fn checkpoint_loop_grad_correct() {
    // f(a) = a * cos(a * cos(a * cos(x0)))  (3-step iteration)
    // Checkpoint after each step
    let a_val = 0.8_f64;
    let x0 = 0.5_f64;
    let steps = 3;
    let mut engine = Engine::new(CpuBackend::new());

    let a = TracedTensor::from_tensor_concrete_shape(f64_scalar(a_val));
    let mut x = TracedTensor::from_tensor_concrete_shape(f64_scalar(x0));

    for _ in 0..steps {
        x = &a * &x.cos();
        x.checkpoint(&mut engine).unwrap();
    }

    let mut grad = x.grad(&a).unwrap();
    let grad_val = get_f64_scalar(grad.eval(&mut engine).unwrap());

    // Finite difference
    let f_concrete = |a_v: f64| -> f64 {
        let mut xc = x0;
        for _ in 0..steps {
            xc = a_v * xc.cos();
        }
        xc
    };
    let fd = (f_concrete(a_val + FD_H) - f_concrete(a_val - FD_H)) / (2.0 * FD_H);

    assert!(
        (grad_val - fd).abs() < TOL,
        "loop grad: actual={grad_val}, fd={fd}"
    );
}
```

- [ ] **Step 4: Run all checkpoint tests**

Run: `cargo test -p tenferro --test checkpoint -- --nocapture`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro/tests/checkpoint.rs
git commit -m "test: add HVP and multi-step loop tests for checkpoint"
```

---

## Task 8: Add `ShapeOf` op (simplest new op — full pipeline)

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Modify: `tenferro-ops/src/ad/mod.rs`
- Modify: `tenferro/src/compiler.rs`
- Modify: `tenferro/src/stablehlo.rs`
- Modify: `tenferro/src/exec.rs`
- Modify: `tenferro-tensor/src/backend.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs` (or exec_session.rs)
- Create: `tenferro/tests/shape_of.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tenferro/tests/shape_of.rs
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::{CpuBackend, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_scalar(t: &Tensor) -> f64 {
    match t {
        Tensor::F64(inner) => {
            assert_eq!(inner.shape(), &[] as &[usize]);
            inner.host_data()[0]
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn shape_of_returns_axis_size() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 5, 7], vec![0.0; 105]));
    let mut s0 = x.shape_of(0);
    let mut s1 = x.shape_of(1);
    let mut s2 = x.shape_of(2);

    assert_eq!(get_f64_scalar(s0.eval(&mut engine).unwrap()), 3.0);
    assert_eq!(get_f64_scalar(s1.eval(&mut engine).unwrap()), 5.0);
    assert_eq!(get_f64_scalar(s2.eval(&mut engine).unwrap()), 7.0);
}

#[test]
fn shape_of_grad_is_zero() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4, 3], vec![0.0; 12]));
    let s = x.shape_of(0);
    // shape_of should not be differentiable
    assert!(s.try_grad(&x).unwrap().is_none());
}
```

- [ ] **Step 2: Add `ShapeOf` variant to `StdTensorOp`**

In `tenferro-ops/src/std_tensor_op.rs`, add after `Reverse`:

```rust
/// Extract the size of a specific axis as a scalar f64 tensor.
ShapeOf { axis: usize },
```

Update `n_inputs()`: add `Self::ShapeOf { .. } => 1,`
Update `n_outputs()`: add `Self::ShapeOf { .. } => 1,` (in the existing 1-output match arm)

- [ ] **Step 3: Add AD rules for ShapeOf**

In `tenferro-ops/src/ad/mod.rs`, add to `linearize_non_semiring`:

```rust
StdTensorOp::ShapeOf { .. } => vec![None],  // shape doesn't depend on values
```

Add to `transpose_non_semiring`:

```rust
StdTensorOp::ShapeOf { .. } => vec![None],  // no gradient
```

- [ ] **Step 4: Add StableHloOp variant and lowering**

In `tenferro/src/stablehlo.rs`, add variant:
```rust
ShapeOf { axis: usize },
```

In `tenferro/src/compiler.rs` `lower_to_stablehlo`, add:
```rust
StdTensorOp::ShapeOf { axis } => StableHloOp::ShapeOf { axis: *axis },
```

- [ ] **Step 5: Add ExecOp variant and compile**

In `tenferro/src/exec.rs`, add variant:
```rust
ShapeOf { axis: usize },
```

In `compile_to_exec` (StableHloOp → ExecOp mapping), add:
```rust
StableHloOp::ShapeOf { axis } => ExecOp::ShapeOf { axis: *axis },
```

- [ ] **Step 6: Add execution dispatch**

In `eval_exec_ir`, add to the match:

```rust
ExecOp::ShapeOf { axis } => {
    let input = get(&slots, &inst.input_slots, 0)?;
    let size = input.shape()[*axis] as f64;
    Tensor::F64(TypedTensor::from_vec(vec![], vec![size]))
}
```

Note: No backend trait method needed — this is purely metadata extraction.

- [ ] **Step 7: Add `shape_of()` method on TracedTensor**

In `tenferro/src/traced.rs`:

```rust
/// Extract the size of this tensor along `axis` as a scalar f64 TracedTensor.
///
/// The result is non-differentiable (gradient is always zero).
///
/// # Examples
///
/// ```ignore
/// let size = x.shape_of(1); // scalar tensor with value x.shape[1]
/// ```
pub fn shape_of(&self, axis: usize) -> TracedTensor {
    apply_unary_with_dtype(
        StdTensorOp::ShapeOf { axis },
        self,
        0,            // scalar output (rank 0)
        Some(vec![]), // shape_hint: scalar
        DType::F64,
    )
}
```

- [ ] **Step 8: Run tests**

Run: `cargo test -p tenferro --test shape_of -- --nocapture`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add tenferro-ops/src/std_tensor_op.rs tenferro-ops/src/ad/mod.rs \
  tenferro/src/compiler.rs tenferro/src/stablehlo.rs tenferro/src/exec.rs \
  tenferro/src/traced.rs tenferro/tests/shape_of.rs
git commit -m "feat: add ShapeOf op (axis size extraction, non-differentiable)"
```

---

## Task 9: Add `DynamicTruncate` op

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Modify: `tenferro/src/compiler.rs`
- Modify: `tenferro/src/stablehlo.rs`
- Modify: `tenferro/src/exec.rs`
- Create: `tenferro/tests/dynamic_truncate.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tenferro/tests/dynamic_truncate.rs
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::{CpuBackend, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn f64_scalar(val: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(vec![], vec![val]))
}

fn get_f64_data(t: &Tensor) -> Vec<f64> {
    match t {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected F64"),
    }
}

#[test]
fn dynamic_truncate_basic() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));

    let mut result = x.dynamic_truncate(&size, 0);
    let data = get_f64_data(result.eval(&mut engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn dynamic_truncate_2d_axis1() {
    let mut engine = Engine::new(CpuBackend::new());
    // 2x4 matrix (col-major): [[1,2,3,4],[5,6,7,8]]
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 4], vec![1.0,5.0, 2.0,6.0, 3.0,7.0, 4.0,8.0]));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(2.0));

    let mut result = x.dynamic_truncate(&size, 1);
    let out = result.eval(&mut engine).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    // First 2 columns: [[1,2],[5,6]] stored col-major: [1,5,2,6]
    assert_eq!(get_f64_data(out), vec![1.0, 5.0, 2.0, 6.0]);
}

#[test]
fn dynamic_truncate_clamps_oversize() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(10.0)); // larger than axis

    let mut result = x.dynamic_truncate(&size, 0);
    let data = get_f64_data(result.eval(&mut engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0]); // clamped to actual size
}
```

- [ ] **Step 2: Add `DynamicTruncate` variant to `StdTensorOp`**

```rust
/// Truncate tensor along axis to first N elements (N from scalar input).
DynamicTruncate { axis: usize },
```

Update `n_inputs()`: `Self::DynamicTruncate { .. } => 2,`
Update `n_outputs()`: add to existing 1-output arm.

- [ ] **Step 3: Add lowering pipeline (StableHlo + ExecOp)**

StableHloOp:
```rust
DynamicTruncate { axis: usize },
```

Compiler `lower_to_stablehlo`:
```rust
StdTensorOp::DynamicTruncate { axis } => StableHloOp::DynamicTruncate { axis: *axis },
```

ExecOp:
```rust
DynamicTruncate { axis: usize },
```

`compile_to_exec`:
```rust
StableHloOp::DynamicTruncate { axis } => ExecOp::DynamicTruncate { axis: *axis },
```

- [ ] **Step 4: Add execution**

In `eval_exec_ir`:

```rust
ExecOp::DynamicTruncate { axis } => {
    let input = get(&slots, &inst.input_slots, 0)?;
    let size_tensor = get(&slots, &inst.input_slots, 1)?;
    // Extract scalar size and clamp
    let size_f64 = match size_tensor {
        Tensor::F64(t) => t.host_data()[0],
        Tensor::F32(t) => t.host_data()[0] as f64,
        _ => return Err(Error::Internal("DynamicTruncate size must be real".into())),
    };
    let size = (size_f64.round() as usize).min(input.shape()[*axis]);
    // Build SliceConfig for 0..size along axis
    let mut starts = vec![0i64; input.rank()];
    let mut limits = input.shape().iter().map(|&d| d as i64).collect::<Vec<_>>();
    let strides = vec![1i64; input.rank()];
    limits[*axis] = size as i64;
    let config = tenferro_tensor::SliceConfig { starts, limits, strides };
    exec.slice(input, &config)?
}
```

- [ ] **Step 5: Add `dynamic_truncate()` method on TracedTensor**

```rust
/// Truncate this tensor along `axis` to the first `size` elements.
///
/// `size` is a scalar TracedTensor whose value determines the truncation
/// point at runtime. Non-integer values are rounded; values exceeding
/// the axis length are clamped.
///
/// # Examples
///
/// ```ignore
/// let truncated = x.dynamic_truncate(&size, 0);
/// ```
pub fn dynamic_truncate(&self, size: &TracedTensor, axis: usize) -> TracedTensor {
    apply_binary(
        StdTensorOp::DynamicTruncate { axis },
        self,
        size,
        self.rank,
        None, // shape_hint unknown along axis
    )
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p tenferro --test dynamic_truncate -- --nocapture`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tenferro-ops/src/std_tensor_op.rs tenferro/src/compiler.rs \
  tenferro/src/stablehlo.rs tenferro/src/exec.rs tenferro/src/traced.rs \
  tenferro/tests/dynamic_truncate.rs
git commit -m "feat: add DynamicTruncate op (forward only)"
```

---

## Task 10: Add `PadToMatch` op

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Modify: `tenferro/src/compiler.rs`
- Modify: `tenferro/src/stablehlo.rs`
- Modify: `tenferro/src/exec.rs`
- Modify: `tenferro/tests/dynamic_truncate.rs`

- [ ] **Step 1: Write the failing test**

```rust
// Add to tenferro/tests/dynamic_truncate.rs
#[test]
fn pad_to_match_basic() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let reference = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![0.0; 5]));

    let mut result = x.pad_to_match(&reference, 0);
    let data = get_f64_data(result.eval(&mut engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn pad_to_match_no_op_when_same_size() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let reference = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![0.0; 4]));

    let mut result = x.pad_to_match(&reference, 0);
    let data = get_f64_data(result.eval(&mut engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}
```

- [ ] **Step 2: Add `PadToMatch` variant to `StdTensorOp`**

```rust
/// Pad tensor with zeros along axis to match reference tensor's size.
PadToMatch { axis: usize },
```

Update `n_inputs()`: `Self::PadToMatch { .. } => 2,`
Update `n_outputs()`: add to 1-output arm.

- [ ] **Step 3: Add lowering pipeline**

StableHloOp: `PadToMatch { axis: usize },`
ExecOp: `PadToMatch { axis: usize },`
Lowering: direct 1:1 mapping.

- [ ] **Step 4: Add execution**

```rust
ExecOp::PadToMatch { axis } => {
    let input = get(&slots, &inst.input_slots, 0)?;
    let reference = get(&slots, &inst.input_slots, 1)?;
    let target_size = reference.shape()[*axis];
    let current_size = input.shape()[*axis];
    if current_size >= target_size {
        input.clone()
    } else {
        let pad_amount = target_size - current_size;
        let mut low = vec![0i64; input.rank()];
        let mut high = vec![0i64; input.rank()];
        let interior = vec![0i64; input.rank()];
        high[*axis] = pad_amount as i64;
        let config = tenferro_tensor::PadConfig {
            edge_padding_low: low,
            edge_padding_high: high,
            interior_padding: interior,
        };
        exec.pad(input, &config)?
    }
}
```

- [ ] **Step 5: Add `pad_to_match()` method on TracedTensor**

```rust
/// Pad this tensor with zeros along `axis` to match `reference.shape[axis]`.
///
/// # Examples
///
/// ```ignore
/// let padded = truncated.pad_to_match(&original, 0);
/// ```
pub fn pad_to_match(&self, reference: &TracedTensor, axis: usize) -> TracedTensor {
    apply_binary(
        StdTensorOp::PadToMatch { axis },
        self,
        reference,
        self.rank,
        reference.shape_hint.clone(),
    )
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p tenferro --test dynamic_truncate pad_to_match -- --nocapture`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tenferro-ops/src/std_tensor_op.rs tenferro/src/compiler.rs \
  tenferro/src/stablehlo.rs tenferro/src/exec.rs tenferro/src/traced.rs \
  tenferro/tests/dynamic_truncate.rs
git commit -m "feat: add PadToMatch op (forward only)"
```

---

## Task 11: AD rules for DynamicTruncate and PadToMatch

**Files:**
- Create: `tenferro-ops/src/ad/dynamic.rs`
- Modify: `tenferro-ops/src/ad/mod.rs`
- Modify: `tenferro/tests/dynamic_truncate.rs`

- [ ] **Step 1: Write the failing AD test**

```rust
// Add to tenferro/tests/dynamic_truncate.rs
const TOL: f64 = 1e-5;
const FD_H: f64 = 1e-5;

#[test]
fn dynamic_truncate_vjp_correct() {
    let mut engine = Engine::new(CpuBackend::new());
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], x_data.clone()));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));

    // f(x) = sum(truncate(x, 3)^2) = 1 + 4 + 9 = 14
    let truncated = x.dynamic_truncate(&size, 0);
    let loss = (&truncated * &truncated).reduce_sum(&[0]);

    let mut grad = loss.grad(&x).unwrap();
    let grad_data = get_f64_data(grad.eval(&mut engine).unwrap());
    // df/dx_i = 2*x_i for i<3, 0 for i>=3
    assert_eq!(grad_data, vec![2.0, 4.0, 6.0, 0.0, 0.0]);
}

#[test]
fn dynamic_truncate_jvp_correct() {
    let mut engine = Engine::new(CpuBackend::new());
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], x_data.clone()));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));

    let truncated = x.dynamic_truncate(&size, 0);
    let loss = (&truncated * &truncated).reduce_sum(&[0]);

    // JVP with direction v = [1,1,1,1,1]
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![1.0; 5]));
    let mut jvp_result = loss.jvp(&x, &v);
    let jvp_val = get_f64_data(jvp_result.eval(&mut engine).unwrap())[0];

    // Expected: dot(grad, v) = 2+4+6+0+0 = 12
    assert!((jvp_val - 12.0).abs() < TOL, "jvp={jvp_val}, expected=12");
}
```

- [ ] **Step 2: Create `dynamic.rs` AD rules module**

```rust
// tenferro-ops/src/ad/dynamic.rs
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

/// Linearize DynamicTruncate: apply same truncation to tangent.
pub fn linearize_dynamic_truncate(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    // input[0] = tensor (differentiable), input[1] = size (non-differentiable)
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::DynamicTruncate { axis },
                vec![
                    ValRef::Local(dx),
                    ValRef::External(primal_in[1].clone()), // same size
                ],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

/// Transpose DynamicTruncate: pad cotangent back to original size.
pub fn transpose_dynamic_truncate(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            // Pad cotangent to match primal input[0]'s shape
            let out = builder.add_op(
                StdTensorOp::PadToMatch { axis },
                vec![
                    ValRef::Local(ct),
                    inputs[0].clone(), // reference: original input tensor
                ],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0]), None] // no gradient for size input
        }
        None => vec![None, None],
    }
}

/// Linearize PadToMatch: apply same padding to tangent.
pub fn linearize_pad_to_match(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::PadToMatch { axis },
                vec![
                    ValRef::Local(dx),
                    ValRef::External(primal_in[1].clone()), // same reference
                ],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

/// Transpose PadToMatch: truncate cotangent to original (pre-pad) size.
pub fn transpose_pad_to_match(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            // Get the original input's size along axis via ShapeOf
            let size = builder.add_op(
                StdTensorOp::ShapeOf { axis },
                vec![inputs[0].clone()], // primal input[0]
                OpMode::Linear {
                    active_mask: vec![false],
                },
            );
            // Truncate cotangent to that size
            let out = builder.add_op(
                StdTensorOp::DynamicTruncate { axis },
                vec![ValRef::Local(ct), ValRef::Local(size[0])],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0]), None] // no gradient for reference input
        }
        None => vec![None, None],
    }
}
```

- [ ] **Step 3: Wire AD rules into dispatch**

In `tenferro-ops/src/ad/mod.rs`:

Add `mod dynamic;` after existing module declarations.

In `linearize_non_semiring`, add:
```rust
StdTensorOp::DynamicTruncate { axis } => {
    dynamic::linearize_dynamic_truncate(builder, primal_in, tangent_in, *axis)
}
StdTensorOp::PadToMatch { axis } => {
    dynamic::linearize_pad_to_match(builder, primal_in, tangent_in, *axis)
}
StdTensorOp::ShapeOf { .. } => vec![None],
```

In `transpose_non_semiring`, add:
```rust
StdTensorOp::DynamicTruncate { axis } => {
    dynamic::transpose_dynamic_truncate(builder, cotangent_out, inputs, *axis)
}
StdTensorOp::PadToMatch { axis } => {
    dynamic::transpose_pad_to_match(builder, cotangent_out, inputs, *axis)
}
StdTensorOp::ShapeOf { .. } => vec![None],
```

- [ ] **Step 4: Run AD tests**

Run: `cargo test -p tenferro --test dynamic_truncate -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tenferro-ops/src/ad/dynamic.rs tenferro-ops/src/ad/mod.rs \
  tenferro/tests/dynamic_truncate.rs
git commit -m "feat: add AD rules for DynamicTruncate and PadToMatch"
```

---

## Task 12: HVP tests for DynamicTruncate

**Files:**
- Modify: `tenferro/tests/dynamic_truncate.rs`

- [ ] **Step 1: Write HVP test**

```rust
#[test]
fn dynamic_truncate_hvp_correct() {
    // f(x) = sum(truncate(x, 3)^2) = x[0]^2 + x[1]^2 + x[2]^2
    // Hessian = diag(2, 2, 2, 0, 0)
    // HVP with v = [1,1,1,1,1] = [2, 2, 2, 0, 0]
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], x_data.clone()));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));
    let truncated = x.dynamic_truncate(&size, 0);
    let loss = (&truncated * &truncated).reduce_sum(&[0]);

    // Forward-over-Reverse HVP
    let grad = loss.grad(&x).unwrap();
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![1.0; 5]));
    let mut hv = grad.jvp(&x, &v);
    let hv_data = get_f64_data(hv.eval(&mut engine).unwrap());

    assert_eq!(hv_data.len(), 5);
    for i in 0..3 {
        assert!(
            (hv_data[i] - 2.0).abs() < TOL,
            "hv[{i}]={}, expected 2.0",
            hv_data[i]
        );
    }
    for i in 3..5 {
        assert!(
            hv_data[i].abs() < TOL,
            "hv[{i}]={}, expected 0.0",
            hv_data[i]
        );
    }
}

#[test]
fn dynamic_truncate_hvp_finite_diff() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v_data = vec![0.1, -0.2, 0.3, 0.4, -0.5];

    // Finite difference of gradient in direction v
    let compute_grad = |x_vals: &[f64]| -> Vec<f64> {
        let mut engine = Engine::new(CpuBackend::new());
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], x_vals.to_vec()));
        let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));
        let truncated = x.dynamic_truncate(&size, 0);
        let loss = (&truncated * &truncated).reduce_sum(&[0]);
        let mut grad = loss.grad(&x).unwrap();
        get_f64_data(grad.eval(&mut engine).unwrap())
    };

    let mut x_plus = x_data.clone();
    let mut x_minus = x_data.clone();
    for i in 0..5 {
        x_plus[i] += FD_H * v_data[i];
        x_minus[i] -= FD_H * v_data[i];
    }
    let grad_plus = compute_grad(&x_plus);
    let grad_minus = compute_grad(&x_minus);
    let fd_hv: Vec<f64> = grad_plus.iter().zip(&grad_minus).map(|(p, m)| (p - m) / (2.0 * FD_H)).collect();

    // AD HVP
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], x_data));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0));
    let truncated = x.dynamic_truncate(&size, 0);
    let loss = (&truncated * &truncated).reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], v_data));
    let mut hv = grad.jvp(&x, &v);
    let hv_data = get_f64_data(hv.eval(&mut engine).unwrap());

    for i in 0..5 {
        assert!(
            (hv_data[i] - fd_hv[i]).abs() < TOL,
            "HVP[{i}]: ad={}, fd={}",
            hv_data[i],
            fd_hv[i]
        );
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p tenferro --test dynamic_truncate hvp -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tenferro/tests/dynamic_truncate.rs
git commit -m "test: add HVP tests for DynamicTruncate"
```

---

## Task 13: Integration test — checkpoint + DynamicTruncate

**Files:**
- Create: `tenferro/tests/checkpoint_truncate_integration.rs`

- [ ] **Step 1: Write integration test**

```rust
//! Integration test: checkpoint + DynamicTruncate in iterative loop.
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::{CpuBackend, Tensor, TypedTensor};

const TOL: f64 = 1e-4;
const FD_H: f64 = 1e-5;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}
fn f64_scalar(val: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(vec![], vec![val]))
}
fn get_f64_scalar(t: &Tensor) -> f64 {
    match t { Tensor::F64(inner) => inner.host_data()[0], _ => panic!() }
}

/// Iterative computation with truncation + checkpoint:
/// x_{k+1} = truncate(a * x_k, size=2)  (keep first 2 of 3 elements)
/// loss = sum(x_final)
#[test]
fn checkpoint_truncate_loop_grad() {
    let steps = 3;
    let a_val = 0.5_f64;
    let x0_data = vec![1.0, 2.0, 3.0];
    let mut engine = Engine::new(CpuBackend::new());

    let a = TracedTensor::from_tensor_concrete_shape(f64_scalar(a_val));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(2.0));
    let mut x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x0_data.clone()));

    for _ in 0..steps {
        // Scale by a (broadcast scalar * vector)
        x = &a * &x;
        // Truncate to first 2 elements
        x = x.dynamic_truncate(&size, 0);
        x.checkpoint(&mut engine).unwrap();
    }

    let loss = x.reduce_sum(&[0]);
    let mut grad = loss.grad(&a).unwrap();
    let grad_val = get_f64_scalar(grad.eval(&mut engine).unwrap());

    // Finite difference
    let f_concrete = |a_v: f64| -> f64 {
        let mut xc = x0_data.clone();
        for _ in 0..steps {
            xc = xc.iter().map(|v| a_v * v).collect();
            xc.truncate(2);
        }
        xc.iter().sum()
    };
    let fd = (f_concrete(a_val + FD_H) - f_concrete(a_val - FD_H)) / (2.0 * FD_H);

    assert!(
        (grad_val - fd).abs() < TOL,
        "integration: grad={grad_val}, fd={fd}"
    );
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p tenferro --test checkpoint_truncate_integration -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tenferro/tests/checkpoint_truncate_integration.rs
git commit -m "test: integration test for checkpoint + DynamicTruncate loop"
```

---

## Task 14: Full workspace test pass and cleanup

**Files:**
- Modify: `Cargo.toml` (remove patch when tidu-rs PR is merged)

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace --release`
Expected: All pass

- [ ] **Step 2: Check formatting**

Run: `cargo fmt --all --check`
Expected: No formatting issues

- [ ] **Step 3: Build docs**

Run: `cargo doc --workspace --no-deps 2>&1 | grep -i error`
Expected: No doc errors

- [ ] **Step 4: Final commit (if any fixups needed)**

```bash
git add -A && git commit -m "chore: fixups from full test pass"
```

---

## Cross-repo Dependency Note

Tasks 1-2 modify tidu-rs locally via worktree + `[patch]`. Before merging
the tenferro-rs PR:

1. Push tidu-rs changes and create a PR
2. Merge tidu-rs PR
3. Update `Cargo.toml` to point to new tidu-rs rev
4. Remove `[patch]` section
5. Run `cargo update -p tidu`

This follows the cross-repo dependency protocol in AGENTS.md.
