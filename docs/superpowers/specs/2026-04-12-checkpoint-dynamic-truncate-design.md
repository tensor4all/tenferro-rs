# Checkpoint + DynamicTruncate Design

## Summary

Add a checkpoint mechanism and dynamic truncation ops to enable memory-efficient
backward passes through iterative TN computations with adaptive truncated SVD.

Related issues: #690 (checkpoint), #691 (DynamicTruncate)

## Implementation Order

Checkpoint first, then DynamicTruncate. Checkpoint involves deeper graph/AD
changes; DynamicTruncate builds on top.

---

## Part 1: Checkpoint Mechanism

### API

```rust
impl TracedTensor {
    /// Evaluate, cache, and promote to leaf for eval efficiency.
    /// AD connectivity preserved via checkpoint_chain.
    pub fn checkpoint(&mut self, engine: &mut Engine<impl TensorBackend>) -> Result<()>;
}
```

### Data Structure: CheckpointNode

```rust
/// Persistent linked list of checkpoint metadata.
/// Shared via Arc — O(1) propagation through downstream ops.
pub(crate) struct CheckpointNode {
    /// The old computation fragment (pre-checkpoint graph).
    pub fragment: Arc<Fragment<StdTensorOp>>,
    /// Maps the new leaf key → old output GlobalValKey for AD continuation.
    pub alias: (TensorInputKey, GlobalValKey<StdTensorOp>),
    /// Concrete input data needed to evaluate the old fragment.
    pub old_inputs: HashMap<TensorInputKey, Tensor>,
    /// Link to previous checkpoint (linked list).
    pub prev: Option<Arc<CheckpointNode>>,
}
```

New field on `TracedTensor`:

```rust
pub(crate) checkpoint_chain: Option<Arc<CheckpointNode>>,
```

### checkpoint() Implementation

```rust
pub fn checkpoint(&mut self, engine: &mut Engine<impl TensorBackend>) -> Result<()> {
    // 1. Evaluate to get concrete data
    self.eval(engine)?;
    let data = self.data.clone().expect("eval populates data");

    // 2. Capture old state
    let old_fragment = self.fragment.clone();
    let old_output_key = old_fragment.vals()[self.val].key.clone();
    let old_inputs = (*self.inputs_map).clone();

    // 3. Create new leaf
    let new_key = next_input_key();
    let mut builder = FragmentBuilder::new();
    let leaf_val = builder.add_input(new_key.clone());
    builder.set_outputs(vec![leaf_val]);
    let new_fragment = Arc::new(builder.build());

    // 4. Build checkpoint node (prepend to chain)
    let node = CheckpointNode {
        fragment: old_fragment,
        alias: (new_key.clone(), old_output_key),
        old_inputs,
        prev: self.checkpoint_chain.take(),
    };

    // 5. Update self
    self.fragment = new_fragment;
    self.val = leaf_val;
    self.data = Some(data.clone());
    self.extra_roots = vec![];  // no longer needed for eval
    self.checkpoint_chain = Some(Arc::new(node));

    // 6. Merge inputs_map: keep old + add new checkpoint key
    let mut merged = (*self.inputs_map).clone();
    merged.insert(new_key, data);
    self.inputs_map = Arc::new(merged);

    Ok(())
}
```

### Propagation

In `apply_unary`, `apply_binary`, and other TracedTensor constructors:
- Clone `checkpoint_chain` from operands (O(1) Arc clone)
- For binary ops: if both operands have chains, keep the longer chain or merge
  (in practice, iterative loops produce a single linear chain shared by all)

### eval() Behavior

`eval()` resolves only `vec![self.fragment.clone()]` — no checkpoint_chain
traversal. Since the checkpointed tensor is a leaf, downstream eval is O(K)
per step.

### AD (grad/vjp) Behavior

When `vjp()`/`grad()` is called:

1. Walk `checkpoint_chain` to collect all fragments and aliases
2. Build alias map: `HashMap<TensorInputKey, GlobalValKey>`
3. Collect all old_inputs into a merged inputs_map
4. Call `resolve()` with: `[self.fragment] + all checkpoint fragments`
5. Pass alias map to `differentiate()`

### tidu::differentiate() Changes

Modify `differentiate()` to accept an alias map:

```rust
pub fn differentiate<Op: GraphOp + PrimitiveOp>(
    view: &ResolvedView<Op>,
    output_key: &GlobalValKey<Op>,
    wrt_keys: &[Op::InputKey],
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,  // NEW
) -> Result<(Arc<Fragment<Op>>, GlobalValKey<Op>)>
```

When the traversal reaches `GlobalValKey::Input(key)`:
- Check `aliases`. If `key` maps to a `GlobalValKey::Derived { ... }`, continue
  traversal from that derived key (expanding the subgraph on demand).
- This enables gradients to flow through checkpoint boundaries.

HVP works recursively: when differentiating a graph that itself contains
checkpoint aliases (from a previous differentiation pass), the same alias
expansion logic applies at the tangent/cotangent level.

---

## Part 2: DynamicTruncate

### Op Definition

```rust
StdTensorOp::DynamicTruncate { axis: usize }
// input[0]: tensor to truncate
// input[1]: scalar tensor (0-rank) — number of elements to keep along axis
// n_inputs: 2, n_outputs: 1
// output shape: runtime-determined (shape_hint = None along axis)
```

### ExecOp

```rust
ExecOp::DynamicTruncate { axis: usize }
```

Execution:
1. Extract `size = input[1]` as usize (round to nearest integer, clamp to
   `0..=input[0].shape[axis]`)
2. Slice `input[0]` along `axis` from `0..size`
3. Return sliced tensor

### Lowering

```
StdTensorOp::DynamicTruncate { axis } → StableHloOp::DynamicTruncate { axis }
                                       → ExecOp::DynamicTruncate { axis }
```

### AD Rules

**Linearize (JVP):**
```rust
// tangent of DynamicTruncate = same truncation applied to tangent
// size input (input[1]) is non-differentiable
fn linearize(tangent_in, primal_in, axis) {
    let dt = tangent_in[0]?;
    DynamicTruncate(dt, primal_in[1], axis)  // same size
    // tangent_in[1] → None (non-differentiable)
}
```

**Transpose (VJP):**
```rust
// adjoint = pad cotangent back to original size
fn transpose(cotangent_out, primal_inputs, axis) {
    let ct = cotangent_out[0]?;
    PadToMatch(ct, primal_inputs[0], axis)
    // no gradient for size input
}
```

---

## Part 3: PadToMatch

### Op Definition

```rust
StdTensorOp::PadToMatch { axis: usize }
// input[0]: tensor to pad with zeros
// input[1]: reference tensor (only shape used, values ignored)
// n_inputs: 2, n_outputs: 1
// output shape: same as input[0] except axis = input[1].shape[axis]
```

### ExecOp

```rust
ExecOp::PadToMatch { axis: usize }
```

Execution:
1. `target_size = input[1].shape[axis]`
2. `pad_amount = target_size - input[0].shape[axis]`
3. Pad `input[0]` with zeros on the high side along `axis`

### AD Rules

**Linearize (JVP):**
```rust
fn linearize(tangent_in, primal_in, axis) {
    let dt = tangent_in[0]?;
    PadToMatch(dt, primal_in[1], axis)  // same padding
    // tangent_in[1] → None
}
```

**Transpose (VJP):**
```rust
fn transpose(cotangent_out, primal_inputs, axis) {
    let ct = cotangent_out[0]?;
    let size = ShapeOf(primal_inputs[0], axis);  // original pre-pad size
    DynamicTruncate(ct, size, axis)
    // no gradient for reference input
}
```

---

## Part 4: ShapeOf

### Op Definition

```rust
StdTensorOp::ShapeOf { axis: usize }
// input[0]: tensor
// n_inputs: 1, n_outputs: 1
// output: scalar tensor (0-rank) = input[0].shape[axis] as f64
```

### ExecOp

```rust
ExecOp::ShapeOf { axis: usize }
```

Execution: `output = Tensor::scalar(input[0].shape[axis] as f64)`

### AD Rules

```rust
// Shape does not depend on tensor values
fn linearize(tangent_in, ..) -> vec![None]
fn transpose(cotangent_out, ..) -> vec![None]
```

---

## Shape Handling

DynamicTruncate output has runtime-determined shape. For the initial
implementation:
- `shape_hint` for the truncated axis = `None` (unknown)
- Downstream ops that need shapes (NaryEinsum, DotGeneral) resolve shapes at
  execution time from concrete tensors

Future: add `SymDim::Dynamic` variant if static shape analysis becomes needed.

---

## Edge Cases

- **size = 0**: DynamicTruncate produces empty tensor along axis. Valid.
- **size > input.shape[axis]**: Clamp to input.shape[axis] (no-op truncation).
- **size negative or non-integer**: Cast to usize with saturation (negative → 0).
- **PadToMatch where input already matches**: pad_amount = 0, no-op. Valid.

---

## Testing Strategy

| Test | Content |
|------|---------|
| checkpoint forward | eval after checkpoint is O(K), not O(kK) |
| checkpoint grad | `loss.grad(&x0)` correct through checkpoint (finite-diff) |
| checkpoint HVP | 2nd-order differentiation through checkpoint (finite-diff) |
| checkpoint loop | L-step loop with sqrt(L) checkpoints, verify correctness |
| DynamicTruncate forward | `[1,2,3,4,5]` + size=3 → `[1,2,3]` |
| DynamicTruncate JVP | finite-diff verification |
| DynamicTruncate VJP | finite-diff verification |
| DynamicTruncate HVP | 2nd-order finite-diff |
| PadToMatch forward | padding correctness |
| PadToMatch JVP/VJP | finite-diff verification |
| ShapeOf forward | extracts correct axis size |
| Integration: truncated SVD | SVD + rank determination + DynamicTruncate pipeline |
| Integration: checkpoint + DynamicTruncate | iterative sweep with adaptive truncation |

All AD tests use finite-difference validation with appropriate tolerances.

---

## Open Items Resolved

| Issue | Resolution |
|-------|-----------|
| inputs_map dropped | Merge, not replace |
| extra_roots O(L^2) | checkpoint_chain (Arc linked list), eval doesn't touch |
| tidu changes understated | Explicit: alias map arg + traversal expansion logic |
| SymDim has no Unknown | shape_hint = None for initial impl |
| PadToMatch transpose needs size | ShapeOf op |
