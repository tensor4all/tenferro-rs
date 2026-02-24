# Parenthesized Einsum Design

**Date**: 2026-02-24
**Issue**: [#207](https://github.com/tensor4all/tenferro-rs/issues/207)

## Summary

Implement parenthesized grouping in einsum expressions so that
`(ij,jk),kl->il` respects user-specified contraction order. Port
OMEinsum.jl's `NestedEinsum` architecture to Rust.

## Background

The current `Subscripts::parse` strips parentheses and feeds a flat operand
list into `ContractionTree`. Users cannot control which operands contract
first without manually constructing a `ContractionTree::from_pairs`.

OMEinsum.jl solves this with `NestedEinsum`: a recursive tree where each
node holds an `EinCode` (subscripts) and children that are either leaf
operands or nested sub-einsums.

## Design

### A. Data Model — `NestedEinsum`

```rust
pub enum NestedEinsum {
    /// Leaf: index into the original operand list.
    Leaf(usize),
    /// Node: an einsum over children (which may themselves be nested).
    Node {
        subscripts: Subscripts,
        children: Vec<NestedEinsum>,
    },
}
```

- **N-ary children**: Each node can have 2+ children (not restricted to
  binary). Internally, multi-child nodes are optimized via
  `ContractionTree::optimize`.
- **Leaf indices** refer to the original operand array passed to `einsum()`.

### B. Parsing — `NestedEinsum::parse`

`NestedEinsum::parse(notation: &str) -> Result<NestedEinsum>`

The parser:
1. Splits on `->` to get `lhs` and `rhs`.
2. Recursively parses `lhs` into groups, handling nested parentheses.
3. Each parenthesized group `(A,B,...)` becomes a `Node` whose output
   labels are inferred: the set of labels that appear in that group AND
   are needed by any sibling or the final output.
4. Top-level comma-separated items become children of the root `Node`.
5. Items without parentheses become `Leaf` nodes with operand index
   assigned in left-to-right order.

Example: `(ij,jk),kl->il`

```
Root Node: subscripts = "ik,kl->il"
  ├── Node: subscripts = "ij,jk->ik"
  │     ├── Leaf(0)  [ij]
  │     └── Leaf(1)  [jk]
  └── Leaf(2)  [kl]
```

The intermediate output labels for each group node are computed as:
labels appearing in the group that also appear outside the group (in
siblings or the final output).

### C. Execution — `execute_nested`

```rust
fn execute_nested<Alg, Backend>(
    ctx: &mut Backend::Context,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
```

Recursive bottom-up execution:
1. For `Leaf(i)`: return `operands[i].clone()`.
2. For `Node { subscripts, children }`:
   a. Recursively execute each child to get intermediate tensors.
   b. Call `einsum_with_subscripts` (or `einsum_with_plan` via
      `ContractionTree::optimize`) on the intermediate tensors with
      the node's subscripts.

### D. API Integration

`einsum()` is extended to detect parentheses and branch:

```rust
pub fn einsum<Alg, Backend>(..., subscripts: &str, ...) -> Result<...> {
    if subscripts.contains('(') {
        let nested = NestedEinsum::parse(subscripts)?;
        execute_nested::<Alg, Backend>(ctx, &nested, operands, size_dict)
    } else {
        // existing flat path
        let subs = Subscripts::parse(subscripts)?;
        ...
    }
}
```

The `einsum_owned`, `einsum_into`, and AD variants (`einsum_rrule`,
`einsum_frule`) are NOT modified in this phase. They continue to strip
parentheses and use the flat path. Nested AD support is future work.

### E. Correspondence with OMEinsum.jl

| OMEinsum.jl | tenferro-rs | Notes |
|-------------|-------------|-------|
| `NestedEinsum` | `NestedEinsum` | Enum with Leaf/Node |
| `parse_nested()` | `NestedEinsum::parse()` | Recursive parser |
| `IndexGroup` | `Subscripts` (reused) | Label arrays |
| `NestedEinsum(args, eins)` | `Node { children, subscripts }` | N-ary node |
| `NestedEinsum(idx)` | `Leaf(usize)` | Operand reference |

### F. Tests

- **Parsing**: `(ij,jk),kl->il` produces correct tree structure.
- **Execution**: `(ij,jk),kl->il` produces same result as flat
  `ij,jk,kl->il`.
- **Deeply nested**: `((ij,jk),kl),lm->im` works.
- **N-ary group**: `(ij,jk,kl)->il` contracts three at once.
- **Single operand group**: `(ij)->ij` is identity.
- **Error cases**: mismatched parentheses, empty groups.
