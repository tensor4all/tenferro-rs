# Parenthesized Einsum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement parenthesized grouping in einsum so `(ij,jk),kl->il` respects user-specified contraction order, porting OMEinsum.jl's NestedEinsum architecture to Rust.

**Architecture:** Add `NestedEinsum` enum (Leaf/Node) with recursive parser and bottom-up executor to `tenferro-einsum`. The `einsum()` function detects parentheses and dispatches to the nested path. Flat (non-parenthesized) path is unchanged.

**Tech Stack:** Rust, tenferro-einsum crate, existing `Subscripts`/`ContractionTree` types

---

### Task 1: NestedEinsum Data Model + Basic Parse Tests

**Files:**
- Modify: `tenferro-einsum/src/lib.rs` (add `NestedEinsum` enum after `Subscripts` impl block, around line 1358)
- Modify: `tenferro-einsum/tests/einsum_tests.rs` (add parse tests)

**Step 1: Write the failing tests**

Add these tests at the end of `tenferro-einsum/tests/einsum_tests.rs`:

```rust
// ============================================================================
// NestedEinsum parsing
// ============================================================================

#[test]
fn nested_parse_flat_no_parens() {
    // Without parentheses, produces a single root node with all leaves
    let nested = tenferro_einsum::NestedEinsum::parse("ij,jk->ik").unwrap();
    match &nested {
        tenferro_einsum::NestedEinsum::Node { subscripts, children } => {
            assert_eq!(children.len(), 2);
            assert_eq!(subscripts.output, tenferro_einsum::Subscripts::parse("ij,jk->ik").unwrap().output);
            // Children are leaves
            assert!(matches!(children[0], tenferro_einsum::NestedEinsum::Leaf(0)));
            assert!(matches!(children[1], tenferro_einsum::NestedEinsum::Leaf(1)));
        }
        _ => panic!("expected Node"),
    }
}

#[test]
fn nested_parse_simple_group() {
    // (ij,jk),kl->il
    // Root: two children, first is a Node (group), second is Leaf(2)
    let nested = tenferro_einsum::NestedEinsum::parse("(ij,jk),kl->il").unwrap();
    match &nested {
        tenferro_einsum::NestedEinsum::Node { subscripts, children } => {
            assert_eq!(children.len(), 2);
            // Root output is "il"
            let i = 8u32; // 'i' - 'a'
            let l = 11u32; // 'l' - 'a'
            assert_eq!(subscripts.output, vec![i, l]);
            // First child is a Node (the group)
            match &children[0] {
                tenferro_einsum::NestedEinsum::Node { subscripts: inner_subs, children: inner_children } => {
                    assert_eq!(inner_children.len(), 2);
                    assert!(matches!(inner_children[0], tenferro_einsum::NestedEinsum::Leaf(0)));
                    assert!(matches!(inner_children[1], tenferro_einsum::NestedEinsum::Leaf(1)));
                    // Inner output should contain labels needed outside: i and k
                    // i appears in final output, k appears in sibling kl
                    let k = 10u32;
                    assert!(inner_subs.output.contains(&i));
                    assert!(inner_subs.output.contains(&k));
                }
                _ => panic!("expected inner Node"),
            }
            // Second child is Leaf(2)
            assert!(matches!(children[1], tenferro_einsum::NestedEinsum::Leaf(2)));
        }
        _ => panic!("expected Node"),
    }
}

#[test]
fn nested_parse_deeply_nested() {
    // ((ij,jk),kl),lm->im
    let nested = tenferro_einsum::NestedEinsum::parse("((ij,jk),kl),lm->im").unwrap();
    // Should have depth 3: root -> group -> group -> leaves
    match &nested {
        tenferro_einsum::NestedEinsum::Node { children, .. } => {
            assert_eq!(children.len(), 2); // outer group + lm
            match &children[0] {
                tenferro_einsum::NestedEinsum::Node { children: mid, .. } => {
                    assert_eq!(mid.len(), 2); // inner group + kl
                    match &mid[0] {
                        tenferro_einsum::NestedEinsum::Node { children: inner, .. } => {
                            assert_eq!(inner.len(), 2); // ij + jk
                            assert!(matches!(inner[0], tenferro_einsum::NestedEinsum::Leaf(0)));
                            assert!(matches!(inner[1], tenferro_einsum::NestedEinsum::Leaf(1)));
                        }
                        _ => panic!("expected inner Node"),
                    }
                    assert!(matches!(mid[1], tenferro_einsum::NestedEinsum::Leaf(2)));
                }
                _ => panic!("expected mid Node"),
            }
            assert!(matches!(children[1], tenferro_einsum::NestedEinsum::Leaf(3)));
        }
        _ => panic!("expected Node"),
    }
}

#[test]
fn nested_parse_error_mismatched_parens() {
    assert!(tenferro_einsum::NestedEinsum::parse("(ij,jk->ik").is_err());
    assert!(tenferro_einsum::NestedEinsum::parse("ij),jk->ik").is_err());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-einsum nested_parse -- --no-capture 2>&1 | head -30`
Expected: FAIL — `NestedEinsum` does not exist yet.

**Step 3: Implement NestedEinsum enum and parse method**

In `tenferro-einsum/src/lib.rs`, after the `Subscripts` impl block (line 1357), add:

```rust
// ============================================================================
// NestedEinsum
// ============================================================================

/// Recursive einsum tree that respects parenthesized contraction order.
///
/// Mirrors OMEinsum.jl's `NestedEinsum`. Each `Node` holds subscripts
/// for one einsum call and children that are either leaf operands or
/// nested sub-einsums.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::NestedEinsum;
///
/// // (ij,jk),kl->il  — contract A*B first, then result*C
/// let nested = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
/// ```
#[derive(Debug, Clone)]
pub enum NestedEinsum {
    /// Leaf operand: index into the original operand array.
    Leaf(usize),
    /// Node: an einsum over children (which may themselves be nested).
    Node {
        /// Subscripts for this einsum node.
        subscripts: Subscripts,
        /// Children: leaves or nested sub-einsums.
        children: Vec<NestedEinsum>,
    },
}

impl NestedEinsum {
    /// Parse a parenthesized einsum notation into a nested tree.
    ///
    /// Supports arbitrary nesting depth: `((ij,jk),kl),lm->im`.
    /// Without parentheses, behaves like a flat einsum with all leaves
    /// as children of a single root node.
    ///
    /// # Errors
    ///
    /// Returns an error if parentheses are mismatched or notation is malformed.
    pub fn parse(notation: &str) -> Result<Self> {
        let parts: Vec<&str> = notation.split("->").collect();
        if parts.len() != 2 {
            return Err(Error::InvalidArgument(format!(
                "einsum notation must contain exactly one '->', got: {notation}"
            )));
        }
        let lhs = parts[0];
        let rhs = parts[1];

        let output_labels: Vec<u32> = rhs.chars().map(char_to_label).collect::<Result<_>>()?;

        // Counter for assigning leaf indices in left-to-right order
        let mut leaf_counter = 0usize;
        let (node, _) = Self::parse_group(lhs, &output_labels, &mut leaf_counter)?;
        Ok(node)
    }

    /// Parse a group (possibly containing nested parentheses) into a NestedEinsum node.
    ///
    /// `outer_needed` is the set of labels that the parent/final output needs
    /// from this group. Used to compute intermediate output labels for sub-groups.
    ///
    /// Returns (NestedEinsum, Vec<Vec<u32>>) where the second element is the
    /// list of label-vectors for each child (used by the parent to build its Subscripts).
    fn parse_group(
        lhs: &str,
        outer_needed: &[u32],
        leaf_counter: &mut usize,
    ) -> Result<(Self, Vec<Vec<u32>>)> {
        // Split the lhs into top-level items (respecting parentheses)
        let items = Self::split_top_level(lhs)?;

        if items.len() == 1 && !items[0].starts_with('(') {
            // Single bare operand — this is a leaf
            let labels: Vec<u32> = items[0].chars().map(char_to_label).collect::<Result<_>>()?;
            let idx = *leaf_counter;
            *leaf_counter += 1;
            return Ok((Self::Leaf(idx), vec![labels]));
        }

        // Multiple items or a single parenthesized group
        // First, collect all children and their labels
        let mut children = Vec::new();
        let mut child_label_sets: Vec<Vec<u32>> = Vec::new();

        for item in &items {
            if item.starts_with('(') && item.ends_with(')') {
                // Parenthesized group — recurse (strip outer parens)
                let inner = &item[1..item.len() - 1];

                // Compute labels needed from this group:
                // labels in this group that appear in siblings or outer_needed
                let group_labels = Self::collect_labels(inner)?;
                let sibling_labels = Self::collect_sibling_labels(&items, item)?;

                let mut needed: Vec<u32> = group_labels
                    .iter()
                    .filter(|l| {
                        outer_needed.contains(l) || sibling_labels.contains(l)
                    })
                    .copied()
                    .collect();
                needed.sort();
                needed.dedup();

                let (child, _) = Self::parse_group(inner, &needed, leaf_counter)?;
                child_label_sets.push(needed);
                children.push(child);
            } else {
                // Bare operand — leaf
                let labels: Vec<u32> = item.chars().map(char_to_label).collect::<Result<_>>()?;
                child_label_sets.push(labels.clone());
                let idx = *leaf_counter;
                *leaf_counter += 1;
                children.push(Self::Leaf(idx));
            }
        }

        // Build Subscripts for this node
        let node_subs = Subscripts {
            inputs: child_label_sets.clone(),
            output: outer_needed.to_vec(),
        };

        Ok((
            Self::Node {
                subscripts: node_subs,
                children,
            },
            child_label_sets,
        ))
    }

    /// Split a string into top-level comma-separated items, respecting parentheses.
    ///
    /// `"ij,(jk,kl),lm"` → `["ij", "(jk,kl)", "lm"]`
    fn split_top_level(s: &str) -> Result<Vec<&str>> {
        let mut items = Vec::new();
        let mut depth = 0usize;
        let mut start = 0;

        for (i, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    if depth == 0 {
                        return Err(Error::InvalidArgument(
                            "mismatched closing parenthesis in einsum notation".into(),
                        ));
                    }
                    depth -= 1;
                }
                ',' if depth == 0 => {
                    items.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        if depth != 0 {
            return Err(Error::InvalidArgument(
                "mismatched opening parenthesis in einsum notation".into(),
            ));
        }
        items.push(&s[start..]);
        Ok(items)
    }

    /// Collect all unique labels from a (possibly nested) lhs string.
    fn collect_labels(s: &str) -> Result<Vec<u32>> {
        let mut labels = Vec::new();
        for c in s.chars() {
            if c == '(' || c == ')' || c == ',' {
                continue;
            }
            let l = char_to_label(c)?;
            if !labels.contains(&l) {
                labels.push(l);
            }
        }
        Ok(labels)
    }

    /// Collect all labels from sibling items (all items except `current`).
    fn collect_sibling_labels(items: &[&str], current: &str) -> Result<Vec<u32>> {
        let mut labels = Vec::new();
        for item in items {
            if std::ptr::eq(*item, current) {
                continue;
            }
            for c in item.chars() {
                if c == '(' || c == ')' || c == ',' {
                    continue;
                }
                let l = char_to_label(c)?;
                if !labels.contains(&l) {
                    labels.push(l);
                }
            }
        }
        Ok(labels)
    }
}
```

Also add `NestedEinsum` to the public exports. Near the top of the file (around line 186 where `pub use` statements are), add it to the exports. Look for the existing public re-exports and add `NestedEinsum`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p tenferro-einsum nested_parse -- --no-capture`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add tenferro-einsum/src/lib.rs tenferro-einsum/tests/einsum_tests.rs
git commit -m "feat(einsum): add NestedEinsum data model and recursive parser (#207)"
```

---

### Task 2: execute_nested + einsum() Integration

**Files:**
- Modify: `tenferro-einsum/src/lib.rs` (add `execute_nested` function, modify `einsum()`)
- Modify: `tenferro-einsum/tests/einsum_tests.rs` (add execution tests)

**Step 1: Write the failing tests**

Add these tests at the end of `tenferro-einsum/tests/einsum_tests.rs`:

```rust
// ============================================================================
// NestedEinsum execution
// ============================================================================

#[test]
fn nested_einsum_simple_group() {
    // (ij,jk),kl->il should produce same result as ij,jk,kl->il
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL,
    ).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[3, 4], COL,
    ).unwrap();
    let c = Tensor::<f64>::from_slice(
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[4, 2], COL,
    ).unwrap();

    let flat = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    let nested = einsum::<S, CpuBackend>(&mut ctx, "(ij,jk),kl->il", &[&a, &b, &c], None).unwrap();

    assert_eq!(flat.dims(), nested.dims());
    let flat_data = flat.buffer().as_slice().unwrap();
    let nested_data = nested.buffer().as_slice().unwrap();
    for (f, n) in flat_data.iter().zip(nested_data.iter()) {
        assert!((f - n).abs() < 1e-10, "flat={f}, nested={n}");
    }
}

#[test]
fn nested_einsum_deeply_nested() {
    // ((ij,jk),kl),lm->im
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[2.0, 0.0, 0.0, 2.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[3.0, 0.0, 0.0, 3.0], &[2, 2], COL).unwrap();
    let d = Tensor::<f64>::from_slice(&[4.0, 0.0, 0.0, 4.0], &[2, 2], COL).unwrap();

    let flat = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl,lm->im", &[&a, &b, &c, &d], None).unwrap();
    let nested = einsum::<S, CpuBackend>(&mut ctx, "((ij,jk),kl),lm->im", &[&a, &b, &c, &d], None).unwrap();

    assert_eq!(flat.dims(), nested.dims());
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (get(&flat, &[i, j]) - get(&nested, &[i, j])).abs() < 1e-10,
                "mismatch at [{i},{j}]"
            );
        }
    }
}

#[test]
fn nested_einsum_nary_group() {
    // (ij,jk,kl)->il — three operands in one group
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();

    let flat = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    let nested = einsum::<S, CpuBackend>(&mut ctx, "(ij,jk,kl)->il", &[&a, &b, &c], None).unwrap();

    assert_eq!(flat.dims(), nested.dims());
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (get(&flat, &[i, j]) - get(&nested, &[i, j])).abs() < 1e-10,
                "mismatch at [{i},{j}]"
            );
        }
    }
}

#[test]
fn nested_einsum_single_operand_group() {
    // (ij)->ij — trivial single-operand group is identity
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();

    let result = einsum::<S, CpuBackend>(&mut ctx, "(ij)->ij", &[&a], None).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert!((get(&result, &[i, j]) - get(&a, &[i, j])).abs() < 1e-10);
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-einsum nested_einsum -- --no-capture 2>&1 | head -30`
Expected: FAIL — `execute_nested` does not exist yet, and `einsum()` doesn't dispatch to it.

**Step 3: Implement execute_nested and modify einsum()**

Add `execute_nested` as a private function in `tenferro-einsum/src/lib.rs`, near the other `execute_*` functions (around line 1168):

```rust
/// Execute a NestedEinsum tree recursively (bottom-up).
fn execute_nested<Alg, Backend>(
    ctx: &mut Backend::Context,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    match nested {
        NestedEinsum::Leaf(idx) => {
            if *idx >= operands.len() {
                return Err(Error::InvalidArgument(format!(
                    "NestedEinsum leaf index {idx} out of bounds (have {} operands)",
                    operands.len()
                )));
            }
            Ok(operands[*idx].clone())
        }
        NestedEinsum::Node {
            subscripts,
            children,
        } => {
            // Recursively execute each child
            let intermediates: Vec<Tensor<Alg::Scalar>> = children
                .iter()
                .map(|child| execute_nested::<Alg, Backend>(ctx, child, operands, size_dict))
                .collect::<Result<_>>()?;

            let refs: Vec<&Tensor<Alg::Scalar>> = intermediates.iter().collect();
            einsum_with_subscripts::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
        }
    }
}
```

Then modify `einsum()` (at line 1585) to detect parentheses:

Replace the body of `einsum()` — specifically the first line `let subs = Subscripts::parse(subscripts)?;` and `let mut output = einsum_with_subscripts::<Alg, Backend>(ctx, &subs, operands, size_dict)?;` — with:

```rust
    let mut output = if subscripts.contains('(') {
        execute_nested::<Alg, Backend>(ctx, &NestedEinsum::parse(subscripts)?, operands, size_dict)?
    } else {
        let subs = Subscripts::parse(subscripts)?;
        einsum_with_subscripts::<Alg, Backend>(ctx, &subs, operands, size_dict)?
    };
```

And update the forward-mode tangent propagation section to handle both paths. For the parenthesized case, tangent propagation is skipped (per design doc):

```rust
    // Auto-propagate forward-mode tangents (flat path only)
    if !subscripts.contains('(') && operands.iter().any(|t| t.has_fw_grad()) {
        let subs = Subscripts::parse(subscripts)?;
        let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
            operands.iter().map(|t| t.fw_grad()).collect();
        if let Ok(output_tangent) =
            einsum_frule_impl::<Alg, Backend>(ctx, &subs, operands, &tangents)
        {
            output.set_fw_grad(output_tangent);
        }
    }

    Ok(output)
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p tenferro-einsum nested_einsum -- --no-capture`
Expected: All 4 execution tests PASS.

Also run the full einsum test suite to check for regressions:

Run: `cargo test -p tenferro-einsum`
Expected: All existing tests PASS (flat path is unchanged).

**Step 5: Commit**

```bash
git add tenferro-einsum/src/lib.rs tenferro-einsum/tests/einsum_tests.rs
git commit -m "feat(einsum): execute_nested and parenthesized dispatch in einsum() (#207)"
```

---

### Task 3: Update Crate Docs + Public Exports

**Files:**
- Modify: `tenferro-einsum/src/lib.rs` (crate-level docs, pub use)

**Step 1: Update crate-level doc comment**

At the top of `tenferro-einsum/src/lib.rs` (line 7-8), replace:

```rust
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` is accepted but
//!   grouping is currently ignored (optimizer picks order)
```

with:

```rust
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` respects user-specified
//!   contraction order via [`NestedEinsum`] (OMEinsum.jl-compatible)
```

**Step 2: Add NestedEinsum to public exports**

Find the existing public re-export block. Search for `pub use` or `pub struct ContractionTree` near the top. Ensure `NestedEinsum` is publicly accessible. It is defined as `pub enum NestedEinsum` so it's already public, but verify it appears in the crate's public API (it's in the root module so it should be accessible as `tenferro_einsum::NestedEinsum`).

**Step 3: Run doc tests and full test suite**

Run: `cargo test -p tenferro-einsum`
Expected: All tests PASS.

Run: `cargo fmt --all --check`
Expected: No formatting issues.

**Step 4: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "docs(einsum): update crate docs for parenthesized notation support (#207)"
```

---

### Task 4: Full Workspace Checks

**Files:** None (verification only)

**Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests PASS.

**Step 2: Run formatting check**

Run: `cargo fmt --all --check`
Expected: Clean.

**Step 3: Run coverage check**

Run: `cargo llvm-cov --workspace --json --output-path coverage.json && python3 scripts/check-coverage.py coverage.json`
Expected: No new coverage failures (pre-existing failures in unrelated files are acceptable).

**Step 4: Commit any fixes if needed**

If formatting or tests fail, fix and commit. Otherwise, no commit needed.
