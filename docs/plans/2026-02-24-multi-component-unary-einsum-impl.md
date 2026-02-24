# Multi-Component Unary Einsum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix multi-component Trace/AntiTrace/AntiDiag execution and add pipeline decomposition for all unary einsum patterns.

**Architecture:** Union-find at plan time computes connected components from `paired` edges. Execution iterates the Cartesian product of component dimensions. Einsum unary lowering decomposes complex patterns into a pipeline of Diag view → Trace/Reduce → Permute → AntiDiag/AntiTrace stages.

**Tech Stack:** Rust, tenferro-prims (CPU backend), tenferro-einsum, tenferro-tensor

---

## Task 1: Add union-find component analysis to CpuPlan (tenferro-prims)

**Files:**
- Modify: `tenferro-prims/src/cpu.rs:44-94` (CpuPlan enum — add component fields to Trace/AntiTrace/AntiDiag variants)
- Modify: `tenferro-prims/src/cpu.rs:286-396` (build_plan for Trace/AntiTrace/AntiDiag — compute components)

**Step 1: Write failing test — multi-component trace `iijj->`**

Add inside the `typed_prims_tests!` macro in `tenferro-prims/tests/prims_tests.rs` (after `trace_2d_matrix` test):

```rust
#[test]
fn trace_multi_component_iijj() {
    // iijj-> : two independent paired components
    // Component 0: axes {0,1} dim=3, Component 1: axes {2,3} dim=4
    // Y = sum_{i,j} A[i,i,j,j]
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 3, 4, 4], |idx| {
        <$T as TestScalar>::from_usize(idx[0] * 100 + idx[1] * 10 + idx[2] + idx[3])
    });
    let mut c = tensor_zeros::<$T>(&[]);

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1, 2, 3],
        modes_c: vec![],
        paired: vec![(0, 1), (2, 3)],
    };
    let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3, 3, 4, 4], &[]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        <$T as TestScalar>::from_f64(1.0),
        &[&a],
        <$T as TestScalar>::from_f64(0.0),
        &mut c,
    )
    .unwrap();

    // Expected: sum_{i=0..3, j=0..4} A[i,i,j,j]
    let mut expected = <$T as TestScalar>::from_f64(0.0);
    for i in 0..3usize {
        for j in 0..4usize {
            expected = expected + tensor_get(&a, &[i, i, j, j]);
        }
    }
    assert!(
        <$T as TestScalar>::approx_eq(tensor_get(&c, &[]), expected),
        "trace = {:?}, expected {:?}, diff = {}",
        tensor_get(&c, &[]),
        expected,
        <$T as TestScalar>::diff_norm(tensor_get(&c, &[]), expected)
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-prims trace_multi_component_iijj -- --include-ignored`
Expected: FAIL — single `d` loop produces wrong sum.

**Step 3: Add component metadata to CpuPlan and build_plan**

In `tenferro-prims/src/cpu.rs`:

a) Add component fields to `CpuPlan::Trace`, `CpuPlan::AntiTrace`, `CpuPlan::AntiDiag`:

```rust
// Inside CpuPlan::Trace:
Trace {
    paired_axes: Vec<(usize, usize)>,
    free_axes: Vec<usize>,
    /// Connected components: each inner Vec contains axis positions that share one diagonal index.
    components: Vec<Vec<usize>>,
    /// Dimension of each component's shared diagonal.
    comp_dims: Vec<usize>,
    _marker: PhantomData<T>,
},
```

Same pattern for `AntiTrace` and `AntiDiag`.

b) In `build_plan` for `PrimDescriptor::Trace` (around line 286), after computing `paired_axes` and `free_axes`, add union-find to compute components:

```rust
// Union-find over paired axes
let all_paired_positions: Vec<usize> = paired_axes
    .iter()
    .flat_map(|&(a, b)| [a, b])
    .collect();
let mut parent: HashMap<usize, usize> = HashMap::new();
for &pos in &all_paired_positions {
    parent.entry(pos).or_insert(pos);
}
fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
    let p = parent[&x];
    if p == x { return x; }
    let root = find(parent, p);
    parent.insert(x, root);
    root
}
fn union(parent: &mut HashMap<usize, usize>, a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent.insert(rb, ra);
    }
}
for &(a, b) in &paired_axes {
    union(&mut parent, a, b);
}

// Group axes by component root
let mut comp_map: HashMap<usize, Vec<usize>> = HashMap::new();
for &pos in &all_paired_positions {
    let root = find(&mut parent, pos);
    comp_map.entry(root).or_default().push(pos);
}
// Deterministic ordering by smallest axis in each component
let mut components: Vec<Vec<usize>> = comp_map.into_values().collect();
components.sort_by_key(|c| c[0]);
for comp in &mut components {
    comp.sort();
    comp.dedup();
}
let comp_dims: Vec<usize> = components
    .iter()
    .map(|c| shapes[0][c[0]])  // shapes[0] for Trace input; shapes[1] for AntiTrace/AntiDiag output
    .collect();
```

Apply the same pattern in `build_plan` for `AntiTrace` and `AntiDiag` (using `shapes[1]` for output dimensions).

**Step 4: Update execute_trace to use Cartesian product**

Replace the body of `execute_trace` (lines 603-636) with multi-component iteration:

```rust
fn execute_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        // Odometer over component dimensions
        let mut comp_idx = vec![0usize; n_comps];
        loop {
            let mut in_idx = vec![0; in_dims.len()];
            // Set free axes from output
            for (out_pos, &in_ax) in free_axes.iter().enumerate() {
                in_idx[in_ax] = out_idx[out_pos];
            }
            // Set component axes
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    in_idx[ax] = comp_idx[t];
                }
            }
            sum = sum + input.get(&in_idx);

            // Increment odometer
            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry { break; }
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}
```

Update the `execute` match arm for `CpuPlan::Trace` to pass `components` and `comp_dims` instead of `paired_axes`.

**Step 5: Run test to verify it passes**

Run: `cargo test -p tenferro-prims trace_multi_component_iijj`
Expected: PASS

**Step 6: Run all existing tests for regression**

Run: `cargo test -p tenferro-prims`
Expected: All pass (single-component cases produce one component, same behavior).

**Step 7: Commit**

```bash
git add tenferro-prims/src/cpu.rs tenferro-prims/tests/prims_tests.rs
git commit -m "feat(prims): multi-component Trace via union-find + Cartesian product"
```

---

## Task 2: Multi-component AntiTrace and AntiDiag execution

**Files:**
- Modify: `tenferro-prims/src/cpu.rs:639-706` (execute_anti_trace, execute_anti_diag)
- Modify: `tenferro-prims/tests/prims_tests.rs` (add multi-component tests)

**Step 1: Write failing tests**

Add inside `typed_prims_tests!` macro:

```rust
#[test]
fn anti_trace_multi_component() {
    // scalar -> [3,3,4,4], paired=[(0,1),(2,3)]
    // Two components: {0,1} dim=3, {2,3} dim=4
    // C[i,j,k,l] = A[] for all i==j AND k==l, else 0
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[], |_| <$T as TestScalar>::from_f64(7.0));
    let mut c = tensor_zeros::<$T>(&[3, 3, 4, 4]);

    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1, 2, 3],
        paired: vec![(0, 1), (2, 3)],
    };
    let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[], &[3, 3, 4, 4]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        <$T as TestScalar>::from_f64(1.0),
        &[&a],
        <$T as TestScalar>::from_f64(0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..4 {
                for l in 0..4 {
                    let expected = if i == j && k == l {
                        <$T as TestScalar>::from_f64(7.0)
                    } else {
                        <$T as TestScalar>::from_f64(0.0)
                    };
                    assert!(
                        <$T as TestScalar>::approx_eq(tensor_get(&c, &[i, j, k, l]), expected),
                        "C[{i},{j},{k},{l}] = {:?}, expected {:?}",
                        tensor_get(&c, &[i, j, k, l]),
                        expected,
                    );
                }
            }
        }
    }
}

#[test]
fn anti_diag_generative_scalar_to_diagonal() {
    // scalar -> [4,4], no input axes, paired=[(0,1)]
    // Generative component: must loop over dim=4
    // C[i,j] = A[] if i==j, else 0
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[], |_| <$T as TestScalar>::from_f64(3.0));
    let mut c = tensor_zeros::<$T>(&[4, 4]);

    let desc = PrimDescriptor::AntiDiag {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[], &[4, 4]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        <$T as TestScalar>::from_f64(1.0),
        &[&a],
        <$T as TestScalar>::from_f64(0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j {
                <$T as TestScalar>::from_f64(3.0)
            } else {
                <$T as TestScalar>::from_f64(0.0)
            };
            assert!(
                <$T as TestScalar>::approx_eq(tensor_get(&c, &[i, j]), expected),
                "C[{i},{j}] = {:?}, expected {:?}",
                tensor_get(&c, &[i, j]),
                expected,
            );
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-prims anti_trace_multi_component anti_diag_generative`
Expected: FAIL

**Step 3: Update execute_anti_trace with Cartesian product**

Replace `execute_anti_trace` (lines 639-671) with component-based iteration:

```rust
fn execute_anti_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    scale_output(output, beta);
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        // Iterate Cartesian product of component dimensions
        let mut comp_idx = vec![0usize; n_comps];
        loop {
            let mut out_idx = vec![0; out_dims.len()];
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    out_idx[ax] = comp_idx[t];
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry { break; }
        }
    });
    Ok(())
}
```

**Step 4: Update execute_anti_diag with generative components**

Replace `execute_anti_diag` (lines 674-706). Key difference: AntiDiag has both anchored components (axes in `modes_a`) and generative components (axes not in `modes_a`). Generative components must be iterated explicitly.

```rust
fn execute_anti_diag<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
    generative_comps: &[usize],  // indices into components that have no input anchor
) -> Result<()> {
    scale_output(output, beta);
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();

    // For generative components, compute their Cartesian product dims
    let gen_dims: Vec<usize> = generative_comps.iter().map(|&c| comp_dims[c]).collect();

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        // Iterate over generative component combinations
        let mut gen_idx = vec![0usize; generative_comps.len()];
        loop {
            let mut out_idx = vec![0; out_dims.len()];
            // Set free axes from input
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            // Set anchored components: copy from the free axis that anchors them
            for (t, comp) in components.iter().enumerate() {
                if generative_comps.contains(&t) {
                    let gi = generative_comps.iter().position(|&c| c == t).unwrap();
                    for &ax in comp {
                        out_idx[ax] = gen_idx[gi];
                    }
                } else {
                    // Anchored: value already set by free_axes assignment
                    // Just propagate paired constraint
                    let anchor_val = out_idx[comp[0]];
                    for &ax in &comp[1..] {
                        out_idx[ax] = anchor_val;
                    }
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            // Increment generative odometer
            if gen_dims.is_empty() { break; }
            let mut carry = true;
            for g in 0..gen_dims.len() {
                if carry {
                    gen_idx[g] += 1;
                    if gen_idx[g] < gen_dims[g] {
                        carry = false;
                    } else {
                        gen_idx[g] = 0;
                    }
                }
            }
            if carry { break; }
        }
    });
    Ok(())
}
```

In `build_plan` for AntiDiag, compute `generative_comps`:

```rust
let free_ax_set: HashSet<usize> = free_axes.iter().copied().collect();
let generative_comps: Vec<usize> = components
    .iter()
    .enumerate()
    .filter(|(_, comp)| comp.iter().all(|ax| !free_ax_set.contains(ax)))
    .map(|(i, _)| i)
    .collect();
```

Store `generative_comps` in `CpuPlan::AntiDiag`.

**Step 5: Run tests to verify they pass**

Run: `cargo test -p tenferro-prims anti_trace_multi_component anti_diag_generative`
Expected: PASS

**Step 6: Run full regression**

Run: `cargo test -p tenferro-prims`
Expected: All pass.

**Step 7: Commit**

```bash
git add tenferro-prims/src/cpu.rs tenferro-prims/tests/prims_tests.rs
git commit -m "feat(prims): multi-component AntiTrace + generative AntiDiag"
```

---

## Task 3: Einsum pipeline — unignore `iijj->` and `->ii`/`->iii` tests

**Files:**
- Modify: `tenferro-einsum/tests/einsum_tests.rs:1515` (remove `#[ignore]` from `einsum_multi_pair_trace_iijj`)
- Modify: `tenferro-einsum/tests/einsum_tests.rs:1696` (remove `#[ignore]` from `einsum_size_dict_scalar_to_diagonal_and_superdiagonal`)

**Step 1: Remove `#[ignore]` attributes**

In `tenferro-einsum/tests/einsum_tests.rs`:
- Line 1515: Remove `#[ignore = "opteinsum parity target: multi-pair trace value differs in current backend"]`
- Line 1696: Remove `#[ignore = "opteinsum parity target: scalar -> repeated-output embedding not yet aligned"]`

**Step 2: Run `iijj->` test**

Run: `cargo test -p tenferro-einsum einsum_multi_pair_trace_iijj`
Expected: PASS (einsum lowering already builds correct multi-pair Trace descriptor; Task 1 fixed execution).

**Step 3: Run `->ii` / `->iii` test**

Run: `cargo test -p tenferro-einsum einsum_size_dict_scalar_to_diagonal_and_superdiagonal`
Expected: PASS (einsum lowering already builds AntiDiag with generative paired; Task 2 fixed execution).

If either fails, debug and fix the einsum lowering in `execute_single_tensor_einsum`.

**Step 4: Run full einsum test suite**

Run: `cargo test -p tenferro-einsum`
Expected: All pass.

**Step 5: Commit**

```bash
git add tenferro-einsum/tests/einsum_tests.rs
git commit -m "test(einsum): unignore iijj-> and ->ii/->iii parity tests"
```

---

## Task 4: Einsum pipeline — handle input+output repeated labels

**Files:**
- Modify: `tenferro-einsum/src/lib.rs:703-707` (replace error path with pipeline decomposition)
- Modify: `tenferro-einsum/tests/einsum_tests.rs` (add new tests)

**Step 1: Write failing tests for input+output repeated**

Add to `tenferro-einsum/tests/einsum_tests.rs`:

```rust
#[test]
fn einsum_input_output_repeated_iij_to_jj() {
    // iij->jj : trace over i, then embed j diagonally
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[3, 3, 2], COL).unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "iij->jj", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[2, 2]);

    for j1 in 0..2 {
        for j2 in 0..2 {
            let expected = if j1 == j2 {
                // sum_i A[i,i,j]
                let mut s = 0.0;
                for i in 0..3 {
                    s += get(&a, &[i, i, j1]);
                }
                s
            } else {
                0.0
            };
            assert!(
                (get(&y, &[j1, j2]) - expected).abs() < 1e-10,
                "y[{j1},{j2}] = {}, expected {expected}",
                get(&y, &[j1, j2])
            );
        }
    }
}

#[test]
fn einsum_input_output_repeated_ii_to_ii() {
    // ii->ii : extract diagonal, then embed back
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[3, 3], COL).unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "ii->ii", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[3, 3]);

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { get(&a, &[i, i]) } else { 0.0 };
            assert!(
                (get(&y, &[i, j]) - expected).abs() < 1e-10,
                "y[{i},{j}] = {}, expected {expected}",
                get(&y, &[i, j])
            );
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-einsum einsum_input_output_repeated`
Expected: FAIL with "simultaneous repeated labels in both input and output not yet supported"

**Step 3: Implement pipeline decomposition**

Replace the error path at `tenferro-einsum/src/lib.rs:703-707` with a pipeline that:

1. Classifies each label into: trace (repeated input, not in output unique), diag-extract (repeated input, in output unique), reduce (unique input, not in output), free (unique input, unique output), duplicate (output repeated).

2. Applies stages:
   - **Stage 1 (Diag)**: For labels repeated in input that appear (uniquely) in output, call `input.diagonal(&axis_pairs)` to extract diagonal view.
   - **Stage 2 (Trace/Reduce)**: For remaining labels in input but not in output, build Trace (if from paired) or Reduce prim and execute.
   - **Stage 3 (Permute)**: If axis order doesn't match output, permute.
   - **Stage 4 (AntiDiag)**: For labels repeated in output, build AntiDiag prim and execute.

```rust
} else {
    // Both input and output have repeated labels — pipeline decomposition
    // Step 1: Classify labels
    let output_unique_set: HashSet<u32> = subs_c.iter().copied().collect();

    // Labels repeated in input that appear uniquely in output → diagonal extraction
    let diag_extract_labels: Vec<u32> = repeated_labels
        .iter()
        .filter(|l| output_unique_set.contains(l))
        .copied()
        .collect();
    // Labels repeated in input that do NOT appear in output → trace
    let trace_labels: Vec<u32> = repeated_labels
        .iter()
        .filter(|l| !output_unique_set.contains(l))
        .copied()
        .collect();

    // Stage 1: Diagonal extraction
    let mut current = input.clone();
    let mut current_subs: Vec<u32> = subs_a.to_vec();

    if !diag_extract_labels.is_empty() {
        let mut axis_pairs = Vec::new();
        for &l in &diag_extract_labels {
            let positions: Vec<usize> = current_subs
                .iter().enumerate()
                .filter(|(_, &s)| s == l)
                .map(|(i, _)| i)
                .collect();
            // Pair consecutive occurrences for extraction
            for pair in positions.windows(2) {
                axis_pairs.push((pair[0], pair[1]));
            }
        }
        current = current.diagonal(&axis_pairs)?;

        // Rebuild subscripts: remove paired positions, append diagonal labels
        let mut used = vec![false; current_subs.len()];
        for &(a, b) in &axis_pairs {
            used[a] = true;
            used[b] = true;
        }
        let mut new_subs: Vec<u32> = Vec::new();
        for (i, &l) in current_subs.iter().enumerate() {
            if !used[i] { new_subs.push(l); }
        }
        for &l in &diag_extract_labels { new_subs.push(l); }
        current_subs = new_subs;
    }

    // Stage 2: Trace over trace_labels, Reduce over non-output unique labels
    let output_unique: HashSet<u32> = subs_c.iter().copied().collect();
    let labels_to_remove: Vec<u32> = current_subs
        .iter()
        .filter(|l| !output_unique.contains(l))
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if !labels_to_remove.is_empty() {
        // Build intermediate output subs (remove labels_to_remove)
        let inter_subs: Vec<u32> = current_subs
            .iter()
            .filter(|l| !labels_to_remove.contains(l))
            .copied()
            .collect();
        let inter_shape: Vec<usize> = inter_subs
            .iter()
            .map(|l| {
                let pos = current_subs.iter().position(|s| s == l).unwrap();
                current.dims()[pos]
            })
            .collect();
        let mut inter = Tensor::<Alg::Scalar>::zeros(
            &inter_shape,
            output.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        );

        // Check if any labels_to_remove are from trace pairs
        let has_trace = trace_labels.iter().any(|l| labels_to_remove.contains(l));
        if has_trace {
            // Use recursive call for trace+reduce (handles Trace prim internally)
            execute_single_tensor_einsum::<Alg, Backend>(
                ctx, &current_subs, &inter_subs, &current,
                Alg::Scalar::one(), Alg::Scalar::zero(), &mut inter,
            )?;
        } else {
            // Pure reduce
            let desc = PrimDescriptor::Reduce {
                modes_a: current_subs.clone(),
                modes_c: inter_subs.clone(),
                op: ReduceOp::Sum,
            };
            let shapes = [current.dims(), inter.dims()];
            let plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &plan, Alg::Scalar::one(), &[&current], Alg::Scalar::zero(), &mut inter)?;
        }

        current = inter;
        current_subs = inter_subs;
    }

    // Stage 3+4: Handle output repeated labels (permute + AntiDiag)
    // Delegate to the existing output-repeated-only path via recursive call
    execute_single_tensor_einsum::<Alg, Backend>(
        ctx, &current_subs, subs_c, &current,
        alpha, beta, output,
    )
}
```

Note: The recursive calls reuse existing code paths — the intermediate tensor after Stage 1+2 has no repeated input labels, so the call falls into the existing `repeated_labels.is_empty() && output_has_repeated` branch.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p tenferro-einsum einsum_input_output_repeated`
Expected: PASS

**Step 5: Run full regression**

Run: `cargo test -p tenferro-einsum`
Expected: All pass.

**Step 6: Commit**

```bash
git add tenferro-einsum/src/lib.rs tenferro-einsum/tests/einsum_tests.rs
git commit -m "feat(einsum): pipeline decomposition for input+output repeated labels"
```

---

## Task 5: Final checks — formatting, coverage, cleanup

**Files:**
- All modified files

**Step 1: Check formatting**

Run: `cargo fmt --all --check`
If fails: `cargo fmt --all`

**Step 2: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All pass with no ignored parity tests.

**Step 3: Run coverage check**

Run: `cargo llvm-cov --workspace --json --output-path coverage.json && python3 scripts/check-coverage.py coverage.json`
Expected: Pass thresholds.

**Step 4: Commit any formatting fixes**

```bash
git add -A && git commit -m "style: format"
```

---

## Summary

| Task | What | Crate |
|------|------|-------|
| 1 | Multi-component Trace (union-find + Cartesian product) | tenferro-prims |
| 2 | Multi-component AntiTrace + generative AntiDiag | tenferro-prims |
| 3 | Unignore `iijj->` and `->ii`/`->iii` tests | tenferro-einsum |
| 4 | Pipeline decomposition for input+output repeated | tenferro-einsum |
| 5 | Formatting, coverage, final checks | workspace |
