# Tropical Unary Einsum AD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify tropical AD to support unary and binary einsum backward, closing the omeinsum-rs parity gap.

**Architecture:** Generalize `tropical_forward_with_argmax` and `tropical_backward` from 2-operand signatures to N-ary `&[&Tensor<T>]`. Dispatch by operand count in backward (unary: scatter, binary: mul_backward). Update `tropical_einsum_rrule`, `TropicalEinsumReverseRule`, and `tracked_tropical_einsum` accordingly.

**Tech Stack:** Rust, tenferro-tropical crate, chainrules AD, TDD with `cargo test -p tenferro-tropical`

---

### Task 1: Generalize forward to N-ary

**Files:**
- Modify: `extension/tenferro-tropical/src/ad.rs:327-437` (`tropical_einsum_forward_with_argmax`)

**Step 1: Update function signature**

Change from `(a: &Tensor<T>, b: &Tensor<T>, subs, batch, free_a, free_b, contracted)` to `(operands: &[&Tensor<T>], subs: &Subscripts, contracted: &[u32])`.

Remove the `_batch`, `_free_a`, `_free_b` parameters — they were unused.

```rust
fn tropical_forward_with_argmax<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<(Tensor<T>, ArgmaxTracker)> {
    let output_modes = &subs.output;

    let views: Vec<_> = operands
        .iter()
        .map(|op| crate::prims::tensor_to_view(*op))
        .collect::<Result<_>>()?;

    // Build output shape: resolve each output mode from the first operand that has it
    let output_shape: Vec<usize> = output_modes
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(operands[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "output mode {m} not found in inputs"
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    // Build contracted dimension sizes from the first operand that has each label
    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(operands[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "contracted mode {m} not in any operand"
            )))
        })
        .collect::<Result<Vec<_>>>()?;
    let contracted_total: usize = contracted_dims.iter().product::<usize>().max(1);

    let total_output: usize = output_shape.iter().product::<usize>().max(1);
    let mut output_data = vec![T::zero(); total_output];
    let mut tracker = ArgmaxTracker::new(&output_shape);

    for_each_index(&output_shape, |out_idx| {
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let mut best = T::zero();
        let mut best_k = 0_usize;

        for k_flat in 0..contracted_total {
            let k_idx = if contracted_dims.is_empty() {
                vec![]
            } else {
                unflatten_index(k_flat, &contracted_dims)
            };

            for (c_pos, &c_mode) in contracted.iter().enumerate() {
                mode_values.insert(c_mode, k_idx[c_pos]);
            }

            // Compute product of all operands at resolved indices
            let mut product = T::from_inner(T::Inner::one());
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                let idx: Vec<usize> = input_modes
                    .iter()
                    .map(|m| *mode_values.get(m).unwrap_or(&0))
                    .collect();
                product = product * views[op_idx].get(&idx);
            }

            let new_sum = best + product;
            if k_flat == 0 || product.inner() == new_sum.inner() {
                best_k = k_flat;
            }
            best = new_sum;
        }

        let out_flat = col_major_flat_index(&output_shape, out_idx);
        output_data[out_flat] = best;
        tracker.indices_mut()[out_flat] = best_k;
    });

    let output = Tensor::<T>::from_slice(&output_data, &output_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    Ok((output, tracker))
}
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-tropical`
Expected: FAIL (callers still pass old signature)

**Step 3: Update callers in `tropical_einsum_rrule`**

Replace the call at line ~308:
```rust
// Before:
let (_output, tracker) =
    tropical_einsum_forward_with_argmax(a, b, &subs, &batch, &free_a, &free_b, &contracted)?;

// After:
let (_output, tracker) =
    tropical_forward_with_argmax(operands, &subs, &contracted)?;
```

Also remove the `batch`, `free_a`, `free_b` computation from `tropical_einsum_rrule` (lines 276-302), and remove the `a`, `b` bindings. The only thing needed before calling forward is:

```rust
// Contracted: labels in any input but not in output
let contracted: Vec<u32> = subs
    .inputs
    .iter()
    .flat_map(|inp| inp.iter())
    .copied()
    .collect::<std::collections::HashSet<_>>()
    .into_iter()
    .filter(|m| !subs.output.contains(m))
    .collect();
```

**Step 4: Run tests**

Run: `cargo test -p tenferro-tropical`
Expected: FAIL (backward caller also needs update)

**Step 5: Commit**

```bash
git add extension/tenferro-tropical/src/ad.rs
git commit -m "refactor(tropical-ad): generalize forward to N-ary operands (#211)"
```

---

### Task 2: Generalize backward to N-ary with unary/binary dispatch

**Files:**
- Modify: `extension/tenferro-tropical/src/ad.rs:439-535` (`tropical_einsum_backward`)

**Step 1: Update function signature and implement dispatch**

```rust
fn tropical_backward<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    match operands.len() {
        1 => tropical_backward_unary(operands[0], cotangent, tracker, subs, contracted),
        2 => tropical_backward_binary(operands, cotangent, tracker, subs, contracted),
        n => Err(Error::InvalidArgument(format!(
            "tropical backward supports 1 or 2 operands, got {n}"
        ))),
    }
}
```

**Step 2: Extract current binary backward into `tropical_backward_binary`**

Move the existing body of `tropical_einsum_backward` into a new `tropical_backward_binary` function. The signature is the same as the current function but takes `operands: &[&Tensor<T>]` and returns `Vec<Tensor<T::Inner>>` (wrapping the `(da, db)` into a Vec):

```rust
fn tropical_backward_binary<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    let a = operands[0];
    let b = operands[1];
    let input_modes_a = &subs.inputs[0];
    let input_modes_b = &subs.inputs[1];
    let output_modes = &subs.output;

    let a_view = crate::prims::tensor_to_view(a)?;
    let b_view = crate::prims::tensor_to_view(b)?;
    let cot_view = crate::prims::tensor_to_view(cotangent)?;

    let output_shape = tracker.output_shape();

    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(operands[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "contracted mode {m} not in any operand"
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut da_data = vec![T::Inner::zero(); a.len()];
    let mut db_data = vec![T::Inner::zero(); b.len()];

    for_each_index(output_shape, |out_idx| {
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let dout = cot_view.get(out_idx);

        let out_flat = col_major_flat_index(output_shape, out_idx);
        let k_winner = tracker.indices()[out_flat];

        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(k_winner, &contracted_dims)
        };

        for (c_pos, &c_mode) in contracted.iter().enumerate() {
            mode_values.insert(c_mode, k_idx[c_pos]);
        }

        let a_idx: Vec<usize> = input_modes_a
            .iter()
            .map(|m| *mode_values.get(m).unwrap_or(&0))
            .collect();
        let b_idx: Vec<usize> = input_modes_b
            .iter()
            .map(|m| *mode_values.get(m).unwrap_or(&0))
            .collect();

        let a_val = a_view.get(&a_idx).inner();
        let b_val = b_view.get(&b_idx).inner();

        let da_contrib = T::mul_backward_a(a_val, b_val, dout);
        let db_contrib = T::mul_backward_b(a_val, b_val, dout);

        let a_flat = col_major_flat_index(a.dims(), &a_idx);
        let b_flat = col_major_flat_index(b.dims(), &b_idx);

        da_data[a_flat] += da_contrib;
        db_data[b_flat] += db_contrib;
    });

    let da = Tensor::<T::Inner>::from_slice(&da_data, a.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    let db = Tensor::<T::Inner>::from_slice(&db_data, b.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;

    Ok(vec![da, db])
}
```

**Step 3: Implement `tropical_backward_unary`**

```rust
fn tropical_backward_unary<T: TropicalScalar>(
    operand: &Tensor<T>,
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    let input_modes = &subs.inputs[0];
    let output_modes = &subs.output;
    let output_shape = tracker.output_shape();

    let cot_view = crate::prims::tensor_to_view(cotangent)?;

    let contracted_dims: Vec<usize> = contracted
        .iter()
        .map(|m| {
            let pos = input_modes
                .iter()
                .position(|x| x == m)
                .ok_or_else(|| Error::InvalidArgument(format!("contracted mode {m} not in input")))?;
            Ok(operand.dims()[pos])
        })
        .collect::<Result<Vec<_>>>()?;

    let mut grad_data = vec![T::Inner::zero(); operand.len()];

    for_each_index(output_shape, |out_idx| {
        let mut mode_values: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (pos, &m) in output_modes.iter().enumerate() {
            mode_values.insert(m, out_idx[pos]);
        }

        let dout = cot_view.get(out_idx);

        let out_flat = col_major_flat_index(output_shape, out_idx);
        let k_winner = tracker.indices()[out_flat];

        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(k_winner, &contracted_dims)
        };

        for (c_pos, &c_mode) in contracted.iter().enumerate() {
            mode_values.insert(c_mode, k_idx[c_pos]);
        }

        let input_idx: Vec<usize> = input_modes
            .iter()
            .map(|m| *mode_values.get(m).unwrap_or(&0))
            .collect();

        let input_flat = col_major_flat_index(operand.dims(), &input_idx);
        grad_data[input_flat] += dout;
    });

    let grad = Tensor::<T::Inner>::from_slice(&grad_data, operand.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    Ok(vec![grad])
}
```

**Step 4: Update `tropical_einsum_rrule` to call new backward**

Replace lines ~311-324:
```rust
// Before:
let (da, db) = tropical_einsum_backward(a, b, cotangent, &tracker, ...)?;
Ok(vec![da, db])

// After:
tropical_backward(operands, cotangent, &tracker, &subs, &contracted)
```

Also change the guard from `operands.len() != 2` to:
```rust
if operands.is_empty() || operands.len() > 2 {
    return Err(Error::InvalidArgument(
        "tropical_einsum_rrule supports 1 or 2 operands".into(),
    ));
}
```

**Step 5: Run tests**

Run: `cargo test -p tenferro-tropical`
Expected: Most pass, but `rrule_rejects_single_operand` now fails (it expects error for 1 operand).

**Step 6: Update `rrule_rejects_single_operand` test**

Rename to `rrule_accepts_single_operand` and verify it succeeds:

```rust
#[test]
fn rrule_accepts_single_operand() {
    let mut ctx = ctx();
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0)], &[2], COL,
    ).unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();
    let result = tropical_einsum_rrule::<
        MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend,
    >(&mut ctx, "i->", &[&a], &grad);
    assert!(result.is_ok());
}
```

**Step 7: Run tests**

Run: `cargo test -p tenferro-tropical`
Expected: PASS

**Step 8: Commit**

```bash
git add extension/tenferro-tropical/src/ad.rs extension/tenferro-tropical/tests/ad_tests.rs
git commit -m "refactor(tropical-ad): generalize backward to N-ary with unary dispatch (#211)"
```

---

### Task 3: Update `TropicalEinsumReverseRule` and `tracked_tropical_einsum`

**Files:**
- Modify: `extension/tenferro-tropical/src/ad.rs:558-747`

**Step 1: Simplify `TropicalEinsumReverseRule` fields**

```rust
pub struct TropicalEinsumReverseRule<T: TropicalScalar> {
    subscripts: Subscripts,
    primals: Vec<Tensor<T>>,
    tracker: ArgmaxTracker,
    input_node_ids: Vec<Option<NodeId>>,
    contracted: Vec<u32>,
}
```

**Step 2: Update pullback to use new backward**

```rust
fn pullback(&self, cotangent: &Tensor<T::Inner>) -> AdResult<Vec<(NodeId, Tensor<T::Inner>)>> {
    let primal_refs: Vec<&Tensor<T>> = self.primals.iter().collect();
    let grads = tropical_backward(
        &primal_refs,
        cotangent,
        &self.tracker,
        &self.subscripts,
        &self.contracted,
    )
    .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    let mut results = Vec::new();
    for (i, grad) in grads.into_iter().enumerate() {
        if let Some(id) = self.input_node_ids[i] {
            results.push((id, grad));
        }
    }
    Ok(results)
}
```

**Step 3: Update `tracked_tropical_einsum`**

Change guard:
```rust
if operands.is_empty() || operands.len() > 2 {
    return Err(chainrules::AutodiffError::InvalidArgument(
        "tracked_tropical_einsum supports 1 or 2 operands".into(),
    ));
}
```

Simplify subscript analysis (remove batch/free_a/free_b):
```rust
let contracted: Vec<u32> = {
    let all_input_labels: std::collections::HashSet<u32> = subs
        .inputs.iter().flat_map(|inp| inp.iter()).copied().collect();
    all_input_labels.into_iter().filter(|m| !subs.output.contains(m)).collect()
};
```

Promote all operands generically:
```rust
let tropical_operands: Vec<Tensor<T>> = operands
    .iter()
    .map(|op| promote_to_tropical::<T>(op.value()))
    .collect::<std::result::Result<_, _>>()
    .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

let tropical_refs: Vec<&Tensor<T>> = tropical_operands.iter().collect();

let (output_tropical, tracker) =
    tropical_forward_with_argmax(&tropical_refs, &subs, &contracted)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;
```

Build rule with simplified fields:
```rust
let rule = TropicalEinsumReverseRule::<T> {
    subscripts: subs,
    primals: tropical_operands,
    tracker,
    input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
    contracted,
};
```

**Step 4: Update `tracked_rejects_single_operand` test**

Rename to `tracked_accepts_single_operand`:
```rust
#[test]
fn tracked_accepts_single_operand() {
    let tape = Tape::<Tensor<f64>>::new();
    let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], COL).unwrap();
    let a = tape.leaf(a_data);

    let result = tracked_tropical_einsum::<
        MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend,
    >("i->", &[&a]);
    assert!(result.is_ok());
}
```

**Step 5: Run tests**

Run: `cargo test -p tenferro-tropical`
Expected: PASS

**Step 6: Commit**

```bash
git add extension/tenferro-tropical/src/ad.rs extension/tenferro-tropical/tests/ad_tests.rs
git commit -m "refactor(tropical-ad): simplify reverse rule and tracked einsum for N-ary (#211)"
```

---

### Task 4: Add unary backward tests

**Files:**
- Modify: `extension/tenferro-tropical/tests/ad_tests.rs`

**Step 1: Add unary rrule tests**

Add these tests after the existing error path tests:

```rust
// ============================================================================
// Unary tropical backward tests
// ============================================================================

#[test]
fn maxplus_unary_trace_backward() {
    // ii-> : max of diagonal elements
    let mut ctx = ctx();
    // A = [[1, 3],    (col-major: [1, 2, 3, 4])
    //      [2, 4]]
    // Diagonal: A[0,0]=1, A[1,1]=4. MaxPlus sum = max(1, 4) = 4
    // Winner: (i=1,j=1) → flat index k=1 in contracted dim (size 2)
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2], COL,
    ).unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend,
    >(&mut ctx, "ii->", &[&a], &grad).unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // Only the winner diagonal element (1,1) = flat index 3 gets gradient
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 0.0); // A[1,0]
    assert_eq!(da[2], 0.0); // A[0,1]
    assert_eq!(da[3], 1.0); // A[1,1] — winner
}

#[test]
fn maxplus_unary_full_contraction_backward() {
    // ij-> : max of all elements
    let mut ctx = ctx();
    // A = [[1, 5],    (col-major: [1, 4, 5, 2])
    //      [4, 2]]
    // Max = 5 at (0,1) → contracted dims [2,2], flat idx for (i=0,j=1) = 0*1+1*2=2
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(4.0), MaxPlus(5.0), MaxPlus(2.0)],
        &[2, 2], COL,
    ).unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend,
    >(&mut ctx, "ij->", &[&a], &grad).unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // Winner is the element with value 5.0
    // Need to verify which element that is: col-major [1, 4, 5, 2]
    // A[0,0]=1, A[1,0]=4, A[0,1]=5, A[1,1]=2
    // Max = 5 at A[0,1] = flat index 2
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 0.0); // A[1,0]
    assert_eq!(da[2], 1.0); // A[0,1] — winner
    assert_eq!(da[3], 0.0); // A[1,1]
}

#[test]
fn maxplus_unary_row_max_backward() {
    // ij->i : max over j for each i (row-wise max)
    let mut ctx = ctx();
    // A = [[1, 5],    (col-major: [1, 4, 5, 2])
    //      [4, 2]]
    // Row 0 (i=0): max(A[0,0]=1, A[0,1]=5) = 5, winner j=1
    // Row 1 (i=1): max(A[1,0]=4, A[1,1]=2) = 4, winner j=0
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(4.0), MaxPlus(5.0), MaxPlus(2.0)],
        &[2, 2], COL,
    ).unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend,
    >(&mut ctx, "ij->i", &[&a], &grad).unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // dA[0,0] = 0 (j=0 didn't win for i=0)
    // dA[1,0] = 1 (j=0 won for i=1)
    // dA[0,1] = 1 (j=1 won for i=0)
    // dA[1,1] = 0 (j=1 didn't win for i=1)
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 1.0); // A[1,0]
    assert_eq!(da[2], 1.0); // A[0,1]
    assert_eq!(da[3], 0.0); // A[1,1]
}

#[test]
fn maxplus_unary_col_max_backward() {
    // ij->j : max over i for each j (column-wise max)
    let mut ctx = ctx();
    // A = [[1, 5],    (col-major: [1, 4, 5, 2])
    //      [4, 2]]
    // Col 0 (j=0): max(A[0,0]=1, A[1,0]=4) = 4, winner i=1
    // Col 1 (j=1): max(A[0,1]=5, A[1,1]=2) = 5, winner i=0
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(4.0), MaxPlus(5.0), MaxPlus(2.0)],
        &[2, 2], COL,
    ).unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend,
    >(&mut ctx, "ij->j", &[&a], &grad).unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // dA[0,0] = 0 (i=0 didn't win for j=0)
    // dA[1,0] = 1 (i=1 won for j=0)
    // dA[0,1] = 1 (i=0 won for j=1)
    // dA[1,1] = 0 (i=1 didn't win for j=1)
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 1.0); // A[1,0]
    assert_eq!(da[2], 1.0); // A[0,1]
    assert_eq!(da[3], 0.0); // A[1,1]
}
```

**Step 2: Add tracked unary test**

```rust
#[test]
fn tracked_maxplus_unary_full_contraction_pullback() {
    let tape = Tape::<Tensor<f64>>::new();
    // A = [[1, 5],    (col-major: [1, 4, 5, 2])
    //      [4, 2]]
    let a_data = Tensor::<f64>::from_slice(&[1.0, 4.0, 5.0, 2.0], &[2, 2], COL).unwrap();
    let a = tape.leaf(a_data);

    // ij-> : max of all = 5
    let c = tracked_tropical_einsum::<
        MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend,
    >("ij->", &[&a]).unwrap();

    assert_eq!(c.value().buffer().as_slice().unwrap()[0], 5.0);

    let grads = tape.pullback(&c).unwrap();
    let ga = grads.get(a.node_id().unwrap()).unwrap();
    let ga_data = ga.buffer().as_slice().unwrap();

    // Winner is A[0,1] = 5.0, flat index 2
    assert_eq!(ga_data[0], 0.0);
    assert_eq!(ga_data[1], 0.0);
    assert_eq!(ga_data[2], 1.0); // winner
    assert_eq!(ga_data[3], 0.0);
}
```

**Step 3: Run tests**

Run: `cargo test -p tenferro-tropical`
Expected: PASS (all old + new tests)

**Step 4: Update doc comments**

Update `tropical_einsum_rrule` doc comment to mention unary support:
```rust
/// Currently supports unary (1 operand) and binary (2 operand) contractions.
/// Unary patterns include trace (`ii->`), full contraction (`ij->`),
/// and partial reduction (`ij->i`, `ij->j`).
```

Update `tracked_tropical_einsum` doc comment similarly.

**Step 5: Run full workspace tests**

Run: `cargo test --workspace`
Expected: PASS

**Step 6: Commit**

```bash
git add extension/tenferro-tropical/src/ad.rs extension/tenferro-tropical/tests/ad_tests.rs
git commit -m "feat(tropical-ad): unary tropical einsum backward with tests (#211)"
```
