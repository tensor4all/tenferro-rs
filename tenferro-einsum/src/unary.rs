use std::collections::{HashMap, HashSet};

use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::{PrimDescriptor, ReduceOp, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::pool::BufferPool;
use crate::util::alloc_tensor_from_pool;

/// Execute a single-tensor einsum operation via TensorPrims.
pub(crate) fn execute_single_tensor_einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subs_a: &[u32],
    subs_c: &[u32],
    input: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Count label occurrences in input and output
    let mut label_positions: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &l) in subs_a.iter().enumerate() {
        label_positions.entry(l).or_default().push(i);
    }
    let repeated_labels: Vec<u32> = label_positions
        .iter()
        .filter(|(_, pos)| pos.len() > 1)
        .map(|(&l, _)| l)
        .collect();

    let mut output_label_counts: HashMap<u32, usize> = HashMap::new();
    for &l in subs_c {
        *output_label_counts.entry(l).or_insert(0) += 1;
    }
    let output_has_repeated = output_label_counts.values().any(|&c| c > 1);

    if repeated_labels.is_empty() && !output_has_repeated {
        // No repeated labels in input or output
        let input_set: HashSet<u32> = subs_a.iter().copied().collect();
        let output_set: HashSet<u32> = subs_c.iter().copied().collect();

        if input_set == output_set {
            // Pure permutation
            let desc = PrimDescriptor::Permute {
                modes_a: subs_a.to_vec(),
                modes_b: subs_c.to_vec(),
            };
            let shapes = [input.dims(), output.dims()];
            let plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &plan, alpha, &[input], beta, output)
        } else if output_set.is_subset(&input_set) {
            // Reduction (sum over labels not in output)
            let desc = PrimDescriptor::Reduce {
                modes_a: subs_a.to_vec(),
                modes_c: subs_c.to_vec(),
                op: ReduceOp::Sum,
            };
            let shapes = [input.dims(), output.dims()];
            let plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &plan, alpha, &[input], beta, output)
        } else {
            Err(tenferro_device::Error::InvalidArgument(
                "output labels contain labels not in input".into(),
            ))
        }
    } else if !repeated_labels.is_empty() && !output_has_repeated {
        // Repeated labels in input, unique labels in output
        let repeated_in_output: Vec<u32> = repeated_labels
            .iter()
            .filter(|l| subs_c.contains(l))
            .copied()
            .collect();

        if repeated_in_output.is_empty() {
            // Pure trace: all repeated labels are summed
            // Assign unique internal labels to each input dimension
            let mut unique_modes_a = Vec::new();
            let mut paired = Vec::new();
            let mut einsum_to_internal: HashMap<(u32, usize), u32> = HashMap::new();

            for (i, &l) in subs_a.iter().enumerate() {
                let internal = 1000 + i as u32;
                unique_modes_a.push(internal);
                einsum_to_internal.insert((l, i), internal);
            }

            // Build paired list from repeated labels
            for &l in &repeated_labels {
                let positions = &label_positions[&l];
                for pair in positions.windows(2) {
                    let m1 = einsum_to_internal[&(l, pair[0])];
                    let m2 = einsum_to_internal[&(l, pair[1])];
                    paired.push((m1, m2));
                }
            }

            // Build modes_c: map output labels to internal labels of non-repeated input dims
            let unique_input_labels: HashMap<u32, u32> = subs_a
                .iter()
                .enumerate()
                .filter(|(_, &l)| label_positions[&l].len() == 1)
                .map(|(i, &l)| (l, einsum_to_internal[&(l, i)]))
                .collect();

            let modes_c: Vec<u32> = subs_c
                .iter()
                .map(|&l| {
                    unique_input_labels.get(&l).copied().ok_or_else(|| {
                        tenferro_device::Error::InvalidArgument(format!(
                            "output label {l} not found among non-repeated input labels"
                        ))
                    })
                })
                .collect::<Result<_>>()?;

            let desc = PrimDescriptor::Trace {
                modes_a: unique_modes_a,
                modes_c,
                paired,
            };
            let shapes = [input.dims(), output.dims()];
            let plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &plan, alpha, &[input], beta, output)
        } else {
            // Diagonal extraction: repeated labels appear in output
            // Diagonal extraction + copy
            let mut axis_pairs = Vec::new();
            for &l in &repeated_in_output {
                let positions = &label_positions[&l];
                if positions.len() != 2 {
                    return Err(tenferro_device::Error::InvalidArgument(format!(
                        "label {} appears {} times in input; only 2-way diagonal supported",
                        l,
                        positions.len()
                    )));
                }
                axis_pairs.push((positions[0], positions[1]));
            }

            // Extract diagonal as a new Tensor (shares buffer via Arc)
            let diag_tensor = input.diagonal(&axis_pairs)?;

            // Build subscripts after diagonal extraction
            let mut used = vec![false; subs_a.len()];
            for &(a, b) in &axis_pairs {
                used[a] = true;
                used[b] = true;
            }
            let mut after_diag_subs: Vec<u32> = Vec::new();
            for (i, &l) in subs_a.iter().enumerate() {
                if !used[i] {
                    after_diag_subs.push(l);
                }
            }
            for &l in &repeated_in_output {
                after_diag_subs.push(l);
            }

            // Check if we need reduction or just permutation
            let after_set: HashSet<u32> = after_diag_subs.iter().copied().collect();
            let output_set: HashSet<u32> = subs_c.iter().copied().collect();
            let to_reduce: HashSet<u32> = after_set.difference(&output_set).copied().collect();

            if to_reduce.is_empty() {
                // Permute from diagonal layout to output layout
                let desc = PrimDescriptor::Permute {
                    modes_a: after_diag_subs,
                    modes_b: subs_c.to_vec(),
                };
                let shapes = [diag_tensor.dims(), output.dims()];
                let plan = Backend::plan(ctx, &desc, &shapes)?;
                Backend::execute(ctx, &plan, alpha, &[&diag_tensor], beta, output)
            } else {
                // Copy diagonal to contiguous temp, then reduce
                let diag_tensor = diag_tensor.contiguous(MemoryOrder::ColumnMajor);
                let desc = PrimDescriptor::Reduce {
                    modes_a: after_diag_subs,
                    modes_c: subs_c.to_vec(),
                    op: ReduceOp::Sum,
                };
                let shapes = [diag_tensor.dims(), output.dims()];
                let plan = Backend::plan(ctx, &desc, &shapes)?;
                Backend::execute(ctx, &plan, alpha, &[&diag_tensor], beta, output)
            }
        }
    } else if repeated_labels.is_empty() && output_has_repeated {
        // Diagonal embedding: "i->ii"
        // Assign unique internal labels to output dimensions
        let mut unique_modes_c = Vec::new();
        let mut paired = Vec::new();
        let mut label_first_internal: HashMap<u32, u32> = HashMap::new();
        let mut next_label: u32 = 1000;

        for &l in subs_c {
            let internal = next_label;
            next_label += 1;
            unique_modes_c.push(internal);

            if let Some(&first) = label_first_internal.get(&l) {
                paired.push((first, internal));
            } else {
                label_first_internal.insert(l, internal);
            }
        }

        // Map input labels to their internal equivalents
        let modes_a: Vec<u32> = subs_a
            .iter()
            .map(|&l| {
                label_first_internal.get(&l).copied().ok_or_else(|| {
                    tenferro_device::Error::InvalidArgument(format!(
                        "input label {l} not found in output for diagonal embedding"
                    ))
                })
            })
            .collect::<Result<_>>()?;

        let desc = PrimDescriptor::AntiDiag {
            modes_a,
            modes_c: unique_modes_c,
            paired,
        };
        let shapes = [input.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &plan, alpha, &[input], beta, output)
    } else {
        // Both input and output have repeated labels — pipeline decomposition.
        //
        // Strategy:
        //   Stage 1: Diagonal extraction — for labels repeated in input AND present in output
        //   Stage 2: Trace/Reduce — for labels still repeated in input (not in output)
        //   Stage 3+4: Delegate — remaining unique-input to unique/repeated-output
        //
        // Each stage delegates to a recursive call that hits a DIFFERENT branch.

        let output_unique_set: HashSet<u32> = subs_c.iter().copied().collect();

        // Labels repeated in input that also appear in the output → diagonal extraction
        let diag_extract_labels: Vec<u32> = repeated_labels
            .iter()
            .filter(|l| output_unique_set.contains(l))
            .copied()
            .collect();

        let mut current = input.clone();
        let mut current_subs: Vec<u32> = subs_a.to_vec();

        // Stage 1: Diagonal extraction
        if !diag_extract_labels.is_empty() {
            let mut axis_pairs = Vec::new();
            for &l in &diag_extract_labels {
                let positions: Vec<usize> = current_subs
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s == l)
                    .map(|(i, _)| i)
                    .collect();
                for pair in positions.windows(2) {
                    axis_pairs.push((pair[0], pair[1]));
                }
            }
            current = current.diagonal(&axis_pairs)?;

            // Rebuild subscripts: unused positions first, then one copy per diagonal label
            let mut used = vec![false; current_subs.len()];
            for &(a, b) in &axis_pairs {
                used[a] = true;
                used[b] = true;
            }
            let mut new_subs = Vec::new();
            for (i, &l) in current_subs.iter().enumerate() {
                if !used[i] {
                    new_subs.push(l);
                }
            }
            for &l in &diag_extract_labels {
                new_subs.push(l);
            }
            current_subs = new_subs;
        }

        // Stage 2: Trace/Reduce for labels remaining in input but not in output.
        // After diagonal extraction, some labels may still appear repeated in
        // current_subs (those that were repeated in input but absent from output).
        let output_label_set: HashSet<u32> = subs_c.iter().copied().collect();
        let labels_not_in_output: Vec<u32> = {
            let mut seen = HashSet::new();
            current_subs
                .iter()
                .filter(|l| !output_label_set.contains(l))
                .filter(|l| seen.insert(**l))
                .copied()
                .collect()
        };

        if !labels_not_in_output.is_empty() {
            // Intermediate subscripts: keep only labels that appear in output
            let inter_subs: Vec<u32> = current_subs
                .iter()
                .filter(|l| output_label_set.contains(l))
                .copied()
                .collect();
            // Compute intermediate shape from current tensor's dimensions
            let inter_shape: Vec<usize> = inter_subs
                .iter()
                .map(|l| {
                    let pos = current_subs.iter().position(|s| s == l).ok_or_else(|| {
                        tenferro_device::Error::InvalidArgument(format!(
                            "label {l} not found in current subscripts during pipeline decomposition"
                        ))
                    })?;
                    Ok(current.dims()[pos])
                })
                .collect::<Result<_>>()?;
            let mut intermediate = alloc_tensor_from_pool::<Alg::Scalar>(
                &inter_shape,
                output.logical_memory_space(),
                pool,
            );
            // Recursive call for trace/reduce: current_subs → inter_subs
            // inter_subs has no repeated labels, so this hits a different branch.
            execute_single_tensor_einsum::<Alg, Backend>(
                ctx,
                &current_subs,
                &inter_subs,
                &current,
                Alg::Scalar::one(),
                Alg::Scalar::zero(),
                &mut intermediate,
                pool,
            )?;
            current = intermediate;
            current_subs = inter_subs;
        }

        // Stage 3+4: Now current_subs has unique labels. Recursive call handles
        // permute + AntiDiag for repeated output labels (or just permute/identity).
        execute_single_tensor_einsum::<Alg, Backend>(
            ctx,
            &current_subs,
            subs_c,
            &current,
            alpha,
            beta,
            output,
            pool,
        )
    }
}
