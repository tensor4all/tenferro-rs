use std::collections::{HashMap, HashSet};

use tenferro_algebra::{Conjugate, Scalar};
use tenferro_device::{LogicalMemorySpace, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::syntax::subscripts::Subscripts;

pub(super) fn make_delta<T: Scalar>(n: usize, _space: LogicalMemorySpace) -> Result<Tensor<T>> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    Tensor::from_slice(&data, &[n, n], MemoryOrder::ColumnMajor)
}

pub(super) struct DeltaContext<T> {
    pub delta_tensors: Vec<Tensor<T>>,
    pub base_subs: Subscripts,
    pub embed_subs: Option<Subscripts>,
}

pub(super) fn build_delta_context<T: Scalar>(
    subs: &Subscripts,
    rev_inputs_subs: &[Vec<u32>],
    rev_output: &[u32],
    size_dict: &HashMap<u32, usize>,
    space: LogicalMemorySpace,
) -> Result<DeltaContext<T>> {
    let all_input_labels: HashSet<u32> = rev_inputs_subs
        .iter()
        .flat_map(|labels| labels.iter().copied())
        .collect();

    let unique_output: Vec<u32> = {
        let mut seen = HashSet::new();
        rev_output
            .iter()
            .filter(|l| seen.insert(**l))
            .copied()
            .collect()
    };

    let mut delta_labels: Vec<u32> = Vec::new();
    let mut delta_tensors: Vec<Tensor<T>> = Vec::new();
    for &label in &unique_output {
        if !all_input_labels.contains(&label) {
            let dim = *size_dict.get(&label).ok_or_else(|| {
                tenferro_device::Error::InvalidArgument(format!(
                    "einsum: missing dimension for label {}",
                    label
                ))
            })?;
            delta_tensors.push(make_delta::<T>(dim, space)?);
            delta_labels.push(label);
        }
    }

    let mut base_inputs: Vec<Vec<u32>> = rev_inputs_subs.to_vec();
    for &label in &delta_labels {
        let max_label = subs
            .inputs
            .iter()
            .flat_map(|v| v.iter())
            .chain(subs.output.iter())
            .copied()
            .max()
            .unwrap_or(0);
        let fresh = max_label + 1 + (base_inputs.len() as u32);
        base_inputs.push(vec![label, fresh]);
    }

    let base_subs = Subscripts {
        inputs: base_inputs,
        output: unique_output.clone(),
    };

    let needs_embedding = unique_output != rev_output;
    let embed_subs = if needs_embedding {
        Some(Subscripts {
            inputs: vec![unique_output],
            output: rev_output.to_vec(),
        })
    } else {
        None
    };

    Ok(DeltaContext {
        delta_tensors,
        base_subs,
        embed_subs,
    })
}

/// Pre-built reverse-mode context for a single operand index `k`.
///
/// Holds the conjugated non-`k` operands (`conj_store`) and the delta-context
/// (`dctx`) that supplies any identity tensors and embedding subscripts needed
/// when the reverse subscript contains output-only labels (e.g. trace `"ii->"`).
pub(super) struct ReverseContext<T> {
    pub conj_store: Vec<Tensor<T>>,
    pub dctx: DeltaContext<T>,
}

impl<T> ReverseContext<T> {
    /// Assemble the full operand list for a reverse einsum, prepending `leading`
    /// followed by the conjugated operands and delta tensors.
    pub fn assemble_rev_operands<'a>(&'a self, leading: &'a Tensor<T>) -> Vec<&'a Tensor<T>> {
        let mut ops: Vec<&Tensor<T>> = vec![leading];
        for c in &self.conj_store {
            ops.push(c);
        }
        for dt in &self.dctx.delta_tensors {
            ops.push(dt);
        }
        ops
    }

    /// Like `assemble_rev_operands`, but replaces the conjugated operand for
    /// primal index `sub_j` with `tangent`.  `skip_k` is the primal index being
    /// differentiated (already excluded from `conj_store`).
    pub fn assemble_rev_operands_with_sub<'a>(
        &'a self,
        leading: &'a Tensor<T>,
        sub_j: usize,
        skip_k: usize,
        tangent: &'a Tensor<T>,
    ) -> Vec<&'a Tensor<T>> {
        let n = self.conj_store.len() + 1;
        let mut ops: Vec<&Tensor<T>> = vec![leading];
        let mut ci = 0;
        for i in 0..n {
            if i == skip_k {
                continue;
            }
            if i == sub_j {
                ops.push(tangent);
            } else {
                ops.push(&self.conj_store[ci]);
            }
            ci += 1;
        }
        for dt in &self.dctx.delta_tensors {
            ops.push(dt);
        }
        ops
    }
}

/// Build a `ReverseContext` for differentiating with respect to operand `k`.
///
/// Constructs the conjugated operand store, the reverse subscript inputs, and
/// any delta (identity) tensors needed when the reverse output subscript
/// contains labels not present in the reverse inputs (e.g. trace `"ii->"`).
pub(super) fn prepare_reverse_context<T: Scalar + Conjugate>(
    subs: &Subscripts,
    operands: &[&Tensor<T>],
    k: usize,
    size_dict: &HashMap<u32, usize>,
) -> Result<ReverseContext<T>> {
    let mut rev_inputs_subs = vec![subs.output.clone()];
    let mut conj_store = Vec::new();
    for (i, &op) in operands.iter().enumerate() {
        if i != k {
            rev_inputs_subs.push(subs.inputs[i].clone());
            conj_store.push(op.conj());
        }
    }
    let rev_output = subs.inputs[k].clone();
    let dctx = build_delta_context::<T>(
        subs,
        &rev_inputs_subs,
        &rev_output,
        size_dict,
        operands[0].logical_memory_space(),
    )?;
    Ok(ReverseContext { conj_store, dctx })
}
