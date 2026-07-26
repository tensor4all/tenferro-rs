//! Tropical einsum execution through tenferro-einsum lowering.
//!
//! This module executes binary tropical einsum contractions over compact host
//! `f32` and `f64` tensors. Subscript parsing, shape validation, contraction
//! ordering, and GEMM layout metadata come from `tenferro-einsum`; this crate
//! only supplies tropical arithmetic and argmax capture.
//!
//! # Examples
//!
//! ```
//! use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
//! use tenferro_tensor::Tensor;
//!
//! let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0])?;
//! let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0])?;
//! let result = tropical_einsum_with_argmax(
//!     TropicalKind::MaxPlus,
//!     &[&a, &b],
//!     "ij,jk->ik",
//! )?;
//!
//! assert_eq!(result.output.as_slice::<f64>().unwrap(), &[11.0, 15.0, 10.0, 6.0]);
//! assert_eq!(result.argmax[0].indices(), &[0, 1, 0, 1]);
//! # Ok::<(), tenferro_tensor::Error>(())
//! ```

use num_traits::Float;
use tenferro_einsum::{ContractionTree, Subscripts};
use tenferro_tensor::{DType, Tensor, TensorScalar};

use crate::cpu::{tropical_gemm_with_argmax, TropicalGemmKind};
use crate::error::{from_einsum_error, unsupported_dtype};
use crate::TropicalKind;

const OP: &str = "tropical_einsum_with_argmax";

/// Argmax metadata captured for one pairwise tropical contraction step.
///
/// Indices are stored in the same column-major order as the step output.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
/// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
/// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
///
/// assert_eq!(result.argmax[0].indices(), &[0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TropicalArgmaxStep {
    indices: Vec<u32>,
    output_shape: Vec<usize>,
    output_subscripts: Vec<u32>,
    contracted_subscripts: Vec<u32>,
    contracted_shape: Vec<usize>,
}

impl TropicalArgmaxStep {
    fn new(
        indices: Vec<u32>,
        output_shape: Vec<usize>,
        output_subscripts: Vec<u32>,
        contracted_subscripts: Vec<u32>,
        contracted_shape: Vec<usize>,
    ) -> Self {
        Self {
            indices,
            output_shape,
            output_subscripts,
            contracted_subscripts,
            contracted_shape,
        }
    }

    /// Return first-winning contracted indices in column-major output order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
    /// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
    ///
    /// assert_eq!(result.argmax[0].indices(), &[0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Return the output shape associated with these argmax indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
    /// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
    ///
    /// assert_eq!(result.argmax[0].output_shape(), &[1, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Return the output subscripts for this pairwise step.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
    /// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
    ///
    /// assert_eq!(result.argmax[0].output_subscripts(), &[b'i' as u32, b'k' as u32]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn output_subscripts(&self) -> &[u32] {
        &self.output_subscripts
    }

    /// Return the contracted subscripts represented by each argmax index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
    /// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
    ///
    /// assert_eq!(result.argmax[0].contracted_subscripts(), &[b'j' as u32]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn contracted_subscripts(&self) -> &[u32] {
        &self.contracted_subscripts
    }

    /// Return the shape fused into each argmax index.
    ///
    /// Coordinates are encoded in column-major order over this shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
    /// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
    ///
    /// assert_eq!(result.argmax[0].contracted_shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn contracted_shape(&self) -> &[usize] {
        &self.contracted_shape
    }

    /// Decode the winning contracted coordinates for one output element.
    ///
    /// Returns `None` when `output_index` is outside the argmax buffer or when
    /// the stored fused winner is outside the contracted shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
    /// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
    ///
    /// assert_eq!(result.argmax[0].winner_coordinates(0).unwrap(), vec![0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn winner_coordinates(&self, output_index: usize) -> Option<Vec<usize>> {
        let fused = *self.indices.get(output_index)? as usize;
        decode_col_major_index(fused, &self.contracted_shape)
    }
}

/// Tropical einsum output and per-step argmax metadata.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f32])?;
/// let b = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f32])?;
/// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
///
/// assert_eq!(result.output.as_slice::<f32>().unwrap(), &[5.0]);
/// assert_eq!(result.argmax.len(), 1);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct TropicalEinsumResult {
    /// Tropical output tensor in the requested output subscript order.
    pub output: Tensor,
    /// Per-step first-winner argmax metadata.
    pub argmax: Vec<TropicalArgmaxStep>,
}

/// Execute a binary tropical einsum and capture first-winner argmax.
///
/// This supports binary contractions over compact host `f32` and `f64`
/// tensors, including batched GEMM-style contractions and a generic fallback
/// for unique-label binary contractions. Unsupported cases return
/// structured validation errors instead of panicking.
///
/// # Errors
///
/// Returns shared validation errors for invalid notation, shapes, and dtypes;
/// unsupported lowering features use a structured argument error.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0])?;
/// let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0])?;
/// let result = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ki")?;
///
/// assert_eq!(result.output.as_slice::<f64>().unwrap(), &[11.0, 10.0, 15.0, 6.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn tropical_einsum_with_argmax(
    kind: TropicalKind,
    inputs: &[&Tensor],
    notation: &str,
) -> tenferro_tensor::Result<TropicalEinsumResult> {
    let subscripts = Subscripts::parse(notation).map_err(|err| from_einsum_error(OP, err))?;
    tropical_einsum_subscripts_with_argmax(kind, inputs, &subscripts)
}

/// Execute a binary tropical einsum from parsed subscripts and capture argmax.
///
/// This is equivalent to [`tropical_einsum_with_argmax`] but accepts the parsed
/// [`tenferro_einsum::Subscripts`] representation directly so traced extension
/// payloads do not need to stringify labels.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::Validation`] for invalid shapes or
/// lowering features, or [`tenferro_tensor::Error::Extension`] containing the
/// typed `UnsupportedDType` source when the input dtype is not `F32` or `F64`.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::Subscripts;
/// use tenferro_ext_tropical::{einsum::tropical_einsum_subscripts_with_argmax, TropicalKind};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
/// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0])?;
/// let subscripts = Subscripts::parse("ij,jk->ik").unwrap();
/// let result =
///     tropical_einsum_subscripts_with_argmax(TropicalKind::MaxPlus, &[&a, &b], &subscripts)?;
///
/// assert_eq!(result.output.as_slice::<f64>().unwrap(), &[3.0]);
/// assert_eq!(result.argmax[0].indices(), &[0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn tropical_einsum_subscripts_with_argmax(
    kind: TropicalKind,
    inputs: &[&Tensor],
    subscripts: &Subscripts,
) -> tenferro_tensor::Result<TropicalEinsumResult> {
    if inputs.len() != 2 {
        return Err(invalid_config(format!(
            "only two-input binary contractions are supported, got {} inputs",
            inputs.len()
        )));
    }
    if subscripts.inputs.len() != inputs.len() {
        return Err(invalid_config(format!(
            "notation describes {} inputs but {} tensors were provided",
            subscripts.inputs.len(),
            inputs.len()
        )));
    }
    ensure_unique_labels(&subscripts.output, "output subscripts")?;
    for (input_index, input_labels) in subscripts.inputs.iter().enumerate() {
        ensure_unique_input_labels(input_labels, input_index)?;
    }

    let shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    let tree =
        ContractionTree::optimize(subscripts, &shapes).map_err(|err| from_einsum_error(OP, err))?;
    if tree.step_count() != 1 {
        return Err(invalid_config(format!(
            "only one pairwise contraction step is supported, got {}",
            tree.step_count()
        )));
    }

    let (lhs_idx, rhs_idx) = tree
        .step_pair(0)
        .ok_or_else(|| invalid_config("missing pairwise contraction step"))?;
    if (lhs_idx, rhs_idx) != (0, 1) {
        return Err(invalid_config(format!(
            "only original input pair (0, 1) is supported, got ({lhs_idx}, {rhs_idx})"
        )));
    }
    let (lhs_subs, rhs_subs, output_subs) = tree
        .step_subscripts(0)
        .ok_or_else(|| invalid_config("missing pairwise contraction subscripts"))?;
    let step = tree
        .step_plan(0)
        .ok_or_else(|| invalid_config("missing pairwise lowering plan"))?;

    if step.lhs_diag().is_some() || step.rhs_diag().is_some() {
        return Err(invalid_config("diagonal extraction is not supported yet"));
    }
    if step.lhs_reduce().is_some() || step.rhs_reduce().is_some() {
        return Err(invalid_config("pre-reduction is not supported yet"));
    }

    let gemm = step.gemm();
    if gemm.contracted_modes().is_empty() {
        return Err(invalid_config(
            "tropical einsum requires at least one contracted mode for argmax capture",
        ));
    }

    match (inputs[0].dtype(), inputs[1].dtype()) {
        (DType::F32, DType::F32) => execute_typed::<f32>(
            kind,
            inputs,
            subscripts,
            lhs_subs,
            rhs_subs,
            output_subs,
            &gemm,
        ),
        (DType::F64, DType::F64) => execute_typed::<f64>(
            kind,
            inputs,
            subscripts,
            lhs_subs,
            rhs_subs,
            output_subs,
            &gemm,
        ),
        (lhs, rhs) if lhs != rhs => Err(tenferro_tensor::Error::dtype_mismatch(OP, lhs, rhs)),
        (dtype, _) => Err(unsupported_dtype(OP, dtype)),
    }
}

fn execute_typed<T>(
    kind: TropicalKind,
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    lhs_subs: &[u32],
    rhs_subs: &[u32],
    output_subs: &[u32],
    gemm: &tenferro_einsum::lowering::GemmPlan<'_>,
) -> tenferro_tensor::Result<TropicalEinsumResult>
where
    T: TropicalFloat,
{
    let lhs = T::host_slice(inputs[0])?;
    let rhs = T::host_slice(inputs[1])?;
    if gemm.lhs_target_modes() == lhs_subs
        && gemm.rhs_target_modes() == rhs_subs
        && gemm_shapes_match(gemm)?
    {
        return execute_target_order_gemm_typed::<T>(
            kind,
            inputs,
            lhs,
            rhs,
            subscripts,
            output_subs,
            gemm,
        );
    }

    execute_fallback_typed::<T>(
        kind,
        inputs,
        lhs,
        rhs,
        subscripts,
        lhs_subs,
        rhs_subs,
        output_subs,
        gemm,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_target_order_gemm_typed<T>(
    kind: TropicalKind,
    inputs: &[&Tensor],
    lhs: &[T],
    rhs: &[T],
    subscripts: &Subscripts,
    output_subs: &[u32],
    gemm: &tenferro_einsum::lowering::GemmPlan<'_>,
) -> tenferro_tensor::Result<TropicalEinsumResult>
where
    T: TropicalFloat,
{
    let canonical_shape = gemm.expanded_output_shape().to_vec();
    let requested_shape = output_shape_for_labels(output_subs, subscripts, inputs)?;
    let contracted_shape = gemm.contracted_shape().to_vec();
    let output_len = element_count(&requested_shape)?;

    if output_len == 0 {
        let argmax_step = TropicalArgmaxStep::new(
            Vec::new(),
            requested_shape.clone(),
            output_subs.to_vec(),
            gemm.contracted_modes().to_vec(),
            contracted_shape,
        );
        return Ok(TropicalEinsumResult {
            output: Tensor::from_vec_col_major(requested_shape, Vec::<T>::new())?,
            argmax: vec![argmax_step],
        });
    }

    ensure_nonzero_contraction(gemm.k())?;

    let batch_len = element_count(gemm.batch_shape())?;
    let lhs_batch_len = checked_len(gemm.m(), gemm.k(), "left GEMM batch slice")?;
    let rhs_batch_len = checked_len(gemm.k(), gemm.n(), "right GEMM batch slice")?;
    let output_batch_len = checked_len(gemm.m(), gemm.n(), "output GEMM batch slice")?;
    let mut canonical_values = Vec::with_capacity(output_len);
    let mut canonical_argmax = Vec::with_capacity(output_len);

    for batch_index in 0..batch_len {
        let lhs_start = checked_len(batch_index, lhs_batch_len, "left batch offset")?;
        let rhs_start = checked_len(batch_index, rhs_batch_len, "right batch offset")?;
        let lhs_end = lhs_start
            .checked_add(lhs_batch_len)
            .ok_or_else(|| invalid_config("left batch slice end overflows usize"))?;
        let rhs_end = rhs_start
            .checked_add(rhs_batch_len)
            .ok_or_else(|| invalid_config("right batch slice end overflows usize"))?;
        let lhs_batch = lhs
            .get(lhs_start..lhs_end)
            .ok_or_else(|| invalid_config("left GEMM batch slice is outside input storage"))?;
        let rhs_batch = rhs
            .get(rhs_start..rhs_end)
            .ok_or_else(|| invalid_config("right GEMM batch slice is outside input storage"))?;
        let out = tropical_gemm_with_argmax(
            TropicalGemmKind::from(kind),
            lhs_batch,
            gemm.m(),
            gemm.k(),
            rhs_batch,
            gemm.n(),
        )?;
        debug_assert_eq!(out.values.len(), output_batch_len);
        canonical_values.extend(out.values);
        canonical_argmax.extend(out.argmax);
    }

    let (values, indices) = if gemm.needs_final_permute() {
        (
            permute_col_major(
                &canonical_values,
                &canonical_shape,
                gemm.canonical_output_modes(),
                &requested_shape,
                output_subs,
            )?,
            permute_col_major(
                &canonical_argmax,
                &canonical_shape,
                gemm.canonical_output_modes(),
                &requested_shape,
                output_subs,
            )?,
        )
    } else {
        if canonical_shape != requested_shape {
            return Err(invalid_config(format!(
                "lowering output shape {canonical_shape:?} does not match requested shape {requested_shape:?}"
            )));
        }
        (canonical_values, canonical_argmax)
    };

    let argmax_step = TropicalArgmaxStep::new(
        indices,
        requested_shape.clone(),
        output_subs.to_vec(),
        gemm.contracted_modes().to_vec(),
        contracted_shape,
    );
    Ok(TropicalEinsumResult {
        output: Tensor::from_vec_col_major(requested_shape, values)?,
        argmax: vec![argmax_step],
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_fallback_typed<T>(
    kind: TropicalKind,
    inputs: &[&Tensor],
    lhs: &[T],
    rhs: &[T],
    subscripts: &Subscripts,
    lhs_subs: &[u32],
    rhs_subs: &[u32],
    output_subs: &[u32],
    gemm: &tenferro_einsum::lowering::GemmPlan<'_>,
) -> tenferro_tensor::Result<TropicalEinsumResult>
where
    T: TropicalFloat,
{
    let requested_shape = output_shape_for_labels(output_subs, subscripts, inputs)?;
    let contracted_modes = gemm.contracted_modes();
    let contracted_shape = gemm.contracted_shape().to_vec();
    let output_len = element_count(&requested_shape)?;

    if output_len == 0 {
        let argmax_step = TropicalArgmaxStep::new(
            Vec::new(),
            requested_shape.clone(),
            output_subs.to_vec(),
            contracted_modes.to_vec(),
            contracted_shape,
        );
        return Ok(TropicalEinsumResult {
            output: Tensor::from_vec_col_major(requested_shape, Vec::<T>::new())?,
            argmax: vec![argmax_step],
        });
    }

    let contracted_len = element_count(&contracted_shape)?;
    ensure_nonzero_contraction(contracted_len)?;
    ensure_argmax_representable(contracted_len)?;

    let lhs_strides = col_major_strides(inputs[0].shape())?;
    let rhs_strides = col_major_strides(inputs[1].shape())?;
    let lhs_axes = axis_sources(lhs_subs, output_subs, contracted_modes)?;
    let rhs_axes = axis_sources(rhs_subs, output_subs, contracted_modes)?;
    if lhs_axes.len() != lhs_strides.len() || rhs_axes.len() != rhs_strides.len() {
        return Err(invalid_config(
            "input rank does not match tensor shape rank for fallback execution",
        ));
    }
    let mut values = Vec::with_capacity(output_len);
    let mut argmax = Vec::with_capacity(output_len);
    let mut output_index = vec![0usize; requested_shape.len()];
    let mut contracted_index = vec![0usize; contracted_shape.len()];

    for _ in 0..output_len {
        let mut best = tropical_identity(kind);
        let mut winner = 0_u32;
        let mut has_ordered_candidate = false;
        contracted_index.fill(0);

        for contracted_flat in 0..contracted_len {
            let lhs_offset =
                offset_for_axes(&lhs_axes, &lhs_strides, &output_index, &contracted_index)?;
            let rhs_offset =
                offset_for_axes(&rhs_axes, &rhs_strides, &output_index, &contracted_index)?;
            let lhs_value = lhs
                .get(lhs_offset)
                .ok_or_else(|| invalid_config("fallback left input offset is out of bounds"))?;
            let rhs_value = rhs
                .get(rhs_offset)
                .ok_or_else(|| invalid_config("fallback right input offset is out of bounds"))?;
            let candidate = *lhs_value + *rhs_value;
            if !candidate.is_nan()
                && tropical_candidate_is_better(kind, candidate, best, has_ordered_candidate)
            {
                best = candidate;
                winner = contracted_flat as u32;
                has_ordered_candidate = true;
            }
            increment_col_major_index(&mut contracted_index, &contracted_shape);
        }

        values.push(best);
        argmax.push(winner);
        increment_col_major_index(&mut output_index, &requested_shape);
    }

    let argmax_step = TropicalArgmaxStep::new(
        argmax,
        requested_shape.clone(),
        output_subs.to_vec(),
        contracted_modes.to_vec(),
        contracted_shape,
    );
    Ok(TropicalEinsumResult {
        output: Tensor::from_vec_col_major(requested_shape, values)?,
        argmax: vec![argmax_step],
    })
}

trait TropicalFloat: Float + TensorScalar {
    fn host_slice(tensor: &Tensor) -> tenferro_tensor::Result<&[Self]>;
}

impl TropicalFloat for f32 {
    fn host_slice(tensor: &Tensor) -> tenferro_tensor::Result<&[Self]> {
        match tensor {
            Tensor::F32(tensor) => tensor.as_view().as_slice(),
            _ => Err(invalid_config(format!(
                "expected F32 input, got {:?}",
                tensor.dtype()
            ))),
        }
    }
}

impl TropicalFloat for f64 {
    fn host_slice(tensor: &Tensor) -> tenferro_tensor::Result<&[Self]> {
        match tensor {
            Tensor::F64(tensor) => tensor.as_view().as_slice(),
            _ => Err(invalid_config(format!(
                "expected F64 input, got {:?}",
                tensor.dtype()
            ))),
        }
    }
}

fn output_shape_for_labels(
    labels: &[u32],
    subscripts: &Subscripts,
    inputs: &[&Tensor],
) -> tenferro_tensor::Result<Vec<usize>> {
    labels
        .iter()
        .map(|label| {
            subscripts
                .inputs
                .iter()
                .zip(inputs)
                .find_map(|(input_labels, tensor)| {
                    input_labels
                        .iter()
                        .position(|candidate| candidate == label)
                        .map(|axis| tensor.shape()[axis])
                })
                .ok_or_else(|| invalid_config(format!("output label {label} is not in any input")))
        })
        .collect()
}

fn ensure_unique_labels(labels: &[u32], role: &'static str) -> tenferro_tensor::Result<()> {
    for (axis, label) in labels.iter().enumerate() {
        if labels[..axis].contains(label) {
            return Err(invalid_config(format!(
                "repeated {role} label {label} is not supported yet"
            )));
        }
    }
    Ok(())
}

fn ensure_unique_input_labels(labels: &[u32], input_index: usize) -> tenferro_tensor::Result<()> {
    for (axis, label) in labels.iter().enumerate() {
        if labels[..axis].contains(label) {
            return Err(invalid_config(format!(
                "repeated input {input_index} label {label} is not supported yet"
            )));
        }
    }
    Ok(())
}

fn ensure_nonzero_contraction(contracted_len: usize) -> tenferro_tensor::Result<()> {
    if contracted_len == 0 {
        return Err(invalid_config(
            "zero-sized contracted modes are supported only when the requested output is empty",
        ));
    }
    Ok(())
}

fn ensure_argmax_representable(contracted_len: usize) -> tenferro_tensor::Result<()> {
    let max_argmax_len = (u32::MAX as usize).saturating_add(1);
    if contracted_len > max_argmax_len {
        return Err(invalid_config(format!(
            "contracted element count {contracted_len} cannot be represented as u32 argmax indices"
        )));
    }
    Ok(())
}

fn gemm_shapes_match(
    gemm: &tenferro_einsum::lowering::GemmPlan<'_>,
) -> tenferro_tensor::Result<bool> {
    let expected_lhs = expected_grouped_shape(gemm.m(), gemm.k(), gemm.batch_shape());
    let expected_rhs = expected_grouped_shape(gemm.k(), gemm.n(), gemm.batch_shape());
    let expected_output = expected_grouped_shape(gemm.m(), gemm.n(), gemm.batch_shape());
    Ok(gemm.lhs_gemm_shape() == expected_lhs.as_slice()
        && gemm.rhs_gemm_shape() == expected_rhs.as_slice()
        && gemm.output_gemm_shape() == expected_output.as_slice()
        && element_count(gemm.expanded_output_shape())? == element_count(gemm.output_gemm_shape())?)
}

fn expected_grouped_shape(first: usize, second: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = Vec::with_capacity(2 + batch_shape.len());
    shape.push(first);
    shape.push(second);
    shape.extend_from_slice(batch_shape);
    shape
}

#[derive(Clone, Copy)]
enum AxisSource {
    Output(usize),
    Contracted(usize),
}

fn axis_sources(
    input_labels: &[u32],
    output_labels: &[u32],
    contracted_labels: &[u32],
) -> tenferro_tensor::Result<Vec<AxisSource>> {
    input_labels
        .iter()
        .map(|label| {
            if let Some(axis) = output_labels
                .iter()
                .position(|candidate| candidate == label)
            {
                Ok(AxisSource::Output(axis))
            } else if let Some(axis) = contracted_labels
                .iter()
                .position(|candidate| candidate == label)
            {
                Ok(AxisSource::Contracted(axis))
            } else {
                Err(invalid_config(format!(
                    "input label {label} would require pre-reduction, which is not supported yet"
                )))
            }
        })
        .collect()
}

fn offset_for_axes(
    axes: &[AxisSource],
    strides: &[usize],
    output_index: &[usize],
    contracted_index: &[usize],
) -> tenferro_tensor::Result<usize> {
    axes.iter()
        .zip(strides)
        .try_fold(0usize, |offset, (axis, stride)| {
            let coordinate = match *axis {
                AxisSource::Output(output_axis) => output_index[output_axis],
                AxisSource::Contracted(contracted_axis) => contracted_index[contracted_axis],
            };
            let term = coordinate
                .checked_mul(*stride)
                .ok_or_else(|| invalid_config("input offset multiplication overflows usize"))?;
            offset
                .checked_add(term)
                .ok_or_else(|| invalid_config("input offset addition overflows usize"))
        })
}

fn tropical_identity<T: Float>(kind: TropicalKind) -> T {
    match kind {
        TropicalKind::MaxPlus => T::neg_infinity(),
        TropicalKind::MinPlus => T::infinity(),
    }
}

fn tropical_candidate_is_better<T: Float>(
    kind: TropicalKind,
    candidate: T,
    best: T,
    has_ordered_candidate: bool,
) -> bool {
    match kind {
        TropicalKind::MaxPlus => !has_ordered_candidate || candidate > best,
        TropicalKind::MinPlus => !has_ordered_candidate || candidate < best,
    }
}

fn permute_col_major<T: Copy>(
    values: &[T],
    source_shape: &[usize],
    source_modes: &[u32],
    target_shape: &[usize],
    target_modes: &[u32],
) -> tenferro_tensor::Result<Vec<T>> {
    if source_modes.len() != source_shape.len() || target_modes.len() != target_shape.len() {
        return Err(invalid_config(
            "mode rank does not match shape rank for final permutation",
        ));
    }
    if values.len() != element_count(source_shape)? || values.len() != element_count(target_shape)?
    {
        return Err(invalid_config(format!(
            "final permutation element count mismatch: values={}, source_shape={source_shape:?}, target_shape={target_shape:?}",
            values.len()
        )));
    }
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let source_to_target_axis: Vec<usize> = source_modes
        .iter()
        .map(|mode| {
            target_modes
                .iter()
                .position(|target| target == mode)
                .ok_or_else(|| {
                    invalid_config(format!(
                        "canonical output mode {mode} is missing from requested output"
                    ))
                })
        })
        .collect::<tenferro_tensor::Result<_>>()?;
    let target_strides = col_major_strides(target_shape)?;
    let mut out = vec![values[0]; values.len()];
    let mut index = vec![0usize; source_shape.len()];

    for &value in values {
        let mut target_flat = 0usize;
        for (source_axis, &coordinate) in index.iter().enumerate() {
            let target_axis = source_to_target_axis[source_axis];
            target_flat += coordinate * target_strides[target_axis];
        }
        out[target_flat] = value;
        increment_col_major_index(&mut index, source_shape);
    }

    Ok(out)
}

fn increment_col_major_index(index: &mut [usize], shape: &[usize]) {
    for (axis_index, extent) in index.iter_mut().zip(shape) {
        *axis_index += 1;
        if *axis_index < *extent {
            return;
        }
        *axis_index = 0;
    }
}

fn col_major_strides(shape: &[usize]) -> tenferro_tensor::Result<Vec<usize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| invalid_config(format!("shape {shape:?} overflows usize")))?;
    }
    Ok(strides)
}

fn element_count(shape: &[usize]) -> tenferro_tensor::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &extent| {
        acc.checked_mul(extent)
            .ok_or_else(|| invalid_config(format!("shape {shape:?} overflows usize")))
    })
}

fn checked_len(lhs: usize, rhs: usize, label: &str) -> tenferro_tensor::Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        invalid_config(format!(
            "{label} element count overflows usize for dimensions {lhs} and {rhs}"
        ))
    })
}

fn decode_col_major_index(mut flat: usize, shape: &[usize]) -> Option<Vec<usize>> {
    let total = shape
        .iter()
        .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))?;
    if flat >= total {
        return None;
    }

    let mut coordinates = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            return None;
        }
        coordinates.push(flat % extent);
        flat /= extent;
    }
    Some(coordinates)
}

fn invalid_config(message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(OP, "configuration", message)
}
