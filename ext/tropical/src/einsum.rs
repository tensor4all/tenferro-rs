//! Tropical einsum execution through tenferro-einsum lowering.
//!
//! This module executes the first supported tropical einsum shape: simple
//! binary GEMM-style contractions over compact host `f32` and `f64` tensors.
//! Subscript parsing, shape validation, contraction ordering, and GEMM layout
//! metadata come from `tenferro-einsum`; this crate only supplies tropical
//! arithmetic and argmax capture.
//!
//! # Examples
//!
//! ```
//! use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
//! use tenferro_tensor::Tensor;
//!
//! let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0]);
//! let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0]);
//! let result = tropical_einsum_with_argmax(
//!     TropicalEinsumKind::MaxPlus,
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

const OP: &str = "tropical_einsum_with_argmax";

/// Tropical einsum semiring flavor.
///
/// This is an alias for the crate-level [`crate::TropicalKind`], kept here so
/// callers can import einsum-specific APIs from one module without introducing a
/// second semiring enum.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::{einsum::TropicalEinsumKind, TropicalKind};
///
/// assert_eq!(TropicalEinsumKind::MaxPlus, TropicalKind::MaxPlus);
/// assert_ne!(TropicalEinsumKind::MaxPlus, TropicalEinsumKind::MinPlus);
/// ```
pub type TropicalEinsumKind = crate::TropicalKind;

/// Argmax metadata captured for one pairwise tropical contraction step.
///
/// Indices are stored in the same column-major order as the step output.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
/// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
/// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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
    /// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
    /// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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
    /// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
    /// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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
    /// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
    /// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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
    /// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
    /// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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
    /// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
    /// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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
    /// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
    /// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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
/// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f32]);
/// let b = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f32]);
/// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
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

/// Execute a simple binary tropical einsum and capture first-winner argmax.
///
/// Task 4 supports exactly binary GEMM-style lowering over compact host `f32`
/// and `f64` tensors, with no diagonal extraction, no pre-reduction, and no
/// batch modes. Unsupported cases return
/// [`tenferro_tensor::Error::InvalidConfig`] instead of panicking.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::InvalidConfig`] when notation, shapes,
/// dtype, or lowering features are outside the Task 4 supported surface.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0]);
/// let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0]);
/// let result = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ki")?;
///
/// assert_eq!(result.output.as_slice::<f64>().unwrap(), &[11.0, 10.0, 15.0, 6.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn tropical_einsum_with_argmax(
    kind: TropicalEinsumKind,
    inputs: &[&Tensor],
    notation: &str,
) -> tenferro_tensor::Result<TropicalEinsumResult> {
    if inputs.len() != 2 {
        return Err(invalid_config(format!(
            "only two-input binary contractions are supported, got {} inputs",
            inputs.len()
        )));
    }

    let subscripts = Subscripts::parse(notation)
        .map_err(|err| invalid_config(format!("invalid einsum notation `{notation}`: {err}")))?;
    if subscripts.inputs.len() != inputs.len() {
        return Err(invalid_config(format!(
            "notation describes {} inputs but {} tensors were provided",
            subscripts.inputs.len(),
            inputs.len()
        )));
    }
    ensure_unique_labels(&subscripts.output, "output subscripts")?;

    let shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    let tree = ContractionTree::optimize(&subscripts, &shapes)
        .map_err(|err| invalid_config(format!("einsum lowering failed: {err}")))?;
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
    if !gemm.batch_modes().is_empty() {
        return Err(invalid_config(
            "batched tropical einsum is not supported yet",
        ));
    }
    if gemm.lhs_target_modes() != lhs_subs {
        return Err(invalid_config(
            "left input would require a pre-GEMM permutation, which is not supported yet",
        ));
    }
    if gemm.rhs_target_modes() != rhs_subs {
        return Err(invalid_config(
            "right input would require a pre-GEMM permutation, which is not supported yet",
        ));
    }
    if gemm.lhs_gemm_shape().len() != 2
        || gemm.rhs_gemm_shape().len() != 2
        || gemm.output_gemm_shape().len() != 2
    {
        return Err(invalid_config(
            "only unbatched two-dimensional GEMM lowering is supported yet",
        ));
    }

    match (inputs[0].dtype(), inputs[1].dtype()) {
        (DType::F32, DType::F32) => {
            execute_typed::<f32>(kind, inputs, &subscripts, output_subs, &gemm)
        }
        (DType::F64, DType::F64) => {
            execute_typed::<f64>(kind, inputs, &subscripts, output_subs, &gemm)
        }
        (lhs, rhs) if lhs != rhs => Err(invalid_config(format!(
            "input dtype mismatch: left is {lhs:?}, right is {rhs:?}"
        ))),
        (dtype, _) => Err(invalid_config(format!(
            "unsupported dtype {dtype:?}; only F32 and F64 are supported"
        ))),
    }
}

fn execute_typed<T>(
    kind: TropicalEinsumKind,
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    output_subs: &[u32],
    gemm: &tenferro_einsum::lowering::GemmPlan<'_>,
) -> tenferro_tensor::Result<TropicalEinsumResult>
where
    T: TropicalFloat,
{
    let lhs = T::host_slice(inputs[0])?;
    let rhs = T::host_slice(inputs[1])?;
    let canonical_shape = gemm.expanded_output_shape().to_vec();
    let requested_shape = output_shape_for_labels(output_subs, subscripts, inputs)?;
    let contracted_shape = output_shape_for_labels(gemm.contracted_modes(), subscripts, inputs)?;
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
            output: Tensor::from_vec_col_major(requested_shape, Vec::<T>::new()),
            argmax: vec![argmax_step],
        });
    }

    if gemm.k() == 0 {
        return Err(invalid_config(
            "zero-sized contracted modes are supported only when the requested output is empty",
        ));
    }

    let out = tropical_gemm_with_argmax(
        TropicalGemmKind::from(kind),
        lhs,
        gemm.m(),
        gemm.k(),
        rhs,
        gemm.n(),
    )?;

    let (values, indices) = if gemm.needs_final_permute() {
        (
            permute_col_major(
                &out.values,
                &canonical_shape,
                gemm.canonical_output_modes(),
                &requested_shape,
                output_subs,
            )?,
            permute_col_major(
                &out.argmax,
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
        (out.values, out.argmax)
    };

    let argmax_step = TropicalArgmaxStep::new(
        indices,
        requested_shape.clone(),
        output_subs.to_vec(),
        gemm.contracted_modes().to_vec(),
        contracted_shape,
    );
    Ok(TropicalEinsumResult {
        output: Tensor::from_vec_col_major(requested_shape, values),
        argmax: vec![argmax_step],
    })
}

trait TropicalFloat: Float + TensorScalar {
    fn host_slice(tensor: &Tensor) -> tenferro_tensor::Result<&[Self]>;
}

impl TropicalFloat for f32 {
    fn host_slice(tensor: &Tensor) -> tenferro_tensor::Result<&[Self]> {
        match tensor {
            Tensor::F32(tensor) => tensor.as_view().as_slice().map_err(|err| {
                invalid_config(format!("input must be a compact host F32 tensor: {err}"))
            }),
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
            Tensor::F64(tensor) => tensor.as_view().as_slice().map_err(|err| {
                invalid_config(format!("input must be a compact host F64 tensor: {err}"))
            }),
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
    tenferro_tensor::Error::InvalidConfig {
        op: OP,
        message: message.into(),
    }
}
