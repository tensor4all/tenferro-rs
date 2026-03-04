use std::collections::{HashMap, HashSet};

use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::Tensor;

use crate::partial_diag::meta::{plan_axis_classes_for_subscripts, OperandAxisClasses};
use crate::runtime::{with_default_runtime, RuntimeContext};
use crate::{Error, Result};

/// PartialDiagonal tensor with fixed scalar type.
///
/// This type stores:
/// - logical (uncompressed) axis metadata: `logical_dims`, `axis_classes`
/// - compressed payload tensor: one axis per distinct axis class
///
/// Dense and diagonal tensors are special cases:
/// - Dense: `axis_classes = [0,1,2,...]`
/// - Diag:  `axis_classes = [0,0,...,0]`
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::partial_diag::AdTensor;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
/// let x = AdTensor::new(vec![3, 3], vec![0, 0], payload).unwrap();
/// assert_eq!(x.class_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct AdTensor<T: Scalar> {
    payload: Tensor<T>,
    logical_dims: Vec<usize>,
    axis_classes: Vec<usize>,
}

impl<T: Scalar> AdTensor<T> {
    /// Construct from logical metadata and compressed payload.
    ///
    /// `axis_classes` is canonicalized to first-appearance order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new(vec![3, 3], vec![10, 10], payload).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn new(
        logical_dims: Vec<usize>,
        axis_classes: Vec<usize>,
        payload: Tensor<T>,
    ) -> Result<Self> {
        let canonical_classes = canonicalize_axis_classes(&axis_classes);
        validate_layout(&logical_dims, &canonical_classes, &payload)?;
        Ok(Self {
            payload,
            logical_dims,
            axis_classes: canonical_classes,
        })
    }

    /// Construct from a dense tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::from_dense(dense);
    /// assert_eq!(x.axis_classes(), &[0, 1]);
    /// ```
    pub fn from_dense(payload: Tensor<T>) -> Self {
        let logical_dims = payload.dims().to_vec();
        let axis_classes: Vec<usize> = (0..logical_dims.len()).collect();
        Self {
            payload,
            logical_dims,
            axis_classes,
        }
    }

    /// Construct a diagonal-like PartialDiagonal tensor from a vector payload.
    ///
    /// `logical_rank` is the logical tensor rank to represent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn from_diagonal_vector(payload: Tensor<T>, logical_rank: usize) -> Result<Self> {
        if payload.dims().len() != 1 {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "from_diagonal_vector expects rank-1 payload, got rank {}",
                    payload.dims().len()
                ),
            });
        }
        if logical_rank == 0 {
            return Err(Error::InvalidAdTensor {
                message: "from_diagonal_vector requires logical_rank >= 1".to_string(),
            });
        }
        let n = payload.dims()[0];
        let logical_dims = vec![n; logical_rank];
        let axis_classes = vec![0; logical_rank];
        Self::new(logical_dims, axis_classes, payload)
    }

    /// Borrow compressed payload tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::from_dense(dense);
    /// assert_eq!(x.payload().dims(), &[2]);
    /// ```
    pub fn payload(&self) -> &Tensor<T> {
        &self.payload
    }

    /// Consume self and return compressed payload tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::from_dense(dense);
    /// let payload = x.into_payload();
    /// assert_eq!(payload.dims(), &[2]);
    /// ```
    pub fn into_payload(self) -> Tensor<T> {
        self.payload
    }

    /// Logical dimensions (uncompressed rank).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert_eq!(x.logical_dims(), &[2, 2]);
    /// ```
    pub fn logical_dims(&self) -> &[usize] {
        &self.logical_dims
    }

    /// Axis classes for logical axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn axis_classes(&self) -> &[usize] {
        &self.axis_classes
    }

    /// Number of distinct axis classes (compressed payload rank).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::from_diagonal_vector(payload, 3).unwrap();
    /// assert_eq!(x.class_count(), 1);
    /// ```
    pub fn class_count(&self) -> usize {
        self.payload.dims().len()
    }

    /// Returns `true` when this tensor is represented as a dense layout.
    pub fn is_dense(&self) -> bool {
        self.axis_classes.len() == self.logical_dims.len()
            && self.logical_dims == self.payload.dims()
            && self
                .axis_classes
                .iter()
                .enumerate()
                .all(|(i, &class_id)| class_id == i)
    }

    /// Returns `true` when this tensor is represented as a pure diagonal layout.
    pub fn is_diag(&self) -> bool {
        if self.logical_dims.is_empty() || self.axis_classes.len() != self.logical_dims.len() {
            return false;
        }
        let first_dim = self.logical_dims[0];
        self.axis_classes.iter().all(|&class_id| class_id == 0)
            && self.logical_dims.iter().all(|&dim| dim == first_dim)
            && self.payload.dims().len() == 1
            && self.payload.dims()[0] == first_dim
    }
}

impl<T> AdTensor<T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    /// Materialize into a dense tensor by expanding axis classes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Requires default runtime to be configured.
    /// let dense = partial_diag.to_dense()?;
    /// ```
    pub fn to_dense(&self) -> Result<Tensor<T>> {
        if self.is_dense() {
            return Ok(self.payload.clone());
        }

        with_cpu_runtime("partial_diag_to_dense", |ctx| {
            let input_labels =
                usize_vec_to_u32(&(0..self.payload.dims().len()).collect::<Vec<_>>())?;
            let output_labels = usize_vec_to_u32(&self.axis_classes)?;
            let inputs = [input_labels.as_slice()];
            let subs = Subscripts::new(&inputs, &output_labels);
            let out = tf_einsum::einsum_with_subscripts::<Standard<T>, CpuBackend>(
                ctx,
                &subs,
                &[&self.payload],
                None,
            )
            .map_err(Error::from)?;
            if out.dims() != self.logical_dims {
                return Err(Error::InvalidAdTensor {
                    message: format!(
                        "to_dense output shape mismatch: expected {:?}, got {:?}",
                        self.logical_dims,
                        out.dims()
                    ),
                });
            }
            Ok(out)
        })
    }

    /// Contract/einsum PartialDiagonal operands while preserving compressed metadata.
    ///
    /// `subscripts.inputs` rank must match each operand logical rank.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Requires default runtime to be configured.
    /// let out = AdTensor::einsum_with_subscripts(&subs, &[&a, &b])?;
    /// ```
    pub fn einsum_with_subscripts(subscripts: &Subscripts, operands: &[&Self]) -> Result<Self> {
        if operands.is_empty() {
            return Err(Error::InvalidAdTensor {
                message: "partial-diagonal einsum requires at least one operand".to_string(),
            });
        }

        let operand_meta: Vec<OperandAxisClasses> = operands
            .iter()
            .map(|operand| {
                OperandAxisClasses::new(operand.logical_dims.clone(), operand.axis_classes.clone())
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::InvalidAdTensor {
                message: format!("invalid PartialDiagonal operand metadata: {e}"),
            })?;
        let plan = plan_axis_classes_for_subscripts(&operand_meta, subscripts).map_err(|e| {
            Error::InvalidAdTensor {
                message: format!("failed to plan PartialDiagonal einsum: {e}"),
            }
        })?;

        with_cpu_runtime("partial_diag_einsum", |ctx| {
            let mut normalized_payloads: Vec<Tensor<T>> = Vec::with_capacity(operands.len());
            let mut normalized_roots: Vec<Vec<usize>> = Vec::with_capacity(operands.len());

            for (operand_idx, operand) in operands.iter().enumerate() {
                let class_roots = &plan.operand_plans[operand_idx].class_roots;
                if operand.payload.dims().len() != class_roots.len() {
                    return Err(Error::InvalidAdTensor {
                        message: format!(
                            "operand {} payload rank {} does not match planned local class count {}",
                            operand_idx,
                            operand.payload.dims().len(),
                            class_roots.len()
                        ),
                    });
                }
                let (normalized, roots) =
                    normalize_payload_for_roots(ctx, &operand.payload, class_roots)?;
                normalized_payloads.push(normalized);
                normalized_roots.push(roots);
            }

            let input_labels_u32: Vec<Vec<u32>> = normalized_roots
                .iter()
                .map(|roots| usize_vec_to_u32(roots))
                .collect::<Result<_>>()?;
            let output_labels_u32 = usize_vec_to_u32(&plan.output_compressed_roots)?;
            let input_refs: Vec<&[u32]> = input_labels_u32.iter().map(Vec::as_slice).collect();
            let payload_refs: Vec<&Tensor<T>> = normalized_payloads.iter().collect();
            let backend_subs = Subscripts::new(&input_refs, &output_labels_u32);

            let compressed_output = tf_einsum::einsum_with_subscripts::<Standard<T>, CpuBackend>(
                ctx,
                &backend_subs,
                &payload_refs,
                None,
            )
            .map_err(Error::from)?;

            AdTensor::new(
                plan.output_dims.clone(),
                plan.output_axis_classes.clone(),
                compressed_output,
            )
        })
    }
}

fn with_cpu_runtime<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    with_default_runtime(|runtime| match runtime {
        RuntimeContext::Cpu(ctx) => f(ctx),
        RuntimeContext::Cuda(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "cuda",
        }),
        RuntimeContext::Rocm(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "rocm",
        }),
    })
}

fn canonicalize_axis_classes(classes: &[usize]) -> Vec<usize> {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    classes
        .iter()
        .map(|&class_id| {
            if let Some(&mapped) = map.get(&class_id) {
                mapped
            } else {
                let mapped = next;
                next += 1;
                map.insert(class_id, mapped);
                mapped
            }
        })
        .collect()
}

fn validate_layout<T: Scalar>(
    logical_dims: &[usize],
    axis_classes: &[usize],
    payload: &Tensor<T>,
) -> Result<()> {
    if logical_dims.len() != axis_classes.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "logical_dims length ({}) must match axis_classes length ({})",
                logical_dims.len(),
                axis_classes.len()
            ),
        });
    }
    if logical_dims.is_empty() && payload.dims().is_empty() {
        return Ok(());
    }

    let class_count = axis_classes
        .iter()
        .copied()
        .max()
        .map(|x| x + 1)
        .unwrap_or(0);
    if payload.dims().len() != class_count {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "payload rank {} must equal number of classes {}",
                payload.dims().len(),
                class_count
            ),
        });
    }

    let mut class_dims: Vec<Option<usize>> = vec![None; class_count];
    for (&dim, &class_id) in logical_dims.iter().zip(axis_classes.iter()) {
        if let Some(existing) = class_dims[class_id] {
            if existing != dim {
                return Err(Error::InvalidAdTensor {
                    message: format!(
                        "axis class {class_id} has inconsistent logical dims: {existing} vs {dim}"
                    ),
                });
            }
        } else {
            class_dims[class_id] = Some(dim);
        }
    }

    for (class_id, maybe_dim) in class_dims.iter().enumerate() {
        let expected = maybe_dim.unwrap_or(0);
        let got = payload.dims()[class_id];
        if expected != got {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "payload dim mismatch for class {class_id}: expected {expected}, got {got}"
                ),
            });
        }
    }
    Ok(())
}

fn unique_ids_first_appearance(ids: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &id in ids {
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

fn first_duplicate_pair(ids: &[usize]) -> Option<(usize, usize)> {
    let mut first_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, &id) in ids.iter().enumerate() {
        if let Some(&first) = first_pos.get(&id) {
            return Some((first, pos));
        }
        first_pos.insert(id, pos);
    }
    None
}

fn normalize_payload_for_roots<T>(
    ctx: &mut CpuContext,
    payload: &Tensor<T>,
    roots: &[usize],
) -> Result<(Tensor<T>, Vec<usize>)>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    if payload.dims().len() != roots.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "payload rank {} must match roots length {}",
                payload.dims().len(),
                roots.len()
            ),
        });
    }
    if unique_ids_first_appearance(roots).len() == roots.len() {
        return Ok((payload.clone(), roots.to_vec()));
    }

    let mut current_payload = payload.clone();
    let mut current_roots = roots.to_vec();
    let mut round = 0u32;

    while let Some((pos_a, pos_b)) = first_duplicate_pair(&current_roots) {
        // Pairwise diagonal extraction to avoid depending on 3-way repeated labels.
        let rank = current_roots.len();
        let base = 1_000_000u32.saturating_add(round.saturating_mul(10_000));
        let mut input_labels: Vec<u32> = (0..rank).map(|i| base + i as u32).collect();
        input_labels[pos_b] = input_labels[pos_a];
        let output_labels: Vec<u32> = input_labels
            .iter()
            .enumerate()
            .filter_map(|(axis, &label)| (axis != pos_b).then_some(label))
            .collect();
        let inputs = [input_labels.as_slice()];
        let subs = Subscripts::new(&inputs, &output_labels);
        current_payload = tf_einsum::einsum_with_subscripts::<Standard<T>, CpuBackend>(
            ctx,
            &subs,
            &[&current_payload],
            None,
        )
        .map_err(Error::from)?;
        current_roots.remove(pos_b);
        round = round.saturating_add(1);
    }

    Ok((current_payload, current_roots))
}

fn usize_vec_to_u32(values: &[usize]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|&v| {
            u32::try_from(v).map_err(|_| Error::InvalidAdTensor {
                message: format!("label id {} does not fit into u32", v),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{set_default_runtime, RuntimeContext};
    use tenferro_prims::CpuContext;
    use tenferro_tensor::MemoryOrder;

    fn vector(data: &[f64]) -> Tensor<f64> {
        Tensor::<f64>::from_slice(data, &[data.len()], MemoryOrder::ColumnMajor).unwrap()
    }

    fn dense3(data: &[f64], d0: usize, d1: usize, d2: usize) -> Tensor<f64> {
        Tensor::<f64>::from_slice(data, &[d0, d1, d2], MemoryOrder::ColumnMajor).unwrap()
    }

    fn scalar_value(t: &Tensor<f64>) -> f64 {
        let c = t.clone().into_contiguous(MemoryOrder::ColumnMajor);
        c.buffer().as_slice().unwrap()[0]
    }

    #[test]
    fn constructor_canonicalizes_axis_classes() {
        let payload =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let x = AdTensor::new(vec![2, 2, 2], vec![7, 7, 9], payload).unwrap();
        assert_eq!(x.axis_classes(), &[0, 0, 1]);
        assert_eq!(x.logical_dims(), &[2, 2, 2]);
    }

    #[test]
    fn to_dense_from_diag_rank2() {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let x = AdTensor::from_diagonal_vector(vector(&[1.0, 2.0, 3.0]), 2).unwrap();
        assert!(!x.is_dense());
        assert!(x.is_diag());
        let dense = x.to_dense().unwrap();
        assert_eq!(dense.dims(), &[3, 3]);
        let s = dense.clone().into_contiguous(MemoryOrder::ColumnMajor);
        assert_eq!(
            s.buffer().as_slice().unwrap(),
            &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );
    }

    #[test]
    fn dense_layout_flags() {
        let x = AdTensor::from_dense(dense3(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 2, 2));
        assert!(x.is_dense());
        assert!(!x.is_diag());
    }

    #[test]
    fn einsum_diag_chain_preserves_compressed_output() {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = AdTensor::from_diagonal_vector(vector(&[1.0, 2.0, 3.0]), 2).unwrap();
        let b = AdTensor::from_diagonal_vector(vector(&[4.0, 5.0, 6.0]), 2).unwrap();
        let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let c = AdTensor::einsum_with_subscripts(&subs, &[&a, &b]).unwrap();

        assert_eq!(c.logical_dims(), &[3, 3]);
        assert_eq!(c.axis_classes(), &[0, 0]);
        assert_eq!(c.payload().dims(), &[3]);
        let payload = c
            .payload()
            .clone()
            .into_contiguous(MemoryOrder::ColumnMajor);
        assert_eq!(payload.buffer().as_slice().unwrap(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn einsum_dense_threeway_repeat_normalizes_payload() {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = AdTensor::from_dense(dense3(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 2, 2));
        let b = AdTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap();
        let c = AdTensor::from_diagonal_vector(vector(&[5.0, 7.0]), 2).unwrap();

        let subs = Subscripts::new(&[&[0, 1, 2], &[0, 1], &[1, 2]], &[]);
        let out = AdTensor::einsum_with_subscripts(&subs, &[&a, &b, &c]).unwrap();

        assert!(out.logical_dims().is_empty());
        assert!(out.axis_classes().is_empty());
        assert!(out.payload().dims().is_empty());
        assert!((scalar_value(out.payload()) - 178.0).abs() < 1e-12);
    }
}
