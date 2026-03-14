use chainrules::{NodeId as ChainNodeId, Tape, TrackedValue};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::structured::StructuredTensor;
use crate::{AdMode, Error, NodeId, Result};

enum TensorAdState<T: Scalar> {
    Primal(StructuredTensor<T>),
    Forward {
        primal: StructuredTensor<T>,
        tangent: StructuredTensor<T>,
    },
    Reverse(TrackedValue<StructuredTensor<T>>),
}

pub(crate) enum AdTensorSnapshot<T: Scalar> {
    Primal(StructuredTensor<T>),
    Forward {
        primal: StructuredTensor<T>,
        tangent: StructuredTensor<T>,
    },
    Reverse {
        primal: StructuredTensor<T>,
        node: ChainNodeId,
        tape: Tape<StructuredTensor<T>>,
        tangent: Option<StructuredTensor<T>>,
    },
}

/// Tensor newtype carrying AD mode information.
///
/// Reverse-mode values participate in a homogeneous
/// `chainrules::Tape<StructuredTensor<T>>`. Scalars in reverse-mode are
/// represented as rank-0 tensors rather than a separate scalar graph type.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdTensor};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x: AdTensor<f64> = t.into();
/// assert_eq!(x.mode(), AdMode::Primal);
/// ```
pub struct AdTensor<T: Scalar>(TensorAdState<T>);

fn ensure_same_structured_layout<T: Scalar>(
    op_name: &'static str,
    primal: &StructuredTensor<T>,
    tangent: &StructuredTensor<T>,
) -> Result<()> {
    if primal.logical_dims() != tangent.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires tangent.logical_dims == primal.logical_dims, got primal={:?}, tangent={:?}",
                primal.logical_dims(),
                tangent.logical_dims()
            ),
        });
    }
    if primal.axis_classes() != tangent.axis_classes() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires tangent.axis_classes == primal.axis_classes, got primal={:?}, tangent={:?}",
                primal.axis_classes(),
                tangent.axis_classes()
            ),
        });
    }
    Ok(())
}

fn validate_reverse_tracked<T: Scalar>(
    op_name: &'static str,
    value: &TrackedValue<StructuredTensor<T>>,
) -> Result<()> {
    if let Some(tangent) = value.tangent() {
        ensure_same_structured_layout(op_name, value.value(), tangent)?;
    }
    Ok(())
}

fn public_node_id(node_id: Option<ChainNodeId>) -> Option<NodeId> {
    node_id
}

impl<T: Scalar> AdTensor<T> {
    /// Creates a primal tensor.
    pub fn new_primal(tensor: impl Into<StructuredTensor<T>>) -> Self {
        Self(TensorAdState::Primal(tensor.into()))
    }

    /// Creates a forward-mode tensor.
    pub fn new_forward(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self> {
        let primal = primal.into();
        let tangent = tangent.into();
        ensure_same_structured_layout("AdTensor::new_forward", &primal, &tangent)?;
        Ok(Self(TensorAdState::Forward { primal, tangent }))
    }

    /// Creates a reverse-mode leaf on a homogeneous tape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Tape;
    /// use tenferro_dyadtensor::{AdTensor, StructuredTensor};
    ///
    /// let tape = Tape::<StructuredTensor<f64>>::new();
    /// let x = AdTensor::new_reverse_leaf(StructuredTensor::from_dense(todo!()), &tape)?;
    /// # Ok::<(), tenferro_dyadtensor::Error>(())
    /// ```
    pub fn new_reverse_leaf(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<StructuredTensor<T>>,
    ) -> Result<Self> {
        Self::from_tracked(tape.leaf(primal.into()))
    }

    /// Creates a reverse-mode leaf with a tangent seed for HVP.
    pub fn new_reverse_leaf_with_tangent(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
        tape: &Tape<StructuredTensor<T>>,
    ) -> Result<Self> {
        let primal = primal.into();
        let tangent = tangent.into();
        ensure_same_structured_layout(
            "AdTensor::new_reverse_leaf_with_tangent",
            &primal,
            &tangent,
        )?;
        let tracked = tape
            .leaf_with_tangent(primal, tangent)
            .map_err(Error::Autodiff)?;
        Self::from_tracked(tracked)
    }

    pub(crate) fn from_tracked(value: TrackedValue<StructuredTensor<T>>) -> Result<Self> {
        validate_reverse_tracked("AdTensor::from_tracked", &value)?;
        Ok(Self(TensorAdState::Reverse(value)))
    }

    pub(crate) fn new_reverse_output(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<StructuredTensor<T>>,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self> {
        let primal = primal.into();
        if let Some(tangent) = &tangent {
            ensure_same_structured_layout("AdTensor::new_reverse_output", &primal, tangent)?;
        }
        Self::from_tracked(tape.placeholder(primal, tangent))
    }

    pub(crate) fn as_tracked(&self) -> Option<&TrackedValue<StructuredTensor<T>>> {
        match &self.0 {
            TensorAdState::Reverse(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the reverse-mode tape handle when this tensor participates in a graph.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Tape;
    /// use tenferro_dyadtensor::{AdTensor, StructuredTensor};
    ///
    /// let tape = Tape::<StructuredTensor<f64>>::new();
    /// let x = AdTensor::new_reverse_leaf(StructuredTensor::from_dense(todo!()), &tape)?;
    /// assert!(x.tape().is_some());
    /// # Ok::<(), tenferro_dyadtensor::Error>(())
    /// ```
    pub fn tape(&self) -> Option<&Tape<StructuredTensor<T>>> {
        self.as_tracked().and_then(TrackedValue::tape)
    }

    /// Returns the reverse-mode node id when this tensor participates in a graph.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Tape;
    /// use tenferro_dyadtensor::{AdTensor, StructuredTensor};
    ///
    /// let tape = Tape::<StructuredTensor<f64>>::new();
    /// let x = AdTensor::new_reverse_leaf(StructuredTensor::from_dense(todo!()), &tape)?;
    /// assert!(x.node_id().is_some());
    /// # Ok::<(), tenferro_dyadtensor::Error>(())
    /// ```
    pub fn node_id(&self) -> Option<ChainNodeId> {
        self.as_tracked().and_then(TrackedValue::node_id)
    }

    pub(crate) fn reverse_tape(&self) -> Option<&Tape<StructuredTensor<T>>> {
        self.tape()
    }

    pub(crate) fn reverse_node_id(&self) -> Option<ChainNodeId> {
        self.node_id()
    }

    pub(crate) fn reverse_handle(&self) -> Option<(ChainNodeId, Tape<StructuredTensor<T>>)> {
        self.as_tracked().map(|value| {
            (
                value
                    .node_id()
                    .expect("reverse values always carry a node id"),
                value
                    .tape()
                    .cloned()
                    .expect("reverse values always carry a tape"),
            )
        })
    }

    pub(crate) fn snapshot(&self) -> AdTensorSnapshot<T> {
        match &self.0 {
            TensorAdState::Primal(value) => AdTensorSnapshot::Primal(value.clone()),
            TensorAdState::Forward { primal, tangent } => AdTensorSnapshot::Forward {
                primal: primal.clone(),
                tangent: tangent.clone(),
            },
            TensorAdState::Reverse(value) => AdTensorSnapshot::Reverse {
                primal: value.value().clone(),
                node: public_node_id(value.node_id())
                    .expect("reverse values always carry a node id"),
                tape: value
                    .tape()
                    .cloned()
                    .expect("reverse values always carry a tape"),
                tangent: value.tangent().cloned(),
            },
        }
    }

    /// Returns AD mode.
    pub fn mode(&self) -> AdMode {
        match &self.0 {
            TensorAdState::Primal(_) => AdMode::Primal,
            TensorAdState::Forward { .. } => AdMode::Forward,
            TensorAdState::Reverse(_) => AdMode::Reverse,
        }
    }

    /// Returns structured primal payload reference.
    pub fn structured_primal(&self) -> &StructuredTensor<T> {
        match &self.0 {
            TensorAdState::Primal(value) => value,
            TensorAdState::Forward { primal, .. } => primal,
            TensorAdState::Reverse(value) => value.value(),
        }
    }

    /// Returns compressed primal payload tensor reference.
    pub fn primal(&self) -> &Tensor<T> {
        self.structured_primal().payload()
    }

    /// Returns structured tangent reference when available.
    pub fn structured_tangent(&self) -> Option<&StructuredTensor<T>> {
        match &self.0 {
            TensorAdState::Primal(_) => None,
            TensorAdState::Forward { tangent, .. } => Some(tangent),
            TensorAdState::Reverse(value) => value.tangent(),
        }
    }

    /// Returns compressed tangent payload tensor reference when available.
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.structured_tangent().map(StructuredTensor::payload)
    }

    /// Returns dimensions of the primal tensor.
    pub fn dims(&self) -> &[usize] {
        self.structured_primal().logical_dims()
    }

    /// Returns number of dimensions of the primal tensor.
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns total number of elements in the primal tensor.
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns axis classes of the structured primal.
    pub fn axis_classes(&self) -> &[usize] {
        self.structured_primal().axis_classes()
    }

    /// Returns `true` when the structured primal is dense.
    pub fn is_dense(&self) -> bool {
        self.structured_primal().is_dense()
    }

    /// Returns `true` when the structured primal is diagonal.
    pub fn is_diag(&self) -> bool {
        self.structured_primal().is_diag()
    }
}

impl<T: Scalar> Clone for AdTensor<T> {
    fn clone(&self) -> Self {
        match &self.0 {
            TensorAdState::Primal(value) => Self::new_primal(value.clone()),
            TensorAdState::Forward { primal, tangent } => {
                Self::new_forward(primal.clone(), tangent.clone())
                    .expect("forward clone should preserve valid structured layout")
            }
            TensorAdState::Reverse(value) => {
                let tape = value
                    .tape()
                    .cloned()
                    .expect("reverse values always carry a tape");
                let node = value
                    .node_id()
                    .expect("reverse values always carry a node id");
                let tracked = tape
                    .tracked_existing(node, value.value().clone(), value.tangent().cloned())
                    .expect("reverse clone should preserve valid tracked value");
                Self::from_tracked(tracked)
                    .expect("reverse clone should preserve valid structured layout")
            }
        }
    }
}

impl<T: Scalar> From<Tensor<T>> for AdTensor<T> {
    fn from(value: Tensor<T>) -> Self {
        Self::new_primal(StructuredTensor::from_dense(value))
    }
}

impl<T: Scalar> From<StructuredTensor<T>> for AdTensor<T> {
    fn from(value: StructuredTensor<T>) -> Self {
        Self::new_primal(value)
    }
}

impl<T: Scalar + 'static> TryFrom<AdTensorSnapshot<T>> for AdTensor<T> {
    type Error = Error;

    fn try_from(value: AdTensorSnapshot<T>) -> Result<Self> {
        match value {
            AdTensorSnapshot::Primal(primal) => Ok(Self::new_primal(primal)),
            AdTensorSnapshot::Forward { primal, tangent } => Self::new_forward(primal, tangent),
            AdTensorSnapshot::Reverse {
                primal,
                node,
                tape,
                tangent,
            } => {
                let tracked = tape
                    .tracked_existing(node, primal, tangent)
                    .map_err(Error::from)?;
                Self::from_tracked(tracked)
            }
        }
    }
}
