use std::sync::{Arc, Mutex, MutexGuard};

use tenferro_algebra::Scalar;
use tenferro_device::{ComputeDevice, LogicalMemorySpace};
use tenferro_internal_error::{Error, Result};
use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, StructuredTensor};
use tenferro_tensor::Tensor;
use tidu::expert::{NodeId as ChainNodeId, Tape, TrackedValue};
use tidu::Value as ReverseValue;

use crate::{AdMode, NodeId};

mod reverse;

#[derive(Clone)]
struct LegacyReverseAttachment {
    node: ChainNodeId,
    tape: Tape<DynTensor>,
}

type EdgeValueHandle<T> = Arc<ReverseValue<StructuredTensor<T>>>;

enum TensorAdState<T: Scalar> {
    Primal(StructuredTensor<T>),
    Forward {
        primal: StructuredTensor<T>,
        tangent: StructuredTensor<T>,
    },
    Reverse {
        primal: StructuredTensor<T>,
        tangent: Option<StructuredTensor<T>>,
        state: Arc<Mutex<ReverseTensorState<T>>>,
    },
}

struct ReverseTensorState<T: Scalar> {
    legacy_attachment: Option<LegacyReverseAttachment>,
    edge_value: Option<EdgeValueHandle<T>>,
    grad: Option<StructuredTensor<T>>,
    hvp: Option<StructuredTensor<T>>,
    is_leaf: bool,
}

#[doc(hidden)]
pub enum AdTensorSnapshot<T: Scalar> {
    Primal(StructuredTensor<T>),
    Forward {
        primal: StructuredTensor<T>,
        tangent: StructuredTensor<T>,
    },
    Reverse {
        primal: StructuredTensor<T>,
        node: ChainNodeId,
        tape: Tape<DynTensor>,
        tangent: Option<StructuredTensor<T>>,
    },
}

/// Tensor newtype carrying AD mode information.
///
/// Reverse-mode values participate in a homogeneous `tidu::expert::Tape<DynTensor>`.
/// Scalars in reverse mode are represented as rank-0 tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_ad_core::{AdMode, AdTensor};
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

fn validate_reverse_state<T: Scalar>(
    op_name: &'static str,
    primal: &StructuredTensor<T>,
    tangent: Option<&StructuredTensor<T>>,
) -> Result<()> {
    if let Some(tangent) = tangent {
        ensure_same_structured_layout(op_name, primal, tangent)?;
    }
    Ok(())
}

fn public_node_id(node_id: Option<ChainNodeId>) -> Option<NodeId> {
    node_id
}

fn lock_reverse_state<T: Scalar>(
    state: &Arc<Mutex<ReverseTensorState<T>>>,
) -> MutexGuard<'_, ReverseTensorState<T>> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl<T: Scalar> AdTensor<T> {
    pub fn new_primal(tensor: impl Into<StructuredTensor<T>>) -> Self {
        Self(TensorAdState::Primal(tensor.into()))
    }

    pub fn new_forward(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self> {
        let primal = primal.into();
        let tangent = tangent.into();
        ensure_same_structured_layout("AdTensor::new_forward", &primal, &tangent)?;
        Ok(Self(TensorAdState::Forward { primal, tangent }))
    }

    pub fn new_reverse_leaf(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
    ) -> Result<Self>
    where
        T: DynTensorTyped,
    {
        let primal = primal.into();
        let tensor = Self::new_pending_reverse(primal, None)?;
        tensor.ensure_reverse_leaf_on(tape)?;
        Ok(tensor)
    }

    pub fn new_reverse_leaf_with_tangent(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
    ) -> Result<Self>
    where
        T: DynTensorTyped,
    {
        let primal = primal.into();
        let tangent = tangent.into();
        ensure_same_structured_layout(
            "AdTensor::new_reverse_leaf_with_tangent",
            &primal,
            &tangent,
        )?;
        let tensor = Self::new_pending_reverse(primal, Some(tangent))?;
        tensor.ensure_reverse_leaf_on(tape)?;
        Ok(tensor)
    }

    pub(crate) fn new_pending_reverse(
        primal: StructuredTensor<T>,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self> {
        validate_reverse_state("AdTensor::new_pending_reverse", &primal, tangent.as_ref())?;
        Ok(Self(TensorAdState::Reverse {
            primal,
            tangent,
            state: Arc::new(Mutex::new(ReverseTensorState {
                legacy_attachment: None,
                edge_value: None,
                grad: None,
                hvp: None,
                is_leaf: true,
            })),
        }))
    }

    #[doc(hidden)]
    pub fn new_reverse_output(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: DynTensorTyped,
    {
        let primal = primal.into();
        if let Some(tangent) = &tangent {
            ensure_same_structured_layout("AdTensor::new_reverse_output", &primal, tangent)?;
        }
        let tracked = tape.placeholder(
            T::into_dyn(primal.clone()),
            tangent.clone().map(T::into_dyn),
        );
        let attachment = LegacyReverseAttachment {
            node: tracked
                .node_id()
                .expect("reverse placeholder carries a node"),
            tape: tracked
                .tape()
                .cloned()
                .expect("reverse placeholder carries a tape"),
        };
        Ok(Self(TensorAdState::Reverse {
            primal,
            tangent,
            state: Arc::new(Mutex::new(ReverseTensorState {
                legacy_attachment: Some(attachment),
                edge_value: None,
                grad: None,
                hvp: None,
                is_leaf: false,
            })),
        }))
    }

    #[doc(hidden)]
    pub fn ensure_reverse_leaf_on(&self, tape: &Tape<DynTensor>) -> Result<()>
    where
        T: DynTensorTyped,
    {
        match &self.0 {
            TensorAdState::Reverse {
                primal,
                tangent,
                state,
            } => {
                let mut guard = lock_reverse_state(state);
                match guard.legacy_attachment.as_ref() {
                    Some(existing) if existing.tape.same_tape(tape) => Ok(()),
                    Some(existing) => Err(Error::MixedReverseTape {
                        expected: existing.tape.id() as u64,
                        found: tape.id() as u64,
                    }),
                    None => {
                        if !guard.is_leaf {
                            return Err(Error::InvalidAdTensor {
                                message: "reverse output is missing an attached tape node"
                                    .to_string(),
                            });
                        }
                        let tracked = match tangent {
                            Some(tangent) => tape
                                .leaf_with_tangent(
                                    T::into_dyn(primal.clone()),
                                    T::into_dyn(tangent.clone()),
                                )
                                .map_err(Error::Autodiff)?,
                            None => tape.leaf(T::into_dyn(primal.clone())),
                        };
                        guard.legacy_attachment = Some(LegacyReverseAttachment {
                            node: tracked.node_id().expect("reverse leaf carries a node"),
                            tape: tracked
                                .tape()
                                .cloned()
                                .expect("reverse leaf carries a tape"),
                        });
                        Ok(())
                    }
                }
            }
            _ => Ok(()),
        }
    }

    fn reverse_attachment(&self) -> Option<LegacyReverseAttachment> {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => {
                lock_reverse_state(state).legacy_attachment.clone()
            }
            _ => None,
        }
    }

    #[doc(hidden)]
    pub fn reverse_edge_value(&self) -> Option<EdgeValueHandle<T>> {
        match &self.0 {
            TensorAdState::Reverse {
                primal,
                tangent,
                state,
            } => {
                if tangent.is_some() {
                    return None;
                }
                let mut guard = lock_reverse_state(state);
                if let Some(edge_value) = guard.edge_value.as_ref() {
                    return Some(edge_value.clone());
                }
                if guard.is_leaf && guard.legacy_attachment.is_none() {
                    let edge_value =
                        Arc::new(ReverseValue::new(primal.clone()).requires_grad_(true));
                    guard.edge_value = Some(edge_value.clone());
                    return Some(edge_value);
                }
                None
            }
            _ => None,
        }
    }

    pub fn tape(&self) -> Option<Tape<DynTensor>> {
        self.reverse_attachment().map(|attachment| attachment.tape)
    }

    pub fn node_id(&self) -> Option<ChainNodeId> {
        self.reverse_attachment().map(|attachment| attachment.node)
    }

    #[doc(hidden)]
    pub fn reverse_tape(&self) -> Option<Tape<DynTensor>> {
        self.tape()
    }

    #[doc(hidden)]
    pub fn reverse_node_id(&self) -> Option<ChainNodeId> {
        self.node_id()
    }

    #[doc(hidden)]
    pub fn reverse_handle(&self) -> Option<(ChainNodeId, Tape<DynTensor>)> {
        self.reverse_attachment()
            .map(|attachment| (attachment.node, attachment.tape))
    }

    #[doc(hidden)]
    pub fn as_tracked(&self) -> Option<TrackedValue<DynTensor>>
    where
        T: DynTensorTyped,
    {
        let (node, tape) = self.reverse_handle()?;
        tape.tracked_existing(
            node,
            T::into_dyn(self.structured_primal().clone()),
            self.structured_tangent().cloned().map(T::into_dyn),
        )
        .ok()
    }

    #[doc(hidden)]
    pub fn snapshot(&self) -> Result<AdTensorSnapshot<T>> {
        match &self.0 {
            TensorAdState::Primal(value) => Ok(AdTensorSnapshot::Primal(value.clone())),
            TensorAdState::Forward { primal, tangent } => Ok(AdTensorSnapshot::Forward {
                primal: primal.clone(),
                tangent: tangent.clone(),
            }),
            TensorAdState::Reverse {
                primal,
                tangent,
                state,
            } => {
                let guard = lock_reverse_state(state);
                let attachment =
                    guard
                        .legacy_attachment
                        .clone()
                        .ok_or_else(|| Error::InvalidAdTensor {
                            message: "reverse tensor is not attached to a graph yet".to_string(),
                        })?;
                Ok(AdTensorSnapshot::Reverse {
                    primal: primal.clone(),
                    node: public_node_id(Some(attachment.node)).expect("public node id"),
                    tape: attachment.tape,
                    tangent: tangent.clone(),
                })
            }
        }
    }

    pub fn mode(&self) -> AdMode {
        match &self.0 {
            TensorAdState::Primal(_) => AdMode::Primal,
            TensorAdState::Forward { .. } => AdMode::Forward,
            TensorAdState::Reverse { .. } => AdMode::Reverse,
        }
    }

    pub fn structured_primal(&self) -> &StructuredTensor<T> {
        match &self.0 {
            TensorAdState::Primal(value) => value,
            TensorAdState::Forward { primal, .. } => primal,
            TensorAdState::Reverse { primal, .. } => primal,
        }
    }

    pub fn primal(&self) -> &Tensor<T> {
        self.structured_primal().payload()
    }

    pub fn structured_tangent(&self) -> Option<&StructuredTensor<T>> {
        match &self.0 {
            TensorAdState::Primal(_) => None,
            TensorAdState::Forward { tangent, .. } => Some(tangent),
            TensorAdState::Reverse { tangent, .. } => tangent.as_ref(),
        }
    }

    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.structured_tangent().map(|tensor| tensor.payload())
    }

    pub fn dims(&self) -> &[usize] {
        self.structured_primal().logical_dims()
    }

    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn axis_classes(&self) -> &[usize] {
        self.structured_primal().axis_classes()
    }

    pub fn is_dense(&self) -> bool {
        self.structured_primal().is_dense()
    }

    pub fn is_diag(&self) -> bool {
        self.structured_primal().is_diag()
    }

    pub fn memory_space(&self) -> LogicalMemorySpace {
        self.structured_primal().memory_space()
    }

    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.structured_primal().preferred_compute_device()
    }

    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        match &mut self.0 {
            TensorAdState::Primal(primal) => primal.set_preferred_compute_device(device),
            TensorAdState::Forward { primal, tangent } => {
                primal.set_preferred_compute_device(device);
                tangent.set_preferred_compute_device(device);
            }
            TensorAdState::Reverse {
                primal, tangent, ..
            } => {
                primal.set_preferred_compute_device(device);
                if let Some(tangent) = tangent {
                    tangent.set_preferred_compute_device(device);
                }
            }
        }
    }

    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Self> {
        match &self.0 {
            TensorAdState::Primal(primal) => {
                Ok(Self::new_primal(primal.to_memory_space_async(target)?))
            }
            TensorAdState::Forward { primal, tangent } => Self::new_forward(
                primal.to_memory_space_async(target)?,
                tangent.to_memory_space_async(target)?,
            ),
            TensorAdState::Reverse {
                primal,
                tangent,
                state,
            } => Ok(Self(TensorAdState::Reverse {
                primal: primal.to_memory_space_async(target)?,
                tangent: tangent
                    .as_ref()
                    .map(|value| value.to_memory_space_async(target))
                    .transpose()?,
                state: Arc::clone(state),
            })),
        }
    }

    pub fn to_memory_space(&self, target: LogicalMemorySpace) -> Result<Self> {
        let moved = self.to_memory_space_async(target)?;
        moved.wait();
        Ok(moved)
    }

    pub fn wait(&self) {
        self.structured_primal().wait();
        if let Some(tangent) = self.structured_tangent() {
            tangent.wait();
        }
    }

    pub fn is_ready(&self) -> bool {
        self.structured_primal().is_ready()
            && self
                .structured_tangent()
                .map(|tangent| tangent.is_ready())
                .unwrap_or(true)
    }

    pub fn requires_grad(&self) -> bool {
        matches!(self.0, TensorAdState::Reverse { .. })
    }

    pub fn is_leaf(&self) -> bool {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => lock_reverse_state(state).is_leaf,
            _ => true,
        }
    }

    pub fn with_requires_grad(&self, enabled: bool) -> Result<Self>
    where
        T: DynTensorTyped,
    {
        if enabled {
            if self.requires_grad() && self.is_leaf() {
                return Ok(self.clone());
            }
            return Self::new_pending_reverse(self.structured_primal().clone(), None);
        }

        Ok(Self::new_primal(self.structured_primal().clone()))
    }

    pub fn set_requires_grad(&mut self, enabled: bool) -> Result<()>
    where
        T: DynTensorTyped,
    {
        match (&self.0, enabled) {
            (TensorAdState::Primal(primal), true) => {
                *self = Self::new_pending_reverse(primal.clone(), None)?;
                Ok(())
            }
            (TensorAdState::Forward { .. }, true) => Err(Error::UnsupportedAdOp {
                op: "set_requires_grad",
            }),
            (TensorAdState::Reverse { primal, .. }, false) => {
                *self = Self::new_primal(primal.clone());
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
