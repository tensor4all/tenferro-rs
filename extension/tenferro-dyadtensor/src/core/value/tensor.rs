use std::sync::{Arc, Mutex, MutexGuard};

use chainrules::{NodeId as ChainNodeId, Tape, TrackedValue};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::core::{AdMode, DynTensor, DynTensorTyped, NodeId};
use crate::structured::StructuredTensor;
use crate::{Error, Result};

mod accessors;
mod reverse_api;

#[derive(Clone)]
struct ReverseAttachment {
    node: ChainNodeId,
    tape: Tape<DynTensor>,
}

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
    attachment: Option<ReverseAttachment>,
    grad: Option<StructuredTensor<T>>,
    hvp: Option<StructuredTensor<T>>,
    is_leaf: bool,
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
        tape: Tape<DynTensor>,
        tangent: Option<StructuredTensor<T>>,
    },
}

/// Tensor newtype carrying AD mode information.
///
/// Reverse-mode values participate in a homogeneous `chainrules::Tape<DynTensor>`.
/// Scalars in reverse-mode are
/// represented as rank-0 tensors rather than a separate scalar graph type.
///
/// # Examples
///
/// ```text
/// use tenferro_dyadtensor::{AdMode, core::AdTensor};
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
    /// ```text
    /// use chainrules::Tape;
    /// use tenferro_dyadtensor::{StructuredTensor, core::{AdTensor, DynTensor}};
    ///
    /// let tape = Tape::<DynTensor>::new();
    /// let x = AdTensor::new_reverse_leaf(StructuredTensor::from_dense(todo!()), &tape)?;
    /// # Ok::<(), tenferro_dyadtensor::Error>(())
    /// ```
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

    /// Creates a reverse-mode leaf with a tangent seed for HVP.
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
                attachment: None,
                grad: None,
                hvp: None,
                is_leaf: true,
            })),
        }))
    }

    pub(crate) fn new_reverse_output(
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
        let attachment = ReverseAttachment {
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
                attachment: Some(attachment),
                grad: None,
                hvp: None,
                is_leaf: false,
            })),
        }))
    }

    pub(crate) fn ensure_reverse_leaf_on(&self, tape: &Tape<DynTensor>) -> Result<()>
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
                match guard.attachment.as_ref() {
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
                        guard.attachment = Some(ReverseAttachment {
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

    fn reverse_attachment(&self) -> Option<ReverseAttachment> {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => lock_reverse_state(state).attachment.clone(),
            _ => None,
        }
    }

    /// Returns the reverse-mode tape handle when this tensor participates in a graph.
    ///
    /// # Examples
    ///
    /// ```text
    /// use chainrules::Tape;
    /// use tenferro_dyadtensor::{StructuredTensor, core::{AdTensor, DynTensor}};
    ///
    /// let tape = Tape::<DynTensor>::new();
    /// let x = AdTensor::new_reverse_leaf(StructuredTensor::from_dense(todo!()), &tape)?;
    /// assert!(x.tape().is_some());
    /// # Ok::<(), tenferro_dyadtensor::Error>(())
    /// ```
    pub fn tape(&self) -> Option<Tape<DynTensor>> {
        self.reverse_attachment().map(|attachment| attachment.tape)
    }

    /// Returns the reverse-mode node id when this tensor participates in a graph.
    ///
    /// # Examples
    ///
    /// ```text
    /// use chainrules::Tape;
    /// use tenferro_dyadtensor::{StructuredTensor, core::{AdTensor, DynTensor}};
    ///
    /// let tape = Tape::<DynTensor>::new();
    /// let x = AdTensor::new_reverse_leaf(StructuredTensor::from_dense(todo!()), &tape)?;
    /// assert!(x.node_id().is_some());
    /// # Ok::<(), tenferro_dyadtensor::Error>(())
    /// ```
    pub fn node_id(&self) -> Option<ChainNodeId> {
        self.reverse_attachment().map(|attachment| attachment.node)
    }

    pub(crate) fn reverse_tape(&self) -> Option<Tape<DynTensor>> {
        self.tape()
    }

    pub(crate) fn reverse_node_id(&self) -> Option<ChainNodeId> {
        self.node_id()
    }

    pub(crate) fn reverse_handle(&self) -> Option<(ChainNodeId, Tape<DynTensor>)> {
        self.reverse_attachment()
            .map(|attachment| (attachment.node, attachment.tape))
    }

    pub(crate) fn as_tracked(&self) -> Option<TrackedValue<DynTensor>>
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

    pub(crate) fn snapshot(&self) -> Result<AdTensorSnapshot<T>> {
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
                        .attachment
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
}

impl<T: Scalar + DynTensorTyped> Clone for AdTensor<T> {
    fn clone(&self) -> Self {
        match &self.0 {
            TensorAdState::Primal(value) => Self::new_primal(value.clone()),
            TensorAdState::Forward { primal, tangent } => Self(TensorAdState::Forward {
                primal: primal.clone(),
                tangent: tangent.clone(),
            }),
            TensorAdState::Reverse {
                primal,
                tangent,
                state,
            } => Self(TensorAdState::Reverse {
                primal: primal.clone(),
                tangent: tangent.clone(),
                state: Arc::clone(state),
            }),
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

impl<T: Scalar + DynTensorTyped + 'static> TryFrom<AdTensorSnapshot<T>> for AdTensor<T> {
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
            } => Ok(Self(TensorAdState::Reverse {
                primal,
                tangent,
                state: Arc::new(Mutex::new(ReverseTensorState {
                    attachment: Some(ReverseAttachment { node, tape }),
                    grad: None,
                    hvp: None,
                    is_leaf: false,
                })),
            })),
        }
    }
}
