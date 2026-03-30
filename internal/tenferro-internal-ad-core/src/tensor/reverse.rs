use std::sync::{Arc, Mutex};

use tenferro_algebra::Scalar;
use tenferro_internal_error::{Error, Result};
use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, StructuredTensor};
use tenferro_tensor::Tensor;
use tidu::expert::Tape;

use super::{
    ensure_same_structured_layout, lock_reverse_state, AdTensor, AdTensorSnapshot, EdgeValueHandle,
    LegacyReverseAttachment, ReverseTensorState, TensorAdState,
};

impl<T: Scalar> AdTensor<T> {
    pub fn grad(&self) -> Option<StructuredTensor<T>>
    where
        T: Clone,
    {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => {
                let (edge_value, grad) = {
                    let guard = lock_reverse_state(state);
                    (guard.edge_value.clone(), guard.grad.clone())
                };
                if let Some(edge_value) = edge_value {
                    if let Ok(Some(edge_grad)) = edge_value.grad() {
                        return Some(edge_grad);
                    }
                }
                grad
            }
            _ => None,
        }
    }

    pub fn hvp(&self) -> Option<StructuredTensor<T>>
    where
        T: Clone,
    {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => lock_reverse_state(state).hvp.clone(),
            _ => None,
        }
    }

    pub fn zero_grad(&self) -> Result<()> {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => {
                let edge_value = {
                    let mut guard = lock_reverse_state(state);
                    if !guard.is_leaf {
                        return Err(Error::InvalidAdTensor {
                            message: "zero_grad is valid on reverse leaf tensors only".to_string(),
                        });
                    }
                    guard.grad = None;
                    guard.hvp = None;
                    guard.edge_value.clone()
                };
                if let Some(edge_value) = edge_value {
                    edge_value.zero_grad().map_err(Error::from)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[doc(hidden)]
    pub fn new_reverse_output_from_edge(
        primal: impl Into<StructuredTensor<T>>,
        edge_value: EdgeValueHandle<T>,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self> {
        let primal = primal.into();
        if let Some(tangent) = &tangent {
            ensure_same_structured_layout(
                "AdTensor::new_reverse_output_from_edge",
                &primal,
                tangent,
            )?;
        }
        Ok(Self(TensorAdState::Reverse {
            primal,
            tangent,
            state: Arc::new(Mutex::new(ReverseTensorState {
                legacy_attachment: None,
                edge_value: Some(edge_value),
                grad: None,
                hvp: None,
                is_leaf: false,
            })),
        }))
    }

    #[doc(hidden)]
    pub fn from_reverse_edge_value(output: tidu::Value<StructuredTensor<T>>) -> Result<Self> {
        let primal = output.primal().clone();
        Self::new_reverse_output_from_edge(primal, Arc::new(output), None)
    }

    #[doc(hidden)]
    pub fn from_reverse_edge_handle(
        primal: impl Into<StructuredTensor<T>>,
        edge_value: EdgeValueHandle<T>,
    ) -> Result<Self> {
        Self::new_reverse_output_from_edge(primal, edge_value, None)
    }

    #[doc(hidden)]
    pub fn from_reverse_output(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: DynTensorTyped,
    {
        Self::new_reverse_output(primal, tape, tangent)
    }

    #[doc(hidden)]
    pub fn has_edge_reverse(&self) -> bool {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => lock_reverse_state(state).edge_value.is_some(),
            _ => false,
        }
    }

    #[doc(hidden)]
    pub fn has_legacy_reverse(&self) -> bool {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => {
                lock_reverse_state(state).legacy_attachment.is_some()
            }
            _ => false,
        }
    }

    #[doc(hidden)]
    pub fn accumulate_leaf_grad(&self, grad: StructuredTensor<T>) -> Result<()>
    where
        T: Clone,
    {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => {
                let mut guard = lock_reverse_state(state);
                if !guard.is_leaf {
                    return Err(Error::InvalidAdTensor {
                        message: "gradient accumulation requires reverse leaf tensor".to_string(),
                    });
                }
                guard.grad = Some(match guard.grad.take() {
                    Some(existing) => {
                        <StructuredTensor<T> as chainrules_core::Differentiable>::accumulate_tangent(
                            existing, &grad,
                        )
                    }
                    None => grad,
                });
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[doc(hidden)]
    pub fn accumulate_input_grad(&self, grad: StructuredTensor<T>) -> Result<()>
    where
        T: Clone,
    {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => {
                let mut guard = lock_reverse_state(state);
                guard.grad = Some(match guard.grad.take() {
                    Some(existing) => {
                        <StructuredTensor<T> as chainrules_core::Differentiable>::accumulate_tangent(
                            existing, &grad,
                        )
                    }
                    None => grad,
                });
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[doc(hidden)]
    pub fn accumulate_leaf_hvp(&self, hvp: StructuredTensor<T>) -> Result<()>
    where
        T: Clone,
    {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => {
                let mut guard = lock_reverse_state(state);
                if !guard.is_leaf {
                    return Err(Error::InvalidAdTensor {
                        message: "HVP accumulation requires reverse leaf tensor".to_string(),
                    });
                }
                guard.hvp = Some(match guard.hvp.take() {
                    Some(existing) => {
                        <StructuredTensor<T> as chainrules_core::Differentiable>::accumulate_tangent(
                            existing, &hvp,
                        )
                    }
                    None => hvp,
                });
                Ok(())
            }
            _ => Ok(()),
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
        Self::new_primal(StructuredTensor::from(value))
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
                    legacy_attachment: Some(LegacyReverseAttachment { node, tape }),
                    edge_value: None,
                    grad: None,
                    hvp: None,
                    is_leaf: false,
                })),
            })),
        }
    }
}
