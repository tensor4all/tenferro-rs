use chainrules_core::Differentiable;
use tenferro_algebra::Scalar;

use super::{
    lock_reverse_state, AdTensor, DynTensorTyped, Error, Result, StructuredTensor, TensorAdState,
};

impl<T: Scalar> AdTensor<T> {
    pub fn requires_grad(&self) -> bool {
        matches!(self.0, TensorAdState::Reverse { .. })
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

    pub fn grad(&self) -> Option<StructuredTensor<T>>
    where
        T: Clone,
    {
        match &self.0 {
            TensorAdState::Reverse { state, .. } => lock_reverse_state(state).grad.clone(),
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
                let mut guard = lock_reverse_state(state);
                if !guard.is_leaf {
                    return Err(Error::InvalidAdTensor {
                        message: "zero_grad is valid on reverse leaf tensors only".to_string(),
                    });
                }
                guard.grad = None;
                guard.hvp = None;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn accumulate_leaf_grad(&self, grad: StructuredTensor<T>) -> Result<()>
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
                        <StructuredTensor<T> as Differentiable>::accumulate_tangent(existing, &grad)
                    }
                    None => grad,
                });
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[cfg(test)]
    pub(crate) fn accumulate_leaf_hvp(&self, hvp: StructuredTensor<T>) -> Result<()>
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
                        <StructuredTensor<T> as Differentiable>::accumulate_tangent(existing, &hvp)
                    }
                    None => hvp,
                });
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
