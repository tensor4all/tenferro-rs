use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use computegraph::{GraphOp, Operand};

use crate::config::DotGeneralConfig;
use crate::semiring_op_kind::SemiringOpKind;
use crate::semiring_ops::SemiringOps;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SemiringInputKey {
    pub id: u64,
}

pub struct SemiringOp<T> {
    pub kind: SemiringOpKind,
    _marker: PhantomData<T>,
}

impl<T> SemiringOp<T> {
    pub fn new(kind: SemiringOpKind) -> Self {
        Self {
            kind,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for SemiringOp<T> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> fmt::Debug for SemiringOp<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemiringOp")
            .field("kind", &self.kind)
            .finish()
    }
}

impl<T> Hash for SemiringOp<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl<T> PartialEq for SemiringOp<T> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl<T> Eq for SemiringOp<T> {}

impl<T: Operand> GraphOp for SemiringOp<T> {
    type Operand = T;
    type Context = ();
    type InputKey = SemiringInputKey;

    fn n_inputs(&self) -> usize {
        todo!()
    }

    fn n_outputs(&self) -> usize {
        todo!()
    }

    fn eval(&self, _ctx: &mut Self::Context, _inputs: &[&Self::Operand]) -> Vec<Self::Operand> {
        todo!()
    }
}

impl<T: Operand> SemiringOps for SemiringOp<T> {
    fn add_op() -> Self {
        Self::new(SemiringOpKind::Add)
    }

    fn mul_op() -> Self {
        Self::new(SemiringOpKind::Mul)
    }

    fn dot_general(config: DotGeneralConfig) -> Self {
        Self::new(SemiringOpKind::DotGeneral(config))
    }

    fn reduce_sum(axes: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::ReduceSum { axes })
    }

    fn transpose_op(perm: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::Transpose { perm })
    }

    fn reshape(shape: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::Reshape { shape })
    }

    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::BroadcastInDim { shape, dims })
    }

    fn extract_diag(axis_a: usize, axis_b: usize) -> Self {
        Self::new(SemiringOpKind::ExtractDiag { axis_a, axis_b })
    }
}
