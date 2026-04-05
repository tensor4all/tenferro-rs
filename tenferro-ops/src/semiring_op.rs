use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use computegraph::GraphOp;
use tenferro_algebra::Algebra;
use tenferro_tensor::{DotGeneralConfig, TypedTensor};

use crate::semiring_op_kind::SemiringOpKind;
use crate::semiring_ops::SemiringOps;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SemiringInputKey {
    pub id: u64,
}

pub struct SemiringOp<Alg: Algebra> {
    pub kind: SemiringOpKind,
    _marker: PhantomData<Alg>,
}

impl<Alg: Algebra> SemiringOp<Alg> {
    pub fn new(kind: SemiringOpKind) -> Self {
        Self {
            kind,
            _marker: PhantomData,
        }
    }
}

impl<Alg: Algebra> Clone for SemiringOp<Alg> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Alg: Algebra> fmt::Debug for SemiringOp<Alg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemiringOp")
            .field("kind", &self.kind)
            .finish()
    }
}

impl<Alg: Algebra> Hash for SemiringOp<Alg> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl<Alg: Algebra> PartialEq for SemiringOp<Alg> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl<Alg: Algebra> Eq for SemiringOp<Alg> {}

impl<Alg> GraphOp for SemiringOp<Alg>
where
    Alg: Algebra + Send + Sync + 'static,
{
    type Operand = TypedTensor<Alg::Scalar>;
    type Context = ();
    type InputKey = SemiringInputKey;

    fn n_inputs(&self) -> usize {
        self.kind.n_inputs()
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl<Alg> SemiringOps for SemiringOp<Alg>
where
    Alg: Algebra + Send + Sync + 'static,
{
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

    fn embed_diag(axis_a: usize, axis_b: usize) -> Self {
        Self::new(SemiringOpKind::EmbedDiag { axis_a, axis_b })
    }
}
