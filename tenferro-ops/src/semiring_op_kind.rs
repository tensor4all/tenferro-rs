use crate::config::DotGeneralConfig;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SemiringOpKind {
    Add,
    Mul,
    DotGeneral(DotGeneralConfig),
    ReduceSum { axes: Vec<usize> },
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    ExtractDiag { axis_a: usize, axis_b: usize },
}
