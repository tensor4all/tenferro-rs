use tenferro_tensor::DotGeneralConfig;

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
    EmbedDiag { axis_a: usize, axis_b: usize },
}

impl SemiringOpKind {
    pub fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul | Self::DotGeneral(_) => 2,
            Self::ReduceSum { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::BroadcastInDim { .. }
            | Self::ExtractDiag { .. }
            | Self::EmbedDiag { .. } => 1,
        }
    }
}
