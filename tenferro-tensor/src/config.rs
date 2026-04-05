/// DotGeneral dimension configuration.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::DotGeneralConfig;
///
/// let config = DotGeneralConfig {
///     lhs_contracting_dims: vec![1],
///     rhs_contracting_dims: vec![0],
///     lhs_batch_dims: vec![],
///     rhs_batch_dims: vec![],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: Vec<usize>,
    pub rhs_contracting_dims: Vec<usize>,
    pub lhs_batch_dims: Vec<usize>,
    pub rhs_batch_dims: Vec<usize>,
}

/// Comparison direction.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::CompareDir;
///
/// let dir = CompareDir::Eq;
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CompareDir {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Gather configuration placeholder.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GatherConfig {}

/// Scatter configuration placeholder.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ScatterConfig {}

/// Slice configuration.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::SliceConfig;
///
/// let config = SliceConfig {
///     starts: vec![0],
///     limits: vec![2],
///     strides: vec![1],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SliceConfig {
    pub starts: Vec<usize>,
    pub limits: Vec<usize>,
    pub strides: Vec<usize>,
}

/// Pad configuration placeholder.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PadConfig {}
