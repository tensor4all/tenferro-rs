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
///     lhs_rank: 2,
///     rhs_rank: 2,
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: Vec<usize>,
    pub rhs_contracting_dims: Vec<usize>,
    pub lhs_batch_dims: Vec<usize>,
    pub rhs_batch_dims: Vec<usize>,
    pub lhs_rank: usize,
    pub rhs_rank: usize,
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

/// StableHLO gather dimension configuration.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::GatherConfig;
///
/// let config = GatherConfig {
///     offset_dims: vec![],
///     collapsed_slice_dims: vec![0],
///     start_index_map: vec![0],
///     index_vector_dim: 1,
///     slice_sizes: vec![1],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GatherConfig {
    pub offset_dims: Vec<usize>,
    pub collapsed_slice_dims: Vec<usize>,
    pub start_index_map: Vec<usize>,
    pub index_vector_dim: usize,
    pub slice_sizes: Vec<usize>,
}

/// StableHLO scatter dimension configuration.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::ScatterConfig;
///
/// let config = ScatterConfig {
///     update_window_dims: vec![],
///     inserted_window_dims: vec![0],
///     scatter_dims_to_operand_dims: vec![0],
///     index_vector_dim: 1,
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ScatterConfig {
    pub update_window_dims: Vec<usize>,
    pub inserted_window_dims: Vec<usize>,
    pub scatter_dims_to_operand_dims: Vec<usize>,
    pub index_vector_dim: usize,
}

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

/// StableHLO pad configuration.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::PadConfig;
///
/// let config = PadConfig {
///     edge_padding_low: vec![1, 1],
///     edge_padding_high: vec![1, 1],
///     interior_padding: vec![0, 0],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PadConfig {
    pub edge_padding_low: Vec<i64>,
    pub edge_padding_high: Vec<i64>,
    pub interior_padding: Vec<i64>,
}
