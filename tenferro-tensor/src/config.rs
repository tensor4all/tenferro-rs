/// DotGeneral dimension configuration.
///
/// The output shape is `[lhs_free..., rhs_free..., batch...]` (col-major
/// batch-trailing convention). Batch dims have the largest stride so that
/// each batch slice occupies a contiguous block of memory.
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

impl DotGeneralConfig {
    /// Validate that `lhs_rank` and `rhs_rank` match the actual tensor ranks.
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
    /// config.validate_ranks(2, 2).unwrap();
    /// ```
    pub fn validate_ranks(
        &self,
        actual_lhs_rank: usize,
        actual_rhs_rank: usize,
    ) -> Result<(), String> {
        if self.lhs_rank != actual_lhs_rank {
            return Err(format!(
                "DotGeneralConfig.lhs_rank ({}) does not match actual lhs tensor rank ({})",
                self.lhs_rank, actual_lhs_rank
            ));
        }
        if self.rhs_rank != actual_rhs_rank {
            return Err(format!(
                "DotGeneralConfig.rhs_rank ({}) does not match actual rhs tensor rank ({})",
                self.rhs_rank, actual_rhs_rank
            ));
        }
        Ok(())
    }

    /// Validate that all dimension indices are within range for the stored ranks
    /// and that no axis appears in multiple roles.
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
    /// config.validate_dims().unwrap();
    /// ```
    fn check_no_duplicates(dims: &[usize], label: &str) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for &d in dims {
            if !seen.insert(d) {
                return Err(format!("{} contains duplicate dim {}", label, d));
            }
        }
        Ok(())
    }

    pub fn validate_dims(&self) -> Result<(), String> {
        for &d in &self.lhs_contracting_dims {
            if d >= self.lhs_rank {
                return Err(format!(
                    "lhs_contracting_dim {} out of bounds for lhs_rank {}",
                    d, self.lhs_rank
                ));
            }
        }
        for &d in &self.rhs_contracting_dims {
            if d >= self.rhs_rank {
                return Err(format!(
                    "rhs_contracting_dim {} out of bounds for rhs_rank {}",
                    d, self.rhs_rank
                ));
            }
        }
        for &d in &self.lhs_batch_dims {
            if d >= self.lhs_rank {
                return Err(format!(
                    "lhs_batch_dim {} out of bounds for lhs_rank {}",
                    d, self.lhs_rank
                ));
            }
        }
        for &d in &self.rhs_batch_dims {
            if d >= self.rhs_rank {
                return Err(format!(
                    "rhs_batch_dim {} out of bounds for rhs_rank {}",
                    d, self.rhs_rank
                ));
            }
        }
        Self::check_no_duplicates(&self.lhs_contracting_dims, "lhs_contracting_dims")?;
        Self::check_no_duplicates(&self.rhs_contracting_dims, "rhs_contracting_dims")?;
        Self::check_no_duplicates(&self.lhs_batch_dims, "lhs_batch_dims")?;
        Self::check_no_duplicates(&self.rhs_batch_dims, "rhs_batch_dims")?;
        for &d in &self.lhs_contracting_dims {
            if self.lhs_batch_dims.contains(&d) {
                return Err(format!(
                    "lhs dim {} appears in both contracting and batch dims",
                    d
                ));
            }
        }
        for &d in &self.rhs_contracting_dims {
            if self.rhs_batch_dims.contains(&d) {
                return Err(format!(
                    "rhs dim {} appears in both contracting and batch dims",
                    d
                ));
            }
        }
        if self.lhs_contracting_dims.len() != self.rhs_contracting_dims.len() {
            return Err(format!(
                "lhs/rhs contracting dim counts differ ({} vs {})",
                self.lhs_contracting_dims.len(),
                self.rhs_contracting_dims.len()
            ));
        }
        if self.lhs_batch_dims.len() != self.rhs_batch_dims.len() {
            return Err(format!(
                "lhs/rhs batch dim counts differ ({} vs {})",
                self.lhs_batch_dims.len(),
                self.rhs_batch_dims.len()
            ));
        }
        Ok(())
    }
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
