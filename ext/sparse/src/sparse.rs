use tenferro_runtime::{Error as RuntimeError, Result as RuntimeResult, TracedTensor};
use tenferro_tensor::{DType, Error, Result, Tensor};

use crate::extension::{apply_sparse_matmul, SparseMatmulPlan};

const OP: &str = "tenferro-ext-sparse";

/// Concrete COO sparse tensor used by the sparse extension tutorial.
///
/// The coordinate tensor is a dense `I64` tenferro tensor with shape
/// `[2, nnz]`. The value tensor is a dense `F64` tensor with shape `[nnz]`.
///
/// # Examples
///
/// ```
/// use tenferro_ext_sparse::SparseCooTensor;
/// use tenferro_tensor::Tensor;
///
/// let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
/// let values = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
/// let sparse = SparseCooTensor::from_parts(vec![1, 1], coords, values).unwrap();
/// assert_eq!(sparse.shape(), &[1, 1]);
/// ```
#[derive(Clone, Debug)]
pub struct SparseCooTensor {
    shape: Vec<usize>,
    coordinates: Tensor,
    values: Tensor,
    entries: Vec<[usize; 2]>,
}

impl SparseCooTensor {
    /// Build a concrete sparse COO tensor from shape, coordinates, and values.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape is not rank-2, coordinates are not `I64`
    /// with shape `[2, nnz]`, coordinates are out of bounds, or values are not
    /// `F64` with shape `[nnz]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_sparse::SparseCooTensor;
    /// use tenferro_tensor::Tensor;
    ///
    /// let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?;
    /// let values = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// let sparse = SparseCooTensor::from_parts(vec![1, 1], coords, values)?;
    /// assert_eq!(sparse.values().as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_parts(shape: Vec<usize>, coordinates: Tensor, values: Tensor) -> Result<Self> {
        let entries = validate_coordinates(&shape, &coordinates)?;
        validate_value_tensor(&values, entries.len())?;
        Ok(Self {
            shape,
            coordinates,
            values,
            entries,
        })
    }

    /// Return the sparse logical shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_ext_sparse::SparseCooTensor;
    /// # use tenferro_tensor::Tensor;
    /// # let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
    /// # let values = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// # let sparse = SparseCooTensor::from_parts(vec![1, 1], coords, values).unwrap();
    /// assert_eq!(sparse.shape(), &[1, 1]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the dense COO coordinate tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_ext_sparse::SparseCooTensor;
    /// # use tenferro_tensor::Tensor;
    /// # let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
    /// # let values = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// # let sparse = SparseCooTensor::from_parts(vec![1, 1], coords, values).unwrap();
    /// assert_eq!(sparse.coordinates().shape(), &[2, 1]);
    /// ```
    pub fn coordinates(&self) -> &Tensor {
        &self.coordinates
    }

    /// Borrow the dense nonzero value tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_ext_sparse::SparseCooTensor;
    /// # use tenferro_tensor::Tensor;
    /// # let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
    /// # let values = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// # let sparse = SparseCooTensor::from_parts(vec![1, 1], coords, values).unwrap();
    /// assert_eq!(sparse.values().shape(), &[1]);
    /// ```
    pub fn values(&self) -> &Tensor {
        &self.values
    }

    pub(crate) fn entries(&self) -> &[[usize; 2]] {
        &self.entries
    }
}

/// Traced COO sparse tensor used by the sparse extension tutorial.
///
/// Coordinates and shape are fixed sparse structure. Values are the traced
/// differentiable data that flow through extension ops.
///
/// # Examples
///
/// ```
/// use tenferro_ext_sparse::SparseCooTracedTensor;
/// use tenferro_runtime::TracedTensor;
/// use tenferro_tensor::Tensor;
///
/// let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
/// let values = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
/// let sparse = SparseCooTracedTensor::from_parts(vec![1, 1], coords, values).unwrap();
/// assert_eq!(sparse.shape(), &[1, 1]);
/// ```
#[derive(Clone, Debug)]
pub struct SparseCooTracedTensor {
    shape: Vec<usize>,
    coordinates: Tensor,
    values: TracedTensor,
    entries: Vec<[usize; 2]>,
}

impl SparseCooTracedTensor {
    /// Build a traced sparse COO tensor from fixed coordinates and traced values.
    ///
    /// # Errors
    ///
    /// Returns an error if sparse metadata is invalid, values are not `F64`, or
    /// the traced value shape is not concretely `[nnz]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_sparse::SparseCooTracedTensor;
    /// use tenferro_runtime::TracedTensor;
    /// use tenferro_tensor::Tensor;
    ///
    /// let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?;
    /// let values = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// let sparse = SparseCooTracedTensor::from_parts(vec![1, 1], coords, values)?;
    /// assert_eq!(sparse.values().rank, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_parts(
        shape: Vec<usize>,
        coordinates: Tensor,
        values: TracedTensor,
    ) -> RuntimeResult<Self> {
        let entries = validate_coordinates(&shape, &coordinates).map_err(RuntimeError::from)?;
        validate_traced_values(&values, entries.len())?;
        Ok(Self {
            shape,
            coordinates,
            values,
            entries,
        })
    }

    /// Return the sparse logical shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_ext_sparse::SparseCooTracedTensor;
    /// # use tenferro_runtime::TracedTensor;
    /// # use tenferro_tensor::Tensor;
    /// # let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
    /// # let values = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// # let sparse = SparseCooTracedTensor::from_parts(vec![1, 1], coords, values).unwrap();
    /// assert_eq!(sparse.shape(), &[1, 1]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the fixed dense COO coordinate tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_ext_sparse::SparseCooTracedTensor;
    /// # use tenferro_runtime::TracedTensor;
    /// # use tenferro_tensor::Tensor;
    /// # let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
    /// # let values = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// # let sparse = SparseCooTracedTensor::from_parts(vec![1, 1], coords, values).unwrap();
    /// assert_eq!(sparse.coordinates().shape(), &[2, 1]);
    /// ```
    pub fn coordinates(&self) -> &Tensor {
        &self.coordinates
    }

    /// Borrow the traced nonzero value tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_ext_sparse::SparseCooTracedTensor;
    /// # use tenferro_runtime::TracedTensor;
    /// # use tenferro_tensor::Tensor;
    /// # let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap();
    /// # let values = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// # let sparse = SparseCooTracedTensor::from_parts(vec![1, 1], coords, values).unwrap();
    /// assert_eq!(sparse.values().rank, 1);
    /// ```
    pub fn values(&self) -> &TracedTensor {
        &self.values
    }

    pub(crate) fn entries(&self) -> &[[usize; 2]] {
        &self.entries
    }
}

/// Multiply two concrete COO sparse matrices.
///
/// # Errors
///
/// Returns an error when the sparse shapes are not matrix-multiplication
/// compatible or when value tensors have unsupported metadata.
///
/// # Examples
///
/// ```
/// use tenferro_ext_sparse::{sparse_matmul_eager, SparseCooTensor};
/// use tenferro_tensor::Tensor;
///
/// let a = SparseCooTensor::from_parts(
///     vec![1, 1],
///     Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?,
///     Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?,
/// )?;
/// let b = SparseCooTensor::from_parts(
///     vec![1, 1],
///     Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?,
///     Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
/// )?;
/// let out = sparse_matmul_eager(&a, &b)?;
/// assert_eq!(out.values().as_slice::<f64>()?, &[6.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn sparse_matmul_eager(
    lhs: &SparseCooTensor,
    rhs: &SparseCooTensor,
) -> Result<SparseCooTensor> {
    let plan = SparseMatmulPlan::new(lhs.shape(), lhs.entries(), rhs.shape(), rhs.entries())?;
    let values = apply_sparse_matmul(&plan, lhs.values(), rhs.values())?;
    let coordinates = coordinates_tensor(&plan.output_coords)?;
    SparseCooTensor::from_parts(plan.output_shape.clone(), coordinates, values)
}

pub(crate) fn validate_coordinates(
    shape: &[usize],
    coordinates: &Tensor,
) -> Result<Vec<[usize; 2]>> {
    if shape.len() != 2 {
        return Err(invalid(format!(
            "expected rank-2 sparse shape, got {shape:?}"
        )));
    }
    if coordinates.dtype() != DType::I64 {
        return Err(invalid(format!(
            "coordinate tensor must have dtype I64, got {:?}",
            coordinates.dtype()
        )));
    }
    let coord_shape = coordinates.shape();
    if coord_shape.len() != 2 || coord_shape[0] != 2 {
        return Err(invalid(format!(
            "coordinate tensor must have shape [2, nnz], got {coord_shape:?}"
        )));
    }
    let mut entries = Vec::with_capacity(coord_shape[1]);
    for pair in coordinates.as_slice::<i64>()?.chunks_exact(2) {
        let row =
            usize::try_from(pair[0]).map_err(|_| invalid("negative sparse row coordinate"))?;
        let col =
            usize::try_from(pair[1]).map_err(|_| invalid("negative sparse column coordinate"))?;
        if row >= shape[0] || col >= shape[1] {
            return Err(invalid(format!(
                "coordinate [{row}, {col}] is out of bounds for shape {shape:?}"
            )));
        }
        entries.push([row, col]);
    }
    Ok(entries)
}

pub(crate) fn validate_value_tensor(values: &Tensor, nnz: usize) -> Result<()> {
    if values.dtype() != DType::F64 {
        return Err(invalid(format!(
            "sparse tutorial supports F64 values, got {:?}",
            values.dtype()
        )));
    }
    if values.shape() != [nnz] {
        return Err(invalid(format!(
            "value tensor must have shape [{nnz}], got {:?}",
            values.shape()
        )));
    }
    Ok(())
}

pub(crate) fn validate_traced_values(values: &TracedTensor, nnz: usize) -> RuntimeResult<()> {
    if values.dtype != DType::F64 {
        return Err(RuntimeError::TensorRuntime(invalid(format!(
            "sparse tutorial supports F64 values, got {:?}",
            values.dtype
        ))));
    }
    if values.rank != 1 {
        return Err(RuntimeError::TensorRuntime(invalid(format!(
            "value tensor must have rank 1, got rank {}",
            values.rank
        ))));
    }
    if let Some(shape) = values.try_concrete_shape() {
        if shape != [nnz] {
            return Err(RuntimeError::TensorRuntime(invalid(format!(
                "value tensor must have shape [{nnz}], got {shape:?}"
            ))));
        }
    }
    Ok(())
}

pub(crate) fn coordinates_tensor(entries: &[[usize; 2]]) -> Result<Tensor> {
    let mut data = Vec::with_capacity(entries.len() * 2);
    for &[row, col] in entries {
        data.push(i64::try_from(row).map_err(|_| invalid("row coordinate exceeds i64"))?);
        data.push(i64::try_from(col).map_err(|_| invalid("column coordinate exceeds i64"))?);
    }
    Tensor::from_vec_col_major(vec![2, entries.len()], data)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig {
        op: OP,
        message: message.into(),
    }
}
