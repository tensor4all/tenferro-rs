//! Public concrete tensor einsum extension API.

use tenferro_tensor::{
    DType, Error, Result, Tensor, TensorBackend, TensorRead, TensorScalar, TypedTensor,
};

use crate::eager::{
    eager_einsum_exec, eager_einsum_exec_read, eager_einsum_read_subscripts,
    eager_einsum_subscripts, plan_subscripts,
};
use crate::{ContractionTree, EinsumSubscripts, Subscripts};

const TENSOR_EINSUM_OP: &str = "TensorEinsumExt::einsum";
const TENSOR_READ_EINSUM_OP: &str = "TensorReadEinsumExt::einsum_read";
const TYPED_TENSOR_EINSUM_OP: &str = "TypedTensorEinsumExt::einsum";
const PLAN_PREPARE_OP: &str = "ConcreteEinsumPlan::prepare";
const PLAN_EXECUTE_OP: &str = "ConcreteEinsumPlan::execute";

/// Backend-explicit einsum methods for dtype-erased concrete tensors.
///
/// Implementations are provided for slices and fixed-size arrays of
/// [`Tensor`] references, so both `inputs.as_slice().einsum(...)` and
/// `[&lhs, &rhs].einsum(...)` work.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TensorEinsumExt;
/// use tenferro_tensor::Tensor;
///
/// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let mut backend = CpuBackend::new();
///
/// let out = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorEinsumExt {
    /// Execute an einsum from string notation.
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor>;

    /// Execute an einsum from parsed integer-label subscripts.
    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor>;
}

impl TensorEinsumExt for [&Tensor] {
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        let subscripts = parse_subscripts(subscripts, TENSOR_EINSUM_OP)?;
        eager_einsum_subscripts(backend, self, &subscripts)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_subscripts(backend, self, &subscripts)
    }
}

impl<const N: usize> TensorEinsumExt for [&Tensor; N] {
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        self.as_slice().einsum(subscripts, backend)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        self.as_slice().einsum_subscripts(subscripts, backend)
    }
}

/// Backend-explicit einsum methods for typed concrete tensors.
///
/// The result keeps the same scalar type as the inputs. Mixed dtypes should use
/// [`TensorEinsumExt`] on dtype-erased [`Tensor`] values instead.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TypedTensorEinsumExt;
/// use tenferro_tensor::TypedTensor;
///
/// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
/// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], vec![1.0; 12]).unwrap();
/// let mut backend = CpuBackend::new();
///
/// let out = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TypedTensorEinsumExt<T: TensorScalar> {
    /// Execute an einsum from string notation.
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B)
        -> Result<TypedTensor<T>>;

    /// Execute an einsum from parsed integer-label subscripts.
    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>>;
}

impl<T: TensorScalar> TypedTensorEinsumExt<T> for [&TypedTensor<T>] {
    fn einsum<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        let subscripts = parse_subscripts(subscripts, TYPED_TENSOR_EINSUM_OP)?;
        typed_einsum_subscripts(backend, self, &subscripts, TYPED_TENSOR_EINSUM_OP)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        let subscripts = Subscripts::from(subscripts);
        typed_einsum_subscripts(backend, self, &subscripts, TYPED_TENSOR_EINSUM_OP)
    }
}

impl<T: TensorScalar, const N: usize> TypedTensorEinsumExt<T> for [&TypedTensor<T>; N] {
    fn einsum<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum(subscripts, backend)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_subscripts(subscripts, backend)
    }
}

/// Backend-explicit einsum methods for [`TensorRead`] inputs.
///
/// Use this surface when an input is a borrowed tensor view rather than an
/// owned compact [`Tensor`]. The `_read` suffix follows the repository-wide
/// convention for APIs that explicitly accept read-oriented borrowed inputs.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TensorReadEinsumExt;
/// use tenferro_tensor::{Tensor, TensorRead, TensorView};
///
/// let shape = [2, 3];
/// let data = [1.0_f64; 6];
/// let rhs = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
/// let inputs = [
///     TensorRead::from_view(TensorView::f64(&shape, &data)?),
///     TensorRead::from_tensor(&rhs),
/// ];
/// let mut backend = CpuBackend::new();
///
/// let out = inputs.einsum_read("ij,j->i", &mut backend)?;
/// assert_eq!(out.shape(), &[2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorReadEinsumExt {
    /// Execute an einsum from string notation over read-only tensor inputs.
    fn einsum_read<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor>;

    /// Execute an einsum from parsed integer-label subscripts over read-only
    /// tensor inputs.
    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor>;
}

impl<'a> TensorReadEinsumExt for [TensorRead<'a>] {
    fn einsum_read<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        let subscripts = parse_subscripts(subscripts, TENSOR_READ_EINSUM_OP)?;
        eager_einsum_read_subscripts(backend, self, &subscripts)
    }

    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_read_subscripts(backend, self, &subscripts)
    }
}

impl<'a, const N: usize> TensorReadEinsumExt for [TensorRead<'a>; N] {
    fn einsum_read<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        self.as_slice().einsum_read(subscripts, backend)
    }

    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        self.as_slice().einsum_read_subscripts(subscripts, backend)
    }
}

/// Prepared concrete einsum plan for repeated executions with fixed input
/// dtype and shape metadata.
///
/// Preparing a plan parses and optimizes the contraction tree once. Execution
/// validates the later inputs against the prepared dtype and shape contract,
/// then runs the stored tree without re-planning.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::ConcreteEinsumPlan;
/// use tenferro_tensor::Tensor;
///
/// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
///
/// let mut backend = CpuBackend::new();
/// let out = plan.execute([&lhs, &rhs], &mut backend)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct ConcreteEinsumPlan {
    tree: ContractionTree,
    inputs: Vec<ConcreteEinsumInputSpec>,
}

impl ConcreteEinsumPlan {
    /// Prepare a plan from dtype-erased concrete tensor inputs and string
    /// notation.
    pub fn prepare<'a, I>(inputs: I, subscripts: &str) -> Result<Self>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let subscripts = parse_subscripts(subscripts, PLAN_PREPARE_OP)?;
        Self::prepare_subscripts_internal(input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from dtype-erased concrete tensor inputs and parsed
    /// integer-label subscripts.
    pub fn prepare_subscripts<'a, I>(inputs: I, subscripts: &EinsumSubscripts) -> Result<Self>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from typed concrete tensor inputs and string notation.
    pub fn prepare_typed<'a, T, I>(inputs: I, subscripts: &str) -> Result<Self>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
    {
        let subscripts = parse_subscripts(subscripts, PLAN_PREPARE_OP)?;
        Self::prepare_subscripts_internal(typed_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from typed concrete tensor inputs and parsed integer-label
    /// subscripts.
    pub fn prepare_typed_subscripts<'a, T, I>(
        inputs: I,
        subscripts: &EinsumSubscripts,
    ) -> Result<Self>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(typed_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from read-only tensor inputs and string notation.
    pub fn prepare_read<'a, I>(inputs: I, subscripts: &str) -> Result<Self>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let subscripts = parse_subscripts(subscripts, PLAN_PREPARE_OP)?;
        Self::prepare_subscripts_internal(read_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from read-only tensor inputs and parsed integer-label
    /// subscripts.
    pub fn prepare_read_subscripts<'a, I>(inputs: I, subscripts: &EinsumSubscripts) -> Result<Self>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(read_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Execute this plan on dtype-erased concrete tensor inputs.
    pub fn execute<'a, I, B>(&self, inputs: I, backend: &mut B) -> Result<Tensor>
    where
        I: AsRef<[&'a Tensor]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        self.validate_inputs(&input_specs(inputs), PLAN_EXECUTE_OP)?;
        backend.with_backend_session(|exec| eager_einsum_exec(exec, inputs, &self.tree))
    }

    /// Execute this plan on typed concrete tensor inputs.
    pub fn execute_typed<'a, T, I, B>(&self, inputs: I, backend: &mut B) -> Result<TypedTensor<T>>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        self.validate_inputs(&typed_input_specs(inputs), PLAN_EXECUTE_OP)?;
        let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
        let result = backend
            .with_backend_session(|exec| eager_einsum_exec_read(exec, &reads, &self.tree))?;
        into_typed_result(result, PLAN_EXECUTE_OP)
    }

    /// Execute this plan on read-only tensor inputs.
    pub fn execute_read<'a, I, B>(&self, inputs: I, backend: &mut B) -> Result<Tensor>
    where
        I: AsRef<[TensorRead<'a>]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        self.validate_inputs(&read_input_specs(inputs), PLAN_EXECUTE_OP)?;
        backend.with_backend_session(|exec| eager_einsum_exec_read(exec, inputs, &self.tree))
    }

    fn prepare_subscripts_internal(
        inputs: Vec<ConcreteEinsumInputSpec>,
        subscripts: &Subscripts,
    ) -> Result<Self> {
        let shapes: Vec<&[usize]> = inputs.iter().map(|input| input.shape.as_slice()).collect();
        let tree = plan_subscripts(subscripts, &shapes)?;
        Ok(Self { tree, inputs })
    }

    fn validate_inputs(&self, actual: &[ConcreteEinsumInputSpec], op: &'static str) -> Result<()> {
        if actual.len() != self.inputs.len() {
            return Err(Error::InvalidConfig {
                op,
                message: format!(
                    "prepared einsum expects {} inputs, got {}",
                    self.inputs.len(),
                    actual.len()
                ),
            });
        }

        for (index, (expected, actual)) in self.inputs.iter().zip(actual.iter()).enumerate() {
            if expected.dtype != actual.dtype {
                return Err(Error::DTypeMismatch {
                    op,
                    lhs: expected.dtype,
                    rhs: actual.dtype,
                });
            }
            if expected.shape != actual.shape {
                return Err(Error::ShapeMismatch {
                    op,
                    lhs: expected.shape.clone(),
                    rhs: actual.shape.clone(),
                });
            }
            if expected.shape.len() != actual.shape.len() {
                return Err(Error::InvalidConfig {
                    op,
                    message: format!("input {index} rank changed during prepared einsum execution"),
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConcreteEinsumInputSpec {
    dtype: DType,
    shape: Vec<usize>,
}

fn parse_subscripts(subscripts: &str, op: &'static str) -> Result<Subscripts> {
    Subscripts::parse(subscripts).map_err(|err| Error::InvalidConfig {
        op,
        message: format!("invalid subscripts: {err}"),
    })
}

fn input_specs(inputs: &[&Tensor]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn typed_input_specs<T: TensorScalar>(inputs: &[&TypedTensor<T>]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: T::dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn read_input_specs(inputs: &[TensorRead<'_>]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn typed_einsum_subscripts<T: TensorScalar>(
    backend: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &Subscripts,
    op: &'static str,
) -> Result<TypedTensor<T>> {
    let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
    let result = eager_einsum_read_subscripts(backend, &reads, subscripts)?;
    into_typed_result(result, op)
}

fn into_typed_result<T: TensorScalar>(result: Tensor, op: &'static str) -> Result<TypedTensor<T>> {
    let actual = result.dtype();
    T::into_typed(result).map_err(|_| Error::DTypeMismatch {
        op,
        lhs: T::dtype(),
        rhs: actual,
    })
}
