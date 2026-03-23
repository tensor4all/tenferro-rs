use tenferro_algebra::{Algebra, Standard};
use tenferro_device::{Error, Generator, Result};
use tenferro_tensor::Tensor;

use crate::{CudaBackend, CudaContext, RocmBackend, RocmContext};

/// Random-number generation descriptors for dense eager tensor construction.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::RngPrimsDescriptor;
///
/// let desc = RngPrimsDescriptor::Uniform;
/// assert!(matches!(desc, RngPrimsDescriptor::Uniform));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RngPrimsDescriptor {
    /// Fill the output tensor with samples from the half-open interval `[0, 1)`.
    Uniform,
    /// Fill the output tensor with standard normal samples.
    Normal,
    /// Fill the output tensor with integer samples from the half-open interval `[low, high)`.
    Integer {
        /// Inclusive lower bound.
        low: i32,
        /// Exclusive upper bound.
        high: i32,
    },
}

/// Tensor RNG execution family.
///
/// The family is plan/execute based like the rest of `tenferro-prims`, but the
/// plan is intentionally small: it only records the descriptor and the output
/// shape so the execution side can validate the destination tensor before
/// writing into it.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_device::Generator;
/// use tenferro_prims::{CpuBackend, CpuContext, RngPrimsDescriptor, TensorRngPrims};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let mut generator = Generator::cpu(1234);
/// let desc = RngPrimsDescriptor::Uniform;
/// let mut output = Tensor::<f64>::zeros(
///     &[8],
///     tenferro_device::LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let plan = <CpuBackend as TensorRngPrims<Standard<f64>>>::plan(
///     &mut ctx,
///     &desc,
///     &[output.dims()],
/// ).unwrap();
/// <CpuBackend as TensorRngPrims<Standard<f64>>>::execute(
///     &mut ctx,
///     &plan,
///     &mut generator,
///     &mut output,
/// ).unwrap();
/// ```
pub trait TensorRngPrims<Alg: Algebra> {
    /// Backend-specific execution plan.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan a tensor RNG operation for the given output shape.
    fn plan(
        ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned RNG operation.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        generator: &mut Generator,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_rng_support(desc: RngPrimsDescriptor) -> bool;
}

impl TensorRngPrims<Standard<f64>> for CudaBackend {
    type Plan = (RngPrimsDescriptor, Vec<usize>);
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "RNG descriptor {desc:?} is not implemented on CudaBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _generator: &mut Generator,
        _output: &mut Tensor<f64>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "RNG execution is not implemented on CudaBackend in phase 1".into(),
        ))
    }

    fn has_rng_support(_desc: RngPrimsDescriptor) -> bool {
        false
    }
}

impl TensorRngPrims<Standard<i32>> for CudaBackend {
    type Plan = (RngPrimsDescriptor, Vec<usize>);
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "RNG descriptor {desc:?} is not implemented on CudaBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _generator: &mut Generator,
        _output: &mut Tensor<i32>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "RNG execution is not implemented on CudaBackend in phase 1".into(),
        ))
    }

    fn has_rng_support(_desc: RngPrimsDescriptor) -> bool {
        false
    }
}

impl TensorRngPrims<Standard<f64>> for RocmBackend {
    type Plan = (RngPrimsDescriptor, Vec<usize>);
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "RNG descriptor {desc:?} is not implemented on RocmBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _generator: &mut Generator,
        _output: &mut Tensor<f64>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "RNG execution is not implemented on RocmBackend in phase 1".into(),
        ))
    }

    fn has_rng_support(_desc: RngPrimsDescriptor) -> bool {
        false
    }
}

impl TensorRngPrims<Standard<i32>> for RocmBackend {
    type Plan = (RngPrimsDescriptor, Vec<usize>);
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "RNG descriptor {desc:?} is not implemented on RocmBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _generator: &mut Generator,
        _output: &mut Tensor<i32>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "RNG execution is not implemented on RocmBackend in phase 1".into(),
        ))
    }

    fn has_rng_support(_desc: RngPrimsDescriptor) -> bool {
        false
    }
}
