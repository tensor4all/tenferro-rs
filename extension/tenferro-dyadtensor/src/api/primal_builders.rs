use super::runtime::*;
use super::*;

/// Builder for primal einsum.
/// # Examples
///
/// ```text
/// // Construct `EinsumBuilder` via its corresponding operation constructor.
/// ```
pub struct EinsumBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    subscripts: &'a str,
    operands: &'a [&'a Tensor<T>],
    size_dict: Option<&'a HashMap<u32, usize>>,
}

impl<'a, T> EinsumBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    /// Sets optional size dictionary for output-only labels.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.size_dict(&size_dict);
    /// ```
    pub fn size_dict(mut self, size_dict: &'a HashMap<u32, usize>) -> Self {
        self.size_dict = Some(size_dict);
        self
    }

    /// Executes the operation with the default runtime.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("einsum", |ctx| {
            tf_einsum::einsum::<Standard<T>, CpuBackend>(
                ctx,
                self.subscripts,
                self.operands,
                self.size_dict,
            )
            .map_err(Error::from)
        })
    }
}

/// Creates a builder for primal einsum.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{einsum, set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let out = einsum("ij,jk->ik", &[&a, &b]).run().unwrap();
/// assert_eq!(out.dims(), &[2, 2]);
/// ```
pub fn einsum<'a, T>(subscripts: &'a str, operands: &'a [&'a Tensor<T>]) -> EinsumBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    EinsumBuilder {
        subscripts,
        operands,
        size_dict: None,
    }
}
