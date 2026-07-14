use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{DType, Tensor};

use crate::exec::ExecProgram;
use crate::graph::lowering_view::GraphProgramLoweringView;

/// A compiled traced graph, independent of any execution backend.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let y = (&x + &x).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// assert_eq!(program.input_count(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct GraphProgram {
    pub(crate) exec: ExecProgram,
    pub(crate) inputs: Vec<GraphProgramInput>,
}

impl GraphProgram {
    pub(crate) fn new(exec: ExecProgram, inputs: Vec<GraphProgramInput>) -> Self {
        Self { exec, inputs }
    }

    /// Return the number of graph inputs expected by this program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert_eq!(program.input_count(), 1);
    /// ```
    #[inline(never)]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Return the number of graph outputs produced by this program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert_eq!(program.output_count(), 1);
    /// ```
    #[inline(never)]
    pub fn output_count(&self) -> usize {
        self.exec.output_slots.len()
    }

    /// Return the ordered input specs expected by this program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[4])])
    ///     .unwrap();
    /// assert_eq!(program.input_specs()[0].shape(), &[4]);
    /// ```
    #[inline(never)]
    pub fn input_specs(&self) -> &[GraphProgramInput] {
        &self.inputs
    }

    /// Return a read-only lowering view for peer executor integrations.
    ///
    /// The view exposes only immutable, lowering-oriented program metadata.
    /// Native execution and mutation remain owned by [`GraphExecutor`](super::GraphExecutor).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert_eq!(program.lowering_view().output_slots().len(), 1);
    /// ```
    #[inline(never)]
    pub fn lowering_view(&self) -> GraphProgramLoweringView<'_> {
        GraphProgramLoweringView::new(&self.exec)
    }
}

/// A single ordered input required by a [`GraphProgram`].
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let y = x.neg().unwrap();
/// let program = compiler.compile(&y).unwrap();
/// let input = &program.input_specs()[0];
/// assert_eq!(input.shape(), &[2]);
/// ```
#[derive(Clone, Debug)]
pub struct GraphProgramInput {
    pub(crate) key: TensorInputKey,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    // Preserved for symbolic-shape diagnostics and future graph-input metadata
    // without exposing `DimExpr` through the stable input-spec accessor.
    #[allow(dead_code)]
    pub(crate) dim_expr_shape: Vec<DimExpr>,
    pub(crate) default_tensor: Option<Arc<Tensor>>,
}

impl GraphProgramInput {
    pub(crate) fn new(
        key: TensorInputKey,
        dtype: DType,
        shape: Vec<usize>,
        dim_expr_shape: Vec<DimExpr>,
        default_tensor: Option<Arc<Tensor>>,
    ) -> Self {
        Self {
            key,
            dtype,
            shape,
            dim_expr_shape,
            default_tensor,
        }
    }

    /// Return the dtype expected for this input.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&x, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// assert_eq!(program.input_specs()[0].dtype(), DType::F64);
    /// ```
    #[inline(never)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the concrete shape expected for this input.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x).unwrap();
    /// assert_eq!(program.input_specs()[0].shape(), &[2]);
    /// ```
    #[inline(never)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
