use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{DType, Tensor};

use crate::exec::ExecProgram;
use crate::graph::lowering_view::GraphProgramLoweringView;
use crate::program::{FrozenProgram, ProgramBindings, SemanticProgram};

/// A backend-neutral compiled semantic graph with runtime-private execution staging.
///
/// Public consumers inspect only the immutable semantic program and its
/// process-local tensor bindings. Native execution staging remains owned by
/// `tenferro-runtime` and is removed in Phase 5.
#[derive(Clone)]
pub struct CompiledGraph {
    pub(crate) staging: ExecProgram,
    pub(crate) frozen: FrozenProgram,
    pub(crate) inputs: Vec<GraphProgramInput>,
}

impl CompiledGraph {
    pub(crate) fn new(
        frozen: FrozenProgram,
        staging: ExecProgram,
        inputs: Vec<GraphProgramInput>,
    ) -> Self {
        Self {
            staging,
            frozen,
            inputs,
        }
    }

    /// Borrow the immutable backend-neutral semantic program.
    pub fn program(&self) -> &SemanticProgram {
        &self.frozen.program
    }

    /// Borrow tensor defaults and large constants kept outside semantic structure.
    pub fn bindings(&self) -> &ProgramBindings {
        &self.frozen.bindings
    }

    /// Return the number of ordered semantic inputs.
    pub fn input_count(&self) -> usize {
        self.frozen.program.inputs().len()
    }

    /// Return the number of ordered semantic outputs.
    pub fn output_count(&self) -> usize {
        self.frozen.program.outputs().len()
    }

    /// Transitional alias for [`Self::program`], removed with legacy callers.
    pub fn semantic_program(&self) -> &SemanticProgram {
        self.program()
    }

    /// Transitional alias for [`Self::bindings`], removed with legacy callers.
    pub fn program_bindings(&self) -> &ProgramBindings {
        self.bindings()
    }

    /// Transitional concrete input descriptors for legacy executor callers.
    pub fn input_specs(&self) -> &[GraphProgramInput] {
        &self.inputs
    }

    /// Transitional runtime-owned lowering view, removed after caller migration.
    pub fn lowering_view(&self) -> GraphProgramLoweringView<'_> {
        GraphProgramLoweringView::new(&self.staging)
    }
}

impl std::fmt::Debug for CompiledGraph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledGraph")
            .field("inputs", &self.input_count())
            .field("outputs", &self.output_count())
            .field("bindings", &self.bindings().len())
            .field(
                "semantic_fingerprint",
                &self.program().semantic_fingerprint(),
            )
            .finish()
    }
}

/// A transitional concrete descriptor for one ordered compiled input.
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
