use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{DType, Tensor};

use crate::exec::ExecProgram;
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

#[derive(Clone, Debug)]
pub(crate) struct GraphProgramInput {
    pub(crate) key: TensorInputKey,
    pub(crate) dtype: DType,
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
        dim_expr_shape: Vec<DimExpr>,
        default_tensor: Option<Arc<Tensor>>,
    ) -> Self {
        Self {
            key,
            dtype,
            dim_expr_shape,
            default_tensor,
        }
    }
}
