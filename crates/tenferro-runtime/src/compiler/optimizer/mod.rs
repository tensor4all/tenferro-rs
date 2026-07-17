use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::DType;

use crate::exec::ExecProgram;
use crate::Result;

use super::options::OptimizerConfig;

/// Run the standard execution-IR optimizer pipeline.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] with `ShapeMismatch`, `DTypeMismatch`,
/// `RankMismatch`, or `InvalidArgument` when shape/dtype metadata is
/// inconsistent with an optimizer pass, [`crate::Error::Unsupported`] when a
/// configured decomposition cannot support an operation, and
/// [`crate::Error::Internal`] when the execution IR violates an invariant such
/// as a missing slot metadata entry.
pub fn optimize_exec_program(
    program: &mut ExecProgram,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
    config: OptimizerConfig,
) -> Result<()> {
    super::conj_sinking(program, input_dtypes, input_shapes)?;
    super::dot_dimension_sorter(program);
    if config.algebraic_layout_simplifier {
        super::algebraic_layout_simplifier(program, input_shapes)?;
    }
    super::transpose_folding(program);
    super::dot_conj_folding(program)?;
    if config.dot_decomposer {
        super::dot_decomposer(program, input_shapes)?;
        if config.algebraic_layout_simplifier {
            super::algebraic_layout_simplifier(program, input_shapes)?;
        }
    }
    super::eliminate_dead_code(program);
    super::populate_last_use(program)?;
    Ok(())
}
