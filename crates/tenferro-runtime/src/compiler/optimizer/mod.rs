use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::DType;

use crate::exec::ExecProgram;

use super::options::OptimizerConfig;

/// Run the standard execution-IR optimizer pipeline.
pub fn optimize_exec_program(
    program: &mut ExecProgram,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
    config: OptimizerConfig,
) {
    super::conj_sinking(program, input_dtypes, input_shapes);
    super::dot_dimension_sorter(program);
    if config.algebraic_layout_simplifier {
        super::algebraic_layout_simplifier(program);
    }
    super::transpose_folding(program);
    if config.layout_chain_transpose_folding {
        super::layout_chain_transpose_folding(program);
    }
    super::dot_conj_folding(program);
    if config.dot_decomposer {
        super::dot_decomposer(program, input_shapes);
        if config.algebraic_layout_simplifier {
            super::algebraic_layout_simplifier(program);
        }
    }
    super::eliminate_dead_code(program);
    super::populate_last_use(program);
}
