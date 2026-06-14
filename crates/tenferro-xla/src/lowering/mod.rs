//! StableHLO lowering from tenferro graph lowering views.

mod emit;
mod program;
mod shape;
mod types;

use tenferro_runtime::GraphProgram;

use crate::{Result, StableHloModule};

pub(crate) fn lower_to_stablehlo(program: &GraphProgram) -> Result<StableHloModule> {
    self::program::lower_graph_program(program)
}
