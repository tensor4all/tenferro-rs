//! StableHLO lowering from tenferro semantic programs.

mod emit;
mod program;
mod types;

use tenferro_runtime::program::SemanticProgram;

use crate::{Result, StableHloModule};

pub(crate) fn lower_to_stablehlo(program: &SemanticProgram) -> Result<StableHloModule> {
    self::program::lower_semantic_program(program)
}
