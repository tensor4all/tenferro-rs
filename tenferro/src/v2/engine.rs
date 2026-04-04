use super::backend::SemiringCore;
use super::exec::ExecProgram;
use super::stablehlo::StableHloProgram;

// --- Standard algebra engine (TracedTensor uses this) ---

pub struct Engine {
    pub(crate) compile_cache: Vec<StableHloProgram>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            compile_cache: Vec::new(),
        }
    }
}

// --- Custom algebra evaluation (no Engine wrapper needed) ---
// Users construct the pipeline explicitly:
//   build_einsum_fragment → resolve → materialize → compile
//     → lower_semiring_to_stablehlo → compile_to_exec → eval_custom

pub fn eval_custom<B: SemiringCore>(
    _backend: &mut B,
    _program: &ExecProgram,
    _inputs: Vec<B::Operand>,
) -> Vec<B::Operand> {
    todo!()
}
