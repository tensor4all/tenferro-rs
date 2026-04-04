use super::backend::SemiringCore;
use super::stablehlo::StableHloProgram;

pub struct Engine<B: SemiringCore> {
    pub(crate) backend: B,
    pub(crate) compile_cache: Vec<StableHloProgram>,
}

impl<B: SemiringCore> Engine<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            compile_cache: Vec::new(),
        }
    }
}
