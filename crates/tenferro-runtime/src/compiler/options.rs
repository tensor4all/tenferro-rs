use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CompilerOptions {
    pub optimizer: OptimizerConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OptimizerConfig {
    pub algebraic_layout_simplifier: bool,
    pub layout_chain_transpose_folding: bool,
    pub dot_decomposer: bool,
}

impl OptimizerConfig {
    pub const VERSION: u64 = 1;

    pub fn fingerprint(self) -> u64 {
        let mut hasher = DefaultHasher::new();
        Self::VERSION.hash(&mut hasher);
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            algebraic_layout_simplifier: true,
            layout_chain_transpose_folding: true,
            dot_decomposer: false,
        }
    }
}
