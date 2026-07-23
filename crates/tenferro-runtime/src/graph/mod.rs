//! Graph compilation APIs.
//!
//! # Examples
//!
//! ```
//! use tenferro_runtime::{GraphCompiler, TracedTensor};
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let y = (&x + &x).unwrap();
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! assert_eq!(program.output_count(), 1);
//! ```

pub(crate) mod cache;
mod compiler;
mod executor;
mod lowering_view;
mod program;

pub use cache::{GraphCompilerCacheStats, GraphExecutorCacheStats};
pub use compiler::GraphCompiler;
pub use executor::GraphExecutor;
pub use lowering_view::{
    GraphInstructionView, GraphOpView, GraphProgramLoweringShapeError, GraphProgramLoweringView,
};
pub use program::{CompiledGraph, GraphProgramInput};
