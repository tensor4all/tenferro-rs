//! Graph compilation APIs.
//!
//! # Examples
//!
//! ```
//! use tenferro::{GraphCompiler, TracedTensor};
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
//! let y = &x + &x;
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! assert_eq!(program.output_count(), 1);
//! ```

pub(crate) mod cache;
mod compiler;
mod program;

pub use cache::GraphCompilerCacheStats;
pub use compiler::GraphCompiler;
pub use program::{GraphProgram, GraphProgramInput};
