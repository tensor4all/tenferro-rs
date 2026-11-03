//! Graph compilation APIs.
//!
//! # Examples
//!
//! ```
//! use tenferro_runtime::{TraceContext, GraphCompiler, TracedTensor};
//!
//! let trace = TraceContext::new();
//! let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let y = (&x + &x).unwrap();
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! assert_eq!(program.output_count(), 1);
//! ```

mod compiler;
mod program;

pub use compiler::GraphCompiler;
pub use program::CompiledGraph;
