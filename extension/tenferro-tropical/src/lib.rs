//! Tropical semiring tensor operations for the tenferro workspace.
//!
//! This crate extends the tenferro algebra-parameterized architecture with
//! three tropical semirings:
//!
//! | Semiring | ⊕ (add) | ⊗ (mul) | Zero | One | Use case |
//! |----------|---------|---------|------|-----|----------|
//! | MaxPlus  | max     | +       | −∞   | 0   | Shortest path, optimal alignment |
//! | MinPlus  | min     | +       | +∞   | 0   | Shortest path (Dijkstra) |
//! | MaxMul   | max     | ×       | 0    | 1   | Viterbi, max-probability paths |
//!
//! # Architecture
//!
//! The crate provides:
//!
//! - **Scalar wrappers** ([`MaxPlus<T>`], [`MinPlus<T>`], [`MaxMul<T>`]):
//!   `#[repr(transparent)]` newtypes that redefine `Add`/`Mul` with tropical
//!   semantics. Satisfy the [`Scalar`](tenferro_algebra::Scalar) blanket impl.
//!
//! - **Algebra markers** ([`MaxPlusAlgebra<T>`], [`MinPlusAlgebra<T>`], [`MaxMulAlgebra<T>`]):
//!   Zero-sized types used as the algebra parameter `Alg` in
//!   [`TensorPrims<Alg>`](tenferro_prims::TensorPrims).
//!
//! - **[`TensorPrims`](tenferro_prims::TensorPrims) implementations**:
//!   `impl TensorPrims<MaxPlusAlgebra<f64>> for CpuBackend` etc. Orphan rule
//!   compatible because algebra markers are defined locally.
//!
//! - **[`ArgmaxTracker`]**: Records winner indices during tropical forward
//!   pass for use in automatic differentiation.
//!
//! # Examples
//!
//! ## Scalar arithmetic
//!
//! ```
//! use tenferro_tropical::MaxPlus;
//!
//! let a = MaxPlus(3.0_f64);
//! let b = MaxPlus(5.0_f64);
//! let c = a + b;   // MaxPlus(5.0) -- tropical add = max
//! assert_eq!(c, MaxPlus(5.0));
//! let d = a * b;   // MaxPlus(8.0) -- tropical mul = ordinary +
//! assert_eq!(d, MaxPlus(8.0));
//! ```
//!
//! ## Algebra dispatch
//!
//! ```
//! use tenferro_algebra::HasAlgebra;
//! use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
//!
//! // MaxPlus<f64> maps to MaxPlusAlgebra<f64>
//! fn check_f64<T: HasAlgebra<Algebra = MaxPlusAlgebra<f64>>>() {}
//! check_f64::<MaxPlus<f64>>();
//!
//! // MaxPlus<f32> maps to MaxPlusAlgebra<f32>
//! fn check_f32<T: HasAlgebra<Algebra = MaxPlusAlgebra<f32>>>() {}
//! check_f32::<MaxPlus<f32>>();
//! ```
//!
//! ## Plan-based tropical contraction
//!
//! ```ignore
//! use tenferro_algebra::Standard;
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_tropical::MaxPlusAlgebra;
//!
//! let mut ctx = CpuContext::new(1);
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], mem, col);
//! let mut c = Tensor::<f64>::zeros(&[3, 5], mem, col);
//! let desc = PrimDescriptor::BatchedGemm {
//!     batch_dims: vec![], m: 3, n: 5, k: 4,
//! };
//! // Under MaxPlusAlgebra<f64>, GEMM computes:
//! //   C[i,j] = max_k (A[i,k] + B[k,j])
//! let plan = <CpuBackend as TensorPrims<MaxPlusAlgebra<f64>>>::plan(
//!     &mut ctx,
//!     &desc,
//!     &[&[3, 4], &[4, 5], &[3, 5]],
//! )
//! .unwrap();
//! <CpuBackend as TensorPrims<MaxPlusAlgebra<f64>>>::execute(
//!     &mut ctx,
//!     &plan,
//!     1.0,
//!     &[&a, &b],
//!     0.0,
//!     &mut c,
//! )
//! .unwrap();
//! ```

pub mod ad;
pub mod algebra;
pub mod argmax;
pub mod prims;
pub mod scalar;

// Re-export primary types at crate root.
pub use algebra::{MaxMulAlgebra, MaxPlusAlgebra, MinPlusAlgebra};
pub use argmax::ArgmaxTracker;
pub use prims::TropicalPlan;
pub use scalar::{MaxMul, MaxPlus, MinPlus};
