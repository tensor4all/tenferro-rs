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
//! - **Algebra markers** ([`MaxPlusAlgebra`], [`MinPlusAlgebra`], [`MaxMulAlgebra`]):
//!   Zero-sized types used as the algebra parameter `Alg` in
//!   [`TensorPrims<Alg>`](tenferro_prims::TensorPrims).
//!
//! - **[`TensorPrims`](tenferro_prims::TensorPrims) implementations**:
//!   `impl TensorPrims<MaxPlusAlgebra> for CpuBackend` etc. Orphan rule
//!   compatible because algebra markers are defined locally.
//!
//! - **[`ArgmaxTracker`]**: Records winner indices during tropical forward
//!   pass for use in automatic differentiation.
//!
//! # Examples
//!
//! ## Scalar arithmetic
//!
//! ```ignore
//! use tenferro_tropical::MaxPlus;
//!
//! let a = MaxPlus(3.0_f64);
//! let b = MaxPlus(5.0_f64);
//! let c = a + b;   // MaxPlus(5.0) — tropical add = max
//! let d = a * b;   // MaxPlus(8.0) — tropical mul = ordinary +
//! ```
//!
//! ## Algebra dispatch
//!
//! ```ignore
//! use tenferro_algebra::HasAlgebra;
//! use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
//!
//! // MaxPlus<f64> automatically maps to MaxPlusAlgebra
//! fn check<T: HasAlgebra<Algebra = MaxPlusAlgebra>>() {}
//! check::<MaxPlus<f64>>();
//! ```
//!
//! ## Plan-based tropical contraction
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor};
//! use tenferro_tropical::MaxPlusAlgebra;
//!
//! let desc = PrimDescriptor::BatchedGemm {
//!     batch_dims: vec![], m: 3, n: 5, k: 4,
//! };
//! // Under MaxPlusAlgebra, GEMM computes:
//! //   C[i,j] = max_k (A[i,k] + B[k,j])
//! let plan = <CpuBackend as TensorPrims<MaxPlusAlgebra>>::plan::<f64>(
//!     &desc, &[&[3, 4], &[4, 5], &[3, 5]],
//! ).unwrap();
//! ```

pub mod algebra;
pub mod argmax;
pub mod prims;
pub mod scalar;

// Re-export primary types at crate root.
pub use algebra::{MaxMulAlgebra, MaxPlusAlgebra, MinPlusAlgebra};
pub use argmax::ArgmaxTracker;
pub use prims::TropicalPlan;
pub use scalar::{MaxMul, MaxPlus, MinPlus};
