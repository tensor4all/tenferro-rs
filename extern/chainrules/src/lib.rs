//! AD engine: tape-based reverse-mode and dual-number forward-mode.
//!
//! This crate provides the AD execution engine, built on top of
//! [`chainrules_core`] traits. It is analogous to Zygote.jl in the Julia
//! ecosystem: a concrete AD engine that uses ChainRulesCore.jl interfaces.
//!
//! - Reverse-mode AD via [`Tape`], [`TrackedValue`], and [`Tape::pullback`]
//! - Forward-mode AD via [`DualValue`]
//! - Forward-over-reverse HVP via [`Tape::hvp`]
//!
//! Operation-specific AD rules (e.g., einsum rrule/frule) live in the crate
//! that defines the operation. See `tenferro-einsum` for einsum AD functions.
//!
//! The reverse-mode graph model is homogeneous: one [`Tape`] carries one value
//! type `V`. This supports both tensor graphs such as `Tape<Tensor<f64>>` and
//! downstream custom-type graphs such as `Tape<MyType>`, as long as
//! `MyType: Differentiable`.
//!
//! For tensor-valued APIs, scalar semantics follow PyTorch conventions:
//! scalar tensors are rank-0 (`shape=[]`), not shape `[1]`. Implicit reverse
//! seed creation remains based on `Differentiable::num_elements() == 1`.
//!
//! # Examples
//!
//! Reverse-mode usage (with operation-specific AD functions from other crates):
//!
//! ```ignore
//! use chainrules::{Tape, TrackedValue};
//! use std::sync::{Arc, Mutex};
//! use tenferro_algebra::Standard;
//! use tenferro_einsum::tracked_einsum;
//! use tenferro_prims::{CpuBackend, CpuContext};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let tape = Tape::<Tensor<f64>>::new();
//! let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
//! let a = tape.leaf(Tensor::ones(
//!     &[2, 3],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ));
//! let b = tape.leaf(Tensor::ones(
//!     &[3, 4],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ));
//! let c =
//!     tracked_einsum::<Standard<f64>, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
//! let loss =
//!     tracked_einsum::<Standard<f64>, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();
//! let grads = tape.pullback(&loss).unwrap();
//! let _ga = grads.get(a.node_id().unwrap()).unwrap();
//! ```
//!
//! Reverse-mode with a downstream custom type:
//!
//! ```ignore
//! use chainrules::{Differentiable, Tape};
//!
//! #[derive(Clone, Copy, Debug, PartialEq)]
//! struct MyScalar(f64);
//!
//! impl Differentiable for MyScalar {
//!     type Tangent = Self;
//!
//!     fn zero_tangent(&self) -> Self::Tangent { Self(0.0) }
//!     fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
//!         Self(a.0 + b.0)
//!     }
//!     fn num_elements(&self) -> usize { 1 }
//!     fn seed_cotangent(&self) -> Self::Tangent { Self(1.0) }
//! }
//!
//! let tape = Tape::<MyScalar>::new();
//! let x = tape.leaf(MyScalar(2.0));
//! let grads = tape.pullback(&x).unwrap();
//! assert_eq!(grads.get(x.node_id().unwrap()).unwrap().0, 1.0);
//! ```
//!
//! Forward-mode usage:
//!
//! ```ignore
//! use chainrules::DualValue;
//! use tenferro_algebra::Standard;
//! use tenferro_einsum::dual_einsum;
//! use tenferro_prims::{CpuBackend, CpuContext};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(1);
//! let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
//! let da = Tensor::<f64>::ones(&[2, 2], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//! let b = Tensor::<f64>::ones(&[2, 2], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//!
//! let a_dual = DualValue::with_tangent(a, da).unwrap();
//! let b_dual = DualValue::new(b);
//! let c_dual =
//!     dual_einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a_dual, &b_dual])
//!         .unwrap();
//! let _jvp = c_dual.tangent();
//! ```
//!
//! Forward-over-reverse HVP (Hessian-vector product):
//!
//! ```ignore
//! use chainrules::Tape;
//! use std::sync::{Arc, Mutex};
//! use tenferro_algebra::Standard;
//! use tenferro_einsum::tracked_einsum;
//! use tenferro_prims::{CpuBackend, CpuContext};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let tape = Tape::<Tensor<f64>>::new();
//! let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
//! let x = tape.leaf_with_tangent(
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),  // direction v
//! ).unwrap();
//! let loss =
//!     tracked_einsum::<Standard<f64>, CpuBackend>(ctx, "i,i->", &[&x, &x]).unwrap();  // f(x) = x·x
//! let result = tape.hvp(&loss).unwrap();
//! let _grad = result.gradients;  // ∇f(x) = 2x
//! let _hv = result.hvp;          // H·v = 2v
//! ```

// Re-export all core traits so downstream can depend on just `chainrules`.
pub use chainrules_core::*;

mod engine;
mod ops;

pub use engine::{
    AutogradGraph, BackwardOptions, DualValue, Gradients, HvpResult, PullbackPlan, Tape,
    TrackedValue, Variable,
};

/// Monomorphic AD operation helpers for [`Variable`].
pub use ops::autograd;

/// Test-only graph builders used by API contract tests.
pub use ops::test_support;
