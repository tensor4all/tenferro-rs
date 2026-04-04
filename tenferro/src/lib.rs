#![allow(clippy::multiple_bound_locations)]

//! `tenferro`: traced tensor computation with StableHLO-style IR.
//!
//! This crate provides a tracing-based tensor computation framework where
//! operations are recorded into a StableHLO-compatible intermediate
//! representation, then compiled and executed on a backend (e.g., CPU).
//!
//! # Examples
//!
//! ```rust,ignore
//! use tenferro::cpu_backend::CpuBackend;
//! use tenferro::einsum::einsum;
//! use tenferro::engine::Engine;
//! use tenferro::traced::TracedTensor;
//!
//! let engine = Engine::new(CpuBackend::default());
//! // ... build and execute traced computations
//! ```

pub mod backend;
pub mod buffer_pool;
pub mod compiler;
pub mod cpu_backend;
pub mod einsum;
pub mod engine;
pub mod exec;
pub(crate) mod gemm;
pub mod indexing;
pub mod linalg;
pub mod reduction;
pub mod stablehlo;
pub mod standard;
pub mod structural;
pub mod traced;
