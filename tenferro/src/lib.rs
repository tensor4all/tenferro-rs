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
//! use tenferro::einsum::einsum;
//! use tenferro::engine::Engine;
//! use tenferro::traced::TracedTensor;
//! use tenferro_tensor::cpu::CpuBackend;
//!
//! let engine = Engine::new(CpuBackend::default());
//! // ... build and execute traced computations
//! ```

pub mod buffer_pool;
pub mod compiler;
pub mod einsum;
pub mod engine;
pub mod error;
pub mod exec;
pub mod stablehlo;
pub mod traced;
