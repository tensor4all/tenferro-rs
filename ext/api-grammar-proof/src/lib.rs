//! Vertical prototypes for the call grammar proposed in tenferro-rs#2552.
//!
//! Not published. Each module is one prototype from inventory v0.2, built as an
//! adapter over the current public API (plus, for prototype 4, the smallest
//! internal hook it needs). The prototypes exist to find where the proposed
//! grammar does not fit the code, so a module may record a finding instead of a
//! working adapter. Findings are collected in `findings/pN.md`.

pub mod policy;

pub mod p1_svd;
pub mod p2_einsum_dest;
pub mod p3_fft_consume;
pub mod p4_plan_policy;
pub mod p5_error_causes;
pub mod p6_trace_einsum;
pub mod p7_borrowed_engine;
