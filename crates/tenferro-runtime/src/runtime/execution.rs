//! Runtime-owned compiled-graph execution boundary.
//!
//! Phase 5 fills this module with the private execution bridge used by
//! `Runtime::run_compiled*`. The legacy `GraphExecutor` compatibility facade
//! must not grow new runtime ownership after this boundary exists.
