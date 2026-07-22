//! Internal, sealed-custom-cfg Phase 2E evidence sink.
//!
//! This module intentionally exports no Rust API. The gate runner supplies a
//! row-specific path and the actual typed-add closure appends its Rayon lane
//! and operating-system CPU before doing kernel work.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const OBSERVATION_FILE: &str = "TENFERRO_PHASE2E_OPERATION_OBSERVATION_FILE";

#[cfg(target_os = "linux")]
pub(crate) fn record_typed_add_worker() {
    let Some(path) = std::env::var_os(OBSERVATION_FILE) else {
        return;
    };
    let Some(lane) = rayon::current_thread_index() else {
        return;
    };
    let Ok(cpu) = crate::affinity::current_cpu() else {
        return;
    };
    let path = PathBuf::from(path);
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(PathBuf, usize, usize)>>> =
        OnceLock::new();
    let mut seen = SEEN
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
        .unwrap();
    if !seen.insert((path.clone(), lane, cpu.as_usize())) {
        return;
    }
    let Ok(mut output) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let _ = writeln!(output, "{lane},{}", cpu.as_usize());
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn record_typed_add_worker() {}
