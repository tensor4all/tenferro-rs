use crate::api::runtime_dispatch::{unsupported_runtime_capability, with_runtime};
use crate::Result;
use tenferro_prims::CpuContext;

pub(crate) fn with_cpu_runtime<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    with_runtime(
        f,
        |_ctx| Err(unsupported_runtime_capability(op, "cuda")),
        |_ctx| Err(unsupported_runtime_capability(op, "rocm")),
    )
}
