pub(crate) use tenferro_internal_runtime::dispatch::*;

macro_rules! runtime_slot_closure {
    ($slot:ty, |$ctx:ident, $backend:ident, $runtime:ident| $body:expr) => {{
        |$ctx| {
            type $backend = <$slot as crate::runtime::dispatch::RuntimeSlot>::SemiringBackend;
            let $runtime = <$slot as crate::runtime::dispatch::RuntimeSlot>::NAME;
            $body
        }
    }};
}

pub(crate) use runtime_slot_closure;

macro_rules! dispatch_einsum_runtime {
    ($ty:ty, $op:expr, |$ctx:ident, $backend:ident| $body:expr) => {{
        dispatch_einsum_runtime!($ty, $op, |$ctx, $backend, _runtime| $body)
    }};
    ($ty:ty, $op:expr, |$ctx:ident, $backend:ident, $runtime:ident| $body:expr) => {{
        crate::runtime::dispatch::with_einsum_runtime::<$ty, _>(
            $op,
            crate::runtime::dispatch::runtime_slot_closure!(
                crate::runtime::dispatch::CpuRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
            crate::runtime::dispatch::runtime_slot_closure!(
                crate::runtime::dispatch::CudaRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
            crate::runtime::dispatch::runtime_slot_closure!(
                crate::runtime::dispatch::RocmRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
        )
    }};
}

pub(crate) use dispatch_einsum_runtime;
