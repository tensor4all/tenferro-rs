//! Planning and launching the kernel catalogue's entries on one runtime.
//!
//! The kernel crate is the single source of validation: every entry is planned by its
//! `plan_*` function, and this module only prepares the artifacts (NVRTC, once per
//! module), binds the entries and marshals the by-value parameters the ABI documents.

use std::collections::HashMap;
use std::sync::Arc;

use t4a_cuda_kernels::source::{compose, module_of, MODULES};
use t4a_cuda_kernels::{CLayout, Entry};
use t4a_cuda_runtime::{Arg, Artifact, BoundFunction, CompileRequest, LaunchConfig, OutputKind};

use super::runtime::CudaRuntime;

/// One runtime's prepared module set and bound functions.
#[doc(hidden)]
pub struct KernelModules {
    runtime: CudaRuntime,
    artifacts: HashMap<&'static str, Arc<Artifact>>,
    functions: HashMap<&'static str, BoundFunction>,
}

impl KernelModules {
    /// Prepare every translation unit for `runtime` and bind nothing yet.
    ///
    /// One NVRTC compilation per module; the runtime's artifact cache serves repeated
    /// calls, so a second `KernelModules` for the same runtime compiles nothing.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeStateSource`] when NVRTC rejects a module or its cubin, and
    /// [`crate::Error::RuntimeStateSource`] when the runtime cannot prepare the artifact.
    pub fn new(runtime: &CudaRuntime) -> crate::Result<KernelModules> {
        // The entries use double atomics and other sm_60+ intrinsics, so the artifacts
        // target the device's own architecture rather than a virtual one.
        let capability = runtime.device_info().compute_capability();
        let arch = format!("sm_{}{}", capability.major, capability.minor);
        let mut artifacts = HashMap::with_capacity(MODULES.len());
        {
            let mut guard = runtime.submit();
            for module in MODULES {
                let mut request = CompileRequest::new(module.name, compose(module));
                request.output = OutputKind::Cubin;
                request.arch = Some(arch.clone());
                let artifact = guard
                    .compile(&request)
                    .map_err(|error| crate::Error::runtime_state_source("cuda prepare", error))?;
                artifacts.insert(module.name, artifact);
            }
        }
        Ok(KernelModules {
            runtime: runtime.clone(),
            artifacts,
            functions: HashMap::new(),
        })
    }

    /// Number of prepared modules.
    pub fn modules(&self) -> usize {
        self.artifacts.len()
    }

    /// Bind `entry` if it is not bound yet and return its function.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Unsupported`] when the entry has no translation unit,
    /// [`crate::Error::RuntimeState`] when its module is not prepared, and
    /// [`crate::Error::RuntimeStateSource`] when the entry point cannot be bound.
    pub fn function(&mut self, entry: Entry) -> crate::Result<&BoundFunction> {
        if !self.functions.contains_key(entry.name()) {
            let module = module_of(entry).ok_or_else(|| {
                crate::Error::unsupported("cuda bind", "the entry has no translation unit")
            })?;
            let artifact = self.artifacts.get(module.name).ok_or_else(|| {
                crate::Error::runtime_state("cuda bind", "the entry's module is not prepared")
            })?;
            let function = {
                let mut guard = self.runtime.submit();
                guard
                    .bind(artifact, entry.name())
                    .map_err(|error| crate::Error::runtime_state_source("cuda bind", error))?
            };
            self.functions.insert(entry.name(), function);
        }
        Ok(self
            .functions
            .get(entry.name())
            .expect("the bound function was just inserted"))
    }

    /// Enqueue one launch of `entry` with `args`.
    ///
    /// # Safety
    ///
    /// The arguments must match the entry's ABI exactly; the planners are the validation
    /// source, and this module does not prove an argument list correct.
    ///
    /// # Errors
    ///
    /// As [`KernelModules::function`] for binding, and
    /// [`crate::Error::RuntimeStateSource`] when the runtime refuses or fails the launch.
    pub unsafe fn launch(
        &mut self,
        entry: Entry,
        config: LaunchConfig,
        args: &[Arg<'_>],
    ) -> crate::Result<()> {
        // Bind first (which borrows the runtime mutably) and only then take the
        // submission lock, so the two borrows never overlap.
        self.function(entry)?;
        let runtime = self.runtime.clone();
        let function = self
            .functions
            .get(entry.name())
            .expect("the function was bound above");
        let mut guard = runtime.submit();
        // SAFETY: forwarded to the caller's contract.
        unsafe { guard.launch(function, config, args) }
            .map_err(|error| crate::Error::runtime_state_source("cuda launch", error))
    }
}

/// A by-value parameter view of a `CLayout`, borrowing the layout for the launch.
///
/// `CLayout` is `repr(C)` plain data with no padding (the kernel crate asserts its size),
/// so its byte view is exactly the kernel's parameter.
#[doc(hidden)]
pub fn layout_arg<'a>(layout: &'a CLayout) -> Arg<'a> {
    // SAFETY: `CLayout` is `repr(C)` plain data and entirely initialised, so every byte
    // of it is readable and the slice borrows the caller's value for `'a`.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (layout as *const CLayout).cast::<u8>(),
            std::mem::size_of::<CLayout>(),
        )
    };
    Arg::Bytes {
        bytes,
        align: std::mem::align_of::<CLayout>(),
    }
}

/// The runtime allocation behind a CUDA tensor, for a launch argument.
///
/// The allocation domain is the tensor's; a launch on another runtime is refused by the
/// runtime's own ownership checks.
#[doc(hidden)]
pub fn device_buffer(tensor: &tenferro_tensor::Tensor) -> Option<&t4a_cuda_runtime::Buffer> {
    super::pending::cuda_storage(tensor).map(|storage| &storage.buffer)
}
