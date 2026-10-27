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
    /// Prepare the distinct modules that export `entries` and bind nothing yet.
    ///
    /// A dispatch names the entries its operation launches, so a run compiles only the
    /// translation units it touches. The runtime's artifact cache serves repeats.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Unsupported`] when an entry has no translation unit,
    /// [`crate::Error::RuntimeStateSource`] when NVRTC rejects a module, and
    /// [`crate::Error::RuntimeState`] when the runtime cannot prepare its artifact.
    pub fn for_entries(runtime: &CudaRuntime, entries: &[Entry]) -> crate::Result<KernelModules> {
        let mut needed: Vec<&'static str> = Vec::new();
        for entry in entries {
            let module = module_of(*entry).ok_or_else(|| {
                crate::Error::unsupported("cuda prepare", "the entry has no translation unit")
            })?;
            if !needed.contains(&module.name) {
                needed.push(module.name);
            }
        }
        let capability = runtime.device_info().compute_capability();
        let arch = format!("sm_{}{}", capability.major, capability.minor);
        let mut artifacts = HashMap::with_capacity(needed.len());
        {
            let mut guard = runtime.submit();
            for module in MODULES
                .iter()
                .filter(|module| needed.contains(&module.name))
            {
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
        self.function_named(entry, entry.name())
    }

    /// Bind the general-path symbol of `entry`.
    ///
    /// The `_any` entries read their geometry from device memory, so the parameters that name
    /// those addresses are only known inside the runtime's raw scope; see
    /// [`KernelModules::in_scope`].
    fn function_named(
        &mut self,
        entry: Entry,
        name: &'static str,
    ) -> crate::Result<&BoundFunction> {
        if !self.functions.contains_key(name) {
            let module = module_of(entry).ok_or_else(|| {
                crate::Error::unsupported("cuda bind", "the entry has no translation unit")
            })?;
            let artifact = self.artifacts.get(module.name).ok_or_else(|| {
                crate::Error::runtime_state("cuda bind", "the entry's module is not prepared")
            })?;
            let function = {
                let mut guard = self.runtime.submit();
                guard
                    .bind(artifact, name)
                    .map_err(|error| crate::Error::runtime_state_source("cuda bind", error))?
            };
            self.functions.insert(name, function);
        }
        Ok(self
            .functions
            .get(name)
            .expect("the bound function was just inserted"))
    }

    /// Enqueue one launch of `symbol` (bound from `entry`'s module) inside the runtime's raw
    /// scope.
    ///
    /// `resources` must list every buffer the parameters name, because a general-path launch
    /// embeds device addresses the caller can only read through the scope.
    ///
    /// # Safety
    ///
    /// As [`KernelModules::launch`], and every address the closure embeds must stay valid for
    /// the scope's work.
    ///
    /// # Errors
    ///
    /// As [`KernelModules::function_named`], plus the runtime's raw-scope failure.
    pub(crate) unsafe fn in_scope<R>(
        &mut self,
        entry: Entry,
        symbol: &'static str,
        resources: Vec<t4a_cuda_runtime::RawResource<'_>>,
        run: impl FnOnce(&t4a_cuda_runtime::RawScope<'_>, &BoundFunction) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.function_named(entry, symbol)?;
        let runtime = self.runtime.clone();
        let function = self.functions.get(symbol).expect("bound above");
        runtime
            .with_raw_locked(resources, |scope| run(&scope, function))
            .and_then(|result| result)
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
    super::pending::cuda_storage(tensor).map(|storage| storage.allocation())
}
