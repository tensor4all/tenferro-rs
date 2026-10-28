//! Fusion and prepare: one emitted artifact per fused region, per-engine module cache,
//! and cold/warm counters.
//!
//! A fused region is planned and emitted by the kernel crate; this module compiles the
//! emitted source once per runtime (the runtime's artifact cache keys it by the request,
//! which includes the source, the options and the target architecture), binds the entry
//! and launches it. A warm run performs no NVRTC compilation.

use std::sync::Arc;

use t4a_cuda_kernels::fusion::{
    artifact_key, emit_fusion, plan_fused, ArtifactKey, FusionPlan, FUSION_ENTRY,
};
use t4a_cuda_kernels::Launch;
use t4a_cuda_runtime::{Arg, Artifact, BoundFunction, CompileRequest, LaunchConfig, OutputKind};

use super::runtime::CudaRuntime;

/// One prepared fused region for one runtime.
#[doc(hidden)]
pub struct PreparedFusion {
    runtime: CudaRuntime,
    /// The compiled artifact, retained so its module stays alive with the binding.
    #[allow(dead_code)]
    artifact: Arc<Artifact>,
    function: BoundFunction,
    key: ArtifactKey,
    outputs: u32,
    inputs: u32,
    dtype: t4a_cuda_kernels::DType,
}

impl PreparedFusion {
    /// Emit `plan`, compile it unless this runtime already has the artifact, and bind it.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the kernel crate refuses the plan or the
    /// emitted source, [`crate::Error::RuntimeStateSource`] when NVRTC rejects the source
    /// or the runtime cannot bind the entry.
    pub fn prepare(runtime: &CudaRuntime, plan: &FusionPlan) -> crate::Result<PreparedFusion> {
        let emitted = emit_fusion(plan)
            .map_err(|error| crate::Error::runtime_state("cuda fusion plan", error.to_string()))?;
        let capability = runtime.device_info().compute_capability();
        let arch = format!("sm_{}{}", capability.major, capability.minor);
        let manifest = emitted.manifest();
        // The artifact key is the kernel crate's canonical identity; the runtime's cache
        // keys on the full request, which carries the same inputs plus the compiler.
        let key = artifact_key(&emitted, "", &arch, "nvrtc");
        let (artifact, function) = {
            let mut guard = runtime.submit();
            let mut request = CompileRequest::new(FUSION_ENTRY, emitted.source());
            request.output = OutputKind::Cubin;
            request.arch = Some(arch.clone());
            let artifact = guard
                .compile(&request)
                .map_err(|error| crate::Error::runtime_state_source("cuda fusion", error))?;
            let function = guard
                .bind(&artifact, FUSION_ENTRY)
                .map_err(|error| crate::Error::runtime_state_source("cuda fusion bind", error))?;
            (artifact, function)
        };
        Ok(PreparedFusion {
            runtime: runtime.clone(),
            artifact,
            function,
            key,
            outputs: manifest.outputs,
            inputs: manifest.inputs,
            dtype: manifest.dtype,
        })
    }

    /// The canonical artifact key of this prepared region.
    pub fn key(&self) -> &ArtifactKey {
        &self.key
    }

    /// Payload dtype of the region.
    pub fn dtype(&self) -> t4a_cuda_kernels::DType {
        self.dtype
    }

    /// Enqueue one launch over `len` elements.
    ///
    /// The first `outputs` entries of `operands` are the destinations, the rest are the
    /// inputs, in the plan's declaration order.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the operand count does not match the plan,
    /// and [`crate::Error::RuntimeStateSource`] when the launch cannot be admitted.
    pub fn launch(
        &self,
        len: i64,
        operands: &[&t4a_cuda_runtime::Buffer],
    ) -> crate::Result<Launch> {
        let expected = (self.outputs + self.inputs) as usize;
        if operands.len() != expected {
            return Err(crate::Error::runtime_state(
                "cuda fusion",
                format!("the plan needs {expected} operands, got {}", operands.len()),
            ));
        }
        let plan: Launch = plan_fused(len)
            .map_err(|error| crate::Error::runtime_state("cuda fusion plan", error.to_string()))?;
        let mut args: Vec<Arg<'_>> = Vec::with_capacity(expected + 1);
        for operand in operands {
            args.push(Arg::DevicePtr(operand));
        }
        args.push(Arg::U64(len as u64));
        let runtime = self.runtime.clone();
        let mut guard = runtime.submit();
        // SAFETY: the emitted entry takes one pointer per output, one per input and the
        // element count, in the plan's order, which is what `operands` provides.
        unsafe {
            guard
                .launch(
                    &self.function,
                    LaunchConfig::new(plan.grid[0], plan.block[0]),
                    &args,
                )
                .map_err(|error| crate::Error::runtime_state_source("cuda fusion", error))
        }?;
        Ok(plan)
    }
}

/// The runtime counters a prepare/warm pair is judged by.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrepareCounters {
    /// NVRTC compilation attempts so far.
    pub compile_attempts: u64,
    /// Artifact-cache hits so far.
    pub artifact_hits: u64,
    /// Module loads so far.
    pub module_loads: u64,
}

#[cfg(test)]
impl PreparedFusion {
    /// The runtime's prepare counters, for a cold/warm comparison.
    pub(crate) fn counters(&self) -> PrepareCounters {
        PrepareCounters {
            compile_attempts: self.runtime.compile_attempts(),
            artifact_hits: self.runtime.artifact_cache_hits(),
            module_loads: self.runtime.module_loads(),
        }
    }
}

#[cfg(test)]
mod tests;
