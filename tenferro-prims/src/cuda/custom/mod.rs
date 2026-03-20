use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use cudarc::runtime::result::version::get_driver_version;
use tenferro_device::{Error, Result};

mod cache;

const POINTWISE_REAL_SOURCE: &str = include_str!("../kernel_src/pointwise_real.cu");

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(super) enum RealUnaryKernelOp {
    Identity = 0,
    ImagZero = 1,
    Expm1 = 2,
    Log1p = 3,
    Rsqrt = 4,
}

#[derive(Clone, Copy)]
pub(super) enum RealBinaryKernelOp {
    Pow = 0,
    Atan2 = 1,
    Hypot = 2,
    Xlogy = 3,
}

pub(super) struct CustomCudaRuntime {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    pointwise_module_key: String,
    pointwise_module: Mutex<Option<Arc<CudaModule>>>,
    functions: Mutex<HashMap<&'static str, CudaFunction>>,
}

impl CustomCudaRuntime {
    pub(super) fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id)
            .map_err(|err| Error::DeviceError(format!("CUDA driver init failed: {err:?}")))?;
        let stream = ctx.default_stream();
        let (sm_major, sm_minor) = ctx.compute_capability().map_err(|err| {
            Error::DeviceError(format!("Failed to query CUDA compute capability: {err:?}"))
        })?;
        let driver_version = get_driver_version().map_err(|err| {
            Error::DeviceError(format!("Failed to query CUDA driver version: {err:?}"))
        })?;
        let pointwise_module_key =
            cache::pointwise_module_key(POINTWISE_REAL_SOURCE, sm_major, sm_minor, driver_version);

        Ok(Self {
            ctx,
            stream,
            pointwise_module_key,
            pointwise_module: Mutex::new(None),
            functions: Mutex::new(HashMap::new()),
        })
    }

    pub(super) fn launch_pointwise_unary_f32(
        &self,
        op: RealUnaryKernelOp,
        input: u64,
        output: u64,
        numel: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        self.launch_pointwise_unary(
            "tf_pointwise_unary_f32",
            op as i32,
            input,
            output,
            numel,
            alpha,
            beta,
        )
    }

    pub(super) fn launch_pointwise_unary_f64(
        &self,
        op: RealUnaryKernelOp,
        input: u64,
        output: u64,
        numel: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        self.launch_pointwise_unary(
            "tf_pointwise_unary_f64",
            op as i32,
            input,
            output,
            numel,
            alpha,
            beta,
        )
    }

    pub(super) fn launch_pointwise_binary_f32(
        &self,
        op: RealBinaryKernelOp,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        self.launch_pointwise_binary(
            "tf_pointwise_binary_f32",
            op as i32,
            lhs,
            rhs,
            output,
            numel,
            alpha,
            beta,
        )
    }

    pub(super) fn launch_pointwise_binary_f64(
        &self,
        op: RealBinaryKernelOp,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        self.launch_pointwise_binary(
            "tf_pointwise_binary_f64",
            op as i32,
            lhs,
            rhs,
            output,
            numel,
            alpha,
            beta,
        )
    }

    fn launch_pointwise_unary<T: cudarc::driver::DeviceRepr>(
        &self,
        function_name: &'static str,
        op: i32,
        input: u64,
        output: u64,
        numel: usize,
        alpha: T,
        beta: T,
    ) -> Result<()> {
        let func = self.load_function(function_name)?;
        let numel_u64 = numel as u64;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&input)
                .arg(&output)
                .arg(&numel_u64)
                .arg(&op)
                .arg(&alpha)
                .arg(&beta)
                .launch(cfg)
        }
        .map_err(|err| Error::DeviceError(format!("Custom CUDA launch failed: {err:?}")))?;
        self.stream.synchronize().map_err(|err| {
            Error::DeviceError(format!("Custom CUDA synchronize failed: {err:?}"))
        })?;
        Ok(())
    }

    fn launch_pointwise_binary<T: cudarc::driver::DeviceRepr>(
        &self,
        function_name: &'static str,
        op: i32,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        alpha: T,
        beta: T,
    ) -> Result<()> {
        let func = self.load_function(function_name)?;
        let numel_u64 = numel as u64;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&lhs)
                .arg(&rhs)
                .arg(&output)
                .arg(&numel_u64)
                .arg(&op)
                .arg(&alpha)
                .arg(&beta)
                .launch(cfg)
        }
        .map_err(|err| Error::DeviceError(format!("Custom CUDA launch failed: {err:?}")))?;
        self.stream.synchronize().map_err(|err| {
            Error::DeviceError(format!("Custom CUDA synchronize failed: {err:?}"))
        })?;
        Ok(())
    }

    fn load_function(&self, function_name: &'static str) -> Result<CudaFunction> {
        let mut functions = self
            .functions
            .lock()
            .map_err(|_| Error::DeviceError("Custom CUDA function cache lock poisoned".into()))?;
        if let Some(function) = functions.get(function_name) {
            return Ok(function.clone());
        }
        let module = self.load_pointwise_module()?;
        let function = module.load_function(function_name).map_err(|err| {
            Error::DeviceError(format!(
                "Failed to load custom CUDA function `{function_name}`: {err:?}"
            ))
        })?;
        functions.insert(function_name, function.clone());
        Ok(function)
    }

    fn load_pointwise_module(&self) -> Result<Arc<CudaModule>> {
        let mut module = self
            .pointwise_module
            .lock()
            .map_err(|_| Error::DeviceError("Custom CUDA module cache lock poisoned".into()))?;
        if let Some(module) = module.as_ref() {
            return Ok(Arc::clone(module));
        }
        let ptx = if let Some(ptx) = cache::load_ptx(&self.pointwise_module_key)? {
            Ptx::from_src(ptx)
        } else {
            let (sm_major, sm_minor) = self.ctx.compute_capability().map_err(|err| {
                Error::DeviceError(format!("Failed to query CUDA compute capability: {err:?}"))
            })?;
            let opts = CompileOptions {
                options: vec![
                    "--std=c++14".into(),
                    format!("--gpu-architecture=compute_{sm_major}{sm_minor}"),
                ],
                ..Default::default()
            };
            let compiled = compile_ptx_with_opts(POINTWISE_REAL_SOURCE, opts)
                .map_err(|err| Error::DeviceError(format!("NVRTC compilation failed: {err:?}")))?;
            cache::store_ptx(&self.pointwise_module_key, &compiled.to_src())?;
            compiled
        };
        let loaded = self.ctx.load_module(ptx).map_err(|err| {
            Error::DeviceError(format!("Failed to load cached PTX module: {err:?}"))
        })?;
        *module = Some(Arc::clone(&loaded));
        Ok(loaded)
    }
}
