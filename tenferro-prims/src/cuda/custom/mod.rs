use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use cudarc::runtime::result::version::get_driver_version;
use num_complex::{Complex32, Complex64};
use tenferro_device::{Error, Result};

mod cache;

const POINTWISE_REAL_SOURCE: &str = include_str!("../kernel_src/pointwise_real.cu");
const POINTWISE_COMPLEX_SOURCE: &str = include_str!("../kernel_src/pointwise_complex.cu");

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

#[derive(Clone, Copy)]
pub(super) enum ComplexUnaryKernelOp {
    Neg = 0,
    Conj = 1,
    Abs = 2,
    Reciprocal = 3,
    Real = 4,
    Imag = 5,
    Sqrt = 6,
    Rsqrt = 7,
    Exp = 8,
    Expm1 = 9,
    Log = 10,
    Log1p = 11,
    Sin = 12,
    Cos = 13,
    Tan = 14,
    Tanh = 15,
    Asin = 16,
    Acos = 17,
    Atan = 18,
    Sinh = 19,
    Cosh = 20,
    Asinh = 21,
    Acosh = 22,
    Atanh = 23,
}

#[derive(Clone, Copy)]
pub(super) enum ComplexBinaryKernelOp {
    Sub = 0,
    Div = 1,
    Pow = 2,
    Xlogy = 3,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KernelComplex32 {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KernelComplex64 {
    re: f64,
    im: f64,
}

unsafe impl cudarc::driver::DeviceRepr for KernelComplex32 {}
unsafe impl cudarc::driver::DeviceRepr for KernelComplex64 {}

pub(super) struct CustomCudaRuntime {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    real_pointwise_module_key: String,
    real_pointwise_module: Mutex<Option<Arc<CudaModule>>>,
    complex_pointwise_module_key: String,
    complex_pointwise_module: Mutex<Option<Arc<CudaModule>>>,
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
        let compile_options = build_compile_options(sm_major, sm_minor);

        Ok(Self {
            ctx,
            stream,
            real_pointwise_module_key: cache::pointwise_real_module_key(
                POINTWISE_REAL_SOURCE,
                sm_major,
                sm_minor,
                driver_version,
                &compile_options,
            ),
            real_pointwise_module: Mutex::new(None),
            complex_pointwise_module_key: cache::pointwise_complex_module_key(
                POINTWISE_COMPLEX_SOURCE,
                sm_major,
                sm_minor,
                driver_version,
                &compile_options,
            ),
            complex_pointwise_module: Mutex::new(None),
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
            self.load_real_function("tf_pointwise_unary_f32")?,
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
            self.load_real_function("tf_pointwise_unary_f64")?,
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
            self.load_real_function("tf_pointwise_binary_f32")?,
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
            self.load_real_function("tf_pointwise_binary_f64")?,
            op as i32,
            lhs,
            rhs,
            output,
            numel,
            alpha,
            beta,
        )
    }

    pub(super) fn launch_pointwise_complex_unary_c32(
        &self,
        op: ComplexUnaryKernelOp,
        input: u64,
        output: u64,
        numel: usize,
        alpha: Complex32,
        beta: Complex32,
    ) -> Result<()> {
        self.launch_pointwise_unary(
            self.load_complex_function("tf_pointwise_unary_c32")?,
            op as i32,
            input,
            output,
            numel,
            KernelComplex32::from(alpha),
            KernelComplex32::from(beta),
        )
    }

    pub(super) fn launch_pointwise_complex_unary_c64(
        &self,
        op: ComplexUnaryKernelOp,
        input: u64,
        output: u64,
        numel: usize,
        alpha: Complex64,
        beta: Complex64,
    ) -> Result<()> {
        self.launch_pointwise_unary(
            self.load_complex_function("tf_pointwise_unary_c64")?,
            op as i32,
            input,
            output,
            numel,
            KernelComplex64::from(alpha),
            KernelComplex64::from(beta),
        )
    }

    pub(super) fn launch_pointwise_complex_binary_c32(
        &self,
        op: ComplexBinaryKernelOp,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        alpha: Complex32,
        beta: Complex32,
    ) -> Result<()> {
        self.launch_pointwise_binary(
            self.load_complex_function("tf_pointwise_binary_c32")?,
            op as i32,
            lhs,
            rhs,
            output,
            numel,
            KernelComplex32::from(alpha),
            KernelComplex32::from(beta),
        )
    }

    pub(super) fn launch_pointwise_complex_binary_c64(
        &self,
        op: ComplexBinaryKernelOp,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        alpha: Complex64,
        beta: Complex64,
    ) -> Result<()> {
        self.launch_pointwise_binary(
            self.load_complex_function("tf_pointwise_binary_c64")?,
            op as i32,
            lhs,
            rhs,
            output,
            numel,
            KernelComplex64::from(alpha),
            KernelComplex64::from(beta),
        )
    }

    fn launch_pointwise_unary<T: cudarc::driver::DeviceRepr>(
        &self,
        func: CudaFunction,
        op: i32,
        input: u64,
        output: u64,
        numel: usize,
        alpha: T,
        beta: T,
    ) -> Result<()> {
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
        func: CudaFunction,
        op: i32,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        alpha: T,
        beta: T,
    ) -> Result<()> {
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

    fn load_real_function(&self, function_name: &'static str) -> Result<CudaFunction> {
        self.load_function(function_name, || self.load_real_pointwise_module())
    }

    fn load_complex_function(&self, function_name: &'static str) -> Result<CudaFunction> {
        self.load_function(function_name, || self.load_complex_pointwise_module())
    }

    fn load_function<F>(&self, function_name: &'static str, load_module: F) -> Result<CudaFunction>
    where
        F: FnOnce() -> Result<Arc<CudaModule>>,
    {
        let mut functions = self
            .functions
            .lock()
            .map_err(|_| Error::DeviceError("Custom CUDA function cache lock poisoned".into()))?;
        if let Some(function) = functions.get(function_name) {
            return Ok(function.clone());
        }
        let module = load_module()?;
        let function = module.load_function(function_name).map_err(|err| {
            Error::DeviceError(format!(
                "Failed to load custom CUDA function `{function_name}`: {err:?}"
            ))
        })?;
        functions.insert(function_name, function.clone());
        Ok(function)
    }

    fn load_real_pointwise_module(&self) -> Result<Arc<CudaModule>> {
        self.load_module(
            &self.real_pointwise_module,
            &self.real_pointwise_module_key,
            POINTWISE_REAL_SOURCE,
        )
    }

    fn load_complex_pointwise_module(&self) -> Result<Arc<CudaModule>> {
        self.load_module(
            &self.complex_pointwise_module,
            &self.complex_pointwise_module_key,
            POINTWISE_COMPLEX_SOURCE,
        )
    }

    fn load_module(
        &self,
        slot: &Mutex<Option<Arc<CudaModule>>>,
        cache_key: &str,
        source: &str,
    ) -> Result<Arc<CudaModule>> {
        let mut module = slot
            .lock()
            .map_err(|_| Error::DeviceError("Custom CUDA module cache lock poisoned".into()))?;
        if let Some(module) = module.as_ref() {
            return Ok(Arc::clone(module));
        }
        let ptx = if let Some(ptx) = cache::load_ptx(cache_key)? {
            Ptx::from_src(ptx)
        } else {
            let (sm_major, sm_minor) = self.ctx.compute_capability().map_err(|err| {
                Error::DeviceError(format!("Failed to query CUDA compute capability: {err:?}"))
            })?;
            let compile_options = build_compile_options(sm_major, sm_minor);
            let opts = CompileOptions {
                options: compile_options,
                ..Default::default()
            };
            let compiled = compile_ptx_with_opts(source, opts)
                .map_err(|err| Error::DeviceError(format!("NVRTC compilation failed: {err:?}")))?;
            cache::store_ptx(cache_key, &compiled.to_src())?;
            compiled
        };
        let loaded = self.ctx.load_module(ptx).map_err(|err| {
            Error::DeviceError(format!("Failed to load cached PTX module: {err:?}"))
        })?;
        *module = Some(Arc::clone(&loaded));
        Ok(loaded)
    }
}

impl From<Complex32> for KernelComplex32 {
    fn from(value: Complex32) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

impl From<Complex64> for KernelComplex64 {
    fn from(value: Complex64) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

fn build_compile_options(sm_major: i32, sm_minor: i32) -> Vec<String> {
    vec![
        "--std=c++14".into(),
        format!("--gpu-architecture=compute_{sm_major}{sm_minor}"),
    ]
}
