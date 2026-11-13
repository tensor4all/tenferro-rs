//! The FFT operation registry: one `api_ops!` invocation, the single source of
//! the public FFT grammar (`op`, `op_with` on three layers, plus the consuming
//! in-place transforms of exception E01).
//!
//! The macro generates the traits, the forwarding impls, the rustdoc and the
//! metadata constants; the bodies are `crate::api::{concrete, traced, eager}`.

use tenferro_extension_macros::api_ops;

use crate::{FftOptions, FftnOptions};

api_ops! {
    family: fft;

    layer concrete {
        trait_doc: "FFTs on anything that implements `AsView`: owned and borrowed tensors, typed tensors and strided views. Results are `Tensor`s for erased inputs and typed tensors of the transform's dtype for typed inputs. Every operation takes the options last-but-one and the engine last.";
        trait_def: pub trait FftOps: tenferro_tensor::AsView;
        impl_def: impl<X: tenferro_tensor::AsView + ?Sized> FftOps for X;
        method_where: [Self::Repr: crate::FftRepr];
        receiver: self;
        tensor_arg: &impl tenferro_tensor::AsView<Repr = Self::Repr>;
        exec: &mut dyn tenferro_tensor::Engine;
        result: tenferro_tensor::Result;
        types {
            C = <Self::Repr as crate::FftRepr>::Complex,
            R = <Self::Repr as crate::FftRepr>::Real
        }
        forward: crate::api::concrete;
    }
    layer traced {
        trait_doc: "FFTs on traced tensors: each call records one FFT node and returns its traced result; execution happens when the graph is prepared and run.";
        trait_def: pub trait TracedFftOps;
        impl_def: impl TracedFftOps for tenferro_runtime::TracedTensor;
        method_where: [];
        receiver: self;
        tensor_arg: &tenferro_runtime::TracedTensor;
        result: tenferro_runtime::Result;
        types {
            C = tenferro_runtime::TracedTensor,
            R = tenferro_runtime::TracedTensor
        }
        forward: crate::api::traced;
    }
    layer eager {
        trait_doc: "FFTs on eager tensors, with the session as receiver. Eager operations execute immediately and record the tape for reverse-mode differentiation. The consuming in-place transforms `fft_in_place` and `ifft_in_place` (exception E01) live here too.";
        trait_def: pub trait EagerFftOps;
        impl_def: impl EagerFftOps for tenferro_ad::EagerSession<'_>;
        method_where: [];
        receiver: session;
        first_arg: &tenferro_ad::EagerTensor;
        tensor_arg: &tenferro_ad::EagerTensor;
        result: tenferro_ad::Result;
        types {
            C = tenferro_ad::EagerTensor,
            R = tenferro_ad::EagerTensor
        }
        forward: crate::api::eager;
        cfg: feature = "autodiff";
        extra_decls {
            /// Consuming in-place forward FFT along `opts.axis` (exception E01).
            ///
            /// Consumes a uniquely owned, untracked, compact column-major CPU
            /// Complex32/Complex64 tensor and overwrites it with its FFT. Shape and
            /// allocation are preserved; no implicit copy occurs. A compact slice of
            /// a larger plain host allocation keeps that allocation: only the
            /// slice's logical elements are overwritten (#2023). A device-backed or
            /// managed owner must be whole-root. Gradient-tracked inputs and active
            /// trace capture are rejected even under `no_grad`. `opts.n` must be
            /// `None`: an in-place transform cannot change the length.
            ///
            /// # Errors
            ///
            /// [`EagerFftInPlaceError::Rejected`](crate::EagerFftInPlaceError::Rejected)
            /// returns the unchanged input with the reason (wrong dtype, layout,
            /// placement, `n` set, shared or tracked input).
            /// [`EagerFftInPlaceError::Execution`](crate::EagerFftInPlaceError::Execution)
            /// means the input was consumed. `?` converts the error to
            /// `tenferro_ad::Error` and drops a rejected input; `match` first to
            /// recover it.
            ///
            /// # Examples
            ///
            /// ```
            /// use num_complex::Complex64;
            /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
            /// use tenferro_cpu::CpuEngine;
            /// use tenferro_fft::{EagerFftInPlaceError, EagerFftOps, FftOptions};
            ///
            /// let runtime = EagerRuntime::new()?;
            /// let mut engine = CpuEngine::serial();
            /// let x = runtime.constant_from(Tensor::from_vec_col_major([2], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)])?)?;
            /// // Match before `?` to recover a rejected input.
            /// let x = match runtime.with_eager_session(&mut engine, |s| {
            ///     Ok::<_, tenferro_ad::Error>(s.fft_in_place(x, &FftOptions::default().n(Some(4))))
            /// })? {
            ///     Err(EagerFftInPlaceError::Rejected { input, .. }) => *input,
            ///     other => panic!("a length change must be rejected: {other:?}"),
            /// };
            /// let y = runtime.with_eager_session(&mut engine, |s| {
            ///     Ok::<_, tenferro_ad::Error>(s.fft_in_place(x, &FftOptions::default())?)
            /// })?;
            /// assert_eq!(y.value()?.as_slice::<Complex64>()?[0], Complex64::new(3.0, 0.0));
            /// # Ok::<(), Box<dyn std::error::Error>>(())
            /// ```
            fn fft_in_place(
                &mut self,
                x: tenferro_ad::EagerTensor,
                opts: &crate::FftOptions,
            ) -> ::std::result::Result<tenferro_ad::EagerTensor, crate::EagerFftInPlaceError>;

            /// Consuming in-place inverse FFT along `opts.axis` (exception E01).
            ///
            /// Same contract as [`fft_in_place`](Self::fft_in_place).
            ///
            /// # Errors
            ///
            /// As [`fft_in_place`](Self::fft_in_place).
            ///
            /// # Examples
            ///
            /// ```
            /// use num_complex::Complex64;
            /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
            /// use tenferro_cpu::CpuEngine;
            /// use tenferro_fft::{EagerFftOps, FftOptions};
            ///
            /// let runtime = EagerRuntime::new()?;
            /// let x = runtime.constant_from(Tensor::from_vec_col_major([2], vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)])?)?;
            /// let y = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
            ///     Ok::<_, tenferro_ad::Error>(s.ifft_in_place(x, &FftOptions::default())?)
            /// })?;
            /// assert_eq!(y.value()?.as_slice::<Complex64>()?[1], Complex64::new(2.0, 0.0));
            /// # Ok::<(), Box<dyn std::error::Error>>(())
            /// ```
            fn ifft_in_place(
                &mut self,
                x: tenferro_ad::EagerTensor,
                opts: &crate::FftOptions,
            ) -> ::std::result::Result<tenferro_ad::EagerTensor, crate::EagerFftInPlaceError>;
        }
        extra_impls {
            fn fft_in_place(
                &mut self,
                x: tenferro_ad::EagerTensor,
                opts: &crate::FftOptions,
            ) -> ::std::result::Result<tenferro_ad::EagerTensor, crate::EagerFftInPlaceError> {
                crate::api::eager::in_place(self, "fft_in_place", x, crate::FftOperation::C2cForward, opts)
            }

            fn ifft_in_place(
                &mut self,
                x: tenferro_ad::EagerTensor,
                opts: &crate::FftOptions,
            ) -> ::std::result::Result<tenferro_ad::EagerTensor, crate::EagerFftInPlaceError> {
                crate::api::eager::in_place(self, "ifft_in_place", x, crate::FftOperation::C2cInverse, opts)
            }
        }
    }

    op fft {
        summary: "One-axis discrete Fourier transform along `axis` (default the last axis).";
        operands: (a);
        options: FftOptions;
        output: C;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result is complex (Complex32 for f32/Complex32 inputs, Complex64 otherwise); a real input is transformed as a complex one with zero imaginary part.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?;
let y = x.fft(&mut engine)?;
let z = x.fft_with(&FftOptions::default().axis(0), &mut engine)?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?;
let y = x.fft()?;
let z = x.fft_with(&FftOptions::default().axis(0))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?)?;
    let y = s.fft(&x)?;
    let z = s.fft_with(&x, &FftOptions::default().axis(0))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.value()?.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op ifft {
        summary: "One-axis inverse discrete Fourier transform along `axis`.";
        operands: (a);
        options: FftOptions;
        output: C;
        dtypes: "`a`: Complex32 or Complex64. The result has the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![4], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)])?;
let y = x.ifft(&mut engine)?;
let z = x.ifft_with(&FftOptions::default().axis(0), &mut engine)?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(2.5, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![4], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)])?;
let y = x.ifft()?;
let z = x.ifft_with(&FftOptions::default().axis(0))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(2.5, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![4], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)])?)?;
    let y = s.ifft(&x)?;
    let z = s.ifft_with(&x, &FftOptions::default().axis(0))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.value()?.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(2.5, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op rfft {
        summary: "One-axis transform of a real signal, returning the one-sided spectrum.";
        operands: (a);
        options: FftOptions;
        output: C;
        dtypes: "`a`: f32 or f64. The result is the one-sided complex spectrum (Complex32 or Complex64) of length `n / 2 + 1` along `axis`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?;
let y = x.rfft(&mut engine)?;
let z = x.rfft_with(&FftOptions::default().axis(0), &mut engine)?;
assert_eq!(y.shape(), &[3]);
assert_eq!(z.shape(), &[3]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?;
let y = x.rfft()?;
let z = x.rfft_with(&FftOptions::default().axis(0))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[3]);
assert_eq!(z.shape(), &[3]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?)?;
    let y = s.rfft(&x)?;
    let z = s.rfft_with(&x, &FftOptions::default().axis(0))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[3]);
assert_eq!(z.shape(), &[3]);
let v = y.value()?.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op irfft {
        summary: "Inverse of `rfft`: a one-sided spectrum back to a real signal.";
        operands: (a);
        options: FftOptions;
        output: R;
        dtypes: "`a`: Complex32 or Complex64 (a one-sided spectrum). The result is real (f32 or f64); its length along `axis` is `n`, by default `2 * (m - 1)`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![3], vec![Complex64::new(10.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-2.0, 0.0)])?;
let y = x.irfft(&mut engine)?;
let z = x.irfft_with(&FftOptions::default().n(Some(4)), &mut engine)?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.as_slice::<f64>()?[0];
assert!((v - 2.0).abs() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![3], vec![Complex64::new(10.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-2.0, 0.0)])?;
let y = x.irfft()?;
let z = x.irfft_with(&FftOptions::default().n(Some(4)))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.as_slice::<f64>()?[0];
assert!((v - 2.0).abs() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![3], vec![Complex64::new(10.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-2.0, 0.0)])?)?;
    let y = s.irfft(&x)?;
    let z = s.irfft_with(&x, &FftOptions::default().n(Some(4)))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[4]);
assert_eq!(z.shape(), &[4]);
let v = y.value()?.as_slice::<f64>()?[0];
assert!((v - 2.0).abs() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op fftn {
        summary: "N-dimensional discrete Fourier transform over `axes` (default every axis).";
        operands: (a);
        options: FftnOptions;
        output: C;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result is complex, as for `fft`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftnOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?;
let y = x.fftn(&mut engine)?;
let z = x.fftn_with(&FftnOptions::default().axes(vec![0, 1]), &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
assert_eq!(z.shape(), &[2, 2]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftnOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?;
let y = x.fftn()?;
let z = x.fftn_with(&FftnOptions::default().axes(vec![0, 1]))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
assert_eq!(z.shape(), &[2, 2]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftnOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])?)?;
    let y = s.fftn(&x)?;
    let z = s.fftn_with(&x, &FftnOptions::default().axes(vec![0, 1]))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[2, 2]);
assert_eq!(z.shape(), &[2, 2]);
let v = y.value()?.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(10.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op ifftn {
        summary: "N-dimensional inverse discrete Fourier transform over `axes`.";
        operands: (a);
        options: FftnOptions;
        output: C;
        dtypes: "`a`: Complex32 or Complex64. The result has the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftnOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)])?;
let y = x.ifftn(&mut engine)?;
let z = x.ifftn_with(&FftnOptions::default().axes(vec![0, 1]), &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
assert_eq!(z.shape(), &[2, 2]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(2.5, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftnOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![2, 2], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)])?;
let y = x.ifftn()?;
let z = x.ifftn_with(&FftnOptions::default().axes(vec![0, 1]))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
assert_eq!(z.shape(), &[2, 2]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(2.5, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftnOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)])?)?;
    let y = s.ifftn(&x)?;
    let z = s.ifftn_with(&x, &FftnOptions::default().axes(vec![0, 1]))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[2, 2]);
assert_eq!(z.shape(), &[2, 2]);
let v = y.value()?.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(2.5, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op rfftn {
        summary: "N-dimensional transform of a real signal; the last transformed axis is one-sided.";
        operands: (a);
        options: FftnOptions;
        output: C;
        dtypes: "`a`: f32 or f64. The result is complex; the last transformed axis is one-sided (`n / 2 + 1`).";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftnOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![2, 4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64])?;
let y = x.rfftn(&mut engine)?;
let z = x.rfftn_with(&FftnOptions::default().axes(vec![0, 1]), &mut engine)?;
assert_eq!(y.shape(), &[2, 3]);
assert_eq!(z.shape(), &[2, 3]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(36.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftnOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![2, 4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64])?;
let y = x.rfftn()?;
let z = x.rfftn_with(&FftnOptions::default().axes(vec![0, 1]))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[2, 3]);
assert_eq!(z.shape(), &[2, 3]);
let v = y.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(36.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftnOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![2, 4], vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64])?)?;
    let y = s.rfftn(&x)?;
    let z = s.rfftn_with(&x, &FftnOptions::default().axes(vec![0, 1]))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[2, 3]);
assert_eq!(z.shape(), &[2, 3]);
let v = y.value()?.as_slice::<Complex64>()?[0];
assert!((v - Complex64::new(36.0, 0.0)).norm() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op irfftn {
        summary: "Inverse of `rfftn`.";
        operands: (a);
        options: FftnOptions;
        output: R;
        dtypes: "`a`: Complex32 or Complex64. The result is real (f32 or f64); the last transformed axis has length `2 * (m - 1)` unless `shape` says otherwise.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`"];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftOps, FftnOptions};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![2, 3], vec![Complex64::new(8.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)])?;
let y = x.irfftn(&mut engine)?;
let z = x.irfftn_with(&FftnOptions::default().shape(Some(vec![2, 4])), &mut engine)?;
assert_eq!(y.shape(), &[2, 4]);
assert_eq!(z.shape(), &[2, 4]);
let v = y.as_slice::<f64>()?[0];
assert!((v - 1.0).abs() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{TracedFftOps, FftnOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let x = trace.from_vec_col_major(vec![2, 3], vec![Complex64::new(8.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)])?;
let y = x.irfftn()?;
let z = x.irfftn_with(&FftnOptions::default().shape(Some(vec![2, 4])))?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&y, &z])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let y = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let z = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(y.shape(), &[2, 4]);
assert_eq!(z.shape(), &[2, 4]);
let v = y.as_slice::<f64>()?[0];
assert!((v - 1.0).abs() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftnOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (y, z) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let x = s.constant_from(Tensor::from_vec_col_major(vec![2, 3], vec![Complex64::new(8.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)])?)?;
    let y = s.irfftn(&x)?;
    let z = s.irfftn_with(&x, &FftnOptions::default().shape(Some(vec![2, 4])))?;
    Ok::<_, tenferro_ad::Error>((y, z))
})?;
assert_eq!(y.shape(), &[2, 4]);
assert_eq!(z.shape(), &[2, 4]);
let v = y.value()?.as_slice::<f64>()?[0];
assert!((v - 1.0).abs() < 1e-12);
# let _ = Complex64::new(0.0, 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    exceptions {
        E01 {
            symbols: [
                "tenferro_fft::EagerFftOps::fft_in_place",
                "tenferro_fft::EagerFftOps::ifft_in_place",
                "tenferro_fft::EagerFftInPlaceError"
            ];
            scope: "consuming eager FFTs that reuse the input storage";
            category: "in-place | consuming";
            owner_module: "tenferro_fft::eager_in_place";
            reason: "an FFT of a large compact complex tensor can reuse its storage; the grammar's `_into` writes to a separate destination and cannot express a consumed input";
            verification: "src/api/tests/registry.rs: a length change and a real input are Rejected with the input returned unchanged; an accepted call equals the out-of-place fft bit for bit";
            exit_condition: "a consuming `Dest` form in the grammar";
            status: retained;
        }
        E02 {
            symbols: [
                "tenferro_fft::FftExecutor::fft_with",
                "tenferro_fft::FftExecutor::ifft_with",
                "tenferro_fft::FftExecutor::rfft_with",
                "tenferro_fft::FftExecutor::irfft_with",
                "tenferro_fft::FftExecutor::fftn_with",
                "tenferro_fft::FftExecutor::ifftn_with",
                "tenferro_fft::FftExecutor::rfftn_with",
                "tenferro_fft::FftExecutor::irfftn_with",
                "tenferro_fft::FftExecutor::new",
                "tenferro_fft::FftExecutor::plan_cache",
                "tenferro_fft::FftExecutor::plan_cache_mut",
                "tenferro_fft::FftExecutor::cache_stats",
                "tenferro_fft::FftExecutor::clear_cache"
            ];
            scope: "the caller-owned FFT plan cache and its transforms";
            category: "cache owner";
            owner_module: "tenferro_fft";
            reason: "the receiver is the plan cache, not the input; the transforms take the input, the options and the engine in grammar order";
            verification: "src/api/tests/registry.rs: every `_with` equals the `FftOps` result bit for bit and fills the caller's plan cache";
            exit_condition: "plan caches owned by the engine for every caller";
            status: retained;
        }
    }
}
