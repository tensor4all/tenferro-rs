//! The core operation registry: one `api_ops!` invocation, the single source of
//! the public core grammar (#2552 §7.2).
//!
//! The concrete layer is generated (`CoreOps`). The traced layer (`TracedTensor`
//! inherent methods) and the eager layer (`EagerSession` inherent methods) stay
//! inherent and are described here, including the per-layer differences of the
//! operation sets. Generated with `/tmp/2552-s6/gen_core.py` for S6 and edited
//! by hand from then on.

use tenferro_extension_macros::api_ops;

use crate::AddOptions;

api_ops! {
    family: core;

    layer concrete {
        trait_doc: "Core tensor operations on anything that implements `AsView`: owned and borrowed tensors, typed tensors and strided views. Results are `Tensor`s for erased inputs and typed tensors for typed inputs. Axes are signed (`-1` is the last axis); the engine is always the last argument.";
        trait_def: pub trait CoreOps: tenferro_tensor::AsView;
        impl_def: impl<X: tenferro_tensor::AsView + ?Sized> CoreOps for X;
        method_where: [Self::Repr: crate::CoreRepr];
        receiver: self;
        tensor_arg: &impl tenferro_tensor::AsView<Repr = Self::Repr>;
        exec: &mut dyn tenferro_tensor::Engine;
        dest: tenferro_tensor::Dest<<Self::Repr as tenferro_tensor::Repr>::Write<'_>>;
        result: tenferro_tensor::Result;
        types {
            V = <Self::Repr as tenferro_tensor::Repr>::Value,
            R = <Self::Repr as crate::CoreRepr>::Real,
            B = <Self::Repr as crate::CoreRepr>::Bool
        }
        forward: crate::api::concrete;
    }
    layer traced { describe_only; }
    layer eager { describe_only; }

    op exp {
        summary: "Elementwise exponential.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.exp(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op neg {
        summary: "Elementwise negation.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.neg(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op sign {
        summary: "Elementwise sign.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.sign(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op conj {
        summary: "Elementwise complex conjugate (identity on real dtypes).";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.conj(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op log {
        summary: "Elementwise natural logarithm.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.log(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op expm1 {
        summary: "Elementwise `exp(x) - 1`.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.expm1(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op log1p {
        summary: "Elementwise `log(1 + x)`.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.log1p(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op erf {
        summary: "Elementwise error function.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.erf(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op sin {
        summary: "Elementwise sine.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.sin(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op cos {
        summary: "Elementwise cosine.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.cos(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op tanh {
        summary: "Elementwise hyperbolic tangent.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.tanh(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op sqrt {
        summary: "Elementwise square root.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.sqrt(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op rsqrt {
        summary: "Elementwise reciprocal square root.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.rsqrt(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op abs {
        summary: "Elementwise absolute value; complex inputs return their modulus.";
        operands: (a);
        options: none;
        output: R;
        dtypes: "Any numeric dtype; complex inputs return the real dtype of the same precision.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.abs(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op sigmoid {
        summary: "Logistic sigmoid.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32 or f64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.sigmoid(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op silu {
        summary: "SiLU (`x * sigmoid(x)`).";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32 or f64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.silu(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op softplus {
        summary: "Softplus `log(1 + exp(x))`.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32 or f64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.softplus(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op gelu {
        summary: "GELU with the exact error function.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32 or f64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.gelu(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op gelu_tanh {
        summary: "GELU with the tanh approximation.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "f32 or f64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.gelu_tanh(&mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op mul {
        summary: "Elementwise product with broadcasting.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.mul(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op sub {
        summary: "Elementwise difference with broadcasting.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.sub(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op div {
        summary: "Elementwise quotient with broadcasting.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.div(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op rem {
        summary: "Elementwise remainder with broadcasting.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.rem(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op pow {
        summary: "Elementwise power with broadcasting.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.pow(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op maximum {
        summary: "Elementwise maximum with broadcasting.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.maximum(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op minimum {
        summary: "Elementwise minimum with broadcasting.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.minimum(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op add {
        summary: "Elementwise sum with broadcasting.";
        operands: (a, b);
        options: AddOptions;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [overwrite];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel. `add_into` writes the result into the destination's storage.";
        errors: [Validation, Unsupported];
        omissions: ["traced and eager: only `add`; `add_with` and `add_into` exist on the concrete layer"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.add(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op matmul {
        summary: "Matrix product of the two leading axes, batched over trailing axes.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.matmul(&a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op compare {
        summary: "Elementwise comparison with broadcasting; returns a `bool` tensor.";
        operands: (a, b, dir: tenferro_tensor::CompareDir);
        options: none;
        output: B;
        dtypes: "Both operands share a dtype; the result is `bool`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{CompareDir, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.compare(&a, CompareDir::Eq, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op clamp {
        summary: "Elementwise clamp to `[lower, upper]` with broadcasting.";
        operands: (a, lower, upper);
        options: none;
        output: V;
        dtypes: "All operands share a real dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.clamp(&a, &a, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op where_select {
        summary: "Elementwise selection: `on_true` where the receiver is true, `on_false` elsewhere, with broadcasting. The receiver is the `bool` condition; a typed receiver must be `TypedTensor<bool>`.";
        generics: <Y: crate::SelectOperand<Self::Repr> + ?Sized>;
        operands: (condition, on_true: &Y, on_false: &impl tenferro_tensor::AsView<Repr = Y::Repr>);
        options: none;
        output: <Y::Repr as tenferro_tensor::Repr>::Value;
        dtypes: "Receiver `bool`; the branches share a dtype, which is the result dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::TypedTensor;

let mut engine = CpuEngine::serial();
let c = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false])?;
let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
let z = c.where_select(&x, &y, &mut engine)?;
assert_eq!(z.host_data()?, &[1.0, 4.0]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op dot_general {
        summary: "General contraction with batch dimensions (`DotGeneralConfig`); `DotGeneralOptions` conjugates an operand.";
        operands: (a, b, config: tenferro_tensor::DotGeneralConfig);
        options: tenferro_tensor::DotGeneralOptions;
        output: V;
        dtypes: "Both operands share a numeric dtype; the result has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: ["traced: no `dot_general_with`; conjugate an operand with `conj()` first"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{DotGeneralConfig, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let cfg = DotGeneralConfig {
    lhs_contracting_dims: [1usize].into_iter().collect(),
    rhs_contracting_dims: [0usize].into_iter().collect(),
    lhs_batch_dims: Default::default(),
    rhs_batch_dims: Default::default(),
};
let y = a.dot_general(&a, cfg, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reduce_sum {
        summary: "Sum over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.";
        operands: (a, axes: Option<&[isize]>);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reduce_sum(Some(&[-1]), &mut engine)?;
assert_eq!(y.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reduce_max {
        summary: "Maximum over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.";
        operands: (a, axes: Option<&[isize]>);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reduce_max(Some(&[-1]), &mut engine)?;
assert_eq!(y.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reduce_min {
        summary: "Minimum over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.";
        operands: (a, axes: Option<&[isize]>);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reduce_min(Some(&[-1]), &mut engine)?;
assert_eq!(y.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reduce_prod {
        summary: "Product over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.";
        operands: (a, axes: Option<&[isize]>);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reduce_prod(Some(&[-1]), &mut engine)?;
assert_eq!(y.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reduce_sum_squares {
        summary: "Sum of squared magnitudes over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.";
        operands: (a, axes: Option<&[isize]>);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reduce_sum_squares(Some(&[-1]), &mut engine)?;
assert_eq!(y.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reduce_mean {
        summary: "Mean over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.";
        operands: (a, axes: Option<&[isize]>);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reduce_mean(Some(&[-1]), &mut engine)?;
assert_eq!(y.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reshape {
        summary: "Reshape to `shape` in column-major element order.";
        operands: (a, shape: &[usize]);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel. A view that is not compact in column-major order is gathered by the engine; reshape is a data operation, not a metadata query.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reshape(&[4], &mut engine)?;
assert_eq!(y.shape(), &[4]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op broadcast_in_dim {
        summary: "Broadcast to `shape`; `dims[i]` is the output axis of input axis `i` (an IR-level dimension map, unsigned like the `*Config` structs).";
        operands: (a, shape: &[usize], dims: &[usize]);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.broadcast_in_dim(&[2, 2, 3], &[0, 1], &mut engine)?;
assert_eq!(y.shape(), &[2, 2, 3]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op transpose {
        summary: "Permute axes; `perm` is signed (`-1` is the last axis).";
        operands: (a, perm: &[isize]);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.transpose(&[1, 0], &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op reverse {
        summary: "Reverse the order of elements along `axes` (signed).";
        operands: (a, axes: &[isize]);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.reverse(&[-1], &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op tril {
        summary: "Lower triangle of the two leading axes relative to diagonal `k`.";
        operands: (a, k: i64);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.tril(0, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op triu {
        summary: "Upper triangle of the two leading axes relative to diagonal `k`.";
        operands: (a, k: i64);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.triu(0, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op extract_diag {
        summary: "Extract the diagonal of axes `axis_a` and `axis_b` (signed).";
        operands: (a, axis_a: isize, axis_b: isize);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.extract_diag(0, 1, &mut engine)?;
assert_eq!(y.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op embed_diag {
        summary: "Embed axis `axis_a` of the receiver as the diagonal of a new axis pair; `axis_b` (signed, against the output rank) is the position of the inserted axis.";
        operands: (a, axis_a: isize, axis_b: isize);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.embed_diag(0, -1, &mut engine)?;
assert_eq!(y.shape(), &[2, 2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op slice {
        summary: "Static strided slice (`SliceConfig`).";
        operands: (a, config: tenferro_tensor::SliceConfig);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{SliceConfig, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.slice(SliceConfig { starts: vec![0, 0], limits: vec![1, 2], strides: vec![1, 1] }, &mut engine)?;
assert_eq!(y.shape(), &[1, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op pad {
        summary: "Pad with edge and interior padding (`PadConfig`).";
        operands: (a, config: tenferro_tensor::PadConfig);
        options: none;
        output: V;
        dtypes: "Any numeric dtype the engine supports for the operation; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{PadConfig, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.pad(PadConfig { edge_padding_low: vec![1, 0], edge_padding_high: vec![0, 0], interior_padding: vec![0, 0] }, &mut engine)?;
assert_eq!(y.shape(), &[3, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op gather {
        summary: "XLA-style gather (`GatherConfig`); `indices` is an integer tensor of any representation.";
        operands: (a, indices: &impl tenferro_tensor::AsView, config: tenferro_tensor::GatherConfig);
        options: none;
        output: V;
        dtypes: "Operand: any dtype; indices: an integer dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_runtime::CoreOps;

// See the `gather` tests for a complete configuration.
fn _uses_gather<X: CoreOps>() {}"#;
    }
    op scatter {
        summary: "XLA-style scatter (`ScatterConfig`); `indices` and `updates` may have any representation.";
        operands: (a, indices: &impl tenferro_tensor::AsView, updates: &impl tenferro_tensor::AsView, config: tenferro_tensor::ScatterConfig);
        options: none;
        output: V;
        dtypes: "Operand and updates share a dtype; indices: an integer dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_runtime::CoreOps;

// See the `scatter` tests for a complete configuration.
fn _uses_scatter<X: CoreOps>() {}"#;
    }
    op dynamic_slice {
        summary: "Slice of `sizes` starting at the run-time `starts` (an integer vector).";
        operands: (a, starts: &impl tenferro_tensor::AsView, sizes: &[usize]);
        options: none;
        output: V;
        dtypes: "Operand: any dtype; starts: an integer dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let starts = Tensor::from_vec_col_major(vec![2], vec![1_i64, 0])?;
let y = a.dynamic_slice(&starts, &[1, 2], &mut engine)?;
assert_eq!(y.shape(), &[1, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op dynamic_update_slice {
        summary: "Copy of the receiver with `update` written at the run-time `starts`.";
        operands: (a, update, starts: &impl tenferro_tensor::AsView);
        options: none;
        output: V;
        dtypes: "Operand and update share a dtype; starts: an integer dtype.";
        layers: [concrete];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: ["traced and eager: not available"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let u = Tensor::from_vec_col_major(vec![1, 2], vec![9.0_f64, 9.0])?;
let starts = Tensor::from_vec_col_major(vec![2], vec![1_i64, 0])?;
let y = a.dynamic_update_slice(&u, &starts, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op concatenate {
        summary: "Concatenate `inputs` along `axis` (signed). Called as `CoreOps::concatenate(&[&a, &b], axis, exec)`.";
        associated;
        operands: (inputs: &[&Self], axis: isize);
        options: none;
        output: V;
        dtypes: "All inputs share a dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = Tensor::concatenate(&[&a, &a], -1, &mut engine)?;
assert_eq!(y.shape(), &[2, 4]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op convert {
        summary: "Checked dtype conversion along tenferro's promotion lattice. The result is a dtype-erased `Tensor` because its dtype is a run-time value.";
        operands: (a, to: tenferro_tensor::DType);
        options: none;
        output: tenferro_tensor::Tensor;
        dtypes: "Any source dtype the conversion supports; the result has dtype `to`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "An owned `Tensor` receiver is converted directly. Any other receiver (a borrowed view or a typed tensor) is first gathered once into a compact tensor of the same placement and dtype, because the engine conversion reads an owned tensor.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{DType, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.convert(DType::C64, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op cast {
        summary: "Explicit, possibly lossy dtype projection. The result is a dtype-erased `Tensor` because its dtype is a run-time value.";
        operands: (a, to: tenferro_tensor::DType);
        options: none;
        output: tenferro_tensor::Tensor;
        dtypes: "Any source dtype the conversion supports; the result has dtype `to`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "An owned `Tensor` receiver is converted directly. Any other receiver (a borrowed view or a typed tensor) is first gathered once into a compact tensor of the same placement and dtype, because the engine conversion reads an owned tensor.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{DType, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.cast(DType::F32, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op scale_real {
        summary: "Multiply by a real scalar.";
        operands: (a, factor: f64);
        options: none;
        output: V;
        dtypes: "f32, f64, Complex32 or Complex64; the result has the input dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.scale_real(2.0, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op scale_complex {
        summary: "Multiply by a complex scalar.";
        operands: (a, factor: num_complex::Complex64);
        options: none;
        output: V;
        dtypes: "Complex32 or Complex64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)])?;
let y = a.scale_complex(Complex64::new(0.0, 1.0), &mut engine)?;
assert_eq!(y.shape(), &[1]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op softmax {
        summary: "Softmax along `axis` (signed).";
        operands: (a, axis: isize);
        options: none;
        output: V;
        dtypes: "f32 or f64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.softmax(-1, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op log_softmax {
        summary: "Log-softmax along `axis` (signed).";
        operands: (a, axis: isize);
        options: none;
        output: V;
        dtypes: "f32 or f64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.log_softmax(-1, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op masked_softmax {
        summary: "Softmax along `axis` over the entries where `mask` is true.";
        operands: (a, mask: &impl tenferro_tensor::AsView, axis: isize);
        options: none;
        output: V;
        dtypes: "Input f32 or f64; mask `bool` of any representation.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let m = Tensor::from_vec_col_major(vec![2, 2], vec![true, true, false, true])?;
let y = a.masked_softmax(&m, -1, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op masked_log_softmax {
        summary: "Log-softmax along `axis` over the entries where `mask` is true.";
        operands: (a, mask: &impl tenferro_tensor::AsView, axis: isize);
        options: none;
        output: V;
        dtypes: "Input f32 or f64; mask `bool` of any representation.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let m = Tensor::from_vec_col_major(vec![2, 2], vec![true, true, false, true])?;
let y = a.masked_log_softmax(&m, -1, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op take_along_axis {
        summary: "Gather along `axis` (signed) with per-position integer `indices`.";
        operands: (a, indices: &impl tenferro_tensor::AsView, axis: isize);
        options: none;
        output: V;
        dtypes: "Input: any dtype; indices: an integer dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let i = Tensor::from_vec_col_major(vec![2, 1], vec![1_i64, 0])?;
let y = a.take_along_axis(&i, -1, &mut engine)?;
assert_eq!(y.shape(), &[2, 1]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op layer_norm {
        summary: "Layer normalization over `axis` (signed) with optional affine `weight` and `bias`.";
        operands: (a, axis: isize, weight: Option<&Self>, bias: Option<&Self>, eps: f64);
        options: none;
        output: V;
        dtypes: "f32 or f64; weight and bias share the input representation.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.layer_norm(-1, None, None, 1e-5, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op rms_norm {
        summary: "RMS normalization over `axis` (signed) with optional affine `weight` and `bias`.";
        operands: (a, axis: isize, weight: Option<&Self>, bias: Option<&Self>, eps: f64);
        options: none;
        output: V;
        dtypes: "f32 or f64; weight and bias share the input representation.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.";
        errors: [Validation, Unsupported];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = a.rms_norm(-1, None, None, 1e-5, &mut engine)?;
assert_eq!(y.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
    }
    op select {
        summary: "Alias of `where_select` with the condition as the first argument.";
        operands: (condition, on_true, on_false);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [traced, eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete: use `where_select`"];
    }
    op stack {
        summary: "Stack equally shaped tensors along a new signed axis.";
        operands: (inputs: &[&Self], axis: isize);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [traced, eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete: not available; use `reshape` and `concatenate`"];
    }
    op index_select {
        summary: "Select positions along a signed axis.";
        operands: (a, axis: isize, positions: &[usize]);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [traced, eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete: not available; use `gather`"];
    }
    op slice_axis {
        summary: "Slice a range along a signed axis.";
        operands: (a, axis: isize, range: Range<usize>);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [traced, eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete: not available; use `slice`"];
    }
    op take_axis {
        summary: "Take the listed positions along a signed axis.";
        operands: (a, axis: isize, indices: &[usize]);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [traced, eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete: not available; use `gather`"];
    }
    op take_rows {
        summary: "Take the listed rows of a matrix.";
        operands: (a, indices: &[usize]);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete and traced: not available"];
    }
    op take_cols {
        summary: "Take the listed columns of a matrix.";
        operands: (a, indices: &[usize]);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete and traced: not available"];
    }
    op take_block {
        summary: "Take a block given row and column positions.";
        operands: (a, rows: &[usize], cols: &[usize]);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete and traced: not available"];
    }
    op dynamic_truncate {
        summary: "Truncate a signed axis to a run-time size.";
        operands: (a, size, axis: isize);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [traced];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete and eager: not available"];
    }
    op pad_to_match {
        summary: "Zero-pad a signed axis to the size of a reference tensor.";
        operands: (a, reference, axis: isize);
        options: none;
        output: V;
        dtypes: "See the method documentation.";
        layers: [traced];
        into: [];
        layout: "Recorded as graph operations (traced) or executed by the eager engine.";
        errors: [Validation, Unsupported];
        omissions: ["concrete and eager: not available"];
    }

    exceptions {
        E03 {
            symbols: [
                "tenferro_einsum::ConcreteEinsumPlan::clear_cpu_cache",
                "tenferro_einsum::ConcreteEinsumPlan::cpu_cache_stats"
            ];
            scope: "plan-cache owner methods of the concrete einsum plan (clearing and inspecting the cache)";
            category: "cache owner | inherent method";
            owner_module: "tenferro_einsum::concrete";
            reason: "the plan owns its CPU numerical descriptor cache; a cache owner exposes its cache, and clearing or inspecting it is not a tensor operation and has no options, inputs or destination";
            verification: "the einsum registry tests, the cache tests and the doctests of both methods; the plan's execute/execute_into follow the grammar (inputs, exec, dest)";
            exit_condition: "kept while ConcreteEinsumPlan owns a CPU cache";
            status: retained;
        }
        E04 {
            symbols: [
                "tenferro_runtime::TraceContext",
                "tenferro_runtime::GraphCompiler::compile",
                "tenferro_runtime::GraphCompiler::compile_many",
                "tenferro_runtime::Runtime::prepare",
                "tenferro_runtime::Runtime::run",
                "tenferro_runtime::TraceContext::finish"
            ];
            scope: "trace, compile, prepare and run lifecycle";
            category: "lifecycle | builder";
            owner_module: "tenferro_runtime::{trace, graph, runtime}";
            reason: "program construction and execution are not tensor operations; their shapes are fixed by the public API snapshot (api/)";
            verification: "scripts/api-snapshot.sh --check and the runtime lifecycle tests";
            exit_condition: "kept";
            status: retained;
        }
        E05 {
            symbols: [
                "tenferro_tensor::Tensor::shape",
                "tenferro_tensor::Tensor::dtype",
                "tenferro_tensor::TypedTensor::as_view",
                "tenferro_tensor::TypedTensorView::transpose_view",
                "tenferro_tensor::TypedTensorView::slice_view",
                "tenferro_runtime::TracedTensor::rank",
                "tenferro_runtime::TracedTensor::sym_size",
                "tenferro_runtime::TracedTensor::axis_sym_dim",
                "tenferro_runtime::TracedTensor::shape_of"
            ];
            scope: "descriptor-only operations and metadata queries";
            category: "metadata | view";
            owner_module: "tenferro_tensor, tenferro_runtime::traced";
            reason: "they never touch element data, so they take no engine and are not operations of the grammar";
            verification: "tenferro-runtime `metadata_queries_and_views_do_not_copy` (pointer identity)";
            exit_condition: "kept";
            status: retained;
        }
        E06 {
            symbols: [
                "tenferro_cpu::with_borrowed_cpu_engine",
                "tenferro_tensor::Dest::into_accumulation",
                "num_complex::Complex"
            ];
            scope: "approved external types and owner-scoped doc-hidden bridges";
            category: "bridge | external type";
            owner_module: "tenferro_cpu, tenferro_tensor::dest";
            reason: "the lane bridge lets tenferro-linalg reach the lane engine (#2552 S2b); into_accumulation maps a Dest to the backend accumulation descriptor (#2552 Q3); num_complex is the public complex scalar";
            verification: "listed symbol by symbol in api/allowlist.txt; the bridges are #[doc(hidden)]";
            exit_condition: "kept while the owners need the bridge";
            status: retained;
        }
        E09 {
            symbols: [
                "tenferro_einsum::Error",
                "tenferro_einsum::PlanningError",
                "tenferro_einsum::parse_einsum_subscripts",
                "tenferro_einsum::parse_einsum_notation",
                "tenferro_einsum::ContractionTree::optimize",
                "tenferro_einsum::ContractionTree::optimize_with_options",
                "tenferro_einsum::ContractionTree::from_pairs",
                "tenferro_linalg::Error",
                "tenferro_fft::EagerFftInPlaceError"
            ];
            scope: "domain errors of parsers and planners (einsum parser and contraction planner), and failures that return ownership";
            category: "error | domain";
            owner_module: "the owning family crates";
            reason: "parsers and planners report einsum-domain errors; the grammar's operations wrap them (concrete: tenferro_tensor::Error::Extension, traced/eager: tenferro_runtime::Error::Extension) as the `source()` of the layer error";
            verification: "the error-cause tests of each family; the einsum error-cause tests reach tenferro_einsum::Error::InvalidSubscripts through source() on all three layers";
            exit_condition: "kept";
            status: retained;
        }
        E10 {
            symbols: [
                "tenferro_tensor::Engine",
                "tenferro_tensor::TensorDot",
                "tenferro_tensor::TensorDot::dot_general_read_into_accum",
                "tenferro_tensor::Engine::axpby_read_into_accum",
                "tenferro_linalg::LinalgBackend",
                "tenferro_fft::FftBackend"
            ];
            scope: "backend and provider traits and their read/accumulate hooks";
            category: "provider contract | trait";
            owner_module: "tenferro_tensor::backend and the family backends";
            reason: "a provider-facing contract, not the user grammar; `*_read` and `*_read_into_accum` name backend hooks";
            verification: "the backend contract tests of each crate";
            exit_condition: "kept";
            status: retained;
        }
    }
}
