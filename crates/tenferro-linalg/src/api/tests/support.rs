//! Shared helpers of the registry tests: sample inputs, bit-level flattening
//! of results, and the traced-graph runner.

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, Tensor, TensorScalar, TypedTensor};

use crate::{Eig, Eigh, FullPivLu, LinalgScalar, Lu, Qr, RankRevealingQrResult, Slogdet, Svd};

pub(super) type TestResult<T> = Result<T, Box<dyn std::error::Error>>;
/// Bits of the plain call and bits of the `_with(default options)` call.
pub(super) type Pair = (Vec<Vec<u64>>, Vec<Vec<u64>>);

/// A scalar the tests build inputs for: `f64` (imaginary parts are dropped) or
/// `Complex64`.
pub(super) trait Sample: LinalgScalar + TensorScalar + Copy {
    fn mk(re: f64, im: f64) -> Self;
}

impl Sample for f64 {
    fn mk(re: f64, _im: f64) -> Self {
        re
    }
}

impl Sample for Complex64 {
    fn mk(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

fn data<T: Sample>(shape: &[usize], entries: &[(f64, f64)]) -> (Vec<usize>, Vec<T>) {
    (
        shape.to_vec(),
        entries.iter().map(|&(re, im)| T::mk(re, im)).collect(),
    )
}

/// A Hermitian positive-definite 2 x 2 matrix.
pub(super) fn sq<T: Sample>() -> (Vec<usize>, Vec<T>) {
    data(&[2, 2], &[(4.0, 0.0), (2.0, -1.0), (2.0, 1.0), (3.0, 0.0)])
}

/// A full-column-rank 3 x 2 matrix.
pub(super) fn tall<T: Sample>() -> (Vec<usize>, Vec<T>) {
    data(
        &[3, 2],
        &[
            (1.0, 0.5),
            (2.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (3.0, 0.5),
        ],
    )
}

pub(super) fn rhs<T: Sample>() -> (Vec<usize>, Vec<T>) {
    data(&[2, 1], &[(1.0, 0.0), (2.0, 1.0)])
}

/// A 3 x 1 right-hand side.
pub(super) fn rhs3<T: Sample>() -> (Vec<usize>, Vec<T>) {
    data(&[3, 1], &[(1.0, 0.0), (-2.0, 0.5), (0.5, 1.0)])
}

pub(super) fn rhs_tall<T: Sample>() -> (Vec<usize>, Vec<T>) {
    data(&[3, 1], &[(1.0, 0.0), (2.0, 0.0), (3.0, 1.0)])
}

pub(super) fn tensor<T: Sample>((shape, values): (Vec<usize>, Vec<T>)) -> TestResult<Tensor> {
    Ok(Tensor::from_vec_col_major(shape, values)?)
}

/// Like [`tensor`] with the tensor crate's error, for `?` inside eager sessions.
#[allow(dead_code)]
pub(super) fn raw<T: Sample>(
    (shape, values): (Vec<usize>, Vec<T>),
) -> tenferro_tensor::Result<Tensor> {
    Tensor::from_vec_col_major(shape, values)
}

pub(super) fn typed<T: Sample>(
    (shape, values): (Vec<usize>, Vec<T>),
) -> TestResult<TypedTensor<T>> {
    Ok(TypedTensor::from_vec_col_major(shape, values)?)
}

/// Shape, dtype tag and every element as raw bits.
pub(super) fn tensor_bits(t: &Tensor) -> TestResult<Vec<u64>> {
    let mut out: Vec<u64> = t.shape().iter().map(|&d| d as u64).collect();
    out.push(u64::MAX);
    match t.dtype() {
        DType::F64 => out.extend(t.as_slice::<f64>()?.iter().map(|x| x.to_bits())),
        DType::F32 => out.extend(t.as_slice::<f32>()?.iter().map(|x| u64::from(x.to_bits()))),
        DType::C64 => out.extend(
            t.as_slice::<Complex64>()?
                .iter()
                .flat_map(|x| [x.re.to_bits(), x.im.to_bits()]),
        ),
        DType::I64 => out.extend(t.as_slice::<i64>()?.iter().map(|&x| x as u64)),
        other => return Err(format!("unexpected result dtype {other:?}").into()),
    }
    Ok(out)
}

/// Results flattened into one bit vector per leaf tensor.
pub(super) trait Bits {
    fn bits(&self) -> TestResult<Vec<Vec<u64>>>;
}

impl Bits for Tensor {
    fn bits(&self) -> TestResult<Vec<Vec<u64>>> {
        Ok(vec![tensor_bits(self)?])
    }
}

#[cfg(feature = "autodiff")]
impl Bits for tenferro_ad::EagerTensor {
    fn bits(&self) -> TestResult<Vec<Vec<u64>>> {
        Ok(vec![tensor_bits(
            &self.to_tensor(&mut CpuEngine::serial())?,
        )?])
    }
}

macro_rules! struct_bits {
    ($ty:ident < $($g:ident),+ > { $($f:ident),+ }) => {
        impl<$($g: Bits),+> Bits for $ty<$($g),+> {
            fn bits(&self) -> TestResult<Vec<Vec<u64>>> {
                let mut out = Vec::new();
                $( out.extend(self.$f.bits()?); )+
                Ok(out)
            }
        }
    };
}

struct_bits!(Svd<V, R> { u, s, vh });
struct_bits!(Qr<V> { q, r });
struct_bits!(Lu<V, R> { p, l, u, parity });
struct_bits!(FullPivLu<V, R> { p, l, u, q, parity });
struct_bits!(Eig<C> { values, vectors });
struct_bits!(Eigh<V, R> { values, vectors });
struct_bits!(Slogdet<V, R> { sign, log_abs_det });
struct_bits!(RankRevealingQrResult<V, M> { q, r, column_permutation, rank });

/// Traced results flattened into their leaf tensors, in field order.
pub(super) trait Leaves {
    fn leaves(&self) -> Vec<TracedTensor>;
}

impl Leaves for TracedTensor {
    fn leaves(&self) -> Vec<TracedTensor> {
        vec![self.clone()]
    }
}

macro_rules! struct_leaves {
    ($ty:ty { $($f:ident),+ }) => {
        impl Leaves for $ty {
            fn leaves(&self) -> Vec<TracedTensor> {
                vec![$(self.$f.clone()),+]
            }
        }
    };
}

struct_leaves!(Svd<TracedTensor, TracedTensor> { u, s, vh });
struct_leaves!(Qr<TracedTensor> { q, r });
struct_leaves!(Lu<TracedTensor, TracedTensor> { p, l, u, parity });
struct_leaves!(FullPivLu<TracedTensor, TracedTensor> { p, l, u, q, parity });
struct_leaves!(Eig<TracedTensor> { values, vectors });
struct_leaves!(Eigh<TracedTensor, TracedTensor> { values, vectors });
struct_leaves!(Slogdet<TracedTensor, TracedTensor> { sign, log_abs_det });
struct_leaves!(RankRevealingQrResult<TracedTensor, TracedTensor> { q, r, column_permutation, rank });

/// Compiles and runs a traced graph that has no inputs; returns one tensor per
/// requested output, in order.
pub(super) fn run_traced(outputs: &[TracedTensor]) -> TestResult<Vec<Tensor>> {
    let refs: Vec<&TracedTensor> = outputs.iter().collect();
    let program = GraphCompiler::new().compile_many(&refs)?;
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    builder.install_extension_module(crate::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    let runtime = builder.build()?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let results = {
        let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
        runtime.run(&mut engines, &prepared, &[])?
    };
    results
        .into_iter()
        .map(|r| r.into_tensor(&mut engine).map_err(Into::into))
        .collect()
}

/// Runs the plain and the `_with` graphs together and returns both bit sets.
pub(super) fn traced_pair(plain: &[TracedTensor], with: &[TracedTensor]) -> TestResult<Pair> {
    let all: Vec<TracedTensor> = plain.iter().chain(with).cloned().collect();
    let tensors = run_traced(&all)?;
    let (p, w) = tensors.split_at(plain.len());
    let bits = |ts: &[Tensor]| ts.iter().map(tensor_bits).collect::<TestResult<Vec<_>>>();
    Ok((bits(p)?, bits(w)?))
}
