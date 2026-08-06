//! Compiled documentation snippets for issue #1609.

#[rustfmt::skip]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    snippet_sparse_extension_1()?;

    // snippet source: docs/tutorials/sparse-extension.md:16
    fn snippet_sparse_extension_1() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:sparse_extension_1
use tenferro_ext_sparse::SparseCooTensor;
use tenferro_tensor::Tensor;

let coords = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![0_i64, 0, 0, 1, 1, 0],
)?;
let values = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 1.0, 3.0])?;
let sparse = SparseCooTensor::from_parts(vec![2, 2], coords, values)?;
        // snippet-end:sparse_extension_1
        Ok(())
    }

    snippet_sparse_extension_2()?;

    // snippet source: docs/tutorials/sparse-extension.md:36
    fn snippet_sparse_extension_2() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:sparse_extension_2
use tenferro_ext_sparse::{sparse_matmul_eager, SparseCooTensor};
use tenferro_tensor::Tensor;

let left = SparseCooTensor::from_parts(
    vec![2, 2],
    Tensor::from_vec_col_major(vec![2, 3], vec![0_i64, 0, 0, 1, 1, 0])?,
    Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 1.0, 3.0])?,
)?;
let right = SparseCooTensor::from_parts(
    vec![2, 2],
    Tensor::from_vec_col_major(vec![2, 3], vec![0_i64, 0, 1, 0, 0, 1])?,
    Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 70.0, 20.0])?,
)?;
let product = sparse_matmul_eager(&left, &right)?;
        // snippet-end:sparse_extension_2
        Ok(())
    }

    snippet_sparse_extension_3()?;

    // snippet source: docs/tutorials/sparse-extension.md:62
    fn snippet_sparse_extension_3() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:sparse_extension_3
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_ext_sparse::{extension_modules, sparse_matmul, SparseCooTracedTensor};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::Tensor;

let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?;
let left = SparseCooTracedTensor::from_parts(
    vec![1, 1],
    Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?,
    TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64])?,
)?;
let right = SparseCooTracedTensor::from_parts(
    vec![1, 1],
    coords,
    TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
)?;
let out = sparse_matmul(&left, &right)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(out.values())?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
for module in extension_modules::<CpuBackend>(runtime_engine_id()?)? {
    builder.install_extension_module(module)?;
}
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let values = outputs.remove(0);
        // snippet-end:sparse_extension_3
        Ok(())
    }

    snippet_tropical_extension_5()?;

    // snippet source: docs/tutorials/tropical-extension.md:43
    fn snippet_tropical_extension_5() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tropical_extension_5
use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
use tenferro_runtime::Tensor;

let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 4.0, 0.0])?;
let b = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, -1.0, 5.0])?;
let out = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
        // snippet-end:tropical_extension_5
        Ok(())
    }

    snippet_tropical_extension_6()?;

    // snippet source: docs/tutorials/tropical-extension.md:55
    fn snippet_tropical_extension_6() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tropical_extension_6
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_ext_tropical::{extension_modules, traced::tropical_dot_general_fused};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 4.0, 0.0])?;
let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, -1.0, 5.0])?;
let out = tropical_dot_general_fused(&a, &b)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&out)?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
for module in extension_modules::<CpuBackend>(runtime_engine_id()?)? {
    builder.install_extension_module(module)?;
}
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let value = outputs.remove(0);
        // snippet-end:tropical_extension_6
        Ok(())
    }

    Ok(())
}
