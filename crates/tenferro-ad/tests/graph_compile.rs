use std::num::NonZeroUsize;
use std::sync::Arc;

use tenferro_runtime::extension::{apply, ExtensionOp};
use tenferro_runtime::SymDim;
use tenferro_runtime::{CompilerOptions, OptimizerConfig};
use tenferro_runtime::{DType, GraphCompiler, TracedTensor};

#[derive(Clone)]
struct ConstantDebugExtension {
    payload: usize,
}

impl std::fmt::Debug for ConstantDebugExtension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ConstantDebugExtension")
    }
}

impl ExtensionOp for ConstantDebugExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.graph_compile_constant_debug.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        hasher.write_u64(0);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<ConstantDebugExtension>()
            .is_some_and(|other| self.payload == other.payload)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(input_dtypes[0], input_shapes[0].to_vec())])
    }
}

#[test]
fn graph_compiler_compiles_without_backend() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 1);
    assert_eq!(compiler.compile_cache_len(), 1);
}

#[test]
fn graph_compiler_validates_placeholder_specs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.input_specs()[0].shape(), &[3]);

    let err = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F32, &[3])])
        .unwrap_err();
    assert!(format!("{err}").contains("dtype"));

    let z = TracedTensor::input_concrete_shape(DType::F64, &[3]).unwrap();
    let err = compiler
        .compile_with_input_specs(&z.neg(), &[(&z, DType::F64, &[2])])
        .unwrap_err();
    assert!(format!("{err}").contains("shape"));
}

#[test]
fn graph_program_input_accessors_report_compiled_contract() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = x.neg();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[4])])
        .unwrap();
    let program = std::hint::black_box(program);
    let input = std::hint::black_box(&program.input_specs()[0]);

    assert_eq!(std::hint::black_box(&program).input_count(), 1);
    assert_eq!(std::hint::black_box(&program).output_count(), 1);
    assert_eq!(input.dtype(), DType::F64);
    assert_eq!(input.shape(), &[4]);
}

#[test]
fn graph_compiler_cache_is_bounded_and_reports_stats() {
    let mut compiler = GraphCompiler::new();
    compiler.set_compile_cache_capacity(NonZeroUsize::new(1).unwrap());

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();
    let _ = compiler.compile(&y).unwrap();
    let _ = compiler.compile(&x.neg()).unwrap();

    let stats = compiler.cache_stats();
    assert_eq!(compiler.compile_cache_capacity().get(), 1);
    assert_eq!(stats.compile.entries, 1);
    assert!(stats.compile.retained_bytes > 0);
}

#[test]
fn graph_compiler_compiler_options_setter_clears_compile_cache() {
    let mut compiler = GraphCompiler::new();
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let _ = compiler.compile(&x.neg()).unwrap();
    assert_eq!(compiler.compile_cache_len(), 1);

    let options = CompilerOptions {
        optimizer: OptimizerConfig {
            dot_decomposer: true,
            ..OptimizerConfig::default()
        },
    };
    compiler.set_compiler_options(options);

    assert_eq!(compiler.compiler_options(), options);
    assert_eq!(compiler.compile_cache_len(), 0);
}

#[test]
fn graph_compiler_cache_distinguishes_symbolic_input_shapes() {
    let mut compiler = GraphCompiler::new();
    compiler.set_compile_cache_capacity(NonZeroUsize::new(4).unwrap());

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();

    let _ = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let _ = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();

    assert_eq!(compiler.compile_cache_len(), 2);
}

#[test]
fn graph_compiler_cache_distinguishes_dtypes() {
    let mut compiler = GraphCompiler::new();
    compiler.set_compile_cache_capacity(NonZeroUsize::new(4).unwrap());

    let x_f64 = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y_f64 = (&x_f64 + &x_f64).unwrap();
    let x_f32 = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let y_f32 = (&x_f32 + &x_f32).unwrap();

    let _ = compiler.compile(&y_f64).unwrap();
    let _ = compiler.compile(&y_f32).unwrap();

    assert_eq!(compiler.compile_cache_len(), 2);
}

#[test]
fn graph_compiler_cache_distinguishes_extension_payload_eq_despite_hash_collision() {
    let mut compiler = GraphCompiler::new();
    compiler.set_compile_cache_capacity(NonZeroUsize::new(4).unwrap());

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y1 = apply(Arc::new(ConstantDebugExtension { payload: 1 }), &[&x])
        .unwrap()
        .remove(0);
    let y2 = apply(Arc::new(ConstantDebugExtension { payload: 2 }), &[&x])
        .unwrap()
        .remove(0);

    let _ = compiler.compile(&y1).unwrap();
    let _ = compiler.compile(&y2).unwrap();

    assert_eq!(compiler.compile_cache_len(), 2);
}

#[test]
fn graph_compiler_compile_many_returns_multi_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();
    let z = x.neg();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&y, &z]).unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 2);
}
