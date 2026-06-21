use std::any::Any;
use std::sync::Arc;

use tenferro_runtime::extension::{apply, ExtensionOp};
use tenferro_runtime::{DType, GraphCompiler, SymDim, Tensor, TracedTensor};
use tenferro_xla::{lower_to_stablehlo, Error};

#[derive(Clone, Debug)]
struct RuntimeOnlyExtension;

impl ExtensionOp for RuntimeOnlyExtension {
    fn family_id(&self) -> &'static str {
        "test.runtime_only.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
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
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(input_dtypes[0], input_shapes[0].to_vec())]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}

#[test]
fn rejects_i64_dtype_before_emitting_mlir() {
    let x = TracedTensor::input_symbolic_shape(DType::I64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::I64, &[2])])
        .unwrap();

    let err = lower_to_stablehlo(&program).unwrap_err();

    assert!(matches!(
        err,
        Error::UnsupportedDType {
            dtype: DType::I64,
            ..
        }
    ));
}

#[test]
fn rejects_dynamic_upper_bound_extents() {
    let data = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let size = TracedTensor::input_symbolic_shape(DType::F64, 0).unwrap();
    let y = data.dynamic_truncate(&size, 0).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&data, DType::F64, &[4]), (&size, DType::F64, &[])])
        .unwrap();

    let err = lower_to_stablehlo(&program).unwrap_err();

    assert!(matches!(
        err,
        Error::NonStaticShape {
            op: "DynamicTruncate",
            kind: "an upper bound",
            ..
        }
    ));
}

#[test]
fn rejects_unsupported_static_op() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = x.maximum(&x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();

    let err = lower_to_stablehlo(&program).unwrap_err();

    assert!(matches!(err, Error::UnsupportedOp { op: "Maximum", .. }));
}

#[test]
fn rejects_extension_without_standard_op_lowering() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let outputs = apply(Arc::new(RuntimeOnlyExtension), &[&x]).unwrap();
    let y = outputs.into_iter().next().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();

    let err = lower_to_stablehlo(&program).unwrap_err();

    assert!(matches!(
        err,
        Error::UnsupportedOp {
            op: "test.runtime_only.v1",
            reason: "extension does not provide a standard-op lowering for exact static shapes"
        }
    ));
}
