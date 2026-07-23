use std::sync::Arc;

use tenferro_runtime::{
    DType, DotGeneralConfig, GatherConfig, GraphCompiler, GraphExecutor, PadConfig, ScatterConfig,
    SliceConfig, Tensor, TensorRead, TracedTensor, TypedTensor,
};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, Buffer, BufferHandle, CompareDir,
    MemoryKind, Placement, TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};

#[derive(Default)]
struct UploadRejectingBackend;

macro_rules! unreachable_backend_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                $(let _ = &$arg;)*
                panic!(concat!(stringify!($name), " should not be called by this test"))
            }
        )+
    };
}

impl TensorDeviceTransfer for UploadRejectingBackend {
    fn upload_host_tensor(&mut self, tensor: &Tensor) -> tenferro_tensor::Result<Tensor> {
        let _ = tensor;
        Err(tenferro_tensor::Error::backend_failure(
            "upload_host_tensor",
            "default scalar inputs must route through backend upload",
        ))
    }
}

impl BackendRuntimeCache for UploadRejectingBackend {
    type RuntimeCache = ();
}

impl TensorElementwise for UploadRejectingBackend {
    unreachable_backend_methods! {
        add(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sub(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        mul(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        neg(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        conj(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        abs(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sign(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> tenferro_tensor::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> tenferro_tensor::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorAnalytic for UploadRejectingBackend {
    unreachable_backend_methods! {
        exp(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        log(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sin(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        cos(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        tanh(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        rsqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        pow(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        expm1(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        log1p(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorStructural for UploadRejectingBackend {
    unreachable_backend_methods! {
        transpose(input: &Tensor, perm: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> tenferro_tensor::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor>;
        cast(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorReduction for UploadRejectingBackend {
    unreachable_backend_methods! {
        reduce_sum(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_prod(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_max(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_min(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorIndexing for UploadRejectingBackend {
    unreachable_backend_methods! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> tenferro_tensor::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> tenferro_tensor::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> tenferro_tensor::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> tenferro_tensor::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> tenferro_tensor::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> tenferro_tensor::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorDot for UploadRejectingBackend {
    unreachable_backend_methods! {
        dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorFusion for UploadRejectingBackend {}
impl TensorBuffer for UploadRejectingBackend {}
impl BackendCachedDot for UploadRejectingBackend {}
impl BackendSessionHost for UploadRejectingBackend {}
impl TensorBackend for UploadRejectingBackend {}

fn scalar_default_program() -> tenferro_runtime::CompiledGraph {
    let scalar = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(),
    )
    .unwrap();
    GraphCompiler::new().compile(&scalar).unwrap()
}

#[test]
fn scalar_default_input_is_uploaded_for_owned_execution() {
    let program = scalar_default_program();
    let err = GraphExecutor::new(UploadRejectingBackend)
        .run_many(&program)
        .unwrap_err();

    assert!(err.to_string().contains("upload_host_tensor"));
}

#[test]
fn scalar_default_input_is_uploaded_for_borrowed_execution() {
    let program = scalar_default_program();
    let err = GraphExecutor::new(UploadRejectingBackend)
        .run_many_with_input_reads(&program, &[])
        .unwrap_err();

    assert!(err.to_string().contains("upload_host_tensor"));
}

#[test]
fn explicit_scalar_binding_is_not_auto_uploaded() {
    let scalar = TracedTensor::input_symbolic_shape(DType::F64, 0).unwrap();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&scalar, &[(&scalar, DType::F64, &[])])
        .unwrap();
    let bound = Tensor::from_vec_col_major(vec![], vec![7.0_f64]).unwrap();

    let owned = GraphExecutor::new(UploadRejectingBackend)
        .run_many_with_inputs(&program, &[&bound])
        .unwrap();
    assert_eq!(owned[0].as_slice::<f64>().unwrap(), &[7.0]);

    let borrowed = GraphExecutor::new(UploadRejectingBackend)
        .run_many_with_input_reads(&program, &[TensorRead::from_tensor(&bound)])
        .unwrap();
    assert_eq!(borrowed[0].as_slice::<f64>().unwrap(), &[7.0]);
}

#[test]
fn non_scalar_default_input_is_not_auto_uploaded() {
    let vector = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
    )
    .unwrap();
    let program = GraphCompiler::new().compile(&vector).unwrap();

    let owned = GraphExecutor::new(UploadRejectingBackend)
        .run_many(&program)
        .unwrap();
    assert_eq!(owned[0].as_slice::<f64>().unwrap(), &[3.0]);

    let borrowed = GraphExecutor::new(UploadRejectingBackend)
        .run_many_with_input_reads(&program, &[])
        .unwrap();
    assert_eq!(borrowed[0].as_slice::<f64>().unwrap(), &[3.0]);
}

#[test]
fn backend_resident_scalar_default_input_is_not_reuploaded() {
    let tensor = Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![],
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(1, 1))),
            Placement {
                memory_kind: MemoryKind::Other("test-backend".to_string()),
                device: None,
                cpu_affinity: None,
            },
        )
        .unwrap(),
    );
    let scalar = TracedTensor::from_tensor_concrete_shape(tensor).unwrap();
    let program = GraphCompiler::new().compile(&scalar).unwrap();

    let owned = GraphExecutor::new(UploadRejectingBackend)
        .run_many(&program)
        .unwrap();
    assert_eq!(owned[0].shape(), &[]);
    assert_eq!(owned[0].dtype(), DType::F64);

    let borrowed = GraphExecutor::new(UploadRejectingBackend)
        .run_many_with_input_reads(&program, &[])
        .unwrap();
    assert_eq!(borrowed[0].shape(), &[]);
    assert_eq!(borrowed[0].dtype(), DType::F64);
}
