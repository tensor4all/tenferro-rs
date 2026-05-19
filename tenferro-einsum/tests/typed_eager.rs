use tenferro_einsum::typed_eager_einsum;
use tenferro_tensor::{
    cpu::CpuBackend, CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, Tensor, TensorBackend, TypedTensor,
};

#[derive(Default)]
struct WrongDTypeBackend;

macro_rules! panic_backend_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                let _ = ($($arg),*);
                panic!(concat!(stringify!($name), " should not be called in this test"))
            }
        )+
    };
}

impl TensorBackend for WrongDTypeBackend {
    type RuntimeCache = ();

    panic_backend_methods! {
        add(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        mul(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        neg(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        abs(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sign(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> tenferro_tensor::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> tenferro_tensor::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> tenferro_tensor::Result<Tensor>;
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
        transpose(input: &Tensor, perm: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> tenferro_tensor::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor>;
        convert(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        reduce_sum(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_prod(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_max(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_min(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> tenferro_tensor::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> tenferro_tensor::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> tenferro_tensor::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> tenferro_tensor::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> tenferro_tensor::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> tenferro_tensor::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        cholesky(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        triangular_solve(a: &Tensor, b: &Tensor, left_side: bool, lower: bool, transpose_a: bool, unit_diagonal: bool) -> tenferro_tensor::Result<Tensor>;
        lu(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
        full_piv_lu(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
        full_piv_lu_solve(a: &Tensor, b: &Tensor, transpose_a: bool) -> tenferro_tensor::Result<Tensor>;
        svd(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
        qr(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
        eigh(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
        eig(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
        solve(a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor>;
    }

    fn dot_general(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        Ok(Tensor::F64(TypedTensor::from_vec(
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )))
    }

    fn conj(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        CpuBackend::new().conj(input)
    }
}

#[test]
fn typed_einsum_f64() {
    let mut ctx = CpuBackend::new();
    let lhs = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = TypedTensor::<f64>::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = typed_eager_einsum(&mut ctx, &[&lhs, &rhs], "ij,jk->ik").unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.as_slice(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn typed_einsum_f64_three_operands() {
    let mut ctx = CpuBackend::new();
    let a = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TypedTensor::<f64>::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = TypedTensor::<f64>::from_vec(vec![2, 2], vec![2.0, 0.0, 1.0, 3.0]);

    let result = typed_eager_einsum(&mut ctx, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.as_slice(), &[152.0, 200.0, 385.0, 508.0]);
}

#[test]
fn typed_einsum_reports_dtype_mismatch_from_backend_result() {
    let mut ctx = WrongDTypeBackend;
    let lhs = TypedTensor::<f32>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = TypedTensor::<f32>::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let err = typed_eager_einsum(&mut ctx, &[&lhs, &rhs], "ij,jk->ik").unwrap_err();

    assert!(matches!(
        err,
        tenferro_tensor::Error::DTypeMismatch {
            op: "typed_eager_einsum",
            lhs: DType::F64,
            rhs: DType::F32,
        }
    ));
}

#[test]
fn tensor_backend_default_cached_methods_delegate_to_backend_ops() {
    let lhs = Tensor::F64(TypedTensor::from_vec(vec![1, 1], vec![1.0]));
    let rhs = Tensor::F64(TypedTensor::from_vec(vec![1, 1], vec![3.0]));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let mut backend = WrongDTypeBackend;
    let mut cache = ();

    let direct =
        TensorBackend::dot_general_cached(&mut backend, &mut cache, Some(7), &lhs, &rhs, &config)
            .unwrap();
    assert_eq!(direct.shape(), &[2, 2]);

    let folded =
        TensorBackend::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, true)
            .unwrap();
    assert_eq!(folded.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let value = TensorBackend::with_exec_session_cached(&mut backend, &mut cache, |exec| {
        let cached = exec
            .dot_general_cached(Some(3), &lhs, &rhs, &config)
            .unwrap();
        let folded = exec
            .dot_general_with_conj_cached(Some(5), &lhs, &rhs, &config, false, false)
            .unwrap();
        cached.shape().len() + folded.shape().len()
    });
    assert_eq!(value, 4);
}
