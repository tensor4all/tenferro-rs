use tenferro_tensor::{
    cpu::CpuBackend, BackendCachedDot, BackendRuntimeCache, BackendSessionHost, CompareDir, DType,
    DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor, TensorAnalytic,
    TensorBackend, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorRead, TensorReduction, TensorStructural, TensorView, TypedTensor,
};

use crate::eager::{
    eager_einsum, eager_einsum_owned, eager_einsum_owned_subscripts, eager_einsum_read_subscripts,
    eager_einsum_subscripts,
};
use crate::typed_eager::typed_eager_einsum;
use crate::Subscripts;

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

impl BackendRuntimeCache for WrongDTypeBackend {
    type RuntimeCache = ();
}

impl TensorElementwise for WrongDTypeBackend {
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
    }

    fn conj(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        CpuBackend::new().conj(input)
    }
}

impl TensorAnalytic for WrongDTypeBackend {
    panic_backend_methods! {
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

impl TensorStructural for WrongDTypeBackend {
    panic_backend_methods! {
        transpose(input: &Tensor, perm: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> tenferro_tensor::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor>;
        convert(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorReduction for WrongDTypeBackend {
    panic_backend_methods! {
        reduce_sum(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_prod(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_max(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_min(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorIndexing for WrongDTypeBackend {
    panic_backend_methods! {
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

impl TensorDot for WrongDTypeBackend {
    fn dot_general(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        Ok(Tensor::F64(TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )))
    }
}

impl BackendCachedDot for WrongDTypeBackend {}

impl BackendSessionHost for WrongDTypeBackend {}

impl TensorDeviceTransfer for WrongDTypeBackend {}

impl TensorBuffer for WrongDTypeBackend {}

impl TensorFusion for WrongDTypeBackend {}

impl TensorBackend for WrongDTypeBackend {}

#[test]
fn typed_einsum_f64() {
    unsafe {
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_AGG", "1");
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_PRINT_EVERY", "1");
    }
    let mut ctx = CpuBackend::new();
    let lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = typed_eager_einsum(&mut ctx, &[&lhs, &rhs], "ij,jk->ik").unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.as_slice(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn eager_einsum_subscripts_and_read_views_use_integer_api() {
    unsafe {
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_AGG", "1");
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_PRINT_EVERY", "1");
    }
    let mut ctx = CpuBackend::new();
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs_shape = [3usize, 2];
    let rhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = Tensor::from_vec_col_major(rhs_shape.to_vec(), rhs_data.to_vec());
    let subscripts = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);

    let borrowed = eager_einsum_subscripts(&mut ctx, &[&lhs, &rhs], &subscripts).unwrap();
    let read = eager_einsum_read_subscripts(
        &mut ctx,
        &[
            TensorRead::from_tensor(&lhs),
            TensorRead::from_view(TensorView::f64(&rhs_shape, &rhs_data).unwrap()),
        ],
        &subscripts,
    )
    .unwrap();

    assert_eq!(
        borrowed.as_slice::<f64>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0]
    );
    assert_eq!(read.as_slice::<f64>(), borrowed.as_slice::<f64>());
}

#[test]
fn eager_einsum_owned_matches_borrowed() {
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut borrowed_ctx = CpuBackend::new();
    let borrowed = eager_einsum(&mut borrowed_ctx, &[&a, &b], "ij,jk->ik").unwrap();

    let mut owned_ctx = CpuBackend::new();
    let owned = eager_einsum_owned(&mut owned_ctx, vec![a, b], "ij,jk->ik").unwrap();

    assert_eq!(owned.shape(), borrowed.shape());
    assert_eq!(owned.as_slice::<f64>(), borrowed.as_slice::<f64>());
    assert!(owned_ctx.buffer_pool_len() >= 2);
}

#[test]
fn eager_einsum_owned_subscripts_handles_three_operands() {
    let mut ctx = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0]);
    let subscripts = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    let result = eager_einsum_owned_subscripts(&mut ctx, vec![a, b, c], &subscripts).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[152.0, 200.0, 385.0, 508.0]
    );
}

#[test]
fn typed_einsum_f64_three_operands() {
    let mut ctx = CpuBackend::new();
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 1.0, 3.0]);

    let result = typed_eager_einsum(&mut ctx, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.as_slice(), &[152.0, 200.0, 385.0, 508.0]);
}

#[test]
fn typed_einsum_reports_dtype_mismatch_from_backend_result() {
    let mut ctx = WrongDTypeBackend;
    let lhs =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs =
        TypedTensor::<f32>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

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
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![1.0]));
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![3.0]));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let mut backend = WrongDTypeBackend;
    let mut cache = ();

    let direct = BackendCachedDot::dot_general_cached(
        &mut backend,
        &mut cache,
        Some(7),
        &lhs,
        &rhs,
        &config,
    )
    .unwrap();
    assert_eq!(direct.shape(), &[2, 2]);

    let read = TensorDot::dot_general_read(
        &mut backend,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
    )
    .unwrap();
    assert_eq!(read.shape(), &[2, 2]);

    let rhs_shape = [1usize, 1];
    let rhs_data = [3.0_f64];
    let read_view = TensorDot::dot_general_read(
        &mut backend,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_view(TensorView::f64(&rhs_shape, &rhs_data).unwrap()),
        &config,
    )
    .unwrap();
    assert_eq!(read_view.shape(), &[2, 2]);

    let folded =
        TensorDot::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, true).unwrap();
    assert_eq!(folded.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let value = BackendSessionHost::with_backend_session_cached(&mut backend, &mut cache, |exec| {
        let cached = exec
            .dot_general_cached(Some(3), &lhs, &rhs, &config)
            .unwrap();
        let folded = exec
            .dot_general_with_conj_cached(Some(5), &lhs, &rhs, &config, false, false)
            .unwrap();
        let read = exec
            .dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_view(TensorView::f64(&rhs_shape, &rhs_data).unwrap()),
                &config,
            )
            .unwrap();
        cached.shape().len() + folded.shape().len() + read.shape().len()
    });
    assert_eq!(value, 6);
}
