use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, Engine, GatherConfig, PadConfig,
    ScatterConfig, SessionCachedDot, SliceConfig, Tensor, TensorAnalytic, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorStructural, TensorView, TensorWrite, TypedTensor,
};

use crate::eager::{
    eager_einsum, eager_einsum_owned, eager_einsum_owned_subscripts, eager_einsum_read_subscripts,
    eager_einsum_subscripts,
};
use crate::typed_eager::typed_eager_einsum;
use crate::Subscripts;

#[test]
fn typed_eager_einsum_does_not_erase_through_host_copies() {
    let source = include_str!("typed_eager.rs");

    assert!(
        !source.contains("host_data().to_vec()"),
        "typed eager einsum must use TensorRead inputs instead of copying host data"
    );
}

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

impl TensorElementwise for WrongDTypeBackend {
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        let _ = (op, inputs, out);
        panic!("elementwise_read_into should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn add_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("add", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("add", rhs)?;
        panic!("add should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sub_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sub", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("sub", rhs)?;
        panic!("sub should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn mul_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("mul", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("mul", rhs)?;
        panic!("mul should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn neg_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("neg", input)?;
        panic!("neg should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn conj_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("conj", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.conj_read(TensorRead::from_tensor(input))
        }
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn div_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("div", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("div", rhs)?;
        panic!("div should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn abs_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("abs", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.abs_read(TensorRead::from_tensor(input))
        }
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sign_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sign", input)?;
        panic!("sign should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn maximum_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("maximum", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("maximum", rhs)?;
        panic!("maximum should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn minimum_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("minimum", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("minimum", rhs)?;
        panic!("minimum should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> tenferro_tensor::Result<Tensor> {
        let lhs = tenferro_tensor::backend::read_owned_tensor("compare", lhs)?;
        let rhs = tenferro_tensor::backend::read_owned_tensor("compare", rhs)?;
        {
            let __s = &mut CpuEngine::serial();
            {
                __s.compare_read(
                    TensorRead::from_tensor(lhs),
                    TensorRead::from_tensor(rhs),
                    dir,
                )
            }
        }
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("select", pred)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_true)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_false)?;
        panic!("select should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("clamp", input)?;
        let lower = tenferro_tensor::backend::read_owned_tensor("clamp", lower)?;
        let upper = tenferro_tensor::backend::read_owned_tensor("clamp", upper)?;
        {
            let __s = &mut CpuEngine::serial();
            {
                __s.clamp_read(
                    TensorRead::from_tensor(input),
                    TensorRead::from_tensor(lower),
                    TensorRead::from_tensor(upper),
                )
            }
        }
    }
}

impl TensorAnalytic for WrongDTypeBackend {
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn exp_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("exp", input)?;
        panic!("exp should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn log_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("log", input)?;
        panic!("log should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sin_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sin", input)?;
        panic!("sin should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn cos_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("cos", input)?;
        panic!("cos should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn tanh_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("tanh", input)?;
        panic!("tanh should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sqrt_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sqrt", input)?;
        panic!("sqrt should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn rsqrt_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("rsqrt", input)?;
        panic!("rsqrt should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn pow_read(
        &mut self,
        lhs: tenferro_tensor::TensorRead<'_>,
        rhs: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("pow", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("pow", rhs)?;
        panic!("pow should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn expm1_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("expm1", input)?;
        panic!("expm1 should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn log1p_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("log1p", input)?;
        panic!("log1p should not be called in this test")
    }

    fn erf_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("erf", input)?;
        panic!("erf should not be called in this test")
    }
}

impl TensorStructural for WrongDTypeBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        {
            let __s = &mut CpuEngine::serial();
            __s.to_contiguous_read(input)
        }
    }

    fn copy_read_into(
        &mut self,
        src: TensorRead<'_>,
        dst: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        {
            let __s = &mut CpuEngine::serial();
            __s.copy_read_into(src, dst)
        }
    }

    panic_backend_methods! {
        cast(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn transpose_read(
        &mut self,
        input: TensorRead<'_>,
        perm: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("transpose", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.transpose_read(TensorRead::from_tensor(input), perm)
        }
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn reshape_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("reshape", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.reshape_read(TensorRead::from_tensor(input), shape)
        }
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("broadcast_in_dim", input)?;
        {
            let __s = &mut CpuEngine::serial();
            {
                __s.broadcast_in_dim_read(TensorRead::from_tensor(input), shape, dims)
            }
        }
    }
}

impl TensorReduction for WrongDTypeBackend {
    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_sum_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("reduce_sum", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.reduce_sum_read(TensorRead::from_tensor(input), axes)
        }
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_prod_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.reduce_prod_read(TensorRead::from_tensor(input), axes)
        }
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_max_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.reduce_max_read(TensorRead::from_tensor(input), axes)
        }
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_min_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?;
        {
            let __s = &mut CpuEngine::serial();
            __s.reduce_min_read(TensorRead::from_tensor(input), axes)
        }
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
    // The previous read-half default delegated an owned pair to the one-shot
    // method and materialized borrowed views through to_contiguous_read before
    // contracting. Reproduce that exactly rather than forwarding a view.
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        _config: &DotGeneralConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(_), Some(_)) => Ok(Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
            _ => {
                let _ = self.to_contiguous_read(lhs)?;
                let _ = self.to_contiguous_read(rhs)?;
                Ok(Tensor::from_typed::<f64>(
                    TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ))
            }
        }
    }
}

impl SessionCachedDot for WrongDTypeBackend {}

impl Engine for WrongDTypeBackend {}

impl TensorDeviceTransfer for WrongDTypeBackend {
    fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "WrongDTypeBackend::download_to_host",
            "test backend does not transfer tensors",
        ))
    }

    fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "WrongDTypeBackend::upload_host_tensor",
            "test backend does not transfer tensors",
        ))
    }
}

impl TensorBuffer for WrongDTypeBackend {}

impl TensorFusion for WrongDTypeBackend {}

#[test]
fn typed_einsum_f64() {
    unsafe {
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_AGG", "1");
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_PRINT_EVERY", "1");
    }
    let mut ctx = CpuEngine::serial();
    let lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

    let result = typed_eager_einsum(&mut ctx, &[&lhs, &rhs], "ij,jk->ik").unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn eager_einsum_subscripts_and_read_views_use_integer_api() {
    unsafe {
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_AGG", "1");
        std::env::set_var("TENFERRO_PROFILE_EAGER_EINSUM_PRINT_EVERY", "1");
    }
    let mut ctx = CpuEngine::serial();
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs_shape = [3usize, 2];
    let rhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = Tensor::from_vec_col_major(rhs_shape.to_vec(), rhs_data.to_vec()).unwrap();
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
    assert_eq!(
        read.as_slice::<f64>().unwrap(),
        borrowed.as_slice::<f64>().unwrap()
    );
}

#[test]
fn eager_einsum_owned_matches_borrowed() {
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let mut borrowed_ctx = CpuEngine::serial();
    let borrowed = eager_einsum(&mut borrowed_ctx, &[&a, &b], "ij,jk->ik").unwrap();

    let mut owned_ctx = CpuEngine::serial();
    let owned = eager_einsum_owned(&mut owned_ctx, vec![a, b], "ij,jk->ik").unwrap();

    assert_eq!(owned.shape(), borrowed.shape());
    assert_eq!(
        owned.as_slice::<f64>().unwrap(),
        borrowed.as_slice::<f64>().unwrap()
    );
    assert!(owned_ctx.buffer_pool_len() >= 2);
}

#[test]
fn eager_einsum_owned_subscripts_handles_three_operands() {
    let mut ctx = CpuEngine::serial();
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0]).unwrap();
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
    let mut ctx = CpuEngine::serial();
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap();
    let c = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 1.0, 3.0]).unwrap();

    let result = typed_eager_einsum(&mut ctx, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice().unwrap(), &[152.0, 200.0, 385.0, 508.0]);
}

#[test]
fn typed_einsum_reports_dtype_mismatch_from_backend_result() {
    let mut ctx = WrongDTypeBackend;
    let lhs =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let rhs =
        TypedTensor::<f32>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

    let err = typed_eager_einsum(&mut ctx, &[&lhs, &rhs], "ij,jk->ik").unwrap_err();

    assert!(matches!(
        err,
        tenferro_tensor::Error::Validation {
            op: "typed_eager_einsum",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        }
    ));
}

#[test]
fn typed_einsum_preserves_typed_parser_source_for_invalid_notation() {
    let mut ctx = CpuEngine::serial();
    let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();

    let error = typed_eager_einsum(&mut ctx, &[&input], "ij,(jk,kl)->il")
        .expect_err("malformed notation must fail before backend execution");

    assert!(matches!(
        error,
        tenferro_tensor::Error::Extension {
            op: "typed_eager_einsum",
            kind: tenferro_tensor::ErrorKind::Validation(
                tenferro_tensor::ValidationKind::InvalidArgument
            ),
            ..
        }
    ));
    assert!(std::error::Error::source(&error).is_some());
}
