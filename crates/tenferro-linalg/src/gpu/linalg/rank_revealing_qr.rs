use cubecl::prelude::{CubeCount, CubeDim, TensorBinding};

use super::*;
use crate::{rank_revealing_qr::validate_rank_revealing_qr_options, RankRevealingQrOptions};

const OP: &str = "rank_revealing_qr";
const RRQR_PLANE_WIDTH: u32 = 32;

type CudaTensorBinding = TensorBinding<CubeclCudaRuntime>;

trait CudaRrqrScalar: LinalgScalar + TensorScalar {
    fn device_imaginary_unit(
        backend: &mut CudaExecSession<'_>,
    ) -> Result<Option<TypedTensor<Self>>>;

    fn launch_norms(
        cubecl: &CubeclSession<'_>,
        work: &TypedTensor<Self>,
        norms: &TypedTensor<<Self as LinalgScalar>::Real>,
        count: CubeCount,
        step: usize,
        m: usize,
        n: usize,
    ) -> Result<()>;

    fn launch_pivot(
        cubecl: &CubeclSession<'_>,
        norms: &TypedTensor<<Self as LinalgScalar>::Real>,
        permutation: &TypedTensor<i64>,
        pivots: &TypedTensor<i64>,
        status: &TypedTensor<i64>,
        count: CubeCount,
        step: usize,
        n: usize,
    ) -> Result<()>;

    fn launch_reflector(
        cubecl: &CubeclSession<'_>,
        work: &TypedTensor<Self>,
        coeff: &TypedTensor<Self>,
        count: CubeCount,
        step: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn launch_apply(
        cubecl: &CubeclSession<'_>,
        packed: &TypedTensor<Self>,
        target: &TypedTensor<Self>,
        coeff: &TypedTensor<Self>,
        imaginary_unit: Option<&TypedTensor<Self>>,
        count: CubeCount,
        step: usize,
        first_column: usize,
        column_count: usize,
        m: usize,
        packed_columns: usize,
        target_columns: usize,
        k: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn launch_rank(
        cubecl: &CubeclSession<'_>,
        r: &TypedTensor<Self>,
        rank: &TypedTensor<i64>,
        status: &TypedTensor<i64>,
        count: CubeCount,
        k: usize,
        n: usize,
        options: RankRevealingQrOptions,
    ) -> Result<()>;
}

fn rrqr_binding<T>(cubecl: &CubeclSession<'_>, tensor: &TypedTensor<T>) -> Result<CudaTensorBinding>
where
    T: cubecl::prelude::CubeElement + TensorScalar + Clone,
{
    cubecl.tensor_binding(tensor, OP)
}

macro_rules! impl_cuda_rrqr_real {
    ($scalar:ty, $variant:ident) => {
        impl CudaRrqrScalar for $scalar {
            fn device_imaginary_unit(
                _backend: &mut CudaExecSession<'_>,
            ) -> Result<Option<TypedTensor<Self>>> {
                Ok(None)
            }

            fn launch_norms(
                cubecl: &CubeclSession<'_>,
                work: &TypedTensor<Self>,
                norms: &TypedTensor<<Self as LinalgScalar>::Real>,
                count: CubeCount,
                step: usize,
                m: usize,
                n: usize,
            ) -> Result<()> {
                let work = rrqr_binding(cubecl, work)?;
                let norms = rrqr_binding(cubecl, norms)?;
                // SAFETY: dense same-runtime bindings are live for this launch;
                // one plane owns each (trailing column, batch) norm.
                unsafe {
                    cubecl_linalg::column_norms_real::launch_unchecked::<Self, CubeclCudaRuntime>(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        work.into_tensor_arg(),
                        norms.into_tensor_arg(),
                        step,
                        m,
                        n,
                    );
                }
                Ok(())
            }

            fn launch_pivot(
                cubecl: &CubeclSession<'_>,
                norms: &TypedTensor<<Self as LinalgScalar>::Real>,
                permutation: &TypedTensor<i64>,
                pivots: &TypedTensor<i64>,
                status: &TypedTensor<i64>,
                count: CubeCount,
                step: usize,
                n: usize,
            ) -> Result<()> {
                launch_pivot_real::<Self>(
                    cubecl,
                    norms,
                    permutation,
                    pivots,
                    status,
                    count,
                    step,
                    n,
                )
            }

            fn launch_reflector(
                cubecl: &CubeclSession<'_>,
                work: &TypedTensor<Self>,
                coeff: &TypedTensor<Self>,
                count: CubeCount,
                step: usize,
                m: usize,
                n: usize,
                k: usize,
            ) -> Result<()> {
                let work = rrqr_binding(cubecl, work)?;
                let coeff = rrqr_binding(cubecl, coeff)?;
                // SAFETY: one plane owns each batch reflector and lanes write
                // disjoint entries of its packed vector.
                unsafe {
                    cubecl_linalg::reflector_real::launch_unchecked::<Self, CubeclCudaRuntime>(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        work.into_tensor_arg(),
                        coeff.into_tensor_arg(),
                        step,
                        m,
                        n,
                        k,
                    );
                }
                Ok(())
            }

            fn launch_apply(
                cubecl: &CubeclSession<'_>,
                packed: &TypedTensor<Self>,
                target: &TypedTensor<Self>,
                coeff: &TypedTensor<Self>,
                _imaginary_unit: Option<&TypedTensor<Self>>,
                count: CubeCount,
                step: usize,
                first_column: usize,
                column_count: usize,
                m: usize,
                packed_columns: usize,
                target_columns: usize,
                k: usize,
            ) -> Result<()> {
                let packed = rrqr_binding(cubecl, packed)?;
                let target = rrqr_binding(cubecl, target)?;
                let coeff = rrqr_binding(cubecl, coeff)?;
                // SAFETY: one plane owns each (target column, batch) update;
                // target columns never overlap the packed reflector column.
                unsafe {
                    cubecl_linalg::apply_reflector_real::launch_unchecked::<
                        Self,
                        CubeclCudaRuntime,
                    >(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        packed.into_tensor_arg(),
                        target.into_tensor_arg(),
                        step,
                        first_column,
                        column_count,
                        m,
                        packed_columns,
                        target_columns,
                        k,
                        coeff.into_tensor_arg(),
                    );
                }
                Ok(())
            }

            fn launch_rank(
                cubecl: &CubeclSession<'_>,
                r: &TypedTensor<Self>,
                rank: &TypedTensor<i64>,
                status: &TypedTensor<i64>,
                count: CubeCount,
                k: usize,
                n: usize,
                options: RankRevealingQrOptions,
            ) -> Result<()> {
                let r = rrqr_binding(cubecl, r)?;
                let rank = rrqr_binding(cubecl, rank)?;
                let status = rrqr_binding(cubecl, status)?;
                // SAFETY: one plane owns each batch rank and scans disjoint
                // diagonal subsequences before a plane minimum reduction.
                unsafe {
                    cubecl_linalg::prefix_rank_real::launch_unchecked::<Self, CubeclCudaRuntime>(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        r.into_tensor_arg(),
                        rank.into_tensor_arg(),
                        status.into_tensor_arg(),
                        k,
                        n,
                        options.rtol as $scalar,
                        options.atol as $scalar,
                    );
                }
                Ok(())
            }
        }
    };
}

impl_cuda_rrqr_real!(f32, F32);
impl_cuda_rrqr_real!(f64, F64);

fn launch_pivot_real<F>(
    cubecl: &CubeclSession<'_>,
    norms: &TypedTensor<F>,
    permutation: &TypedTensor<i64>,
    pivots: &TypedTensor<i64>,
    status: &TypedTensor<i64>,
    count: CubeCount,
    step: usize,
    n: usize,
) -> Result<()>
where
    F: cubecl::prelude::Float
        + cubecl::prelude::CubeElement
        + cubecl::prelude::CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>
        + TensorScalar
        + Clone,
{
    let norms = rrqr_binding(cubecl, norms)?;
    let permutation = rrqr_binding(cubecl, permutation)?;
    let pivots = rrqr_binding(cubecl, pivots)?;
    let status = rrqr_binding(cubecl, status)?;
    // SAFETY: one plane owns each batch pivot/status output; permutation is
    // read-only and the lowest original index resolves exact norm ties.
    unsafe {
        cubecl_linalg::select_pivot::launch_unchecked::<F, CubeclCudaRuntime>(
            cubecl.client(),
            count,
            rrqr_cube_dim(),
            norms.into_tensor_arg(),
            permutation.into_tensor_arg(),
            pivots.into_tensor_arg(),
            status.into_tensor_arg(),
            step,
            n,
        );
    }
    Ok(())
}

macro_rules! impl_cuda_rrqr_complex {
    ($scalar:ty, $variant:ident, $real:ty, $norms:ident, $reflector:ident, $apply:ident, $rank_kernel:ident) => {
        impl CudaRrqrScalar for $scalar {
            fn device_imaginary_unit(
                backend: &mut CudaExecSession<'_>,
            ) -> Result<Option<TypedTensor<Self>>> {
                let host = Tensor::$variant(TypedTensor::from_vec_col_major(
                    vec![1],
                    vec![<$scalar>::new(0.0 as $real, 1.0 as $real)],
                )?);
                match tenferro_gpu::cuda::upload_tensor(backend.runtime(), &host)? {
                    Tensor::$variant(tensor) => Ok(Some(tensor)),
                    _ => Err(Error::Internal("RRQR constant upload changed dtype".into())),
                }
            }

            fn launch_norms(
                cubecl: &CubeclSession<'_>,
                work: &TypedTensor<Self>,
                norms: &TypedTensor<<Self as LinalgScalar>::Real>,
                count: CubeCount,
                step: usize,
                m: usize,
                n: usize,
            ) -> Result<()> {
                let work = rrqr_binding(cubecl, work)?;
                let norms = rrqr_binding(cubecl, norms)?;
                // SAFETY: one plane owns each (trailing column, batch) norm.
                unsafe {
                    cubecl_linalg::$norms::launch_unchecked::<CubeclCudaRuntime>(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        work.into_tensor_arg(),
                        norms.into_tensor_arg(),
                        step,
                        m,
                        n,
                    );
                }
                Ok(())
            }

            fn launch_pivot(
                cubecl: &CubeclSession<'_>,
                norms: &TypedTensor<<Self as LinalgScalar>::Real>,
                permutation: &TypedTensor<i64>,
                pivots: &TypedTensor<i64>,
                status: &TypedTensor<i64>,
                count: CubeCount,
                step: usize,
                n: usize,
            ) -> Result<()> {
                launch_pivot_real::<$real>(
                    cubecl,
                    norms,
                    permutation,
                    pivots,
                    status,
                    count,
                    step,
                    n,
                )
            }

            fn launch_reflector(
                cubecl: &CubeclSession<'_>,
                work: &TypedTensor<Self>,
                coeff: &TypedTensor<Self>,
                count: CubeCount,
                step: usize,
                m: usize,
                n: usize,
                k: usize,
            ) -> Result<()> {
                let work = rrqr_binding(cubecl, work)?;
                let coeff = rrqr_binding(cubecl, coeff)?;
                // SAFETY: one plane owns each batch reflector and lanes write
                // disjoint entries of its packed vector.
                unsafe {
                    cubecl_linalg::$reflector::launch_unchecked::<CubeclCudaRuntime>(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        work.into_tensor_arg(),
                        coeff.into_tensor_arg(),
                        step,
                        m,
                        n,
                        k,
                    );
                }
                Ok(())
            }

            fn launch_apply(
                cubecl: &CubeclSession<'_>,
                packed: &TypedTensor<Self>,
                target: &TypedTensor<Self>,
                coeff: &TypedTensor<Self>,
                imaginary_unit: Option<&TypedTensor<Self>>,
                count: CubeCount,
                step: usize,
                first_column: usize,
                column_count: usize,
                m: usize,
                packed_columns: usize,
                target_columns: usize,
                k: usize,
            ) -> Result<()> {
                let packed = rrqr_binding(cubecl, packed)?;
                let target = rrqr_binding(cubecl, target)?;
                let coeff = rrqr_binding(cubecl, coeff)?;
                let imaginary_unit = imaginary_unit.ok_or_else(|| {
                    Error::Internal("complex RRQR requires an imaginary-unit constant".into())
                })?;
                let imaginary_unit = rrqr_binding(cubecl, imaginary_unit)?;
                // SAFETY: one plane owns each (target column, batch) update;
                // target columns never overlap the packed reflector column.
                unsafe {
                    cubecl_linalg::$apply::launch_unchecked::<CubeclCudaRuntime>(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        packed.into_tensor_arg(),
                        target.into_tensor_arg(),
                        step,
                        first_column,
                        column_count,
                        m,
                        packed_columns,
                        target_columns,
                        k,
                        coeff.into_tensor_arg(),
                        imaginary_unit.into_tensor_arg(),
                    );
                }
                Ok(())
            }

            fn launch_rank(
                cubecl: &CubeclSession<'_>,
                r: &TypedTensor<Self>,
                rank: &TypedTensor<i64>,
                status: &TypedTensor<i64>,
                count: CubeCount,
                k: usize,
                n: usize,
                options: RankRevealingQrOptions,
            ) -> Result<()> {
                let r = rrqr_binding(cubecl, r)?;
                let rank = rrqr_binding(cubecl, rank)?;
                let status = rrqr_binding(cubecl, status)?;
                // SAFETY: one plane owns each batch rank and status output.
                unsafe {
                    cubecl_linalg::$rank_kernel::launch_unchecked::<CubeclCudaRuntime>(
                        cubecl.client(),
                        count,
                        rrqr_cube_dim(),
                        r.into_tensor_arg(),
                        rank.into_tensor_arg(),
                        status.into_tensor_arg(),
                        k,
                        n,
                        options.rtol as $real,
                        options.atol as $real,
                    );
                }
                Ok(())
            }
        }
    };
}

impl_cuda_rrqr_complex!(
    Complex32,
    C32,
    f32,
    column_norms_c32,
    reflector_c32,
    apply_reflector_c32,
    prefix_rank_c32
);
impl_cuda_rrqr_complex!(
    Complex64,
    C64,
    f64,
    column_norms_c64,
    reflector_c64,
    apply_reflector_c64,
    prefix_rank_c64
);

pub(super) fn rank_revealing_qr(
    backend: &mut CudaExecSession<'_>,
    input: &Tensor,
    options: RankRevealingQrOptions,
) -> Result<Vec<Tensor>> {
    validate_rank_revealing_qr_options(OP, options)?;
    match input {
        Tensor::F32(input) => {
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::F32(q),
                    Tensor::F32(r),
                    Tensor::I64(p),
                    Tensor::I64(rank),
                ]
            })
        }
        Tensor::F64(input) => {
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::F64(q),
                    Tensor::F64(r),
                    Tensor::I64(p),
                    Tensor::I64(rank),
                ]
            })
        }
        Tensor::C32(input) => {
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::C32(q),
                    Tensor::C32(r),
                    Tensor::I64(p),
                    Tensor::I64(rank),
                ]
            })
        }
        Tensor::C64(input) => {
            rank_revealing_qr_typed(backend, input, options).map(|(q, r, p, rank)| {
                vec![
                    Tensor::C64(q),
                    Tensor::C64(r),
                    Tensor::I64(p),
                    Tensor::I64(rank),
                ]
            })
        }
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_linalg_dtype(OP, input))
        }
    }
}

fn rank_revealing_qr_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
    options: RankRevealingQrOptions,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<i64>,
    TypedTensor<i64>,
)>
where
    T: CudaRrqrScalar + RrqrTensorVariant,
{
    ensure_cubecl_resident_typed(OP, input)?;
    let (m, n) = matrix_dims(OP, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let actual_batch_count = checked_shape_product(OP, "batch shape", batch_shape)?;
    let batch_total = actual_batch_count.max(1);
    u32::try_from(n).map_err(|_| {
        Error::invalid_argument(OP, "column count", "column index exceeds u32 range")
    })?;
    u32::try_from(k).map_err(|_| Error::invalid_argument(OP, "rank", "rank exceeds u32 range"))?;

    let q_shape = matrix_shape(m, k, batch_shape);
    let r_shape = matrix_shape(k, n, batch_shape);
    let permutation_shape = vector_shape(n, batch_shape);
    let rank_shape = batch_shape.to_vec();

    if m == 0 || n == 0 || actual_batch_count == 0 {
        return backend.with_cubecl(OP, |cubecl| {
            let q = cubecl.alloc_output::<T>(&q_shape)?;
            let r = cubecl.alloc_output::<T>(&r_shape)?;
            let permutation = cubecl.alloc_output::<i64>(&permutation_shape)?;
            let rank = cubecl.alloc_zero_output::<i64>(&rank_shape)?;
            if permutation.n_elements() > 0 {
                launch_initialize_permutation(cubecl, &permutation, n)?;
            }
            Ok((q, r, permutation, rank))
        });
    }

    let imaginary_unit = T::device_imaginary_unit(backend)?;
    let work = clone_device_tensor(backend, input, OP)?;
    let (q, r, permutation, rank, status) = backend.with_cubecl(OP, |cubecl| {
        let coeff = cubecl.alloc_output::<T>(&vector_shape(k, batch_shape))?;
        let norms =
            cubecl.alloc_output::<<T as LinalgScalar>::Real>(&vector_shape(n, batch_shape))?;
        let pivots = cubecl.alloc_output::<i64>(&rank_shape)?;
        let status = cubecl.alloc_zero_output::<i64>(&rank_shape)?;
        let permutation = cubecl.alloc_output::<i64>(&permutation_shape)?;
        launch_initialize_permutation(cubecl, &permutation, n)?;

        let batch_planes = rrqr_cube_count(batch_total)?;
        for step in 0..k {
            let norm_planes = rrqr_cube_count(checked_mul_usize(
                OP,
                "RRQR norm planes",
                n - step,
                batch_total,
            )?)?;
            T::launch_norms(cubecl, &work, &norms, norm_planes, step, m, n)?;
            T::launch_pivot(
                cubecl,
                &norms,
                &permutation,
                &pivots,
                &status,
                batch_planes.clone(),
                step,
                n,
            )?;
            launch_swap_columns(
                cubecl,
                &work,
                &permutation,
                &pivots,
                step,
                m,
                n,
                batch_total,
            )?;
            T::launch_reflector(cubecl, &work, &coeff, batch_planes.clone(), step, m, n, k)?;
            let trailing = n - step - 1;
            if trailing > 0 {
                let apply_planes = rrqr_cube_count(checked_mul_usize(
                    OP,
                    "RRQR trailing update planes",
                    trailing,
                    batch_total,
                )?)?;
                T::launch_apply(
                    cubecl,
                    &work,
                    &work,
                    &coeff,
                    imaginary_unit.as_ref(),
                    apply_planes,
                    step,
                    step + 1,
                    trailing,
                    m,
                    n,
                    n,
                    k,
                )?;
            }
        }

        let r = cubecl.alloc_output::<T>(&r_shape)?;
        launch_extract_r(cubecl, &work, &r, m, n, k)?;
        let q = cubecl.alloc_output::<T>(&q_shape)?;
        launch_initialize_q(cubecl, &q, m, k)?;
        for step in (0..k).rev() {
            let q_planes = rrqr_cube_count(checked_mul_usize(
                OP,
                "RRQR Q update planes",
                k,
                batch_total,
            )?)?;
            T::launch_apply(
                cubecl,
                &work,
                &q,
                &coeff,
                imaginary_unit.as_ref(),
                q_planes,
                step,
                0,
                k,
                m,
                n,
                k,
                k,
            )?;
        }

        let rank = cubecl.alloc_output::<i64>(&rank_shape)?;
        T::launch_rank(cubecl, &r, &rank, &status, batch_planes, k, n, options)?;
        Ok((q, r, permutation, rank, status))
    })?;

    // The only CUDA-to-host read is this bounded provider-status vector. Matrix
    // payloads, norms, pivots, permutation, and rank remain device-resident.
    backend.runtime().synchronize()?;
    let host_status = download_tensor(backend.runtime(), &Tensor::I64(status))?;
    let Tensor::I64(host_status) = host_status else {
        return Err(Error::Internal(
            "rank_revealing_qr: unexpected provider-status dtype".into(),
        ));
    };
    if host_status.host_data()?.iter().any(|&value| value != 0) {
        return Err(crate::error::into_tensor_error(
            OP,
            crate::Error::NonFinite {
                op: OP,
                role: "input or computed R diagonal",
            },
        ));
    }

    let mut factors = vec![wrap_tensor(q), wrap_tensor(r)];
    if options.gauge == QrGauge::PositiveDiagonal {
        apply_qr_gauge_device(backend, &mut factors, 0, OP)?;
    }
    let q = unwrap_tensor(factors.remove(0))?;
    let r = unwrap_tensor(factors.remove(0))?;
    Ok((q, r, permutation, rank))
}

fn launch_initialize_permutation(
    cubecl: &CubeclSession<'_>,
    permutation: &TypedTensor<i64>,
    n: usize,
) -> Result<()> {
    let permutation_arg = rrqr_binding(cubecl, permutation)?;
    let count = cubecl.cube_count_1d(permutation.n_elements())?;
    // SAFETY: each worker initializes one distinct permutation element.
    unsafe {
        cubecl_linalg::initialize_permutation::launch_unchecked::<CubeclCudaRuntime>(
            cubecl.client(),
            count,
            cubecl.cube_dim_1d(),
            permutation_arg.into_tensor_arg(),
            n,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_swap_columns<T>(
    cubecl: &CubeclSession<'_>,
    work: &TypedTensor<T>,
    permutation: &TypedTensor<i64>,
    pivots: &TypedTensor<i64>,
    step: usize,
    m: usize,
    n: usize,
    batch_total: usize,
) -> Result<()>
where
    T: CudaRrqrScalar,
{
    let work_arg = rrqr_binding(cubecl, work)?;
    let permutation_arg = rrqr_binding(cubecl, permutation)?;
    let pivots_arg = rrqr_binding(cubecl, pivots)?;
    let len = checked_mul_usize(OP, "RRQR swap domain", m, batch_total)?;
    let count = cubecl.cube_count_1d(len)?;
    // SAFETY: each worker swaps one row in one batch; only row zero updates
    // that batch's permutation, and pivot indices came from the bounded scan.
    unsafe {
        cubecl_linalg::swap_columns::launch_unchecked::<T, CubeclCudaRuntime>(
            cubecl.client(),
            count,
            cubecl.cube_dim_1d(),
            work_arg.into_tensor_arg(),
            permutation_arg.into_tensor_arg(),
            pivots_arg.into_tensor_arg(),
            step,
            m,
            n,
        );
    }
    Ok(())
}

fn launch_initialize_q<T>(
    cubecl: &CubeclSession<'_>,
    q: &TypedTensor<T>,
    m: usize,
    k: usize,
) -> Result<()>
where
    T: CudaRrqrScalar,
{
    let q_arg = rrqr_binding(cubecl, q)?;
    let count = cubecl.cube_count_1d(q.n_elements())?;
    // SAFETY: each worker initializes one distinct Q element.
    unsafe {
        cubecl_linalg::initialize_q::launch_unchecked::<T, CubeclCudaRuntime>(
            cubecl.client(),
            count,
            cubecl.cube_dim_1d(),
            q_arg.into_tensor_arg(),
            m,
            k,
        );
    }
    Ok(())
}

fn launch_extract_r<T>(
    cubecl: &CubeclSession<'_>,
    work: &TypedTensor<T>,
    r: &TypedTensor<T>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()>
where
    T: CudaRrqrScalar,
{
    let work_arg = rrqr_binding(cubecl, work)?;
    let r_arg = rrqr_binding(cubecl, r)?;
    let count = cubecl.cube_count_1d(r.n_elements())?;
    // SAFETY: each worker writes one R element from the corresponding packed
    // upper trapezoid or writes zero below it.
    unsafe {
        cubecl_linalg::extract_r::launch_unchecked::<T, CubeclCudaRuntime>(
            cubecl.client(),
            count,
            cubecl.cube_dim_1d(),
            work_arg.into_tensor_arg(),
            r_arg.into_tensor_arg(),
            m,
            n,
            k,
        );
    }
    Ok(())
}

fn rrqr_cube_count(planes: usize) -> Result<CubeCount> {
    let planes = u32::try_from(planes).map_err(|_| {
        Error::invalid_argument(OP, "launch domain", "plane count exceeds u32 range")
    })?;
    Ok(CubeCount::Static(planes.max(1), 1, 1))
}

fn rrqr_cube_dim() -> CubeDim {
    CubeDim::new_1d(RRQR_PLANE_WIDTH)
}

fn matrix_shape(rows: usize, columns: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, columns];
    shape.extend_from_slice(batch_shape);
    shape
}

fn vector_shape(length: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![length];
    shape.extend_from_slice(batch_shape);
    shape
}

trait RrqrTensorVariant: Sized {
    fn wrap(tensor: TypedTensor<Self>) -> Tensor;
    fn unwrap(tensor: Tensor) -> Result<TypedTensor<Self>>;
}

macro_rules! impl_rrqr_tensor_variant {
    ($scalar:ty, $variant:ident) => {
        impl RrqrTensorVariant for $scalar {
            fn wrap(tensor: TypedTensor<Self>) -> Tensor {
                Tensor::$variant(tensor)
            }

            fn unwrap(tensor: Tensor) -> Result<TypedTensor<Self>> {
                match tensor {
                    Tensor::$variant(tensor) => Ok(tensor),
                    other => Err(Error::dtype_mismatch(
                        OP,
                        <$scalar as TensorScalar>::dtype(),
                        other.dtype(),
                    )),
                }
            }
        }
    };
}

impl_rrqr_tensor_variant!(f32, F32);
impl_rrqr_tensor_variant!(f64, F64);
impl_rrqr_tensor_variant!(Complex32, C32);
impl_rrqr_tensor_variant!(Complex64, C64);

fn wrap_tensor<T: RrqrTensorVariant>(tensor: TypedTensor<T>) -> Tensor {
    T::wrap(tensor)
}

fn unwrap_tensor<T: RrqrTensorVariant>(tensor: Tensor) -> Result<TypedTensor<T>> {
    T::unwrap(tensor)
}
