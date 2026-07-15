use super::*;

pub(super) enum FaerPlan {
    F32(StackReq),
    F64(StackReq),
    C32(StackReq),
    C64(StackReq),
}

impl FaerPlan {
    pub(super) fn new(
        dtype: DType,
        shape: [usize; 2],
        par: faer::Par,
    ) -> tenferro_tensor::Result<Self> {
        let [m, n] = shape;
        let req = match dtype {
            DType::F32 => Self::F32(faer::linalg::svd::svd_scratch::<f32>(
                m,
                n,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                par,
                Default::default(),
            )),
            DType::F64 => Self::F64(faer::linalg::svd::svd_scratch::<f64>(
                m,
                n,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                par,
                Default::default(),
            )),
            DType::C32 => Self::C32(faer::linalg::svd::svd_scratch::<faer::c32>(
                m,
                n,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                par,
                Default::default(),
            )),
            DType::C64 => Self::C64(faer::linalg::svd::svd_scratch::<faer::c64>(
                m,
                n,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                par,
                Default::default(),
            )),
            _ => return Err(Error::backend_failure(PREPARE_OP, "unsupported SVD dtype")),
        };
        Ok(req)
    }

    pub(super) fn scratch_bytes(&self) -> usize {
        match self {
            Self::F32(req) | Self::F64(req) | Self::C32(req) | Self::C64(req) => req.size_bytes(),
        }
    }

    pub(super) fn allocate(&self, shape: [usize; 2]) -> tenferro_tensor::Result<FaerWorkspace> {
        macro_rules! alloc {
            ($variant:ident, $req:expr, $input:ty, $faer:ty, $stage_singular:expr) => {
                Ok(FaerWorkspace::$variant(
                    FaerTypedWorkspace::<$input, $faer>::new(*$req, shape, $stage_singular)?,
                ))
            };
        }
        match self {
            Self::F32(req) => alloc!(F32, req, f32, f32, false),
            Self::F64(req) => alloc!(F64, req, f64, f64, false),
            Self::C32(req) => alloc!(C32, req, Complex32, faer::c32, true),
            Self::C64(req) => alloc!(C64, req, Complex64, faer::c64, true),
        }
    }
}

pub(super) enum FaerWorkspace {
    F32(FaerTypedWorkspace<f32, f32>),
    F64(FaerTypedWorkspace<f64, f64>),
    C32(FaerTypedWorkspace<Complex32, faer::c32>),
    C64(FaerTypedWorkspace<Complex64, faer::c64>),
}

impl FaerWorkspace {
    pub(super) fn retained_bytes(&self) -> usize {
        match self {
            Self::F32(w) => w.retained_bytes(),
            Self::F64(w) => w.retained_bytes(),
            Self::C32(w) => w.retained_bytes(),
            Self::C64(w) => w.retained_bytes(),
        }
    }
}

pub(super) struct FaerTypedWorkspace<T, F> {
    scratch: MemBuffer,
    packed: Vec<T>,
    singular: Vec<F>,
    v: Vec<F>,
}

impl<T: Default + Clone, F: Default + Clone> FaerTypedWorkspace<T, F> {
    fn new(
        req: StackReq,
        [m, n]: [usize; 2],
        stage_singular: bool,
    ) -> tenferro_tensor::Result<Self> {
        let matrix_len = checked_shape_product(PREPARE_OP, "input pack", &[m, n])?;
        let k = m.min(n);
        let v_len = checked_shape_product(PREPARE_OP, "right singular vectors", &[n, k])?;
        let scratch = MemBuffer::try_new(req).map_err(|err| {
            Error::backend_failure(PREPARE_OP, format!("Faer scratch allocation failed: {err}"))
        })?;
        Ok(Self {
            scratch,
            packed: vec![T::default(); matrix_len],
            singular: vec![F::default(); if stage_singular { k } else { 0 }],
            v: vec![F::default(); v_len],
        })
    }

    fn retained_bytes(&self) -> usize {
        self.scratch.len()
            + self.packed.capacity() * size_of::<T>()
            + self.singular.capacity() * size_of::<F>()
            + self.v.capacity() * size_of::<F>()
    }
}

pub(super) fn execute_faer(
    plan: &PreparedSvd,
    workspace: &mut FaerWorkspace,
    par: faer::Par,
    input: TensorRead<'_>,
    outputs: SvdOutputWrites<'_>,
) -> tenferro_tensor::Result<()> {
    match plan.dtype {
        DType::F32 => execute_f32(plan, workspace, par, input, outputs),
        DType::F64 => execute_f64(plan, workspace, par, input, outputs),
        DType::C32 => execute_c32(plan, workspace, par, input, outputs),
        DType::C64 => execute_c64(plan, workspace, par, input, outputs),
        _ => Err(Error::backend_failure(
            EXECUTE_OP,
            "unsupported prepared SVD dtype",
        )),
    }
}

// Boxing the view variant would make dispatch smaller but allocate on every warm call.
#[allow(clippy::large_enum_variant)]
enum PreparedRead<'a, T> {
    Tensor(&'a TypedTensor<T>),
    View(TypedTensorView<'a, T>),
}

// Output views stay inline because prepared execution must not allocate wrappers.
#[allow(clippy::large_enum_variant)]
enum PreparedWrite<'a, T> {
    Tensor(&'a mut TypedTensor<T>),
    View(TypedTensorViewMut<'a, T>),
}

macro_rules! impl_dtype_dispatch {
    (
        $fn_name:ident,
        $workspace_variant:ident,
        $tensor_variant:ident,
        $view_variant:ident,
        $scalar:ty,
        $real_tensor_variant:ident,
        $real_view_variant:ident,
        $real:ty
    ) => {
        fn $fn_name(
            plan: &PreparedSvd,
            workspace: &mut FaerWorkspace,
            par: faer::Par,
            input: TensorRead<'_>,
            outputs: SvdOutputWrites<'_>,
        ) -> tenferro_tensor::Result<()> {
            let input = match input {
                TensorRead::Tensor(Tensor::$tensor_variant(tensor)) => PreparedRead::Tensor(tensor),
                TensorRead::View(TensorView::$view_variant(view)) => PreparedRead::View(view),
                _ => {
                    return Err(Error::backend_failure(
                        EXECUTE_OP,
                        "validated input dtype changed",
                    ));
                }
            };
            let SvdOutputWrites { u, s, vt } = outputs;
            let u = match u {
                TensorWrite::Tensor(Tensor::$tensor_variant(tensor)) => {
                    PreparedWrite::Tensor(tensor)
                }
                TensorWrite::View(TensorViewMut::$view_variant(view)) => PreparedWrite::View(view),
                _ => {
                    return Err(Error::backend_failure(
                        EXECUTE_OP,
                        "validated U dtype changed",
                    ));
                }
            };
            let s = match s {
                TensorWrite::Tensor(Tensor::$real_tensor_variant(tensor)) => {
                    PreparedWrite::Tensor(tensor)
                }
                TensorWrite::View(TensorViewMut::$real_view_variant(view)) => {
                    PreparedWrite::View(view)
                }
                _ => {
                    return Err(Error::backend_failure(
                        EXECUTE_OP,
                        "validated S dtype changed",
                    ));
                }
            };
            let vt = match vt {
                TensorWrite::Tensor(Tensor::$tensor_variant(tensor)) => {
                    PreparedWrite::Tensor(tensor)
                }
                TensorWrite::View(TensorViewMut::$view_variant(view)) => PreparedWrite::View(view),
                _ => {
                    return Err(Error::backend_failure(
                        EXECUTE_OP,
                        "validated Vt dtype changed",
                    ));
                }
            };
            match workspace {
                FaerWorkspace::$workspace_variant(workspace) => {
                    execute_typed::<$scalar, $real>(plan, workspace, par, input, u, s, vt)
                }
                _ => Err(Error::backend_failure(
                    EXECUTE_OP,
                    "validated workspace dtype changed",
                )),
            }
        }
    };
}

impl_dtype_dispatch!(execute_f32, F32, F32, F32, f32, F32, F32, f32);
impl_dtype_dispatch!(execute_f64, F64, F64, F64, f64, F64, F64, f64);
impl_dtype_dispatch!(execute_c32, C32, C32, C32, Complex32, F32, F32, f32);
impl_dtype_dispatch!(execute_c64, C64, C64, C64, Complex64, F64, F64, f64);

trait PreparedSvdScalar<R>: Copy + Default + Send + Sync + 'static {
    type Faer: Copy + Default;

    fn copy_singular(staging: &[Self::Faer], output: &mut [R]);
    fn conjugated_output(value: Self::Faer) -> Self;
    fn apply_gauge(
        gauge: SvdGauge,
        u: &mut [Self],
        vt: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) -> tenferro_tensor::Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn run_svd(
        input: *const Self,
        row_stride: isize,
        col_stride: isize,
        u: *mut Self,
        singular_output: &mut [R],
        singular_staging: &mut [Self::Faer],
        v: &mut [Self::Faer],
        m: usize,
        n: usize,
        k: usize,
        par: faer::Par,
        scratch: &mut MemBuffer,
    ) -> tenferro_tensor::Result<()>;
}

macro_rules! impl_real_prepared_scalar {
    ($ty:ty, $gauge_fn:ident) => {
        impl PreparedSvdScalar<$ty> for $ty {
            type Faer = $ty;

            fn copy_singular(_staging: &[Self::Faer], _output: &mut [$ty]) {}

            fn conjugated_output(value: Self::Faer) -> Self {
                value
            }

            fn run_svd(
                input: *const Self,
                row_stride: isize,
                col_stride: isize,
                u: *mut Self,
                singular_output: &mut [$ty],
                _singular_staging: &mut [Self::Faer],
                v: &mut [Self::Faer],
                m: usize,
                n: usize,
                k: usize,
                par: faer::Par,
                scratch: &mut MemBuffer,
            ) -> tenferro_tensor::Result<()> {
                // SAFETY: prepared preflight validates all input/output layouts and aliasing.
                let input = unsafe { MatRef::from_raw_parts(input, m, n, row_stride, col_stride) };
                // SAFETY: `u` points to a compact, non-overlapping `m x k` destination.
                let u = unsafe { MatMut::from_raw_parts_mut(u, m, k, 1, m as isize) };
                let singular = DiagMut::from_slice_mut(singular_output);
                let v = MatMut::from_column_major_slice_mut(v, n, k);
                faer::linalg::svd::svd(
                    input,
                    singular,
                    Some(u),
                    Some(v),
                    par,
                    MemStack::new(scratch),
                    Default::default(),
                )
                .map_err(|_| Error::backend_failure(EXECUTE_OP, "Faer SVD decomposition failed"))
            }

            fn apply_gauge(
                gauge: SvdGauge,
                u: &mut [Self],
                vt: &mut [Self],
                m: usize,
                k: usize,
                n: usize,
            ) -> tenferro_tensor::Result<()> {
                if gauge == SvdGauge::Raw {
                    return Ok(());
                }
                $gauge_fn(u, vt, m, k, n, 1)
            }
        }
    };
}

impl_real_prepared_scalar!(f32, canonicalize_svd_gauge_f32);
impl_real_prepared_scalar!(f64, canonicalize_svd_gauge_f64);

macro_rules! impl_complex_prepared_scalar {
    ($complex:ty, $faer:ty, $real:ty, $gauge_fn:ident) => {
        const _: () = {
            assert!(size_of::<$complex>() == size_of::<$faer>());
            assert!(std::mem::align_of::<$complex>() == std::mem::align_of::<$faer>());
            assert!(std::mem::offset_of!($complex, re) == std::mem::offset_of!($faer, re));
            assert!(std::mem::offset_of!($complex, im) == std::mem::offset_of!($faer, im));
        };

        impl PreparedSvdScalar<$real> for $complex {
            type Faer = $faer;

            fn copy_singular(staging: &[Self::Faer], output: &mut [$real]) {
                for (dst, value) in output.iter_mut().zip(staging) {
                    *dst = value.re;
                }
            }

            fn conjugated_output(value: Self::Faer) -> Self {
                Self::new(value.re, -value.im)
            }

            fn run_svd(
                input: *const Self,
                row_stride: isize,
                col_stride: isize,
                u: *mut Self,
                _singular_output: &mut [$real],
                singular_staging: &mut [Self::Faer],
                v: &mut [Self::Faer],
                m: usize,
                n: usize,
                k: usize,
                par: faer::Par,
                scratch: &mut MemBuffer,
            ) -> tenferro_tensor::Result<()> {
                // SAFETY: compile-time layout assertions above prove compatible complex scalars;
                // prepared preflight validates all input/output layouts and aliasing.
                let input = unsafe {
                    MatRef::from_raw_parts(input.cast::<$faer>(), m, n, row_stride, col_stride)
                };
                // SAFETY: `u` points to a compact, non-overlapping `m x k` destination.
                let u =
                    unsafe { MatMut::from_raw_parts_mut(u.cast::<$faer>(), m, k, 1, m as isize) };
                let singular = DiagMut::from_slice_mut(singular_staging);
                let v = MatMut::from_column_major_slice_mut(v, n, k);
                faer::linalg::svd::svd(
                    input,
                    singular,
                    Some(u),
                    Some(v),
                    par,
                    MemStack::new(scratch),
                    Default::default(),
                )
                .map_err(|_| Error::backend_failure(EXECUTE_OP, "Faer SVD decomposition failed"))
            }

            fn apply_gauge(
                gauge: SvdGauge,
                u: &mut [Self],
                vt: &mut [Self],
                m: usize,
                k: usize,
                n: usize,
            ) -> tenferro_tensor::Result<()> {
                if gauge == SvdGauge::Raw {
                    return Ok(());
                }
                $gauge_fn(u, vt, m, k, n, 1)
            }
        }
    };
}

impl_complex_prepared_scalar!(Complex32, faer::c32, f32, canonicalize_svd_gauge_c32);
impl_complex_prepared_scalar!(Complex64, faer::c64, f64, canonicalize_svd_gauge_c64);

fn execute_typed<T, R>(
    plan: &PreparedSvd,
    workspace: &mut FaerTypedWorkspace<T, T::Faer>,
    par: faer::Par,
    input: PreparedRead<'_, T>,
    mut u: PreparedWrite<'_, T>,
    mut s: PreparedWrite<'_, R>,
    mut vt: PreparedWrite<'_, T>,
) -> tenferro_tensor::Result<()>
where
    T: PreparedSvdScalar<R>,
    R: Copy + Default + 'static,
{
    let [m, n] = plan.shape;
    let k = m.min(n);
    let (input_ptr, row_stride, col_stride) =
        prepared_input_parts::<T, R>(&input, workspace, m, n)?;
    let u_slice = prepared_write_slice(&mut u)?;
    let s_slice = prepared_write_slice(&mut s)?;
    let vt_slice = prepared_write_slice(&mut vt)?;
    T::run_svd(
        input_ptr,
        row_stride,
        col_stride,
        u_slice.as_mut_ptr(),
        s_slice,
        workspace.singular.as_mut_slice(),
        workspace.v.as_mut_slice(),
        m,
        n,
        k,
        par,
        &mut workspace.scratch,
    )?;

    T::copy_singular(workspace.singular.as_slice(), s_slice);
    for col in 0..n {
        for row in 0..k {
            vt_slice[row + k * col] = T::conjugated_output(workspace.v[col + n * row]);
        }
    }
    T::apply_gauge(plan.options.gauge, u_slice, vt_slice, m, k, n)?;
    Ok(())
}

fn prepared_input_parts<T, R>(
    input: &PreparedRead<'_, T>,
    workspace: &mut FaerTypedWorkspace<T, T::Faer>,
    m: usize,
    n: usize,
) -> tenferro_tensor::Result<(*const T, isize, isize)>
where
    T: PreparedSvdScalar<R>,
{
    match input {
        PreparedRead::Tensor(tensor) => Ok((tensor.host_data()?.as_ptr(), 1, m as isize)),
        PreparedRead::View(view) if view.strides().iter().all(|&stride| stride > 0) => {
            let storage = view.host_storage()?;
            let offset = usize::try_from(view.offset()).map_err(|_| Error::InvalidConfig {
                op: EXECUTE_OP,
                message: "input view offset is negative".to_owned(),
            })?;
            Ok((
                storage.as_ptr().wrapping_add(offset),
                view.strides()[0],
                view.strides()[1],
            ))
        }
        PreparedRead::View(view) => {
            let storage = view.host_storage()?;
            let row_stride = view.strides()[0];
            let col_stride = view.strides()[1];
            let base = view.offset();
            for col in 0..n {
                for row in 0..m {
                    let row_offset = isize::try_from(row)
                        .ok()
                        .and_then(|row| row.checked_mul(row_stride));
                    let col_offset = isize::try_from(col)
                        .ok()
                        .and_then(|col| col.checked_mul(col_stride));
                    let offset = row_offset
                        .and_then(|row| col_offset.and_then(|col| row.checked_add(col)))
                        .and_then(|delta| base.checked_add(delta))
                        .and_then(|offset| usize::try_from(offset).ok())
                        .ok_or_else(|| Error::InvalidConfig {
                            op: EXECUTE_OP,
                            message: "input view offset overflows during prepared packing"
                                .to_owned(),
                        })?;
                    workspace.packed[row + m * col] = storage[offset];
                }
            }
            Ok((workspace.packed.as_ptr(), 1, m as isize))
        }
    }
}

fn prepared_write_slice<'a, T: Clone + 'static>(
    write: &'a mut PreparedWrite<'_, T>,
) -> tenferro_tensor::Result<&'a mut [T]> {
    match write {
        PreparedWrite::Tensor(tensor) => tensor.host_data_mut(),
        PreparedWrite::View(view) => {
            let offset = usize::try_from(view.offset()).map_err(|_| Error::InvalidConfig {
                op: EXECUTE_OP,
                message: "output view offset is negative".to_owned(),
            })?;
            let len = checked_shape_product(EXECUTE_OP, "output shape", view.shape())?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| Error::InvalidConfig {
                    op: EXECUTE_OP,
                    message: "output view range overflows usize".to_owned(),
                })?;
            Ok(&mut view.host_storage_mut()?[offset..end])
        }
    }
}
