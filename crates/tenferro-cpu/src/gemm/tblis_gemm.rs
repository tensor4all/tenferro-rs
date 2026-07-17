use core::ffi::{c_char, c_int, c_void};

use num_traits::{One, Zero};
use smallvec::SmallVec;
use tblis_ffi::tblis::{
    label_type, len_type, stride_type, tblis_scalar, tblis_scalar_scalar, tblis_tensor,
    tblis_tensor_mult, type_t, TYPE_DCOMPLEX, TYPE_DOUBLE, TYPE_SCOMPLEX, TYPE_SINGLE,
};

use super::{checked_product, validate_dot_general, TypedTensorRead, OP};
use crate::{Error, Result};
use tenferro_tensor::{DotGeneralConfig, TypedTensorViewMut};

use num_complex::{Complex32, Complex64};

pub(crate) trait TblisGemm: Copy + One + Zero + 'static {
    const TYPE: type_t;

    fn scalar(value: Self) -> tblis_scalar;
}

impl TblisGemm for f32 {
    const TYPE: type_t = TYPE_SINGLE;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { s: value },
            type_: Self::TYPE,
        }
    }
}

impl TblisGemm for f64 {
    const TYPE: type_t = TYPE_DOUBLE;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { d: value },
            type_: Self::TYPE,
        }
    }
}

impl TblisGemm for Complex32 {
    const TYPE: type_t = TYPE_SCOMPLEX;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { c: value },
            type_: Self::TYPE,
        }
    }
}

impl TblisGemm for Complex64 {
    const TYPE: type_t = TYPE_DCOMPLEX;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { z: value },
            type_: Self::TYPE,
        }
    }
}

pub(crate) struct TblisPlan {
    lhs_len: SmallVec<[len_type; 8]>,
    rhs_len: SmallVec<[len_type; 8]>,
    out_len: SmallVec<[len_type; 8]>,
    lhs_stride: SmallVec<[stride_type; 8]>,
    rhs_stride: SmallVec<[stride_type; 8]>,
    out_stride: SmallVec<[stride_type; 8]>,
    lhs_labels: SmallVec<[label_type; 8]>,
    rhs_labels: SmallVec<[label_type; 8]>,
    out_labels: SmallVec<[label_type; 8]>,
}

#[derive(Clone, Copy)]
pub(crate) struct TblisExecution<T> {
    alpha: T,
    beta: T,
    lhs_conj: bool,
    rhs_conj: bool,
}

impl<T> TblisExecution<T> {
    pub(crate) fn new(alpha: T, beta: T, lhs_conj: bool, rhs_conj: bool) -> Self {
        Self {
            alpha,
            beta,
            lhs_conj,
            rhs_conj,
        }
    }

    pub(crate) fn beta(&self) -> T
    where
        T: Copy,
    {
        self.beta
    }
}

pub(crate) fn output_shape<L, R, T>(
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
) -> Result<SmallVec<[usize; 8]>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    validate_dot_general(lhs, rhs, config)?;

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_free = free_dims(
        lhs_shape.len(),
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_dims(
        rhs_shape.len(),
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );

    let mut out = SmallVec::<[usize; 8]>::new();
    out.extend(lhs_free.iter().map(|&axis| lhs_shape[axis]));
    out.extend(rhs_free.iter().map(|&axis| rhs_shape[axis]));
    out.extend(config.lhs_batch_dims.iter().map(|&axis| lhs_shape[axis]));
    Ok(out)
}

pub(crate) fn plan<L, R, T>(
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
    out_shape: &[usize],
    out_strides: &[isize],
) -> Result<Option<TblisPlan>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    let expected_out_shape = output_shape(lhs, rhs, config)?;
    if expected_out_shape.as_slice() != out_shape {
        return Ok(None);
    }

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    if lhs_rank == 0 || rhs_rank == 0 || out_shape.is_empty() {
        return Ok(None);
    }
    if lhs_shape
        .iter()
        .chain(rhs_shape)
        .chain(out_shape)
        .any(|&dim| dim == 0)
    {
        return Ok(None);
    }
    if lhs.offset() < 0 || rhs.offset() < 0 {
        return Ok(None);
    }

    let lhs_strides = lhs.strides()?;
    let rhs_strides = rhs.strides()?;
    if lhs_strides
        .iter()
        .chain(rhs_strides.iter())
        .chain(out_strides.iter())
        .any(|&stride| stride <= 0)
    {
        return Ok(None);
    }

    let out_rank = out_shape.len();
    if lhs_rank > c_int::MAX as usize
        || rhs_rank > c_int::MAX as usize
        || out_rank > c_int::MAX as usize
    {
        return Ok(None);
    }

    let mut labels = TblisLabelAllocator::new();
    let mut lhs_labels = SmallVec::<[label_type; 8]>::from_elem(0, lhs_rank);
    let mut rhs_labels = SmallVec::<[label_type; 8]>::from_elem(0, rhs_rank);
    let mut out_labels = SmallVec::<[label_type; 8]>::new();

    let lhs_free = free_dims(
        lhs_rank,
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_dims(
        rhs_rank,
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );

    for &axis in &lhs_free {
        let label = match labels.next() {
            Some(label) => label,
            None => return Ok(None),
        };
        lhs_labels[axis] = label;
        out_labels.push(label);
    }
    for &axis in &rhs_free {
        let label = match labels.next() {
            Some(label) => label,
            None => return Ok(None),
        };
        rhs_labels[axis] = label;
        out_labels.push(label);
    }
    for (&lhs_axis, &rhs_axis) in config
        .lhs_batch_dims
        .iter()
        .zip(config.rhs_batch_dims.iter())
    {
        let label = match labels.next() {
            Some(label) => label,
            None => return Ok(None),
        };
        lhs_labels[lhs_axis] = label;
        rhs_labels[rhs_axis] = label;
        out_labels.push(label);
    }
    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(config.rhs_contracting_dims.iter())
    {
        let label = match labels.next() {
            Some(label) => label,
            None => return Ok(None),
        };
        lhs_labels[lhs_axis] = label;
        rhs_labels[rhs_axis] = label;
    }

    if lhs_labels.contains(&0) || rhs_labels.contains(&0) {
        return Ok(None);
    }

    lhs_labels.push(0);
    rhs_labels.push(0);
    out_labels.push(0);

    Ok(Some(TblisPlan {
        lhs_len: dims_to_tblis(lhs_shape)?,
        rhs_len: dims_to_tblis(rhs_shape)?,
        out_len: dims_to_tblis(out_shape)?,
        lhs_stride: strides_to_tblis(lhs_strides.as_slice())?,
        rhs_stride: strides_to_tblis(rhs_strides.as_slice())?,
        out_stride: strides_to_tblis(out_strides)?,
        lhs_labels,
        rhs_labels,
        out_labels,
    }))
}

pub(crate) fn execute<T>(
    mut plan: TblisPlan,
    lhs_ptr: *const T,
    rhs_ptr: *const T,
    out: &mut TypedTensorViewMut<'_, T>,
    execution: TblisExecution<T>,
) -> Result<()>
where
    T: TblisGemm,
{
    ensure_runtime_available()?;
    if out.offset() < 0 {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output offset must be non-negative".into(),
        ));
    }
    let out_offset = usize::try_from(out.offset()).map_err(|_| {
        Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output offset does not fit in usize".into(),
        )
    })?;
    let out_storage = out.host_storage_mut()?;
    if out_offset > out_storage.len() {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output offset is outside host storage".into(),
        ));
    }
    let out_ptr = out_storage.as_mut_ptr();

    let mut lhs_tensor = tblis_tensor {
        type_: T::TYPE,
        conj: c_int::from(execution.lhs_conj),
        scalar: T::scalar(execution.alpha),
        data: lhs_ptr.cast_mut() as *mut c_void,
        ndim: c_int::try_from(plan.lhs_len.len()).map_err(|_| {
            Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS lhs rank exceeds c_int range".into(),
            )
        })?,
        len: plan.lhs_len.as_mut_ptr(),
        stride: plan.lhs_stride.as_mut_ptr(),
    };
    let mut rhs_tensor = tblis_tensor {
        type_: T::TYPE,
        conj: c_int::from(execution.rhs_conj),
        scalar: T::scalar(T::one()),
        data: rhs_ptr.cast_mut() as *mut c_void,
        ndim: c_int::try_from(plan.rhs_len.len()).map_err(|_| {
            Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS rhs rank exceeds c_int range".into(),
            )
        })?,
        len: plan.rhs_len.as_mut_ptr(),
        stride: plan.rhs_stride.as_mut_ptr(),
    };
    let mut out_tensor = tblis_tensor {
        type_: T::TYPE,
        conj: 0,
        scalar: T::scalar(execution.beta),
        data: out_ptr.wrapping_add(out_offset) as *mut c_void,
        ndim: c_int::try_from(plan.out_len.len()).map_err(|_| {
            Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS output rank exceeds c_int range".into(),
            )
        })?,
        len: plan.out_len.as_mut_ptr(),
        stride: plan.out_stride.as_mut_ptr(),
    };

    // SAFETY: all tensors are host-backed for the duration of the call; ranks,
    // shapes, positive strides, non-negative offsets, dtype, and label counts
    // are validated before construction. TBLIS owns execution only and receives
    // null comm/context for its default runtime behavior.
    unsafe {
        tblis_tensor_mult(
            std::ptr::null(),
            std::ptr::null(),
            &lhs_tensor,
            plan.lhs_labels.as_ptr(),
            &rhs_tensor,
            plan.rhs_labels.as_ptr(),
            &mut out_tensor,
            plan.out_labels.as_ptr(),
        );
    }
    // Keep variables visibly live across the FFI call.
    let _ = (&mut lhs_tensor, &mut rhs_tensor);
    Ok(())
}

fn ensure_runtime_available() -> Result<()> {
    if !runtime_available()? {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS runtime library is unavailable".into(),
        ));
    }
    Ok(())
}

pub(crate) fn runtime_available() -> Result<bool> {
    #[cfg(feature = "cpu-tblis-runtime")]
    {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

        Ok(*AVAILABLE.get_or_init(|| {
            // INVARIANT: `tblis-ffi` 0.2.6 exposes only panic-based dynamic
            // loading. This one-time, pre-FFI availability probe is a temporary
            // exception to the no-panic-catching audit: it caches the result and
            // never catches a panic from a native TBLIS call. It is effective only
            // with `panic = "unwind"`; remove it after adopting a `tblis-ffi`
            // release containing https://github.com/RESTGroup/tblis-rs/pull/4.
            std::panic::catch_unwind(|| unsafe {
                tblis_ffi::tblis::dyload_lib();
            })
            .is_ok()
        }))
    }
    #[cfg(feature = "cpu-tblis-linked")]
    {
        Ok(true)
    }
}

pub(crate) fn output_element_count(shape: &[usize]) -> Result<usize> {
    checked_product(shape).ok_or_else(|| {
        Error::backend_failure(OP, "TBLIS output element count overflow while allocating")
    })
}

fn free_dims(rank: usize, contracting: &[usize], batch: &[usize]) -> SmallVec<[usize; 8]> {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn dims_to_tblis(dims: &[usize]) -> Result<SmallVec<[len_type; 8]>> {
    dims.iter()
        .map(|&dim| {
            len_type::try_from(dim).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "configuration",
                    format!("TBLIS dimension {dim} exceeds len_type range"),
                )
            })
        })
        .collect()
}

fn strides_to_tblis(strides: &[isize]) -> Result<SmallVec<[stride_type; 8]>> {
    strides
        .iter()
        .map(|&stride| {
            stride_type::try_from(stride).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "configuration",
                    format!("TBLIS stride {stride} exceeds stride_type range"),
                )
            })
        })
        .collect()
}

struct TblisLabelAllocator {
    next: usize,
}

impl TblisLabelAllocator {
    fn new() -> Self {
        Self { next: 0 }
    }

    fn next(&mut self) -> Option<label_type> {
        const LABELS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let byte = *LABELS.get(self.next)?;
        self.next += 1;
        Some(byte as c_char)
    }
}
