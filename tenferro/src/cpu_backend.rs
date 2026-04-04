use computegraph::Operand;
use tenferro_ops::config::DotGeneralConfig;
use tenferro_tensor::Tensor;

#[cfg(feature = "cpu-faer")]
use num_traits::{One, Zero};
#[cfg(feature = "cpu-faer")]
use tenferro_tensor::{col_major_strides, Buffer, TypedTensor};

use super::backend::SemiringCore;

#[cfg(feature = "cpu-faer")]
use super::gemm::faer_gemm::FaerGemm;

/// CPU backend for the v2 engine.
///
/// When the `cpu-faer` feature is enabled (default), batched GEMM uses
/// faer's strided matmul for zero-copy, zero-allocation execution.
/// Otherwise a naive fallback is used.
///
/// # Examples
///
/// ```ignore
/// use tenferro::cpu_backend::CpuBackend;
/// use tenferro::engine::Engine;
///
/// let engine = Engine::new(CpuBackend::new());
/// ```
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend instance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::cpu_backend::CpuBackend;
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiringCore for CpuBackend {
    type Operand = Tensor;

    fn batched_gemm(&mut self, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor {
        #[cfg(feature = "cpu-faer")]
        {
            if let Some(result) = try_faer_gemm(lhs, rhs, config) {
                return result;
            }
        }

        // Naive fallback
        lhs.dot_general(
            rhs,
            &config.lhs_contracting_dims,
            &config.rhs_contracting_dims,
            &config.lhs_batch_dims,
            &config.rhs_batch_dims,
        )
    }

    fn reduce_sum(&mut self, operand: &Tensor, axes: &[usize]) -> Tensor {
        operand.reduce_sum(axes)
    }

    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Operand::add(lhs, rhs)
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Operand::multiply(lhs, rhs)
    }
}

// ---------------------------------------------------------------------------
// GEMM dimension analysis
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-faer")]
/// Computed GEMM parameters derived from a DotGeneralConfig.
struct GemmDims {
    m: usize,
    n: usize,
    k: usize,
    batch_total: usize,
    /// Strides: (m_stride, k_stride, batch_stride) for lhs
    a_rs: isize,
    a_cs: isize,
    a_bs: isize,
    /// Strides: (k_stride, n_stride, batch_stride) for rhs
    b_rs: isize,
    b_cs: isize,
    b_bs: isize,
    /// Strides: (m_stride, n_stride, batch_stride) for output
    c_rs: isize,
    c_cs: isize,
    c_bs: isize,
    /// Output shape: batch ++ lhs_free ++ rhs_free
    out_shape: Vec<usize>,
}

#[cfg(feature = "cpu-faer")]
/// Try to fuse a group of dimensions into a single (total_count, stride) pair.
///
/// Returns `Some((total, step))` when the dims form a contiguous column-major
/// block: dim[0] has stride `step`, dim[1] has stride `step * shape[0]`, etc.
/// Returns `None` if the dimensions cannot be fused.
fn try_fuse_dims(shapes: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    if shapes.is_empty() {
        return Some((1, 0));
    }
    if shapes.len() == 1 {
        return Some((shapes[0], strides[0]));
    }
    // Check that each successive dim's stride == previous stride * previous size
    let base_stride = strides[0];
    let mut expected = base_stride;
    for i in 0..shapes.len() {
        if strides[i] != expected {
            return None;
        }
        expected = strides[i]
            .checked_mul(shapes[i] as isize)
            .unwrap_or(isize::MAX);
    }
    let total: usize = shapes.iter().product();
    Some((total, base_stride))
}

#[cfg(feature = "cpu-faer")]
/// Analyse DotGeneralConfig + tensor shapes/strides to compute GEMM layout.
///
/// Returns `None` if the layout is not amenable to direct GEMM dispatch
/// (e.g. multi-dim groups that cannot be fused).
fn analyse_gemm<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<GemmDims> {
    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();

    // Identify free dims (not batch, not contracting)
    let lhs_free: Vec<usize> = (0..lhs_rank)
        .filter(|d| !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d))
        .collect();
    let rhs_free: Vec<usize> = (0..rhs_rank)
        .filter(|d| !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d))
        .collect();

    // Compute batch shape
    let batch_shapes: Vec<usize> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs.shape[d])
        .collect();
    let batch_total: usize = batch_shapes.iter().product();

    // M, K, N sizes
    let lhs_free_shapes: Vec<usize> = lhs_free.iter().map(|&d| lhs.shape[d]).collect();
    let rhs_free_shapes: Vec<usize> = rhs_free.iter().map(|&d| rhs.shape[d]).collect();
    let contract_shapes: Vec<usize> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs.shape[d])
        .collect();

    let m: usize = lhs_free_shapes.iter().product();
    let n: usize = rhs_free_shapes.iter().product();
    let k: usize = contract_shapes.iter().product();

    // Try to fuse each group of dimensions into a single stride
    let lhs_free_strides: Vec<isize> = lhs_free.iter().map(|&d| lhs.strides[d]).collect();
    let rhs_free_strides: Vec<isize> = rhs_free.iter().map(|&d| rhs.strides[d]).collect();
    let lhs_contract_strides: Vec<isize> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs.strides[d])
        .collect();
    let rhs_contract_strides: Vec<isize> = config
        .rhs_contracting_dims
        .iter()
        .map(|&d| rhs.strides[d])
        .collect();
    let lhs_batch_strides: Vec<isize> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs.strides[d])
        .collect();
    let rhs_batch_strides: Vec<isize> = config
        .rhs_batch_dims
        .iter()
        .map(|&d| rhs.strides[d])
        .collect();

    // Try fusing free, contracting, batch dims in each tensor
    let (_, a_rs) = try_fuse_dims(&lhs_free_shapes, &lhs_free_strides)?;
    let (_, a_cs) = try_fuse_dims(&contract_shapes, &lhs_contract_strides)?;
    let (_, b_rs) = try_fuse_dims(&contract_shapes, &rhs_contract_strides)?;
    let (_, b_cs) = try_fuse_dims(&rhs_free_shapes, &rhs_free_strides)?;
    let (_, a_bs) = try_fuse_dims(&batch_shapes, &lhs_batch_strides)?;
    let (_, b_bs) = try_fuse_dims(&batch_shapes, &rhs_batch_strides)?;

    // Output shape: batch ++ lhs_free ++ rhs_free
    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&batch_shapes);
    out_shape.extend_from_slice(&lhs_free_shapes);
    out_shape.extend_from_slice(&rhs_free_shapes);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_strides = col_major_strides(&out_shape);
    let nb = batch_shapes.len();
    let nm = lhs_free_shapes.len();

    // Compute fused output strides for M, N, batch groups
    let out_m_shapes = &out_shape[nb..nb + nm];
    let out_m_strides = &out_strides[nb..nb + nm];
    let out_n_shapes = &out_shape[nb + nm..];
    let out_n_strides = &out_strides[nb + nm..];
    let out_b_shapes = &out_shape[..nb];
    let out_b_strides = &out_strides[..nb];

    let (_, c_rs) = try_fuse_dims(out_m_shapes, out_m_strides)?;
    let (_, c_cs) = try_fuse_dims(out_n_shapes, out_n_strides)?;
    let (_, c_bs) = try_fuse_dims(out_b_shapes, out_b_strides)?;

    Some(GemmDims {
        m: m.max(1),
        n: n.max(1),
        k: k.max(1),
        batch_total: batch_total.max(1),
        a_rs,
        a_cs,
        a_bs,
        b_rs,
        b_cs,
        b_bs,
        c_rs,
        c_cs,
        c_bs,
        out_shape,
    })
}

// ---------------------------------------------------------------------------
// faer GEMM dispatch
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-faer")]
fn try_faer_gemm(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Option<Tensor> {
    match (lhs, rhs) {
        (Tensor::F64(a), Tensor::F64(b)) => typed_faer_gemm(a, b, config).map(Tensor::F64),
        (Tensor::F32(a), Tensor::F32(b)) => typed_faer_gemm(a, b, config).map(Tensor::F32),
        (Tensor::C64(a), Tensor::C64(b)) => typed_faer_gemm(a, b, config).map(Tensor::C64),
        (Tensor::C32(a), Tensor::C32(b)) => typed_faer_gemm(a, b, config).map(Tensor::C32),
        _ => None,
    }
}

#[cfg(feature = "cpu-faer")]
fn typed_faer_gemm<T: FaerGemm + Copy + Zero + One>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<TypedTensor<T>> {
    let dims = analyse_gemm(lhs, rhs, config)?;

    // Host-only: both buffers must be Host
    let a_data = match &lhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
    };
    let b_data = match &rhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
    };

    // Allocate output
    let out_n: usize = dims.out_shape.iter().product();
    let mut out_data = vec![T::zero(); out_n];
    let c_ptr = out_data.as_mut_ptr();

    let alpha = T::one();
    let beta = T::zero();

    for b_idx in 0..dims.batch_total {
        let a_off = (b_idx as isize) * dims.a_bs;
        let b_off = (b_idx as isize) * dims.b_bs;
        let c_off = (b_idx as isize) * dims.c_bs;

        unsafe {
            T::strided_gemm(
                alpha,
                a_data.offset(a_off),
                dims.m,
                dims.k,
                dims.a_rs,
                dims.a_cs,
                b_data.offset(b_off),
                dims.n,
                dims.b_rs,
                dims.b_cs,
                beta,
                c_ptr.offset(c_off),
                dims.c_rs,
                dims.c_cs,
            );
        }
    }

    let strides = col_major_strides(&dims.out_shape);
    Some(TypedTensor {
        buffer: Buffer::Host(out_data),
        shape: dims.out_shape,
        strides,
        placement: lhs.placement.clone(),
        preferred_compute_device: None,
    })
}
