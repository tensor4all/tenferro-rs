use std::any::TypeId;
use tenferro_tensor::backend::{
    BackendSession, BackendSessionHost, ElementwiseReadOp, SessionCachedDot, TensorAnalytic,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};
use tenferro_tensor::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::DType;
use tenferro_tensor::{with_session_entry_guard, Tensor, TensorRead, TensorWrite};

use super::{
    gemm, structural, unsupported, unsupported_op, WebGpuBackend, WebGpuRuntime,
    WebGpuRuntimeIdentity,
};

/// Marker for the concrete erased WebGPU execution-session target.
#[doc(hidden)]
pub(super) struct WebGpuExecSessionMarker;

/// Borrowed WebGPU execution capability.
#[doc(hidden)]
#[derive(Debug)]
pub struct WebGpuExecSession<'a> {
    backend: &'a mut WebGpuBackend,
}

impl WebGpuExecSession<'_> {
    /// Borrow the provider runtime without exposing the owning backend.
    #[doc(hidden)]
    pub fn runtime(&self) -> &WebGpuRuntime {
        self.backend.runtime()
    }

    /// Return the identity of the borrowed provider runtime.
    #[doc(hidden)]
    pub fn runtime_identity(&self) -> WebGpuRuntimeIdentity {
        self.backend.runtime_identity()
    }
}

/// Visit a WebGPU execution session through the erased backend-session surface.
///
/// The callback receives only the lifetime-bound session capability. The
/// owning [`WebGpuBackend`] never crosses this boundary.
#[doc(hidden)]
pub fn with_webgpu_exec_session<B, R>(
    session: &mut B,
    f: impl for<'a> FnOnce(&'a mut WebGpuExecSession<'a>) -> R,
) -> Option<R>
where
    B: BackendSession + ?Sized,
{
    if session.session_type_id() != std::any::TypeId::of::<WebGpuExecSessionMarker>() {
        return None;
    }
    let data = unsafe { session.session_data_mut() };
    // SAFETY: the exact marker check and BackendSession erased-pointer contract
    // identify the value as WebGpuExecSession for this scoped visit.
    Some(unsafe { f(&mut *(data.cast::<WebGpuExecSession<'static>>())) })
}

macro_rules! delegate {
    ($trait:path {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;)*
    }) => {
        impl $trait for WebGpuExecSession<'_> {
            $(
                fn $method(&mut self, $($arg: $arg_ty),*) -> $ret {
                    self.backend.$method($($arg),*)
                }
            )*
        }
    };
}

impl TensorElementwise for WebGpuExecSession<'_> {
    fn elementwise_read_into(
        &mut self,
        _op: ElementwiseReadOp,
        _inputs: &[TensorRead<'_>],
        _out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        unsupported!("webgpu_elementwise_read_into")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("add", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("add", rhs)?;
        unsupported!("webgpu_add")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sub", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("sub", rhs)?;
        unsupported!("webgpu_sub")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("mul", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("mul", rhs)?;
        unsupported!("webgpu_mul")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("neg", input)?;
        unsupported!("webgpu_neg")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("conj", input)?;
        unsupported!("webgpu_conj")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("div", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("div", rhs)?;
        unsupported!("webgpu_div")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("abs", input)?;
        unsupported!("webgpu_abs")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sign", input)?;
        unsupported!("webgpu_sign")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("maximum", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("maximum", rhs)?;
        unsupported!("webgpu_maximum")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("minimum", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("minimum", rhs)?;
        unsupported!("webgpu_minimum")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        _dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("compare", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("compare", rhs)?;
        unsupported!("webgpu_compare")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("select", pred)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_true)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_false)?;
        unsupported!("webgpu_select")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read inputs first and then raise the same
    // unsupported error. Written against the read inputs directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("clamp", input)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("clamp", lower)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("clamp", upper)?;
        unsupported!("webgpu_clamp")
    }
}

impl TensorAnalytic for WebGpuExecSession<'_> {
    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn exp_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("exp", input)?;
        unsupported!("webgpu_exp")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn log_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("log", input)?;
        unsupported!("webgpu_log")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn sin_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sin", input)?;
        unsupported!("webgpu_sin")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn cos_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("cos", input)?;
        unsupported!("webgpu_cos")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn tanh_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("tanh", input)?;
        unsupported!("webgpu_tanh")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn sqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sqrt", input)?;
        unsupported!("webgpu_sqrt")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn rsqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("rsqrt", input)?;
        unsupported!("webgpu_rsqrt")
    }

    // See the unary analytic read halves above: evaluate the read inputs and then
    // raise the same unsupported error.
    fn pow_read(
        &mut self,
        lhs: tenferro_tensor::TensorRead<'_>,
        rhs: tenferro_tensor::TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("pow", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("pow", rhs)?;
        unsupported!("webgpu_pow")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn expm1_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("expm1", input)?;
        unsupported!("webgpu_expm1")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn log1p_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("log1p", input)?;
        unsupported!("webgpu_log1p")
    }
}

impl TensorStructural for WebGpuExecSession<'_> {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        structural::to_contiguous_read(self.backend, input)
    }

    fn copy_read_into(&mut self, _src: TensorRead<'_>, _dst: TensorWrite<'_>) -> crate::Result<()> {
        unsupported!("WebGpuBackend::copy_read_into")
    }

    fn cast(&mut self, _input: &Tensor, _to: DType) -> crate::Result<Tensor> {
        unsupported!("webgpu_cast")
    }

    fn extract_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_extract_diagonal")
    }

    fn embed_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_embed_diagonal")
    }

    fn tril(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        unsupported!("webgpu_tril")
    }

    fn triu(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        unsupported!("webgpu_triu")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        // The one-shot entry used to run the device transpose; the read half is
        // now that entry, so it must keep the operation rather than report it
        // unsupported.
        let input = tenferro_tensor::backend::read_owned_tensor("transpose", input)?;
        structural::transpose(self.backend, input, perm)
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn reshape_read(&mut self, input: TensorRead<'_>, _shape: &[usize]) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("reshape", input)?;
        unsupported!("webgpu_reshape")
    }

    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the operation
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        _shape: &[usize],
        _dims: &[usize],
    ) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("broadcast_in_dim", input)?;
        unsupported!("webgpu_broadcast_in_dim")
    }
}

impl TensorReduction for WebGpuExecSession<'_> {
    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the reduction
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, _axes: &[usize]) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("reduce_sum", input)?;
        unsupported!("webgpu_reduce_sum")
    }
    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the reduction
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn reduce_prod_read(
        &mut self,
        input: TensorRead<'_>,
        _axes: &[usize],
    ) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?;
        unsupported!("webgpu_reduce_prod")
    }
    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the reduction
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn reduce_max_read(&mut self, input: TensorRead<'_>, _axes: &[usize]) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?;
        unsupported!("webgpu_reduce_max")
    }
    // The old chain was: owned input -> the one-shot method -> its unsupported
    // error; view input -> the read-boundary error. WebGPU rejects the reduction
    // either way, so evaluate the read input first and then raise the same
    // unsupported error. Written against the read input directly so the later
    // removal of the one-shot methods does not need to revisit this.
    fn reduce_min_read(&mut self, input: TensorRead<'_>, _axes: &[usize]) -> crate::Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?;
        unsupported!("webgpu_reduce_min")
    }
}

impl TensorDot for WebGpuExecSession<'_> {
    // The previous read-half default delegated an owned pair to the one-shot
    // method and materialized views through to_contiguous_read before
    // contracting. That default is inlined here so the later removal of the
    // one-shot methods does not need to revisit WebGPU.
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => gemm::dot_general(self.backend, lhs, rhs, config),
            _ => {
                let lhs = self.to_contiguous_read(lhs)?;
                let rhs = self.to_contiguous_read(rhs)?;
                gemm::dot_general(self.backend, &lhs, &rhs, config)
            }
        }
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_with_conj(self.backend, lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

impl TensorIndexing for WebGpuExecSession<'_> {
    fn gather(
        &mut self,
        _operand: &Tensor,
        _start_indices: &Tensor,
        _config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_gather")
    }

    fn scatter(
        &mut self,
        _operand: &Tensor,
        _scatter_indices: &Tensor,
        _updates: &Tensor,
        _config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_scatter")
    }

    fn slice(&mut self, _input: &Tensor, _config: &SliceConfig) -> crate::Result<Tensor> {
        unsupported!("webgpu_slice")
    }

    fn dynamic_slice(
        &mut self,
        _input: &Tensor,
        _starts: &Tensor,
        _slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_dynamic_slice")
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_dynamic_update_slice")
    }

    fn pad(&mut self, _input: &Tensor, _config: &PadConfig) -> crate::Result<Tensor> {
        unsupported!("webgpu_pad")
    }

    fn concatenate(&mut self, _inputs: &[&Tensor], _axis: usize) -> crate::Result<Tensor> {
        unsupported!("webgpu_concatenate")
    }

    fn reverse(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_reverse")
    }
}

impl TensorFusion for WebGpuExecSession<'_> {}

delegate!(TensorBuffer {
    fn reclaim_buffer(tensor: Tensor) -> ();
});

delegate!(TensorDeviceTransfer {
    fn download_to_host(tensor: TensorRead<'_>) -> crate::Result<Tensor>;
    fn upload_host_tensor(tensor: TensorRead<'_>) -> crate::Result<Tensor>;
});

// The cached reads are the trait's provided defaults here: WebGPU owns no
// runtime cache of its own, so the session implements them directly.
impl SessionCachedDot for WebGpuExecSession<'_> {}

impl BackendSession for WebGpuExecSession<'_> {
    fn session_type_id(&self) -> TypeId {
        TypeId::of::<WebGpuExecSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

impl BackendSessionHost for WebGpuBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        let mut session = WebGpuExecSession { backend: self };
        // Nested entry is caught by the portable in-session guard in debug
        // builds; the WebGPU runtime must never re-enter a session closure.
        with_session_entry_guard(|| f(&mut session))
    }
}
