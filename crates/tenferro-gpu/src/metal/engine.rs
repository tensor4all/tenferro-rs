//! The Metal engine: the operation surface the Metal catalogue serves.
//!
//! Every operation admits its operands first (same domain, compact, f32 or Complex32), then
//! plans through `t4a_cuda_kernels::metal` — the same planning the other vendors use — encodes
//! one checked launch against `t4a-metal-runtime`, and commits exactly one command buffer. A
//! dtype or operation this slice does not serve returns a typed error naming what serves it;
//! nothing falls back to the host and nothing is approximated.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use t4a_cuda_kernels::{metal, Entry, UnaryOp};
use t4a_metal_runtime::{Arg, CompileRequest, LaunchConfig, Runtime};
use tenferro_tensor::{
    BackendId, DType, DotGeneralConfig, ElementwiseReadOp, Engine, Error, OperationCapability,
    Result as TensorResult, Tensor, TensorAnalytic, TensorBackendCapability, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorScalar, TensorStructural, TensorWrite,
};

use super::context::{check_own_domain, require_compact, MetalContext};
use super::pending::PendingTouches;
use super::storage::MetalStorage;

/// The Metal engine of one [`MetalContext`].
///
/// An engine borrows the context's device, queue and domain: two engines over one context
/// share them and serialise on the context's one runtime lock.
#[derive(Debug, Clone)]
pub struct MetalEngine {
    context: MetalContext,
}

impl MetalEngine {
    pub(crate) fn new(context: MetalContext) -> Self {
        Self { context }
    }

    /// The context this engine runs on.
    pub fn context(&self) -> &MetalContext {
        &self.context
    }

    /// Refuse a dtype this backend does not compute, naming who does.
    fn kernel_dtype(dtype: DType, op: &'static str) -> TensorResult<t4a_cuda_kernels::DType> {
        match dtype {
            DType::F32 => Ok(t4a_cuda_kernels::DType::F32),
            DType::C32 => Ok(t4a_cuda_kernels::DType::C32),
            DType::F64 | DType::C64 => Err(Error::unsupported(
                op,
                format!(
                    "{dtype:?} has no Apple GPU path (MSL rejects `double`); run this operation on \
                     the paired CpuEngine (MetalContext::cpu_engine), which reads these tensors \
                     without a transfer"
                ),
            )),
            other => Err(Error::unsupported_dtype(
                op,
                other,
                "the Metal backend computes f32 and Complex32 only",
            )),
        }
    }

    fn catalogue(op: &'static str, error: t4a_cuda_kernels::Error) -> Error {
        match error {
            t4a_cuda_kernels::Error::UnsupportedDType => {
                Error::unsupported(op, "the Metal catalogue does not serve this dtype")
            }
            t4a_cuda_kernels::Error::LaunchLimit => Error::unsupported(
                op,
                "the element count is above the 32-bit index a Metal kernel reads",
            ),
            other => Error::runtime_state(op, format!("the Metal plan failed: {other:?}")),
        }
    }

    /// One operand: an owned tensor of this domain with compact strides.
    fn operand<'a>(&self, read: TensorRead<'a>, op: &'static str) -> TensorResult<&'a Tensor> {
        let tensor = read.as_tensor().ok_or_else(|| {
            Error::unsupported(
                op,
                "a borrowed view would need a materialization (U4 slice 2)",
            )
        })?;
        check_own_domain(&self.context, tensor, op)?;
        require_compact(tensor, op)?;
        Ok(tensor)
    }

    /// The element count, as the catalogue's `i64`.
    fn len(tensor: &Tensor, op: &'static str) -> TensorResult<i64> {
        let count = tenferro_tensor::validate::checked_shape_product(op, "shape", tensor.shape())?;
        i64::try_from(count).map_err(|_| Error::unsupported(op, "the element count exceeds i64"))
    }

    /// Encode, admit, commit and record one submission.
    ///
    /// The runtime lock is held across the admission check and the commit: a host guard
    /// takes the same lock to raise an allocation's live count, so a guard cannot appear
    /// between them. Every allocation the submission named records the completion, so a
    /// later host touch of any of them waits for this launch. A `commit` error is handled
    /// before returning: the runtime is synchronised, and when that cannot prove the queue
    /// idle the named allocations are poisoned so host access fails instead of trusting a
    /// stale completion (`commit` submits before it back-pressure-waits, so an error there
    /// does not prove the command buffer never ran).
    fn submit(
        &self,
        op: &'static str,
        admission: &[Arc<AtomicUsize>],
        touched: &[&Arc<PendingTouches>],
        encode: impl FnOnce(&mut Runtime) -> TensorResult<()>,
    ) -> TensorResult<()> {
        let mut runtime = self.context.cell().lock();
        for counter in admission {
            if counter.load(Ordering::SeqCst) != 0 {
                return Err(Error::host_access(
                    op,
                    tenferro_tensor::HostAccessError::MappedForHost,
                ));
            }
        }
        encode(&mut runtime)?;
        // `commit` runs under the lock and may wait for the runtime's in-flight bound
        // (back-pressure). Only host completion waits leave the lock; keeping encoding and
        // commit on the one runtime is what orders submissions.
        let completion = match runtime.commit() {
            Ok(completion) => completion,
            Err(error) => {
                let failure = format!("the Metal submission failed: {error}");
                // `commit` submits the command buffer *before* it back-pressure-waits, so
                // this error does not prove the work is unsubmitted. Prove the queue idle
                // before the caller can touch the buffers; when that cannot be proven,
                // poison every allocation this call named so a later host access reports a
                // backend failure instead of waiting on a stale completion and overwriting
                // bytes the unrecorded launch still reads.
                if let Err(sync_error) = runtime.synchronize() {
                    let reason = format!(
                        "{failure}; the runtime could not be synchronized afterwards \
                         ({sync_error}), so this allocation's pending work is unknown"
                    );
                    for pending in touched {
                        pending.poison(reason.clone());
                    }
                }
                return Err(Error::runtime_state(op, failure));
            }
        };
        for pending in touched {
            pending.record(&completion);
        }
        Ok(())
    }

    /// Compile, bind and launch one flat entry, then submit it.
    fn launch_flat(
        &self,
        op: &'static str,
        entry: Entry,
        plan: metal::Plan,
        admission: &[Arc<AtomicUsize>],
        touched: &[&Arc<PendingTouches>],
        args: &[Arg<'_>],
    ) -> TensorResult<()> {
        let module = metal::source::module_of(entry).ok_or_else(|| {
            Error::unsupported(op, "the Metal source set has no module for this entry")
        })?;
        let symbol = metal::symbol(entry).ok_or_else(|| {
            Error::unsupported(op, "the Metal source set has no symbol for this entry")
        })?;
        let source = metal::source::compose(module);
        let config = LaunchConfig::tiled(
            plan.dispatch.threadgroups,
            plan.dispatch.threads_per_threadgroup,
        );
        self.submit(op, admission, touched, |runtime| {
            let artifact = runtime
                .compile(&CompileRequest::new(module.name, source))
                .map_err(|error| {
                    Error::runtime_state(op, format!("the MSL compile failed: {error}"))
                })?;
            let bound = runtime.bind(&artifact, symbol).map_err(|error| {
                Error::runtime_state(op, format!("the Metal pipeline failed: {error}"))
            })?;
            // SAFETY: the arguments mirror the entry's MSL signature — the destination, then
            // the source buffers, then the by-value scalars, each at its declared index — and
            // `plan.dispatch` is the plan's checked geometry for this element count.
            unsafe { runtime.launch(&bound, config, args) }.map_err(|error| {
                Error::runtime_state(op, format!("the Metal launch failed: {error}"))
            })
        })
    }

    /// The flat elementwise binary family over two same-shaped operands.
    fn binary(
        &self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        entry: Entry,
        op: &'static str,
    ) -> TensorResult<Tensor> {
        let left = self.operand(lhs, op)?;
        let right = self.operand(rhs, op)?;
        if left.shape() != right.shape() {
            return Err(Error::shape_mismatch(op, left.shape(), right.shape()));
        }
        let dtype = left.dtype();
        if dtype != right.dtype() {
            return Err(Error::dtype_mismatch(op, dtype, right.dtype()));
        }
        let kernel = Self::kernel_dtype(dtype, op)?;
        let len = Self::len(left, op)?;
        let plan =
            metal::plan_elementwise(entry, kernel, len).map_err(|e| Self::catalogue(op, e))?;
        let shape = left.shape().to_vec();
        if plan.dispatch.is_empty() {
            // A zero-length operand has nothing to launch; the result is an empty allocation
            // and no submission, which is why no completion is recorded.
            return self.context.cell().allocate(op, dtype, &shape);
        }
        let touches = Arc::new(PendingTouches::new(Arc::clone(self.context.cell().waits())));
        let uninit = self
            .context
            .cell()
            .allocate_uninit(op, plan.bytes as usize)?;
        macro_rules! launch {
            ($t:ty) => {{
                let left = storage_of::<$t>(left, op)?;
                let right = storage_of::<$t>(right, op)?;
                let admission = [left.live_counter(), right.live_counter()];
                let args = [
                    Arg::BufferUninit {
                        buffer: &uninit,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: left.allocation(),
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: right.allocation(),
                        offset: 0,
                    },
                    Arg::I32(kernel.tag()),
                    Arg::U64(len as u64),
                ];
                self.launch_flat(
                    op,
                    entry,
                    plan,
                    &admission,
                    &[left.touches(), right.touches(), &touches],
                    &args,
                )
            }};
        }
        match dtype {
            DType::F32 => launch!(f32),
            DType::C32 => launch!(num_complex::Complex32),
            other => Err(Error::unsupported_dtype(
                op,
                other,
                "the Metal backend computes f32 and Complex32 only",
            )),
        }?;
        // SAFETY: `plan.dispatch` covers every destination element and the entry guards its
        // tail, so the launch above wrote all of them.
        let buffer = unsafe { uninit.assume_init() };
        self.wrap_by_dtype(op, dtype, shape, buffer, touches)
    }

    /// The flat elementwise unary family.
    fn unary(
        &self,
        input: TensorRead<'_>,
        operation: UnaryOp,
        src: DType,
        dst: DType,
        op: &'static str,
    ) -> TensorResult<Tensor> {
        let source = self.operand(input, op)?;
        let kernel_src = Self::kernel_dtype(src, op)?;
        let kernel_dst = Self::kernel_dtype(dst, op)?;
        let len = Self::len(source, op)?;
        let plan = metal::plan_unary(operation, kernel_dst, kernel_src, len)
            .map_err(|e| Self::catalogue(op, e))?;
        let shape = source.shape().to_vec();
        if plan.dispatch.is_empty() {
            return self.context.cell().allocate(op, dst, &shape);
        }
        let touches = Arc::new(PendingTouches::new(Arc::clone(self.context.cell().waits())));
        let uninit = self
            .context
            .cell()
            .allocate_uninit(op, plan.bytes as usize)?;
        macro_rules! launch {
            ($src:ty) => {{
                let source = storage_of::<$src>(source, op)?;
                let admission = [source.live_counter()];
                let args = [
                    Arg::BufferUninit {
                        buffer: &uninit,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: source.allocation(),
                        offset: 0,
                    },
                    Arg::I32(kernel_dst.tag()),
                    Arg::I32(kernel_src.tag()),
                    Arg::I32(operation.tag()),
                    Arg::U64(len as u64),
                ];
                self.launch_flat(
                    op,
                    Entry::Unary,
                    plan,
                    &admission,
                    &[source.touches(), &touches],
                    &args,
                )
            }};
        }
        match src {
            DType::F32 => match dst {
                DType::F32 => launch!(f32),
                other => Err(Error::unsupported_dtype(
                    op,
                    other,
                    "the Metal unary entry writes an f32 source to f32 or Complex32 only",
                )),
            },
            DType::C32 => match dst {
                DType::F32 => launch!(num_complex::Complex32),
                DType::C32 => launch!(num_complex::Complex32),
                other => Err(Error::unsupported_dtype(
                    op,
                    other,
                    "the Metal unary entry writes a Complex32 source to f32 or Complex32 only",
                )),
            },
            other => Err(Error::unsupported_dtype(
                op,
                other,
                "the Metal backend computes f32 and Complex32 only",
            )),
        }?;
        // SAFETY: `plan.dispatch` covers every destination element and the entry guards its
        // tail, so the launch above wrote all of them.
        let buffer = unsafe { uninit.assume_init() };
        self.wrap_by_dtype(op, dst, shape, buffer, touches)
    }

    /// An exact copy of a compact operand into a fresh allocation.
    fn copy_exact(
        &self,
        source: &Tensor,
        shape: Vec<usize>,
        op: &'static str,
    ) -> TensorResult<Tensor> {
        let dtype = source.dtype();
        let kernel = Self::kernel_dtype(dtype, op)?;
        let len = Self::len(source, op)?;
        let plan = metal::plan_copy_exact(kernel, len).map_err(|e| Self::catalogue(op, e))?;
        if plan.dispatch.is_empty() {
            return self.context.cell().allocate(op, dtype, &shape);
        }
        let touches = Arc::new(PendingTouches::new(Arc::clone(self.context.cell().waits())));
        let uninit = self
            .context
            .cell()
            .allocate_uninit(op, plan.bytes as usize)?;
        macro_rules! launch {
            ($t:ty) => {{
                let source = storage_of::<$t>(source, op)?;
                let admission = [source.live_counter()];
                let args = [
                    Arg::BufferUninit {
                        buffer: &uninit,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: source.allocation(),
                        offset: 0,
                    },
                    Arg::I32(kernel.tag()),
                    Arg::U64(len as u64),
                ];
                self.launch_flat(
                    op,
                    Entry::CopyExact,
                    plan,
                    &admission,
                    &[source.touches(), &touches],
                    &args,
                )
            }};
        }
        match dtype {
            DType::F32 => launch!(f32),
            DType::C32 => launch!(num_complex::Complex32),
            other => Err(Error::unsupported_dtype(
                op,
                other,
                "the Metal backend computes f32 and Complex32 only",
            )),
        }?;
        // SAFETY: `plan.dispatch` covers every destination element and the entry guards its
        // tail, so the launch above wrote all of them.
        let buffer = unsafe { uninit.assume_init() };
        self.wrap_by_dtype(op, dtype, shape, buffer, touches)
    }

    fn wrap_by_dtype(
        &self,
        op: &'static str,
        dtype: DType,
        shape: Vec<usize>,
        buffer: t4a_metal_runtime::Buffer,
        touches: Arc<PendingTouches>,
    ) -> TensorResult<Tensor> {
        match dtype {
            DType::F32 => self.context.cell().wrap::<f32>(op, shape, buffer, touches),
            DType::C32 => self
                .context
                .cell()
                .wrap::<num_complex::Complex32>(op, shape, buffer, touches),
            other => Err(Error::unsupported_dtype(
                op,
                other,
                "the Metal backend computes f32 and Complex32 only",
            )),
        }
    }

    /// The one refusal every family this slice does not serve uses.
    fn not_yet(op: &'static str, who: &'static str) -> Error {
        Error::unsupported(
            op,
            format!("the Metal backend does not serve this yet: {who}"),
        )
    }
}

/// The Metal storage behind an operand of a known element type.
fn storage_of<'a, T>(tensor: &'a Tensor, op: &'static str) -> TensorResult<&'a MetalStorage<T>>
where
    T: TensorScalar + 'static,
{
    tensor
        .as_typed::<T>()
        .and_then(|typed| typed.backend_buffer())
        .and_then(|storage| storage.as_any().downcast_ref::<MetalStorage<T>>())
        .ok_or_else(|| Error::unsupported(op, "the operand is not a Metal allocation"))
}

impl TensorElementwise for MetalEngine {
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> TensorResult<()> {
        const NAME: &str = "elementwise_read_into";
        let destination = out
            .as_read()
            .as_tensor()
            .ok_or_else(|| Self::not_yet(NAME, "a borrowed write target (U4 slice 2)"))?;
        check_own_domain(&self.context, destination, NAME)?;
        require_compact(destination, NAME)?;
        let computed = match (op, inputs) {
            (ElementwiseReadOp::Add, [lhs, rhs]) => self.add_read(lhs.clone(), rhs.clone())?,
            (ElementwiseReadOp::Subtract, [lhs, rhs]) => self.sub_read(lhs.clone(), rhs.clone())?,
            (ElementwiseReadOp::Multiply, [lhs, rhs]) => self.mul_read(lhs.clone(), rhs.clone())?,
            (ElementwiseReadOp::Divide, [lhs, rhs]) => self.div_read(lhs.clone(), rhs.clone())?,
            (ElementwiseReadOp::Negate, [input]) => self.neg_read(input.clone())?,
            (ElementwiseReadOp::Conj, [input]) => self.conj_read(input.clone())?,
            _ => {
                return Err(Self::not_yet(
                    NAME,
                    "this elementwise entry (the Metal catalogue carries binary arithmetic and \
                     unary neg/conj only)",
                ))
            }
        };
        if computed.shape() != destination.shape() {
            return Err(Error::shape_mismatch(
                NAME,
                computed.shape(),
                destination.shape(),
            ));
        }
        if computed.dtype() != destination.dtype() {
            return Err(Error::dtype_mismatch(
                NAME,
                computed.dtype(),
                destination.dtype(),
            ));
        }
        // The result is a fresh allocation and its submission is recorded, so the host copy
        // below reads a finished result; a later GPU submission naming `destination` waits on
        // the guard this copy takes.
        copy_tensor_into(&computed, destination, NAME)
    }

    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.binary(lhs, rhs, Entry::Add, "add_read")
    }

    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.binary(lhs, rhs, Entry::Sub, "sub_read")
    }

    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.binary(lhs, rhs, Entry::Mul, "mul_read")
    }

    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        let dtype = lhs.dtype();
        // The catalogue's `div` entry is the real-only division; a complex divide needs the
        // scaled Gauss pair, which is a different entry.
        if matches!(dtype, DType::C32 | DType::C64) {
            return Err(Self::not_yet(
                "div_read",
                "a complex divide (the scaled Gauss entry, U4 slice 2)",
            ));
        }
        self.binary(lhs, rhs, Entry::Div, "div_read")
    }

    fn neg_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        let dtype = input.dtype();
        self.unary(input, UnaryOp::Neg, dtype, dtype, "neg_read")
    }

    fn conj_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        let dtype = input.dtype();
        if matches!(dtype, DType::C32 | DType::C64) {
            return self.unary(input, UnaryOp::Conj, dtype, dtype, "conj_read");
        }
        // Conjugation is the identity on a real dtype, but the result is still a fresh
        // allocation, never a borrow of the input.
        let source = self.operand(input, "conj_read")?;
        let shape = source.shape().to_vec();
        self.copy_exact(source, shape, "conj_read")
    }

    fn abs_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        let dtype = input.dtype();
        let dst = if dtype == DType::C32 {
            DType::F32
        } else {
            dtype
        };
        self.unary(input, UnaryOp::Abs, dtype, dst, "abs_read")
    }

    fn sign_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        let dtype = input.dtype();
        self.unary(input, UnaryOp::Sign, dtype, dtype, "sign_read")
    }

    fn maximum_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "maximum_read",
            "the binary max entry (U4 slice 2)",
        ))
    }

    fn minimum_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "minimum_read",
            "the binary min entry (U4 slice 2)",
        ))
    }

    fn compare_read(
        &mut self,
        _lhs: TensorRead<'_>,
        _rhs: TensorRead<'_>,
        _dir: &tenferro_tensor::CompareDir,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "compare_read",
            "the compare entry (U4 slice 2)",
        ))
    }

    fn select_read(
        &mut self,
        _pred: TensorRead<'_>,
        _on_true: TensorRead<'_>,
        _on_false: TensorRead<'_>,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "select_read",
            "the select entry (U4 slice 2)",
        ))
    }

    fn clamp_read(
        &mut self,
        _input: TensorRead<'_>,
        _lower: TensorRead<'_>,
        _upper: TensorRead<'_>,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet("clamp_read", "the clamp entry (U4 slice 2)"))
    }
}

impl TensorStructural for MetalEngine {
    fn cast(&mut self, input: &Tensor, to: DType) -> TensorResult<Tensor> {
        const OP: &str = "cast";
        let source = self.operand(TensorRead::from_tensor(input), OP)?;
        if source.dtype() != to {
            return Err(Error::unsupported_dtype_conversion(
                OP,
                source.dtype(),
                to,
                "the Metal catalogue has no conversion entry yet; a same-dtype cast is one exact \
                 copy, and a conversion is U4 slice 2"
                    .to_string(),
            ));
        }
        let shape = source.shape().to_vec();
        self.copy_exact(source, shape, OP)
    }

    fn transpose_read(&mut self, _input: TensorRead<'_>, _perm: &[usize]) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "transpose_read",
            "a permuted view is a stride relabelling, which lands with the strided kernels \
             (U4 slice 2)",
        ))
    }

    fn reshape_read(&mut self, _input: TensorRead<'_>, _shape: &[usize]) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "reshape_read",
            "a reshaped view is a stride relabelling, which lands with the strided kernels \
             (U4 slice 2)",
        ))
    }

    fn broadcast_in_dim_read(
        &mut self,
        _input: TensorRead<'_>,
        _shape: &[usize],
        _dims: &[usize],
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "broadcast_in_dim_read",
            "the broadcast entry (U4 slice 2)",
        ))
    }

    fn extract_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "extract_diagonal",
            "the diagonal entries (U4 slice 2)",
        ))
    }

    fn embed_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "embed_diagonal",
            "the diagonal entries (U4 slice 2)",
        ))
    }

    fn tril(&mut self, _input: &Tensor, _k: i64) -> TensorResult<Tensor> {
        Err(Self::not_yet("tril", "the triangular entries (U4 slice 2)"))
    }

    fn triu(&mut self, _input: &Tensor, _k: i64) -> TensorResult<Tensor> {
        Err(Self::not_yet("triu", "the triangular entries (U4 slice 2)"))
    }
}

impl TensorReduction for MetalEngine {
    fn reduce_sum_read(&mut self, _input: TensorRead<'_>, _axes: &[usize]) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "reduce_sum_read",
            "reductions, which go through MetalPerformanceShaders (U4 slice 3)",
        ))
    }

    fn reduce_prod_read(
        &mut self,
        _input: TensorRead<'_>,
        _axes: &[usize],
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "reduce_prod_read",
            "reductions, which go through MetalPerformanceShaders (U4 slice 3)",
        ))
    }

    fn reduce_max_read(&mut self, _input: TensorRead<'_>, _axes: &[usize]) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "reduce_max_read",
            "reductions, which go through MetalPerformanceShaders (U4 slice 3)",
        ))
    }

    fn reduce_min_read(&mut self, _input: TensorRead<'_>, _axes: &[usize]) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "reduce_min_read",
            "reductions, which go through MetalPerformanceShaders (U4 slice 3)",
        ))
    }
}

impl TensorAnalytic for MetalEngine {
    fn exp_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "exp_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn log_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "log_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn sin_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "sin_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn cos_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "cos_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn tanh_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "tanh_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn sqrt_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "sqrt_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn rsqrt_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "rsqrt_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn pow_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "pow_read",
            "the analytic family (U4 slice 2)",
        ))
    }

    fn expm1_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "expm1_read",
            "MSL has no expm1; the entry is refused rather than approximated",
        ))
    }

    fn log1p_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "log1p_read",
            "MSL has no log1p; the entry is refused rather than approximated",
        ))
    }

    fn erf_read(&mut self, _input: TensorRead<'_>) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "erf_read",
            "MSL has no erf; the entry is refused rather than approximated",
        ))
    }
}

impl TensorIndexing for MetalEngine {
    fn gather(
        &mut self,
        _operand: &Tensor,
        _start_indices: &Tensor,
        _config: &tenferro_tensor::GatherConfig,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "gather",
            "the indexing family, which is not a U4 slice",
        ))
    }

    fn scatter(
        &mut self,
        _operand: &Tensor,
        _scatter_indices: &Tensor,
        _updates: &Tensor,
        _config: &tenferro_tensor::ScatterConfig,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "scatter",
            "the indexing family, which is not a U4 slice",
        ))
    }

    fn slice(
        &mut self,
        _input: &Tensor,
        _config: &tenferro_tensor::SliceConfig,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "slice",
            "the indexing family, which is not a U4 slice",
        ))
    }

    fn dynamic_slice(
        &mut self,
        _input: &Tensor,
        _starts: &Tensor,
        _slice_sizes: &[usize],
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "dynamic_slice",
            "the indexing family, which is not a U4 slice",
        ))
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "dynamic_update_slice",
            "the indexing family, which is not a U4 slice",
        ))
    }

    fn pad(
        &mut self,
        _input: &Tensor,
        _config: &tenferro_tensor::PadConfig,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "pad",
            "the indexing family, which is not a U4 slice",
        ))
    }

    fn concatenate(&mut self, _inputs: &[&Tensor], _axis: usize) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "concatenate",
            "the indexing family, which is not a U4 slice",
        ))
    }

    fn reverse(&mut self, _input: &Tensor, _axes: &[usize]) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "reverse",
            "the indexing family, which is not a U4 slice",
        ))
    }
}

impl TensorDot for MetalEngine {
    fn dot_general_read(
        &mut self,
        _lhs: TensorRead<'_>,
        _rhs: TensorRead<'_>,
        _config: &DotGeneralConfig,
    ) -> TensorResult<Tensor> {
        Err(Self::not_yet(
            "dot_general_read",
            "contractions, which go through MPSMatrixMultiplication (U5)",
        ))
    }
}

impl TensorFusion for MetalEngine {}

impl TensorBuffer for MetalEngine {}

impl tenferro_tensor::SessionCachedDot for MetalEngine {}

impl Engine for MetalEngine {}

impl TensorDeviceTransfer for MetalEngine {
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> TensorResult<Tensor> {
        self.context.download(tensor)
    }

    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> TensorResult<Tensor> {
        self.context.upload(tensor)
    }
}

impl TensorBackendCapability for MetalEngine {
    fn backend_id(&self) -> BackendId {
        super::capability::METAL_BACKEND
    }

    fn capabilities(&self) -> &'static [OperationCapability] {
        super::metal_capabilities()
    }
}

/// Copy one computed tensor's elements into a same-shaped shared destination.
fn copy_tensor_into(source: &Tensor, destination: &Tensor, op: &'static str) -> TensorResult<()> {
    macro_rules! copy {
        ($t:ty) => {{
            let values = storage_of::<$t>(source, op)?
                .host_slice()
                .map_err(|error| Error::host_access(op, error))?;
            storage_of::<$t>(destination, op)?
                .write_from_slice(values)
                .map_err(|error| Error::host_access(op, error))
        }};
    }
    match source.dtype() {
        DType::F32 => copy!(f32),
        DType::C32 => copy!(num_complex::Complex32),
        other => Err(Error::unsupported_dtype(
            op,
            other,
            "the Metal backend computes f32 and Complex32 only",
        )),
    }
}
