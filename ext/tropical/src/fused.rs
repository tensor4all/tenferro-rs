//! Fused tropical dot-general as an `ExtensionOp`.
//!
//! This module implements `FusedTropicalDotGeneralOp`, the first canonical
//! out-of-tree [`ExtensionOp`] in the `tenferro` ecosystem. It is the
//! committed self-test for the `ExtensionOp` substrate specified in
//! `docs/spec/extension-op.md`.
//!
//! # What this provides
//!
//! - [`TropicalKind`] enum selecting the tropical semiring (`MaxPlus` or
//!   `MinPlus`). Max-mul is a possible later addition.
//! - [`FusedTropicalDotGeneralOp`] — the [`ExtensionOp`] payload, carrying
//!   only the [`TropicalKind`].
//! - [`FusedTropicalDotGeneralFactory`] — the [`ExtensionFactory`] that
//!   registers the family in the process-local registry.
//! - [`register_fused_tropical`] — explicit registration entry point. Tests
//!   should call it once before use; duplicate registrations are tolerated.
//!
//! # Semantics
//!
//! For rank-2 inputs `a: [M, K]` and `b: [K, N]`:
//!
//! - `MaxPlus`: `out[i, j] = max_k (a[i, k] + b[k, j])`
//! - `MinPlus`: `out[i, j] = min_k (a[i, k] + b[k, j])`
//!
//! Output shape is `[M, N]`. Contracting axes are `a` axis `1` and `b` axis
//! `0` (same as `matmul`). The op currently supports rank-2 inputs.
//!
//! # Primal implementation
//!
//! [`FusedTropicalDotGeneralOp::eager_execute`] computes the primal directly
//! from host data for `F64`. This is a correctness-first implementation, not
//! a performance-optimised fused GEMM kernel. The "fused" label is a
//! packaging decision: a real fused kernel would live behind the same trait,
//! and no call site would need to change.
//!
//! # Argmax capture in AD (spec Section 14 informative sketch)
//!
//! The `ExtensionOp` spec's worked example (Section 14, **informative**) sketches
//! `Gather` / `Scatter` on saved argmax indices. The core op vocabulary
//! does not include an `ArgMax` op, so this implementation uses the
//! mathematically equivalent **indicator-mask** approach — the same construction used by
//! the core `ReduceMax` / `ReduceMin` AD rule
//! (`tenferro-ops/src/ad/contraction.rs::linearize_reduce_chooser`) —
//! which produces the same (subgradient) gradient as the argmax-based
//! formulation, modulo how ties are split. Argmax-based Scatter and
//! indicator-based `Compare(Eq) + Mul + ReduceSum + Div` are two
//! implementations of the same mathematical VJP; only the indicator path
//! is expressible in the core op vocabulary.
//!
//! The AD path therefore emits only core `StdTensorOp` variants
//! (`BroadcastInDim`, `Add`, `ReduceMax` / `ReduceMin`, `Compare(Eq)`,
//! `Select`, `ReduceSum`, `Div`, `Mul`, `Constant`, `Transpose`), preserving
//! the ad-contract.md closure rule.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    register_extension, ExtensionFactory, ExtensionOp, ExtensionRegistryError,
};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim};
use tenferro_tensor::{CompareDir, DType, Tensor, TypedTensor};

/// The tropical semiring flavor selected by a [`FusedTropicalDotGeneralOp`].
///
/// The fused op supports [`TropicalKind::MaxPlus`] (the usual tropical semiring)
/// and [`TropicalKind::MinPlus`] (the min-plus, or "shortest-path",
/// semiring). A `MaxMul` variant is a straightforward extension that can
/// land later without breaking the v1 family contract.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::fused::TropicalKind;
///
/// let k = TropicalKind::MaxPlus;
/// assert_eq!(k, TropicalKind::MaxPlus);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TropicalKind {
    /// Max-plus: `out[i,j] = max_k (a[i,k] + b[k,j])`.
    MaxPlus,
    /// Min-plus: `out[i,j] = min_k (a[i,k] + b[k,j])`.
    MinPlus,
}

/// Fused tropical dot-general [`ExtensionOp`].
///
/// Carries only the [`TropicalKind`]. The contracting axes are fixed by
/// the rank-2 shape rule (see module docs): `a: [M, K]` contracts over
/// axis 1 with `b: [K, N]` along axis 0.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::fused::{FusedTropicalDotGeneralOp, TropicalKind};
///
/// let op = FusedTropicalDotGeneralOp::new(TropicalKind::MaxPlus);
/// assert_eq!(op.kind(), TropicalKind::MaxPlus);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FusedTropicalDotGeneralOp {
    kind: TropicalKind,
}

impl FusedTropicalDotGeneralOp {
    /// Create a new fused tropical dot-general op with the given kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::fused::{FusedTropicalDotGeneralOp, TropicalKind};
    ///
    /// let op = FusedTropicalDotGeneralOp::new(TropicalKind::MinPlus);
    /// assert_eq!(op.kind(), TropicalKind::MinPlus);
    /// ```
    pub fn new(kind: TropicalKind) -> Self {
        Self { kind }
    }

    /// Return the tropical kind this op implements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::fused::{FusedTropicalDotGeneralOp, TropicalKind};
    ///
    /// let op = FusedTropicalDotGeneralOp::new(TropicalKind::MaxPlus);
    /// assert_eq!(op.kind(), TropicalKind::MaxPlus);
    /// ```
    pub fn kind(&self) -> TropicalKind {
        self.kind
    }
}

/// Stable family identifier (spec Section 5).
///
/// Kept as a `const` so the `&'static str` return site in
/// [`ExtensionOp::family_id`] is cheap and deterministic.
pub const FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID: &str = "tenferro-ext-tropical.fused_dot_general.v1";

impl ExtensionOp for FusedTropicalDotGeneralOp {
    fn family_id(&self) -> &'static str {
        FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        // Entire payload is just the kind discriminant. Use `write_u8` so
        // the call goes through the `&mut dyn Hasher` vtable without
        // requiring `Sized` on `hasher`.
        let disc: u8 = match self.kind {
            TropicalKind::MaxPlus => 0,
            TropicalKind::MinPlus => 1,
        };
        hasher.write_u8(disc);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<FusedTropicalDotGeneralOp>()
            .is_some_and(|that| self.kind == that.kind)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        assert_eq!(
            input_dtypes.len(),
            2,
            "FusedTropicalDotGeneral expects 2 inputs, got {}",
            input_dtypes.len()
        );
        assert_eq!(
            input_dtypes[0], input_dtypes[1],
            "FusedTropicalDotGeneral expects matching input dtypes; got {:?} and {:?}",
            input_dtypes[0], input_dtypes[1]
        );
        assert_eq!(
            input_shapes[0].len(),
            2,
            "FusedTropicalDotGeneral.a must be rank-2, got rank {}",
            input_shapes[0].len()
        );
        assert_eq!(
            input_shapes[1].len(),
            2,
            "FusedTropicalDotGeneral.b must be rank-2, got rank {}",
            input_shapes[1].len()
        );
        // Contracting axes are a[1] and b[0]; output is [a[0], b[1]].
        let out_shape = vec![input_shapes[0][0].clone(), input_shapes[1][1].clone()];
        vec![(input_dtypes[0], out_shape)]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        assert_eq!(
            inputs.len(),
            2,
            "FusedTropicalDotGeneral::eager_execute expects 2 inputs"
        );
        let a = inputs[0];
        let b = inputs[1];
        assert_eq!(
            a.shape().len(),
            2,
            "FusedTropicalDotGeneral.a must be rank-2"
        );
        assert_eq!(
            b.shape().len(),
            2,
            "FusedTropicalDotGeneral.b must be rank-2"
        );
        assert_eq!(
            a.shape()[1],
            b.shape()[0],
            "FusedTropicalDotGeneral: contracting-axis mismatch: a[1]={} vs b[0]={}",
            a.shape()[1],
            b.shape()[0]
        );

        match (a, b) {
            (Tensor::F64(ta), Tensor::F64(tb)) => {
                Ok(vec![Tensor::F64(tropical_gemm_f64(ta, tb, self.kind))])
            }
            (Tensor::F32(ta), Tensor::F32(tb)) => {
                Ok(vec![Tensor::F32(tropical_gemm_f32(ta, tb, self.kind))])
            }
            _ => Err(tenferro_tensor::Error::BackendFailure {
                op: "extension",
                message: format!(
                    "family_id={}: unsupported dtypes {:?} / {:?} (only F32/F64)",
                    FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID,
                    a.dtype(),
                    b.dtype()
                ),
            }),
        }
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<StdTensorOp>,
        primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        debug_assert_eq!(tangent_in.len(), 2);
        // Deferred zero-tangent policy (spec Section 10): if both tangents
        // are inactive, return `None`.
        if tangent_in[0].is_none() && tangent_in[1].is_none() {
            return vec![None];
        }
        let (a_ref, b_ref) = (
            ValRef::External(primal_in[0].clone()),
            ValRef::External(primal_in[1].clone()),
        );
        let dtype = ctx.dtype_of(&a_ref);

        // --- Build the 3D sum tensor S[i, k, j] = a[i, k] + b[k, j] ---
        // Broadcast `a: [M, K]` to [M, K, N] (axes 0, 1), reading N from b.
        let a_3d = emit_broadcast_mkn_from_mk(builder, a_ref.clone(), b_ref.clone(), 0);
        // Broadcast `b: [K, N]` to [M, K, N] (axes 1, 2), reading M from a.
        let b_3d = emit_broadcast_mkn_from_kn(builder, b_ref.clone(), a_ref.clone(), 1);
        let sum_3d = builder.add_op(
            StdTensorOp::Add,
            vec![ValRef::Local(a_3d), ValRef::Local(b_3d)],
            OpMode::Primal,
        )[0];
        // --- Compute the reduction answer M[i, j] (rank-2) and its 3D broadcast ---
        let reduce_op = match self.kind {
            TropicalKind::MaxPlus => StdTensorOp::ReduceMax { axes: vec![1] },
            TropicalKind::MinPlus => StdTensorOp::ReduceMin { axes: vec![1] },
        };
        let answer_2d = builder.add_op(reduce_op, vec![ValRef::Local(sum_3d)], OpMode::Primal)[0];
        // Broadcast answer back to 3D: [M, N] -> [M, K, N] along axes [0, 2].
        // We need the 3D shape; use S as shape source.
        let answer_3d = emit_broadcast_2d_to_3d(
            builder,
            ValRef::Local(answer_2d),
            ValRef::Local(sum_3d),
            &[0, 2],
        );
        // --- Indicator mask W[i,k,j] = 1 iff S[i,k,j] == M[i,j] ---
        let indicators_bool = builder.add_op(
            StdTensorOp::Compare(CompareDir::Eq),
            vec![ValRef::Local(sum_3d), ValRef::Local(answer_3d)],
            OpMode::Primal,
        )[0];
        let indicators = emit_bool_to_scalar(builder, indicators_bool, dtype);
        // --- counts[i, j] = sum_k W[i, k, j] ---
        let counts_2d = builder.add_op(
            StdTensorOp::ReduceSum { axes: vec![1] },
            vec![ValRef::Local(indicators)],
            OpMode::Primal,
        )[0];
        // --- Compute the tangent S_dot[i, k, j] = (a_dot[i,k] + b_dot[k,j]) ---
        // (treating missing tangents as zero; if only one is present, use it alone.)
        let s_dot = emit_sum_3d_tangent(builder, tangent_in, a_ref.clone(), b_ref.clone());
        // --- weighted_tangent[i, k, j] = indicators * S_dot ---
        let weighted = builder.add_op(
            StdTensorOp::Mul,
            vec![ValRef::Local(indicators), ValRef::Local(s_dot)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        )[0];
        // --- reduce along K: tangent_sum[i, j] = sum_k weighted[i, k, j] ---
        let tangent_sum = builder.add_op(
            StdTensorOp::ReduceSum { axes: vec![1] },
            vec![ValRef::Local(weighted)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        )[0];
        // --- out_dot[i, j] = tangent_sum[i, j] / counts[i, j] ---
        let out_dot = builder.add_op(
            StdTensorOp::Div,
            vec![ValRef::Local(tangent_sum), ValRef::Local(counts_2d)],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        )[0];
        vec![Some(out_dot)]
    }

    fn transpose_rule(
        &self,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        mode: &OpMode,
        ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        let Some(ct) = cotangent_out[0] else {
            return vec![None, None];
        };
        let active_mask = match mode {
            OpMode::Linear { active_mask } => active_mask.clone(),
            OpMode::Primal => return vec![None, None],
        };

        let (a_ref, b_ref) = (inputs[0].clone(), inputs[1].clone());
        let dtype = ctx.dtype_of(&a_ref);

        // Rebuild the 3D sum tensor, answer broadcast, indicators, and counts
        // (all primal-mode; the same construction as in `linearize`).
        let a_3d = emit_broadcast_mkn_from_mk_emitter(emitter, a_ref.clone(), b_ref.clone(), 0);
        let b_3d = emit_broadcast_mkn_from_kn_emitter(emitter, b_ref.clone(), a_ref.clone(), 1);
        let sum_3d = emitter.add_op(
            StdTensorOp::Add,
            vec![ValRef::Local(a_3d), ValRef::Local(b_3d)],
            OpMode::Primal,
        )[0];
        let reduce_op = match self.kind {
            TropicalKind::MaxPlus => StdTensorOp::ReduceMax { axes: vec![1] },
            TropicalKind::MinPlus => StdTensorOp::ReduceMin { axes: vec![1] },
        };
        let answer_2d = emitter.add_op(reduce_op, vec![ValRef::Local(sum_3d)], OpMode::Primal)[0];
        let answer_3d = emit_broadcast_2d_to_3d_emitter(
            emitter,
            ValRef::Local(answer_2d),
            ValRef::Local(sum_3d),
            &[0, 2],
        );
        let indicators_bool = emitter.add_op(
            StdTensorOp::Compare(CompareDir::Eq),
            vec![ValRef::Local(sum_3d), ValRef::Local(answer_3d)],
            OpMode::Primal,
        )[0];
        let indicators = emit_bool_to_scalar_emitter(emitter, indicators_bool, dtype);
        let counts_2d = emitter.add_op(
            StdTensorOp::ReduceSum { axes: vec![1] },
            vec![ValRef::Local(indicators)],
            OpMode::Primal,
        )[0];
        let counts_3d = emit_broadcast_2d_to_3d_emitter(
            emitter,
            ValRef::Local(counts_2d),
            ValRef::Local(sum_3d),
            &[0, 2],
        );
        // weights[i, k, j] = indicators[i, k, j] / counts[i, j]
        let weights = emitter.add_op(
            StdTensorOp::Div,
            vec![ValRef::Local(indicators), ValRef::Local(counts_3d)],
            OpMode::Primal,
        )[0];
        // Broadcast cot_out[i, j] -> 3D along [0, 2].
        let cot_3d = emit_broadcast_2d_to_3d_emitter(
            emitter,
            ValRef::Local(ct),
            ValRef::Local(sum_3d),
            &[0, 2],
        );
        let weighted = emitter.add_op(
            StdTensorOp::Mul,
            vec![ValRef::Local(weights), ValRef::Local(cot_3d)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        )[0];

        let mut result = vec![None, None];
        if active_mask.first().copied().unwrap_or(false) {
            // cot_a[i, k] = sum_j weighted[i, k, j]
            let cot_a = emitter.add_op(
                StdTensorOp::ReduceSum { axes: vec![2] },
                vec![ValRef::Local(weighted)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            )[0];
            result[0] = Some(cot_a);
        }
        if active_mask.get(1).copied().unwrap_or(false) {
            // cot_b[k, j] = sum_i weighted[i, k, j]
            let cot_b = emitter.add_op(
                StdTensorOp::ReduceSum { axes: vec![0] },
                vec![ValRef::Local(weighted)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            )[0];
            result[1] = Some(cot_b);
        }
        result
    }
}

// ----------------------------------------------------------------------
// Helpers for AD-rule graph emission.
// ----------------------------------------------------------------------

/// Emit `BroadcastInDim(a, [M, K, N], [0, 1])` reading `N` from `b`'s axis 1.
///
/// This is the `linearize` version. See `emit_broadcast_mkn_from_mk_emitter`
/// for the `transpose_rule` flavour (same op, different emitter trait).
///
/// `input_idx_of_a` tells the emitted `BroadcastInDim` where `a` sits in the
/// input list so the `InputDim` shape references line up.
fn emit_broadcast_mkn_from_mk(
    builder: &mut FragmentBuilder<StdTensorOp>,
    a: ValRef<StdTensorOp>,
    b: ValRef<StdTensorOp>,
    input_idx_of_a: usize,
) -> LocalValId {
    let (a_idx, b_idx) = (input_idx_of_a, 1 - input_idx_of_a);
    let shape = vec![
        DimExpr::InputDim {
            input_idx: a_idx,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: a_idx,
            axis: 1,
        },
        DimExpr::InputDim {
            input_idx: b_idx,
            axis: 1,
        },
    ];
    let inputs = if input_idx_of_a == 0 {
        vec![a, b]
    } else {
        vec![b, a]
    };
    builder.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![0, 1],
        },
        inputs,
        OpMode::Primal,
    )[0]
}

fn emit_broadcast_mkn_from_mk_emitter(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    a: ValRef<StdTensorOp>,
    b: ValRef<StdTensorOp>,
    input_idx_of_a: usize,
) -> LocalValId {
    let (a_idx, b_idx) = (input_idx_of_a, 1 - input_idx_of_a);
    let shape = vec![
        DimExpr::InputDim {
            input_idx: a_idx,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: a_idx,
            axis: 1,
        },
        DimExpr::InputDim {
            input_idx: b_idx,
            axis: 1,
        },
    ];
    let inputs = if input_idx_of_a == 0 {
        vec![a, b]
    } else {
        vec![b, a]
    };
    emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![0, 1],
        },
        inputs,
        OpMode::Primal,
    )[0]
}

/// Emit `BroadcastInDim(b, [M, K, N], [1, 2])` reading `M` from `a`'s axis 0.
fn emit_broadcast_mkn_from_kn(
    builder: &mut FragmentBuilder<StdTensorOp>,
    b: ValRef<StdTensorOp>,
    a: ValRef<StdTensorOp>,
    input_idx_of_b: usize,
) -> LocalValId {
    let (b_idx, a_idx) = (input_idx_of_b, 1 - input_idx_of_b);
    let shape = vec![
        DimExpr::InputDim {
            input_idx: a_idx,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: b_idx,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: b_idx,
            axis: 1,
        },
    ];
    let inputs = if input_idx_of_b == 0 {
        vec![b, a]
    } else {
        vec![a, b]
    };
    builder.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![1, 2],
        },
        inputs,
        OpMode::Primal,
    )[0]
}

fn emit_broadcast_mkn_from_kn_emitter(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    b: ValRef<StdTensorOp>,
    a: ValRef<StdTensorOp>,
    input_idx_of_b: usize,
) -> LocalValId {
    let (b_idx, a_idx) = (input_idx_of_b, 1 - input_idx_of_b);
    let shape = vec![
        DimExpr::InputDim {
            input_idx: a_idx,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: b_idx,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: b_idx,
            axis: 1,
        },
    ];
    let inputs = if input_idx_of_b == 0 {
        vec![b, a]
    } else {
        vec![a, b]
    };
    emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![1, 2],
        },
        inputs,
        OpMode::Primal,
    )[0]
}

/// Emit a broadcast of a rank-2 tensor to rank-3, using a rank-3 shape source.
///
/// `dims` gives the axes in the rank-3 target that the rank-2 input maps
/// onto (length-2, strictly increasing).
fn emit_broadcast_2d_to_3d(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    shape_source_3d: ValRef<StdTensorOp>,
    dims: &[usize],
) -> LocalValId {
    debug_assert_eq!(dims.len(), 2);
    // Shape is read from shape_source_3d at input index 1.
    let shape: Vec<DimExpr> = (0..3)
        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    builder.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: dims.to_vec(),
        },
        vec![input, shape_source_3d],
        OpMode::Primal,
    )[0]
}

fn emit_broadcast_2d_to_3d_emitter(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    shape_source_3d: ValRef<StdTensorOp>,
    dims: &[usize],
) -> LocalValId {
    debug_assert_eq!(dims.len(), 2);
    let shape: Vec<DimExpr> = (0..3)
        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: dims.to_vec(),
        },
        vec![input, shape_source_3d],
        OpMode::Primal,
    )[0]
}

/// Convert a boolean indicator tensor (from `Compare(Eq)`) to a numeric
/// 0/1 tensor in `target_dtype` via `Select(pred, one, zero)`.
///
/// `Compare` in the core op vocabulary produces a boolean-valued tensor.
/// `Select` accepts a boolean predicate and returns a tensor with the
/// element dtype of the `on_true` / `on_false` branches. We therefore use
/// constant scalars of the same dtype as the surrounding computation and
/// broadcast them to match the predicate's shape.
fn emit_bool_to_scalar(
    builder: &mut FragmentBuilder<StdTensorOp>,
    pred: LocalValId,
    target_dtype: DType,
) -> LocalValId {
    // Broadcast scalar constants to the same shape as `pred` using the
    // pred as the shape source.
    let zero = builder.add_op(
        StdTensorOp::Constant {
            dtype: target_dtype,
            bytes: zero_bytes(target_dtype),
        },
        vec![],
        OpMode::Primal,
    )[0];
    let one = builder.add_op(
        StdTensorOp::Constant {
            dtype: target_dtype,
            bytes: one_bytes(target_dtype),
        },
        vec![],
        OpMode::Primal,
    )[0];
    // Scalar -> 3D with shape read from `pred` at input index 1.
    let shape: Vec<DimExpr> = (0..3)
        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    let zero_bc = builder.add_op(
        StdTensorOp::BroadcastInDim {
            shape: shape.clone(),
            dims: vec![],
        },
        vec![ValRef::Local(zero), ValRef::Local(pred)],
        OpMode::Primal,
    )[0];
    let one_bc = builder.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![],
        },
        vec![ValRef::Local(one), ValRef::Local(pred)],
        OpMode::Primal,
    )[0];
    builder.add_op(
        StdTensorOp::Select,
        vec![
            ValRef::Local(pred),
            ValRef::Local(one_bc),
            ValRef::Local(zero_bc),
        ],
        OpMode::Primal,
    )[0]
}

fn emit_bool_to_scalar_emitter(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    pred: LocalValId,
    target_dtype: DType,
) -> LocalValId {
    let zero = emitter.add_op(
        StdTensorOp::Constant {
            dtype: target_dtype,
            bytes: zero_bytes(target_dtype),
        },
        vec![],
        OpMode::Primal,
    )[0];
    let one = emitter.add_op(
        StdTensorOp::Constant {
            dtype: target_dtype,
            bytes: one_bytes(target_dtype),
        },
        vec![],
        OpMode::Primal,
    )[0];
    let shape: Vec<DimExpr> = (0..3)
        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    let zero_bc = emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape: shape.clone(),
            dims: vec![],
        },
        vec![ValRef::Local(zero), ValRef::Local(pred)],
        OpMode::Primal,
    )[0];
    let one_bc = emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![],
        },
        vec![ValRef::Local(one), ValRef::Local(pred)],
        OpMode::Primal,
    )[0];
    emitter.add_op(
        StdTensorOp::Select,
        vec![
            ValRef::Local(pred),
            ValRef::Local(one_bc),
            ValRef::Local(zero_bc),
        ],
        OpMode::Primal,
    )[0]
}

/// Emit the 3D tangent `S_dot[i, k, j]` from per-input tangents.
///
/// - If only `tangent_in[0]` (`a_dot`) is active: emit
///   `BroadcastInDim(a_dot, [M, K, N], [0, 1])`.
/// - If only `tangent_in[1]` (`b_dot`) is active: emit
///   `BroadcastInDim(b_dot, [M, K, N], [1, 2])`.
/// - If both are active: emit both broadcasts and sum them with `Add`.
/// - Neither active: unreachable (the caller guards against it).
fn emit_sum_3d_tangent(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    a_ref: ValRef<StdTensorOp>,
    b_ref: ValRef<StdTensorOp>,
) -> LocalValId {
    match (tangent_in[0], tangent_in[1]) {
        (Some(da), None) => {
            // `da` is local; shape refs `a` at 1, `b` at 2.
            let shape = vec![
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 1,
                },
                DimExpr::InputDim {
                    input_idx: 2,
                    axis: 1,
                },
            ];
            builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: vec![0, 1],
                },
                vec![ValRef::Local(da), a_ref, b_ref],
                OpMode::Linear {
                    active_mask: vec![true, false, false],
                },
            )[0]
        }
        (None, Some(db)) => {
            let shape = vec![
                DimExpr::InputDim {
                    input_idx: 2,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 1,
                },
            ];
            builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: vec![1, 2],
                },
                vec![ValRef::Local(db), b_ref, a_ref],
                OpMode::Linear {
                    active_mask: vec![true, false, false],
                },
            )[0]
        }
        (Some(da), Some(db)) => {
            let da_3d = {
                let shape = vec![
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 1,
                    },
                    DimExpr::InputDim {
                        input_idx: 2,
                        axis: 1,
                    },
                ];
                builder.add_op(
                    StdTensorOp::BroadcastInDim {
                        shape,
                        dims: vec![0, 1],
                    },
                    vec![ValRef::Local(da), a_ref.clone(), b_ref.clone()],
                    OpMode::Linear {
                        active_mask: vec![true, false, false],
                    },
                )[0]
            };
            let db_3d = {
                let shape = vec![
                    DimExpr::InputDim {
                        input_idx: 2,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 1,
                    },
                ];
                builder.add_op(
                    StdTensorOp::BroadcastInDim {
                        shape,
                        dims: vec![1, 2],
                    },
                    vec![ValRef::Local(db), b_ref, a_ref],
                    OpMode::Linear {
                        active_mask: vec![true, false, false],
                    },
                )[0]
            };
            builder.add_op(
                StdTensorOp::Add,
                vec![ValRef::Local(da_3d), ValRef::Local(db_3d)],
                OpMode::Linear {
                    active_mask: vec![true, true],
                },
            )[0]
        }
        (None, None) => unreachable!("caller already returned None for both-inactive case"),
    }
}

fn zero_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 0.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 0.0_f64.to_le_bytes().to_vec(),
        DType::C32 => {
            let mut b = Vec::with_capacity(8);
            b.extend_from_slice(&0.0_f32.to_le_bytes());
            b.extend_from_slice(&0.0_f32.to_le_bytes());
            b
        }
        DType::C64 => {
            let mut b = Vec::with_capacity(16);
            b.extend_from_slice(&0.0_f64.to_le_bytes());
            b.extend_from_slice(&0.0_f64.to_le_bytes());
            b
        }
    }
}

fn one_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 1.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 1.0_f64.to_le_bytes().to_vec(),
        DType::C32 => {
            let mut b = Vec::with_capacity(8);
            b.extend_from_slice(&1.0_f32.to_le_bytes());
            b.extend_from_slice(&0.0_f32.to_le_bytes());
            b
        }
        DType::C64 => {
            let mut b = Vec::with_capacity(16);
            b.extend_from_slice(&1.0_f64.to_le_bytes());
            b.extend_from_slice(&0.0_f64.to_le_bytes());
            b
        }
    }
}

/// Direct-from-host `F64` tropical GEMM: `out[i, j] = ⊕_k (a[i, k] + b[k, j])`
/// where `⊕` is `max` for [`TropicalKind::MaxPlus`] and `min` for
/// [`TropicalKind::MinPlus`].
///
/// Column-major layout: `a[i, k]` at offset `i + k * m`; `b[k, j]` at offset
/// `k + j * k_dim`; `out[i, j]` at offset `i + j * m`.
fn tropical_gemm_f64(
    a: &TypedTensor<f64>,
    b: &TypedTensor<f64>,
    kind: TropicalKind,
) -> TypedTensor<f64> {
    let m = a.shape[0];
    let k_dim = a.shape[1];
    let n = b.shape[1];
    let a_data = a.host_data();
    let b_data = b.host_data();
    let mut out = vec![0.0_f64; m * n];
    for j in 0..n {
        for i in 0..m {
            let start_k = 0;
            let a_ik0 = a_data[i + start_k * m];
            let b_k0j = b_data[start_k + j * k_dim];
            let mut acc = a_ik0 + b_k0j;
            for k in 1..k_dim {
                let s = a_data[i + k * m] + b_data[k + j * k_dim];
                acc = match kind {
                    TropicalKind::MaxPlus => {
                        if s > acc {
                            s
                        } else {
                            acc
                        }
                    }
                    TropicalKind::MinPlus => {
                        if s < acc {
                            s
                        } else {
                            acc
                        }
                    }
                };
            }
            out[i + j * m] = acc;
        }
    }
    TypedTensor::from_vec(vec![m, n], out)
}

fn tropical_gemm_f32(
    a: &TypedTensor<f32>,
    b: &TypedTensor<f32>,
    kind: TropicalKind,
) -> TypedTensor<f32> {
    let m = a.shape[0];
    let k_dim = a.shape[1];
    let n = b.shape[1];
    let a_data = a.host_data();
    let b_data = b.host_data();
    let mut out = vec![0.0_f32; m * n];
    for j in 0..n {
        for i in 0..m {
            let start_k = 0;
            let a_ik0 = a_data[i + start_k * m];
            let b_k0j = b_data[start_k + j * k_dim];
            let mut acc = a_ik0 + b_k0j;
            for k in 1..k_dim {
                let s = a_data[i + k * m] + b_data[k + j * k_dim];
                acc = match kind {
                    TropicalKind::MaxPlus => {
                        if s > acc {
                            s
                        } else {
                            acc
                        }
                    }
                    TropicalKind::MinPlus => {
                        if s < acc {
                            s
                        } else {
                            acc
                        }
                    }
                };
            }
            out[i + j * m] = acc;
        }
    }
    TypedTensor::from_vec(vec![m, n], out)
}

// ----------------------------------------------------------------------
// Factory and registration.
// ----------------------------------------------------------------------

/// [`ExtensionFactory`] for [`FusedTropicalDotGeneralOp`].
#[derive(Debug, Default)]
pub struct FusedTropicalDotGeneralFactory;

impl ExtensionFactory for FusedTropicalDotGeneralFactory {
    fn family_id(&self) -> &'static str {
        FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID
    }

    fn version(&self) -> u32 {
        1
    }
}

/// Explicitly register the `FusedTropicalDotGeneral` family with the
/// extension registry. Idempotent across calls within a single process.
///
/// Registration style: **explicit** (user / test code calls this, usually
/// once at startup). Implementation tolerates duplicate calls by silently
/// swallowing the `Duplicate` error after the first successful registration
/// — this keeps test-harness ordering simple (multiple integration test
/// binaries may run in the same address space on some runners).
///
/// Returns `Err(MalformedFamilyId)` if the compiled-in family id ever
/// drifts from the normative format — a defensive check, not a normal
/// failure mode. Duplicate registration is mapped to `Ok(())`.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::fused::register_fused_tropical;
///
/// register_fused_tropical().expect("register");
/// ```
pub fn register_fused_tropical() -> Result<(), ExtensionRegistryError> {
    static REGISTERED: OnceLock<Mutex<bool>> = OnceLock::new();
    let done = REGISTERED.get_or_init(|| Mutex::new(false));
    let mut guard = done.lock().expect("stage7 registration mutex");
    if *guard {
        return Ok(());
    }
    match register_extension(Arc::new(FusedTropicalDotGeneralFactory)) {
        Ok(()) => {
            *guard = true;
            Ok(())
        }
        Err(ExtensionRegistryError::Duplicate { .. }) => {
            // Another path already registered the family — still fine.
            *guard = true;
            Ok(())
        }
        Err(other) => Err(other),
    }
}
