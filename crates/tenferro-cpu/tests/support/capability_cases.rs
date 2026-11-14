//! The shared backend-capability conformance cases.
//!
//! Included by path from `tenferro-cpu/tests/integration/capability_conformance.rs` and from
//! `tenferro-gpu/tests/seam_capability.rs`, so both lanes run the identical fixture against the
//! identical rule (issue #2477). It names no backend: the engine is generic and the caller
//! passes the placement step (clone on the host, upload on a device).

use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{
    capability_output_dtype, BackendId, CompareDir, DType, DotGeneralConfig, OperationCapability,
    Result, Tensor, TensorAnalytic, TensorDot, TensorElementwise, TensorRead, TensorReduction,
};

/// Every public `Tensor` dtype: the capability tables are checked on the full cross product.
pub const DTYPES: [DType; 7] = [
    DType::F32,
    DType::F64,
    DType::C32,
    DType::C64,
    DType::I32,
    DType::I64,
    DType::Bool,
];

/// Whether the descriptor has a rule for `(op, dtype)` at all.
///
/// `capability_output_dtype` is the core-op catalog's semantic dtype policy, and the descriptor
/// adds backend implementation status on top of it rather than redefining it. A pair the policy
/// does not admit is outside the descriptor's vocabulary: the table may not name it, and there is
/// no success/refusal rule to apply. (An engine that serves such a pair is an engine gap against
/// the catalog, not a missing table row.)
pub fn catalog_admits(op: PrimitiveOpKind, dtype: DType) -> bool {
    capability_output_dtype(op, dtype).is_some()
}

/// A two-element column vector of `dtype` holding `[1, 2]` (`[true, false]` for `Bool`).
///
/// The values keep integer division, remainder and `pow` domain-valid, so a refusal can only
/// be the dtype refusal the table declares.
fn host_fixture(dtype: DType) -> Tensor {
    let shape = vec![2];
    match dtype {
        DType::F32 => Tensor::from_vec_col_major(shape, vec![1.0_f32, 2.0]).expect("fixture"),
        DType::F64 => Tensor::from_vec_col_major(shape, vec![1.0_f64, 2.0]).expect("fixture"),
        DType::C32 => Tensor::from_vec_col_major(
            shape,
            vec![
                tenferro_tensor::Complex32::new(1.0, 1.0),
                tenferro_tensor::Complex32::new(2.0, -1.0),
            ],
        )
        .expect("fixture"),
        DType::C64 => Tensor::from_vec_col_major(
            shape,
            vec![
                tenferro_tensor::Complex64::new(1.0, 1.0),
                tenferro_tensor::Complex64::new(2.0, -1.0),
            ],
        )
        .expect("fixture"),
        DType::I32 => Tensor::from_vec_col_major(shape, vec![1_i32, 2]).expect("fixture"),
        DType::I64 => Tensor::from_vec_col_major(shape, vec![1_i64, 2]).expect("fixture"),
        DType::Bool => Tensor::from_vec_col_major(shape, vec![true, false]).expect("fixture"),
        other => panic!("no fixture for {other:?}"),
    }
}

/// The `clamp` bounds: `0.5` and `1.5` over the fixture's shape. Only the float dtypes that
/// the tables declare for `clamp` reach this.
fn clamp_bounds(dtype: DType) -> (Tensor, Tensor) {
    let shape = vec![2];
    match dtype {
        DType::F32 => (
            Tensor::from_vec_col_major(shape.clone(), vec![0.5_f32, 0.5]).expect("bound"),
            Tensor::from_vec_col_major(shape, vec![1.5_f32, 1.5]).expect("bound"),
        ),
        DType::F64 => (
            Tensor::from_vec_col_major(shape.clone(), vec![0.5_f64, 0.5]).expect("bound"),
            Tensor::from_vec_col_major(shape, vec![1.5_f64, 1.5]).expect("bound"),
        ),
        other => panic!("clamp has no bounds fixture for {other:?}"),
    }
}

/// Run one catalogued operation on the fixture of `dtype`, placing every input with `place`.
///
/// `panic!("no case for {op:?}")` is the point of the single dispatcher: a table entry for an
/// operation it cannot run fails the test instead of passing unexamined.
pub fn run_case<E, P>(
    engine: &mut E,
    op: PrimitiveOpKind,
    dtype: DType,
    place: &mut P,
) -> Result<Tensor>
where
    E: TensorElementwise + TensorAnalytic + TensorReduction + TensorDot,
    P: FnMut(&mut E, &Tensor) -> Tensor,
{
    let host = host_fixture(dtype);
    let left = place(engine, &host);
    match op {
        PrimitiveOpKind::Add => engine.add_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Sub => engine.sub_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Mul => engine.mul_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Div => engine.div_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Rem => engine.rem_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Maximum => engine.maximum_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Minimum => engine.minimum_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Pow => engine.pow_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
        ),
        PrimitiveOpKind::Neg => engine.neg_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Conj => engine.conj_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Abs => engine.abs_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Sign => engine.sign_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Exp => engine.exp_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Log => engine.log_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Sin => engine.sin_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Cos => engine.cos_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Tanh => engine.tanh_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Sqrt => engine.sqrt_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Rsqrt => engine.rsqrt_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Expm1 => engine.expm1_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Log1p => engine.log1p_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Erf => engine.erf_read(TensorRead::from_tensor(&left)),
        PrimitiveOpKind::Compare => engine.compare_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
            &CompareDir::Eq,
        ),
        PrimitiveOpKind::Select => {
            let predicate = place(engine, &host_fixture(DType::Bool));
            engine.select_read(
                TensorRead::from_tensor(&predicate),
                TensorRead::from_tensor(&left),
                TensorRead::from_tensor(&left),
            )
        }
        PrimitiveOpKind::Clamp => {
            let (lower, upper) = clamp_bounds(dtype);
            let lower = place(engine, &lower);
            let upper = place(engine, &upper);
            engine.clamp_read(
                TensorRead::from_tensor(&left),
                TensorRead::from_tensor(&lower),
                TensorRead::from_tensor(&upper),
            )
        }
        PrimitiveOpKind::ReduceSum => engine.reduce_sum_read(TensorRead::from_tensor(&left), &[0]),
        PrimitiveOpKind::ReduceSumSquares => {
            engine.reduce_sum_squares_read(TensorRead::from_tensor(&left), &[0])
        }
        PrimitiveOpKind::ReduceProd => {
            engine.reduce_prod_read(TensorRead::from_tensor(&left), &[0])
        }
        PrimitiveOpKind::ReduceMax => engine.reduce_max_read(TensorRead::from_tensor(&left), &[0]),
        PrimitiveOpKind::ReduceMin => engine.reduce_min_read(TensorRead::from_tensor(&left), &[0]),
        PrimitiveOpKind::DotGeneral => engine.dot_general_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&left),
            &DotGeneralConfig {
                lhs_contracting_dims: [0].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        ),
        other => panic!("no case for {other:?}"),
    }
}

/// Hold one engine outcome to its descriptor entry.
///
/// A supported entry must run and return the declared output dtype; an `Unsupported` entry, or
/// no entry at all, must be refused. How a backend spells a refusal is its own business — CPU
/// returns `ErrorKind::Unsupported` or `Validation(DTypeMismatch)`, and a backend that leaves the
/// dtype policy to its kernels reports theirs inside a runtime-state error — so the rule held
/// here is "not served".
pub fn assert_entry_matches(
    backend: BackendId,
    op: PrimitiveOpKind,
    dtype: DType,
    entry: Option<OperationCapability>,
    outcome: Result<Tensor>,
) {
    let name = tenferro_core_ops::descriptor(op).name;
    match entry {
        Some(entry) if entry.result.is_supported() => match outcome {
            Ok(value) => assert_eq!(
                value.dtype(),
                entry.output_dtype,
                "{backend} {name}/{dtype:?} is declared with output dtype {:?} but returned {:?}",
                entry.output_dtype,
                value.dtype()
            ),
            Err(error) => panic!(
                "{backend} {name}/{dtype:?} is declared {:?} but the engine refused it: {error}",
                entry.result
            ),
        },
        declared => {
            let expected = match declared {
                Some(entry) => format!("declared {:?}", entry.result),
                None => "absent from the descriptor".to_string(),
            };
            assert!(
                outcome.is_err(),
                "{backend} {name}/{dtype:?} is {expected} but the engine served it"
            );
        }
    }
}
