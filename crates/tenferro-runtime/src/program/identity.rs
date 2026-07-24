use std::hash::Hasher;

use sha2::{Digest, Sha256};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

use super::op::{SemanticOp, SemanticOperation};
use super::semantic::SemanticProgram;
use super::{
    Alias, AliasKind, CoreSemanticOp, Effect, EffectAccess, ProgramShapeRelation, ProgramValue,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticPlacementKind, ShapeGuard,
};

/// Cached fixed-size identity of normalized semantic program structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SemanticFingerprint([u8; 32]);

impl SemanticFingerprint {
    /// Borrow the cached SHA-256 bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

pub(crate) struct SemanticIdentity {
    pub(crate) fingerprint: SemanticFingerprint,
    ordinals: Box<[u32]>,
    #[cfg(test)]
    pub(crate) fingerprint_computations: usize,
}

impl SemanticIdentity {
    pub(crate) fn build(
        inputs: &[ProgramValue],
        outputs: &[ProgramValue],
        values: &[ProgramValueMetadata],
        operations: &[SemanticOperation],
        shape_guards: &[ShapeGuard],
    ) -> Self {
        let ordinals = normalized_ordinals(inputs, values.len(), operations);
        let mut encoder = CanonicalEncoder::new();
        encoder.bytes(b"tenferro.semantic-program.v1");
        encoder.usize(inputs.len());
        for input in inputs {
            encode_metadata(&mut encoder, &values[input.slot as usize]);
        }
        encoder.usize(operations.len());
        for operation in operations {
            encode_operation(&mut encoder, operation, values, &ordinals);
        }
        encoder.usize(outputs.len());
        for output in outputs {
            encoder.u32(ordinals[output.slot as usize]);
        }
        encode_shape_guards(&mut encoder, shape_guards);
        Self {
            fingerprint: SemanticFingerprint(encoder.finalize()),
            ordinals,
            #[cfg(test)]
            fingerprint_computations: 1,
        }
    }

    pub(crate) fn exact_eq(
        &self,
        left: &SemanticProgram,
        other: &Self,
        right: &SemanticProgram,
    ) -> bool {
        if self.fingerprint != other.fingerprint
            || left.inputs.len() != right.inputs.len()
            || left.outputs.len() != right.outputs.len()
            || left.operations.len() != right.operations.len()
            || left.shape_guards != right.shape_guards
        {
            return false;
        }
        if !left
            .inputs
            .iter()
            .zip(&right.inputs)
            .all(|(left_value, right_value)| {
                left.values[left_value.slot as usize] == right.values[right_value.slot as usize]
            })
        {
            return false;
        }
        if !left.operations.iter().zip(&right.operations).all(
            |(left_operation, right_operation)| {
                operation_exact_eq(
                    left_operation,
                    &left.values,
                    &self.ordinals,
                    right_operation,
                    &right.values,
                    &other.ordinals,
                )
            },
        ) {
            return false;
        }
        left.outputs
            .iter()
            .zip(&right.outputs)
            .all(|(left_value, right_value)| {
                self.ordinals[left_value.slot as usize] == other.ordinals[right_value.slot as usize]
            })
    }

    // INVARIANT: this helper is introduced ahead of the P4-C1 cache owner
    // wiring so retained-byte accounting can use the exact semantic identity
    // payload size without duplicating ordinal internals.
    #[allow(
        dead_code,
        reason = "P4-C1 preparation accounting consumes this helper"
    )]
    pub(crate) fn ordinals_retained_bytes(&self) -> Option<usize> {
        self.ordinals.len().checked_mul(std::mem::size_of::<u32>())
    }
}

fn normalized_ordinals(
    inputs: &[ProgramValue],
    value_count: usize,
    operations: &[SemanticOperation],
) -> Box<[u32]> {
    let mut ordinals = vec![u32::MAX; value_count];
    let mut next = 0_u32;
    for input in inputs {
        ordinals[input.slot as usize] = next;
        next += 1;
    }
    for operation in operations {
        for output in &operation.outputs {
            ordinals[output.slot as usize] = next;
            next += 1;
        }
    }
    ordinals.into()
}

fn operation_exact_eq(
    left: &SemanticOperation,
    left_values: &[ProgramValueMetadata],
    left_ordinals: &[u32],
    right: &SemanticOperation,
    right_values: &[ProgramValueMetadata],
    right_ordinals: &[u32],
) -> bool {
    semantic_op_exact_eq(&left.op, &right.op)
        && left.inputs.len() == right.inputs.len()
        && left.outputs.len() == right.outputs.len()
        && left.inputs.iter().zip(&right.inputs).all(|(left, right)| {
            left_ordinals[left.slot as usize] == right_ordinals[right.slot as usize]
        })
        && left
            .outputs
            .iter()
            .zip(&right.outputs)
            .all(|(left, right)| {
                left_values[left.slot as usize] == right_values[right.slot as usize]
            })
        && left.effects == right.effects
        && left.aliases == right.aliases
        && left.shape_guards == right.shape_guards
        && left.placement == right.placement
}

fn semantic_op_exact_eq(left: &SemanticOp, right: &SemanticOp) -> bool {
    match (left, right) {
        (SemanticOp::Core(left), SemanticOp::Core(right)) => left == right,
        (SemanticOp::Extension(left), SemanticOp::Extension(right)) => {
            left.family_id() == right.family_id() && left.payload_eq(right.as_ref())
        }
        _ => false,
    }
}

struct CanonicalEncoder {
    digest: Sha256,
}

impl CanonicalEncoder {
    fn new() -> Self {
        Self {
            digest: Sha256::new(),
        }
    }

    fn finalize(self) -> [u8; 32] {
        self.digest.finalize().into()
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.digest.update(bytes);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.usize(bytes.len());
        self.raw(bytes);
    }

    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn u8(&mut self, value: u8) {
        self.raw(&[value]);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn usize_slice(&mut self, values: &[usize]) {
        self.usize(values.len());
        for &value in values {
            self.usize(value);
        }
    }

    fn i64_slice(&mut self, values: &[i64]) {
        self.usize(values.len());
        for &value in values {
            self.i64(value);
        }
    }
}

impl Hasher for CanonicalEncoder {
    fn finish(&self) -> u64 {
        let digest = self.digest.clone().finalize();
        let mut prefix = [0_u8; 8];
        prefix.copy_from_slice(&digest[..8]);
        u64::from_le_bytes(prefix)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        self.u8(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.u32(value);
    }

    fn write_u64(&mut self, value: u64) {
        self.u64(value);
    }

    fn write_u128(&mut self, value: u128) {
        self.raw(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.usize(value);
    }

    fn write_i8(&mut self, value: i8) {
        self.raw(&value.to_le_bytes());
    }

    fn write_i16(&mut self, value: i16) {
        self.raw(&value.to_le_bytes());
    }

    fn write_i32(&mut self, value: i32) {
        self.raw(&value.to_le_bytes());
    }

    fn write_i64(&mut self, value: i64) {
        self.i64(value);
    }

    fn write_i128(&mut self, value: i128) {
        self.raw(&value.to_le_bytes());
    }

    fn write_isize(&mut self, value: isize) {
        self.i64(value as i64);
    }
}

fn encode_operation(
    encoder: &mut CanonicalEncoder,
    operation: &SemanticOperation,
    values: &[ProgramValueMetadata],
    ordinals: &[u32],
) {
    match &operation.op {
        SemanticOp::Core(op) => encode_core_op(encoder, op),
        SemanticOp::Extension(op) => {
            encoder.u8(1);
            encoder.string(op.family_id());
            op.payload_hash(encoder);
        }
    }
    encoder.usize(operation.inputs.len());
    for input in &operation.inputs {
        encoder.u32(ordinals[input.slot as usize]);
    }
    encoder.usize(operation.outputs.len());
    for output in &operation.outputs {
        encode_metadata(encoder, &values[output.slot as usize]);
    }
    encode_effects(encoder, &operation.effects);
    encode_aliases(encoder, &operation.aliases);
    encode_shape_guards(encoder, &operation.shape_guards);
    encode_placement(encoder, operation.placement);
}

fn encode_core_op(encoder: &mut CanonicalEncoder, op: &CoreSemanticOp) {
    encoder.u8(0);
    match op {
        CoreSemanticOp::Add => encoder.u8(0),
        CoreSemanticOp::Sub => encoder.u8(1),
        CoreSemanticOp::Mul => encoder.u8(2),
        CoreSemanticOp::Neg => encoder.u8(3),
        CoreSemanticOp::Conj => encoder.u8(4),
        CoreSemanticOp::DotGeneral { config } => {
            encoder.u8(5);
            encode_dot_general(encoder, config);
        }
        CoreSemanticOp::Transpose { perm } => {
            encoder.u8(6);
            encoder.usize_slice(perm);
        }
        CoreSemanticOp::Reshape { to_shape } => {
            encoder.u8(7);
            encode_dim_exprs(encoder, to_shape);
        }
        CoreSemanticOp::BroadcastInDim { shape, dims } => {
            encoder.u8(8);
            encode_dim_exprs(encoder, shape);
            encoder.usize_slice(dims);
        }
        CoreSemanticOp::Convert { from, to } => {
            encoder.u8(9);
            encode_dtype(encoder, *from);
            encode_dtype(encoder, *to);
        }
        CoreSemanticOp::Constant { dtype, bytes } => {
            encoder.u8(10);
            encode_dtype(encoder, *dtype);
            encoder.bytes(bytes);
        }
        CoreSemanticOp::ReduceSum { axes } => {
            encoder.u8(11);
            encoder.usize_slice(axes);
        }
        CoreSemanticOp::Div => encoder.u8(12),
        CoreSemanticOp::Rem => encoder.u8(13),
        CoreSemanticOp::Abs => encoder.u8(14),
        CoreSemanticOp::Sign => encoder.u8(15),
        CoreSemanticOp::Maximum => encoder.u8(16),
        CoreSemanticOp::Minimum => encoder.u8(17),
        CoreSemanticOp::Compare(direction) => {
            encoder.u8(18);
            encode_compare(encoder, direction);
        }
        CoreSemanticOp::Select => encoder.u8(19),
        CoreSemanticOp::Clamp => encoder.u8(20),
        CoreSemanticOp::Exp => encoder.u8(21),
        CoreSemanticOp::Log => encoder.u8(22),
        CoreSemanticOp::Sin => encoder.u8(23),
        CoreSemanticOp::Cos => encoder.u8(24),
        CoreSemanticOp::Tanh => encoder.u8(25),
        CoreSemanticOp::Sqrt => encoder.u8(26),
        CoreSemanticOp::Rsqrt => encoder.u8(27),
        CoreSemanticOp::Pow => encoder.u8(28),
        CoreSemanticOp::Expm1 => encoder.u8(29),
        CoreSemanticOp::Log1p => encoder.u8(30),
        CoreSemanticOp::ExtractDiag { axis_a, axis_b } => {
            encoder.u8(31);
            encoder.usize(*axis_a);
            encoder.usize(*axis_b);
        }
        CoreSemanticOp::EmbedDiag { axis_a, axis_b } => {
            encoder.u8(32);
            encoder.usize(*axis_a);
            encoder.usize(*axis_b);
        }
        CoreSemanticOp::Tril { k } => {
            encoder.u8(33);
            encoder.i64(*k);
        }
        CoreSemanticOp::Triu { k } => {
            encoder.u8(34);
            encoder.i64(*k);
        }
        CoreSemanticOp::Gather(config) => {
            encoder.u8(35);
            encode_gather(encoder, config);
        }
        CoreSemanticOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        } => {
            encoder.u8(36);
            encoder.usize_slice(offset_dims);
            encoder.usize_slice(collapsed_slice_dims);
            encoder.usize_slice(start_index_map);
            encoder.usize(*index_vector_dim);
            encode_dim_exprs(encoder, slice_sizes);
        }
        CoreSemanticOp::Scatter(config) => {
            encoder.u8(37);
            encode_scatter(encoder, config);
        }
        CoreSemanticOp::Slice(config) => {
            encoder.u8(38);
            encode_slice(encoder, config);
        }
        CoreSemanticOp::DynamicSlice { slice_sizes } => {
            encoder.u8(39);
            encoder.usize_slice(slice_sizes);
        }
        CoreSemanticOp::DynamicUpdateSlice => encoder.u8(40),
        CoreSemanticOp::Pad(config) => {
            encoder.u8(41);
            encode_pad(encoder, config);
        }
        CoreSemanticOp::Concatenate { axis, input_count } => {
            encoder.u8(42);
            encoder.usize(*axis);
            encoder.usize(*input_count);
        }
        CoreSemanticOp::Reverse { axes } => {
            encoder.u8(43);
            encoder.usize_slice(axes);
        }
        CoreSemanticOp::ShapeOf { axis } => {
            encoder.u8(44);
            encoder.usize(*axis);
        }
        CoreSemanticOp::DynamicTruncate { axis } => {
            encoder.u8(45);
            encoder.usize(*axis);
        }
        CoreSemanticOp::PadToMatch { axis } => {
            encoder.u8(46);
            encoder.usize(*axis);
        }
        CoreSemanticOp::ReduceProd { axes } => {
            encoder.u8(47);
            encoder.usize_slice(axes);
        }
        CoreSemanticOp::ReduceMax { axes } => {
            encoder.u8(48);
            encoder.usize_slice(axes);
        }
        CoreSemanticOp::ReduceMin { axes } => {
            encoder.u8(49);
            encoder.usize_slice(axes);
        }
    }
}

fn encode_metadata(encoder: &mut CanonicalEncoder, metadata: &ProgramValueMetadata) {
    encode_dtype(encoder, metadata.dtype());
    encoder.usize(metadata.shape().len());
    for extent in metadata.shape() {
        match extent {
            ShapeExtent::Exact(expression) => {
                encoder.u8(0);
                encode_dim_expr(encoder, expression);
            }
            ShapeExtent::UpperBound(expression) => {
                encoder.u8(1);
                encode_dim_expr(encoder, expression);
            }
            ShapeExtent::Unknown => encoder.u8(2),
        }
    }
}

fn encode_dtype(encoder: &mut CanonicalEncoder, dtype: DType) {
    encoder.u8(match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::I32 => 2,
        DType::I64 => 3,
        DType::Bool => 4,
        DType::C32 => 5,
        DType::C64 => 6,
    });
}

fn encode_dim_exprs(encoder: &mut CanonicalEncoder, expressions: &[DimExpr]) {
    encoder.usize(expressions.len());
    for expression in expressions {
        encode_dim_expr(encoder, expression);
    }
}

fn encode_dim_expr(encoder: &mut CanonicalEncoder, expression: &DimExpr) {
    match expression {
        DimExpr::Const(value) => {
            encoder.u8(0);
            encoder.usize(*value);
        }
        DimExpr::InputDim { input_idx, axis } => {
            encoder.u8(1);
            encoder.usize(*input_idx);
            encoder.usize(*axis);
        }
        DimExpr::Add(left, right) => encode_binary_dim_expr(encoder, 2, left, right),
        DimExpr::Sub(left, right) => encode_binary_dim_expr(encoder, 3, left, right),
        DimExpr::Mul(left, right) => encode_binary_dim_expr(encoder, 4, left, right),
        DimExpr::FloorDiv(left, right) => encode_binary_dim_expr(encoder, 5, left, right),
        DimExpr::Min(left, right) => encode_binary_dim_expr(encoder, 6, left, right),
        DimExpr::Max(left, right) => encode_binary_dim_expr(encoder, 7, left, right),
    }
}

fn encode_binary_dim_expr(
    encoder: &mut CanonicalEncoder,
    tag: u8,
    left: &DimExpr,
    right: &DimExpr,
) {
    encoder.u8(tag);
    encode_dim_expr(encoder, left);
    encode_dim_expr(encoder, right);
}

fn encode_effects(encoder: &mut CanonicalEncoder, effects: &[Effect]) {
    encoder.usize(effects.len());
    for effect in effects {
        encoder.string(effect.resource().family());
        encoder.u64(effect.resource().key());
        encoder.u8(match effect.access() {
            EffectAccess::Read => 0,
            EffectAccess::Write => 1,
        });
    }
}

fn encode_aliases(encoder: &mut CanonicalEncoder, aliases: &[Alias]) {
    encoder.usize(aliases.len());
    for alias in aliases {
        encoder.u8(match alias.kind() {
            AliasKind::Fresh => 0,
            AliasKind::ViewOf => 1,
            AliasKind::MustAlias => 2,
            AliasKind::ExternalAlias => 3,
        });
        encoder.usize(alias.output());
        encode_option_usize(encoder, alias.input());
        match alias.resource() {
            Some(resource) => {
                encoder.u8(1);
                encoder.string(resource.family());
                encoder.u64(resource.key());
            }
            None => encoder.u8(0),
        }
    }
}

fn encode_shape_guards(encoder: &mut CanonicalEncoder, guards: &[ShapeGuard]) {
    encoder.usize(guards.len());
    for guard in guards {
        encoder.u8(match guard.relation() {
            ProgramShapeRelation::Equal => 0,
            ProgramShapeRelation::LessEqual => 1,
            ProgramShapeRelation::GreaterEqual => 2,
        });
        encode_dim_expr(encoder, guard.lhs());
        encode_dim_expr(encoder, guard.rhs());
    }
}

fn encode_placement(encoder: &mut CanonicalEncoder, placement: SemanticPlacementConstraint) {
    encoder.u8(match placement.kind() {
        SemanticPlacementKind::Any => 0,
        SemanticPlacementKind::SameAsInput => 1,
    });
    encode_option_usize(encoder, placement.input());
}

fn encode_option_usize(encoder: &mut CanonicalEncoder, value: Option<usize>) {
    match value {
        Some(value) => {
            encoder.u8(1);
            encoder.usize(value);
        }
        None => encoder.u8(0),
    }
}

fn encode_compare(encoder: &mut CanonicalEncoder, direction: &CompareDir) {
    encoder.u8(match direction {
        CompareDir::Eq => 0,
        CompareDir::Lt => 1,
        CompareDir::Le => 2,
        CompareDir::Gt => 3,
        CompareDir::Ge => 4,
    });
}

fn encode_dot_general(encoder: &mut CanonicalEncoder, config: &DotGeneralConfig) {
    encoder.usize_slice(&config.lhs_contracting_dims);
    encoder.usize_slice(&config.rhs_contracting_dims);
    encoder.usize_slice(&config.lhs_batch_dims);
    encoder.usize_slice(&config.rhs_batch_dims);
}

fn encode_gather(encoder: &mut CanonicalEncoder, config: &GatherConfig) {
    encoder.usize_slice(&config.offset_dims);
    encoder.usize_slice(&config.collapsed_slice_dims);
    encoder.usize_slice(&config.start_index_map);
    encoder.usize(config.index_vector_dim);
    encoder.usize_slice(&config.slice_sizes);
}

fn encode_scatter(encoder: &mut CanonicalEncoder, config: &ScatterConfig) {
    encoder.usize_slice(&config.update_window_dims);
    encoder.usize_slice(&config.inserted_window_dims);
    encoder.usize_slice(&config.scatter_dims_to_operand_dims);
    encoder.usize(config.index_vector_dim);
}

fn encode_slice(encoder: &mut CanonicalEncoder, config: &SliceConfig) {
    encoder.usize_slice(&config.starts);
    encoder.usize_slice(&config.limits);
    encoder.usize_slice(&config.strides);
}

fn encode_pad(encoder: &mut CanonicalEncoder, config: &PadConfig) {
    encoder.i64_slice(&config.edge_padding_low);
    encoder.i64_slice(&config.edge_padding_high);
    encoder.i64_slice(&config.interior_padding);
}
