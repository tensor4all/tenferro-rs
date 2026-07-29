use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use super::identity::SemanticIdentity;
use super::op::{CoreSemanticOp, SemanticOp, SemanticOperation};
use super::value::ProgramBuilderNonce;
use super::{
    Alias, Effect, ProgramBindings, ProgramQueryError, ProgramValue, ProgramValueMetadata,
    SemanticFingerprint, SemanticOperationView, ShapeGuard,
};

/// Immutable backend-neutral semantic SSA program.
pub struct SemanticProgram {
    pub(crate) owner: ProgramBuilderNonce,
    pub(crate) inputs: Box<[ProgramValue]>,
    pub(crate) outputs: Box<[ProgramValue]>,
    pub(crate) values: Box<[ProgramValueMetadata]>,
    pub(crate) operations: Box<[SemanticOperation]>,
    pub(crate) shape_guards: Box<[ShapeGuard]>,
    pub(crate) identity: SemanticIdentity,
}

impl SemanticProgram {
    /// Borrow ordered external inputs.
    pub fn inputs(&self) -> &[ProgramValue] {
        &self.inputs
    }

    /// Borrow ordered program outputs.
    pub fn outputs(&self) -> &[ProgramValue] {
        &self.outputs
    }

    /// Iterate over operations in semantic order.
    pub fn operations(&self) -> impl ExactSizeIterator<Item = SemanticOperationView<'_>> + '_ {
        self.operations.iter().map(SemanticOperationView::new)
    }

    /// Borrow metadata for a value owned by this program.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramQueryError::ForeignValue`] for a foreign token.
    pub fn value_metadata(
        &self,
        value: ProgramValue,
    ) -> Result<&ProgramValueMetadata, ProgramQueryError> {
        if value.owner != self.owner {
            return Err(ProgramQueryError::ForeignValue);
        }
        self.values
            .get(value.slot as usize)
            .ok_or(ProgramQueryError::ForeignValue)
    }

    /// Borrow all symbolic guards in semantic operation order.
    pub fn shape_guards(&self) -> &[ShapeGuard] {
        &self.shape_guards
    }

    /// Return the cached normalized structural fingerprint.
    pub const fn semantic_fingerprint(&self) -> SemanticFingerprint {
        self.identity.fingerprint
    }

    /// Compare exact normalized semantics after the compact fingerprint check.
    pub fn semantic_eq(&self, other: &Self) -> bool {
        self.identity.exact_eq(self, &other.identity, other)
    }

    pub(crate) fn logical_retained_bytes(&self) -> Option<usize> {
        checked_sum([
            size_of::<SemanticProgram>(),
            self.inputs.len().checked_mul(size_of::<ProgramValue>())?,
            self.outputs.len().checked_mul(size_of::<ProgramValue>())?,
            self.values
                .len()
                .checked_mul(size_of::<ProgramValueMetadata>())?,
            checked_sum_options(
                self.values
                    .iter()
                    .map(ProgramValueMetadata::logical_retained_bytes),
            )?,
            self.operations
                .len()
                .checked_mul(size_of::<SemanticOperation>())?,
            checked_sum_options(
                self.operations
                    .iter()
                    .map(semantic_operation_retained_bytes),
            )?,
            self.shape_guards
                .len()
                .checked_mul(size_of::<ShapeGuard>())?,
            checked_sum_options(
                self.shape_guards
                    .iter()
                    .map(ShapeGuard::logical_retained_bytes),
            )?,
            self.identity.ordinals_retained_bytes()?,
        ])
    }

    #[cfg(test)]
    pub(crate) fn set_fingerprint_for_test(&mut self, fingerprint: SemanticFingerprint) {
        self.identity.fingerprint = fingerprint;
    }

    #[cfg(test)]
    pub(crate) fn set_first_provenance_for_test(&mut self, label: &str) {
        if let Some(operation) = self.operations.first_mut() {
            operation.provenance = super::metadata::SemanticProvenance::builder(Some(label));
        }
    }

    #[cfg(test)]
    pub(crate) fn fingerprint_computations_for_test(&self) -> usize {
        self.identity.fingerprint_computations
    }
}

fn semantic_operation_retained_bytes(operation: &SemanticOperation) -> Option<usize> {
    checked_sum([
        semantic_op_retained_bytes(&operation.op)?,
        operation
            .inputs
            .len()
            .checked_mul(size_of::<ProgramValue>())?,
        operation
            .outputs
            .len()
            .checked_mul(size_of::<ProgramValue>())?,
        operation.effects.len().checked_mul(size_of::<Effect>())?,
        operation.aliases.len().checked_mul(size_of::<Alias>())?,
        operation
            .shape_guards
            .len()
            .checked_mul(size_of::<ShapeGuard>())?,
        checked_sum_options(
            operation
                .shape_guards
                .iter()
                .map(ShapeGuard::logical_retained_bytes),
        )?,
        operation.provenance.view().label().map_or(0, str::len),
    ])
}

fn semantic_op_retained_bytes(op: &SemanticOp) -> Option<usize> {
    match op {
        SemanticOp::Core(op) => core_semantic_op_retained_bytes(op),
        SemanticOp::Extension(_) => Some(0),
    }
}

fn core_semantic_op_retained_bytes(op: &CoreSemanticOp) -> Option<usize> {
    match op {
        CoreSemanticOp::DotGeneral { config } => dot_general_config_retained_bytes(config),
        CoreSemanticOp::Transpose { perm } => vec_bytes::<usize>(perm.len()),
        CoreSemanticOp::Reshape { to_shape } => dim_expr_vec_bytes(to_shape),
        CoreSemanticOp::BroadcastInDim { shape, dims } => {
            checked_sum([dim_expr_vec_bytes(shape)?, vec_bytes::<usize>(dims.len())?])
        }
        CoreSemanticOp::Constant { bytes, .. } => vec_bytes::<u8>(bytes.len()),
        CoreSemanticOp::ReduceSum { axes }
        | CoreSemanticOp::ReduceSumSquares { axes }
        | CoreSemanticOp::ReduceProd { axes }
        | CoreSemanticOp::ReduceMax { axes }
        | CoreSemanticOp::ReduceMin { axes }
        | CoreSemanticOp::Reverse { axes } => vec_bytes::<usize>(axes.len()),
        CoreSemanticOp::Gather(config) => gather_config_retained_bytes(config),
        CoreSemanticOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            slice_sizes,
            ..
        } => checked_sum([
            vec_bytes::<usize>(offset_dims.len())?,
            vec_bytes::<usize>(collapsed_slice_dims.len())?,
            vec_bytes::<usize>(start_index_map.len())?,
            dim_expr_vec_bytes(slice_sizes)?,
        ]),
        CoreSemanticOp::Scatter(config) => scatter_config_retained_bytes(config),
        CoreSemanticOp::Slice(config) => slice_config_retained_bytes(config),
        CoreSemanticOp::DynamicSlice { slice_sizes } => vec_bytes::<usize>(slice_sizes.len()),
        CoreSemanticOp::Pad(config) => pad_config_retained_bytes(config),
        CoreSemanticOp::Add
        | CoreSemanticOp::Sub
        | CoreSemanticOp::Mul
        | CoreSemanticOp::Neg
        | CoreSemanticOp::Conj
        | CoreSemanticOp::Convert { .. }
        | CoreSemanticOp::Div
        | CoreSemanticOp::Rem
        | CoreSemanticOp::Abs
        | CoreSemanticOp::Sign
        | CoreSemanticOp::Maximum
        | CoreSemanticOp::Minimum
        | CoreSemanticOp::Compare(_)
        | CoreSemanticOp::Select
        | CoreSemanticOp::Clamp
        | CoreSemanticOp::Exp
        | CoreSemanticOp::Log
        | CoreSemanticOp::Sin
        | CoreSemanticOp::Cos
        | CoreSemanticOp::Tanh
        | CoreSemanticOp::Sqrt
        | CoreSemanticOp::Rsqrt
        | CoreSemanticOp::Pow
        | CoreSemanticOp::Expm1
        | CoreSemanticOp::Log1p
        | CoreSemanticOp::ExtractDiag { .. }
        | CoreSemanticOp::EmbedDiag { .. }
        | CoreSemanticOp::Tril { .. }
        | CoreSemanticOp::Triu { .. }
        | CoreSemanticOp::DynamicUpdateSlice
        | CoreSemanticOp::Concatenate { .. }
        | CoreSemanticOp::ShapeOf { .. }
        | CoreSemanticOp::DynamicTruncate { .. }
        | CoreSemanticOp::PadToMatch { .. } => Some(0),
    }
}

fn dot_general_config_retained_bytes(config: &tenferro_tensor::DotGeneralConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.lhs_contracting_dims.len())?,
        vec_bytes::<usize>(config.rhs_contracting_dims.len())?,
        vec_bytes::<usize>(config.lhs_batch_dims.len())?,
        vec_bytes::<usize>(config.rhs_batch_dims.len())?,
    ])
}

fn gather_config_retained_bytes(config: &tenferro_tensor::GatherConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.offset_dims.len())?,
        vec_bytes::<usize>(config.collapsed_slice_dims.len())?,
        vec_bytes::<usize>(config.start_index_map.len())?,
        vec_bytes::<usize>(config.slice_sizes.len())?,
    ])
}

fn scatter_config_retained_bytes(config: &tenferro_tensor::ScatterConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.update_window_dims.len())?,
        vec_bytes::<usize>(config.inserted_window_dims.len())?,
        vec_bytes::<usize>(config.scatter_dims_to_operand_dims.len())?,
    ])
}

fn slice_config_retained_bytes(config: &tenferro_tensor::SliceConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.starts.len())?,
        vec_bytes::<usize>(config.limits.len())?,
        vec_bytes::<usize>(config.strides.len())?,
    ])
}

fn pad_config_retained_bytes(config: &tenferro_tensor::PadConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<i64>(config.edge_padding_low.len())?,
        vec_bytes::<i64>(config.edge_padding_high.len())?,
        vec_bytes::<i64>(config.interior_padding.len())?,
    ])
}

fn dim_expr_vec_bytes(values: &[tenferro_ops::dim_expr::DimExpr]) -> Option<usize> {
    checked_sum([
        vec_bytes::<tenferro_ops::dim_expr::DimExpr>(values.len())?,
        checked_sum_options(values.iter().map(dim_expr_logical_retained_bytes))?,
    ])
}

fn dim_expr_logical_retained_bytes(expression: &tenferro_ops::dim_expr::DimExpr) -> Option<usize> {
    use tenferro_ops::dim_expr::DimExpr;

    match expression {
        DimExpr::Const(_) | DimExpr::InputDim { .. } => Some(0),
        DimExpr::Add(left, right)
        | DimExpr::Sub(left, right)
        | DimExpr::Mul(left, right)
        | DimExpr::FloorDiv(left, right)
        | DimExpr::Min(left, right)
        | DimExpr::Max(left, right) => checked_sum([
            2usize.checked_mul(size_of::<DimExpr>())?,
            dim_expr_logical_retained_bytes(left)?,
            dim_expr_logical_retained_bytes(right)?,
        ]),
    }
}

fn vec_bytes<T>(len: usize) -> Option<usize> {
    len.checked_mul(size_of::<T>())
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
}

fn checked_sum_options(values: impl IntoIterator<Item = Option<usize>>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value?))
}

impl fmt::Debug for SemanticProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SemanticProgram")
            .field("inputs", &self.inputs.len())
            .field("outputs", &self.outputs.len())
            .field("values", &self.values.len())
            .field("operations", &self.operations.len())
            .field("shape_guards", &self.shape_guards.len())
            .finish()
    }
}

/// One atomically frozen semantic program and its separate tensor bindings.
#[derive(Clone)]
pub struct FrozenProgram {
    /// Immutable backend-neutral semantic structure.
    pub program: Arc<SemanticProgram>,
    /// Process-local tensor defaults and large constants.
    pub bindings: ProgramBindings,
}

impl fmt::Debug for FrozenProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenProgram")
            .field("program", &self.program)
            .field("bindings", &self.bindings)
            .finish()
    }
}
