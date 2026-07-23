use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::DType;

use super::EffectResourceError;

/// Diagnostic origin class of one semantic operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SemanticProvenanceKind {
    /// Operation was added directly through a semantic-program builder.
    Builder,
    /// Operation was preserved by importing another semantic program.
    Imported,
    /// Operation was produced by an internal semantic derivation.
    Derived,
}

#[derive(Clone)]
pub(crate) struct SemanticProvenance {
    kind: SemanticProvenanceKind,
    label: Option<std::sync::Arc<str>>,
}

impl SemanticProvenance {
    pub(crate) fn builder(label: Option<&str>) -> Self {
        Self {
            kind: SemanticProvenanceKind::Builder,
            label: label.map(std::sync::Arc::from),
        }
    }

    pub(crate) fn view(&self) -> SemanticProvenanceView<'_> {
        SemanticProvenanceView {
            kind: self.kind,
            label: self.label.as_deref(),
        }
    }
}

/// Bounded, allocation-free diagnostic provenance view.
#[derive(Clone, Copy)]
pub struct SemanticProvenanceView<'a> {
    kind: SemanticProvenanceKind,
    label: Option<&'a str>,
}

impl<'a> SemanticProvenanceView<'a> {
    /// Return the origin class without exposing source value or operation IDs.
    pub const fn kind(self) -> SemanticProvenanceKind {
        self.kind
    }

    /// Return an optional operation-family or transform label.
    pub const fn label(self) -> Option<&'a str> {
        self.label
    }
}

impl std::fmt::Debug for SemanticProvenanceView<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticProvenanceView")
            .field("kind", &self.kind)
            .field("has_label", &self.label.is_some())
            .finish()
    }
}

/// Dtype and ordered symbolic extents of one semantic SSA value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramValueMetadata {
    dtype: DType,
    shape: Box<[ShapeExtent<DimExpr>]>,
}

impl ProgramValueMetadata {
    /// Construct value metadata from a dtype and symbolic shape.
    pub fn new(dtype: DType, shape: impl IntoIterator<Item = DimExpr>) -> Self {
        Self {
            dtype,
            shape: shape.into_iter().map(ShapeExtent::Exact).collect(),
        }
    }

    /// Construct metadata with explicit exact, bounded, or unknown extents.
    pub fn from_extents(
        dtype: DType,
        shape: impl IntoIterator<Item = ShapeExtent<DimExpr>>,
    ) -> Self {
        Self {
            dtype,
            shape: shape.into_iter().collect(),
        }
    }

    /// Return the value dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Borrow the ordered symbolic extents without allocation.
    pub fn shape(&self) -> &[ShapeExtent<DimExpr>] {
        &self.shape
    }
}

/// Metadata for one externally supplied semantic-program input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramInputSpec {
    metadata: ProgramValueMetadata,
}

impl ProgramInputSpec {
    /// Construct an input specification.
    pub fn new(dtype: DType, shape: impl IntoIterator<Item = DimExpr>) -> Self {
        Self {
            metadata: ProgramValueMetadata::new(dtype, shape),
        }
    }

    /// Construct an input specification with exact, bounded, or unknown extents.
    pub fn from_metadata(metadata: ProgramValueMetadata) -> Self {
        Self { metadata }
    }

    /// Borrow this input's value metadata.
    pub const fn metadata(&self) -> &ProgramValueMetadata {
        &self.metadata
    }
}

/// Relation enforced between two symbolic dimension expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ProgramShapeRelation {
    /// Both expressions must evaluate to the same extent.
    Equal,
    /// The left expression must not exceed the right expression.
    LessEqual,
    /// The left expression must be at least the right expression.
    GreaterEqual,
}

/// A backend-neutral symbolic-shape obligation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapeGuard {
    relation: ProgramShapeRelation,
    lhs: DimExpr,
    rhs: DimExpr,
}

impl ShapeGuard {
    /// Construct a shape guard.
    pub fn new(relation: ProgramShapeRelation, lhs: DimExpr, rhs: DimExpr) -> Self {
        Self { relation, lhs, rhs }
    }

    /// Return the required relation.
    pub const fn relation(&self) -> ProgramShapeRelation {
        self.relation
    }

    /// Borrow the left expression.
    pub const fn lhs(&self) -> &DimExpr {
        &self.lhs
    }

    /// Borrow the right expression.
    pub const fn rhs(&self) -> &DimExpr {
        &self.rhs
    }
}

/// Typed identity of an observable state resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectResource {
    family: &'static str,
    key: u64,
}

impl EffectResource {
    /// Construct a versioned resource identity.
    ///
    /// # Errors
    ///
    /// Returns [`EffectResourceError::InvalidFamily`] when `family` is empty
    /// or has no `.v<version>` suffix.
    pub fn new(family: &'static str, key: u64) -> Result<Self, EffectResourceError> {
        let version = family.rsplit_once(".v").map(|(_, version)| version);
        if family.is_empty()
            || !version.is_some_and(|version| {
                !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit())
            })
        {
            return Err(EffectResourceError::InvalidFamily);
        }
        Ok(Self { family, key })
    }

    /// Return the stable resource family.
    pub const fn family(self) -> &'static str {
        self.family
    }

    /// Return the family-local resource key.
    pub const fn key(self) -> u64 {
        self.key
    }
}

/// Access mode of one semantic effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectAccess {
    /// Read-only access.
    Read,
    /// Mutating access.
    Write,
}

/// One typed observable state access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Effect {
    resource: EffectResource,
    access: EffectAccess,
}

impl Effect {
    /// Construct a typed effect.
    pub const fn new(resource: EffectResource, access: EffectAccess) -> Self {
        Self { resource, access }
    }

    /// Return the affected resource.
    pub const fn resource(self) -> EffectResource {
        self.resource
    }

    /// Return the access mode.
    pub const fn access(self) -> EffectAccess {
        self.access
    }
}

/// Semantic alias class for one operation output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AliasKind {
    /// The output has fresh value semantics.
    Fresh,
    /// The output is a view of an input.
    ViewOf,
    /// The output must alias an input.
    MustAlias,
    /// The output aliases an external typed resource.
    ExternalAlias,
}

/// Alias contract for one operation output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Alias {
    kind: AliasKind,
    output: usize,
    input: Option<usize>,
    resource: Option<EffectResource>,
}

impl Alias {
    /// Declare a semantically fresh output.
    pub const fn fresh(output: usize) -> Self {
        Self {
            kind: AliasKind::Fresh,
            output,
            input: None,
            resource: None,
        }
    }

    /// Declare an output view of an input.
    pub const fn view_of(output: usize, input: usize) -> Self {
        Self {
            kind: AliasKind::ViewOf,
            output,
            input: Some(input),
            resource: None,
        }
    }

    /// Require an output to alias an input.
    pub const fn must_alias(output: usize, input: usize) -> Self {
        Self {
            kind: AliasKind::MustAlias,
            output,
            input: Some(input),
            resource: None,
        }
    }

    /// Declare an alias of an external typed resource.
    pub const fn external(output: usize, resource: EffectResource) -> Self {
        Self {
            kind: AliasKind::ExternalAlias,
            output,
            input: None,
            resource: Some(resource),
        }
    }

    /// Return the alias class.
    pub const fn kind(self) -> AliasKind {
        self.kind
    }

    /// Return the operation-local output index.
    pub const fn output(self) -> usize {
        self.output
    }

    /// Return the aliased input index, when applicable.
    pub const fn input(self) -> Option<usize> {
        self.input
    }

    /// Return the external resource, when applicable.
    pub const fn resource(self) -> Option<EffectResource> {
        self.resource
    }
}

/// Kind of unresolved semantic placement requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SemanticPlacementKind {
    /// No semantic placement restriction.
    Any,
    /// Placement must match one operation input.
    SameAsInput,
}

/// Backend-neutral placement requirement retained for runtime preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SemanticPlacementConstraint {
    kind: SemanticPlacementKind,
    input: Option<usize>,
}

impl SemanticPlacementConstraint {
    /// Construct an unrestricted placement constraint.
    pub const fn any() -> Self {
        Self {
            kind: SemanticPlacementKind::Any,
            input: None,
        }
    }

    /// Require placement compatible with one operation input.
    pub const fn same_as_input(input: usize) -> Self {
        Self {
            kind: SemanticPlacementKind::SameAsInput,
            input: Some(input),
        }
    }

    /// Return the placement kind.
    pub const fn kind(self) -> SemanticPlacementKind {
        self.kind
    }

    /// Return the referenced input index, when applicable.
    pub const fn input(self) -> Option<usize> {
        self.input
    }
}
