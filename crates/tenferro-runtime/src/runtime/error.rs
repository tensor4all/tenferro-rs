use super::identity::EngineId;
use tenferro_tensor::Error as TensorError;

/// Classifies a malformed runtime identifier without retaining its input.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::EngineId;
///
/// let error = EngineId::new("not namespaced").unwrap_err();
/// assert_eq!(error.kind(), tenferro_runtime::IdentityKind::Engine);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("malformed {kind:?} identifier")]
pub struct IdentityError {
    kind: IdentityKind,
}

impl IdentityError {
    /// Return the kind of identifier that failed validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, IdentityKind};
    ///
    /// assert_eq!(EngineId::new("engine").unwrap_err().kind(), IdentityKind::Engine);
    /// ```
    pub fn kind(&self) -> IdentityKind {
        self.kind
    }

    pub(super) const fn malformed(kind: IdentityKind) -> Self {
        Self { kind }
    }
}

/// Identifies the runtime namespace validated by an opaque identifier.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::IdentityKind;
///
/// assert_eq!(IdentityKind::Engine, IdentityKind::Engine);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum IdentityKind {
    /// An execution engine identifier.
    Engine,
    /// A hardware-class identifier.
    HardwareClass,
    /// A storage-class identifier.
    StorageClass,
    /// A layout-class identifier.
    LayoutClass,
}

/// Reports invalid placement constraints.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{EngineId, PlacementConstraintError, ProgramPlacementConstraint};
///
/// let engine = EngineId::new("tenferro.cpu").unwrap();
/// let error = ProgramPlacementConstraint::new(vec![engine.clone(), engine], None).unwrap_err();
/// assert!(matches!(error, PlacementConstraintError::DuplicateEngine { .. }));
/// ```
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PlacementConstraintError {
    /// The same engine appears more than once in a preference list.
    #[error(
        "engine {engine_id:?} is duplicated at positions \
         {first_index} and {duplicate_index}"
    )]
    DuplicateEngine {
        /// The duplicated engine identifier.
        engine_id: EngineId,
        /// The first position where the engine appeared.
        first_index: usize,
        /// The duplicate position that failed validation.
        duplicate_index: usize,
    },
}

/// Reports failures while building value-free input signatures.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
/// use tenferro_tensor::Placement;
///
/// let error = InputSignatureEntry::new(
///     DType::F64,
///     [2_usize].into_iter().collect(),
///     Placement::default(),
///     LayoutClass::new("tenferro.layout.strided").unwrap(),
///     [1_isize, 2].into_iter().collect(),
///     None,
/// )
/// .unwrap_err();
/// assert!(matches!(error, tenferro_runtime::InputSignatureError::ShapeStrideRankMismatch { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum InputSignatureError {
    /// Shape rank and stride rank disagree.
    #[error("shape rank {rank} does not match stride count {stride_count}")]
    ShapeStrideRankMismatch {
        /// Number of shape axes.
        rank: usize,
        /// Number of stride axes.
        stride_count: usize,
    },
    /// The alignment class is outside the finite `usize` alignment lattice.
    #[error("alignment class {alignment_log2} is outside the usize alignment lattice")]
    InvalidAlignmentClass {
        /// Base-2 logarithm of the requested alignment.
        alignment_log2: u8,
    },
    /// Tensor metadata could not be read for an input.
    #[error("input {input} metadata is unavailable")]
    TensorMetadata {
        /// Input position in the prepare request.
        input: usize,
        /// Original typed tensor error.
        source: TensorError,
    },
}

/// Explains why rank specialization is required by a requested field.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::RankRequirement;
///
/// assert_eq!(RankRequirement::ExactStrides.to_string(), "exact strides");
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RankRequirement {
    /// A concrete axis dimension was requested.
    ConcreteAxis {
        /// Axis that requires rank specialization.
        axis: u32,
    },
    /// Exact strides were requested.
    ExactStrides,
}

impl std::fmt::Display for RankRequirement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConcreteAxis { axis } => write!(formatter, "concrete axis {axis}"),
            Self::ExactStrides => formatter.write_str("exact strides"),
        }
    }
}

/// Reports invalid input-specialization requirements.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{
///     InputSpecializationRequirements, InputSpecializationRequirementsError, RankRequirement,
/// };
///
/// let mut builder = InputSpecializationRequirements::builder();
/// builder.rank(false).concrete_dimensions(vec![2]);
/// assert_eq!(
///     builder.build().unwrap_err(),
///     InputSpecializationRequirementsError::RankRequired {
///         reason: RankRequirement::ConcreteAxis { axis: 2 },
///     }
/// );
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum InputSpecializationRequirementsError {
    /// The same concrete axis appears more than once.
    #[error("axis {axis} is duplicated at positions {first_index} and {duplicate_index}")]
    DuplicateAxis {
        /// Duplicated concrete axis.
        axis: u32,
        /// First position carrying the axis.
        first_index: usize,
        /// Duplicate position that failed validation.
        duplicate_index: usize,
    },
    /// A requested specialization field requires rank specialization.
    #[error("{reason} requires rank specialization")]
    RankRequired {
        /// Requirement that needs rank specialization.
        reason: RankRequirement,
    },
    /// The alignment class is outside the finite `usize` alignment lattice.
    #[error("alignment class {alignment_log2} is outside the usize alignment lattice")]
    InvalidAlignmentClass {
        /// Base-2 logarithm of the requested alignment.
        alignment_log2: u8,
    },
}

/// Reports failures while projecting signatures through specialization requirements.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{InputSignature, SpecializationRequirements};
///
/// let error = SpecializationRequirements::polymorphic(1)
///     .project(&InputSignature::new(Vec::new()))
///     .unwrap_err();
/// assert!(matches!(error, tenferro_runtime::PrepareError::Specialization { .. }));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SpecializationError {
    /// The signature does not have the required input arity.
    #[error("expected {expected} inputs but got {actual}")]
    WrongInputCount {
        /// Required input count.
        expected: usize,
        /// Actual input count.
        actual: usize,
    },
    /// A concrete-axis request names an axis outside the actual rank.
    #[error("input {input} axis {axis} is outside rank {rank}")]
    AxisOutOfRange {
        /// Input position in the signature.
        input: usize,
        /// Requested axis.
        axis: u32,
        /// Actual rank.
        rank: usize,
    },
    /// A required host-pointer alignment class is unavailable.
    #[error("input {input} has unknown alignment; required class {required_alignment_log2}")]
    AlignmentUnavailable {
        /// Input position in the signature.
        input: usize,
        /// Required base-2 logarithm alignment class.
        required_alignment_log2: u8,
    },
}

/// Reports failures while preparing runtime execution artifacts.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{InputSignature, PrepareError, SpecializationRequirements};
///
/// let error = SpecializationRequirements::polymorphic(1)
///     .project(&InputSignature::new(Vec::new()))
///     .unwrap_err();
/// assert!(matches!(error, PrepareError::Specialization { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    /// Input signature construction failed.
    #[error("input signature failed")]
    InputSignature {
        /// Typed source error.
        source: InputSignatureError,
    },
    /// Signature projection through specialization requirements failed.
    #[error("specialization failed")]
    Specialization {
        /// Typed source error.
        source: SpecializationError,
    },
}
