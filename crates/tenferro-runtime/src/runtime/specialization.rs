use tenferro_tensor::{DType, MemoryKind, Placement, StrideVec};

use super::{
    InputSignature, InputSpecializationRequirementsError, PrepareError, RankRequirement,
    SpecializationError, StorageClass,
};

/// Selects how much placement metadata enters a specialization key.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::PlacementSpecialization;
///
/// assert_eq!(PlacementSpecialization::None, PlacementSpecialization::None);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementSpecialization {
    /// Do not specialize on placement.
    #[default]
    None,
    /// Specialize only on the storage class derived from memory kind.
    StorageClass,
    /// Specialize on full device placement metadata.
    Device,
}

/// Selects how much layout metadata enters a specialization key.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::LayoutSpecialization;
///
/// assert_eq!(LayoutSpecialization::Class, LayoutSpecialization::Class);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LayoutSpecialization {
    /// Do not specialize on layout.
    #[default]
    None,
    /// Specialize on layout class.
    Class,
    /// Specialize on exact strides.
    ExactStrides,
}

/// Per-input specialization requirements.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::InputSpecializationRequirements;
///
/// let requirements = InputSpecializationRequirements::builder().build()?;
/// assert!(!requirements.specializes_dtype());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSpecializationRequirements {
    dtype: bool,
    rank: bool,
    concrete_dimensions: Vec<u32>,
    placement: PlacementSpecialization,
    layout: LayoutSpecialization,
    alignment_log2: Option<u8>,
}

impl InputSpecializationRequirements {
    /// Return a new requirements builder.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirements;
    ///
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.dtype(true);
    /// assert!(builder.build()?.specializes_dtype());
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> InputSpecializationRequirementsBuilder {
        InputSpecializationRequirementsBuilder::new()
    }

    /// Return whether dtype is specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirements;
    ///
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.dtype(true);
    /// assert!(builder.build()?.specializes_dtype());
    /// # Ok(())
    /// # }
    /// ```
    pub fn specializes_dtype(&self) -> bool {
        self.dtype
    }

    /// Return whether rank is specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirements;
    ///
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.rank(true);
    /// assert!(builder.build()?.specializes_rank());
    /// # Ok(())
    /// # }
    /// ```
    pub fn specializes_rank(&self) -> bool {
        self.rank
    }

    /// Return concrete-dimension axes.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirements;
    ///
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.rank(true).concrete_dimensions(vec![1]);
    /// assert_eq!(builder.build()?.concrete_dimensions(), &[1]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn concrete_dimensions(&self) -> &[u32] {
        &self.concrete_dimensions
    }

    /// Return the placement specialization mode.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSpecializationRequirements, PlacementSpecialization};
    ///
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.placement(PlacementSpecialization::Device);
    /// assert_eq!(builder.build()?.placement(), PlacementSpecialization::Device);
    /// # Ok(())
    /// # }
    /// ```
    pub fn placement(&self) -> PlacementSpecialization {
        self.placement
    }

    /// Return the layout specialization mode.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSpecializationRequirements, LayoutSpecialization};
    ///
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.layout(LayoutSpecialization::Class);
    /// assert_eq!(builder.build()?.layout(), LayoutSpecialization::Class);
    /// # Ok(())
    /// # }
    /// ```
    pub fn layout(&self) -> LayoutSpecialization {
        self.layout
    }

    /// Return the required alignment class, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirements;
    ///
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.alignment_log2(Some(2));
    /// assert_eq!(builder.build()?.alignment_log2(), Some(2));
    /// # Ok(())
    /// # }
    /// ```
    pub fn alignment_log2(&self) -> Option<u8> {
        self.alignment_log2
    }
}

/// Builder for per-input specialization requirements.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::InputSpecializationRequirementsBuilder;
///
/// let mut builder = InputSpecializationRequirementsBuilder::new();
/// builder.dtype(true);
/// assert!(builder.build()?.specializes_dtype());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InputSpecializationRequirementsBuilder {
    dtype: bool,
    rank: bool,
    concrete_dimensions: Vec<u32>,
    placement: PlacementSpecialization,
    layout: LayoutSpecialization,
    alignment_log2: Option<u8>,
}

impl InputSpecializationRequirementsBuilder {
    /// Return a builder with no specialization fields enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirementsBuilder;
    ///
    /// assert!(!InputSpecializationRequirementsBuilder::new()
    ///     .build()
    ///     ?
    ///     .specializes_dtype());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether dtype should be specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirementsBuilder;
    ///
    /// let mut builder = InputSpecializationRequirementsBuilder::new();
    /// builder.dtype(true);
    /// assert!(builder.build()?.specializes_dtype());
    /// # Ok(())
    /// # }
    /// ```
    pub fn dtype(&mut self, dtype: bool) -> &mut Self {
        self.dtype = dtype;
        self
    }

    /// Set whether rank should be specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirementsBuilder;
    ///
    /// let mut builder = InputSpecializationRequirementsBuilder::new();
    /// builder.rank(true);
    /// assert!(builder.build()?.specializes_rank());
    /// # Ok(())
    /// # }
    /// ```
    pub fn rank(&mut self, rank: bool) -> &mut Self {
        self.rank = rank;
        self
    }

    /// Set concrete axes whose dimensions should be specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirementsBuilder;
    ///
    /// let mut builder = InputSpecializationRequirementsBuilder::new();
    /// builder.rank(true).concrete_dimensions(vec![0]);
    /// assert_eq!(builder.build()?.concrete_dimensions(), &[0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn concrete_dimensions(&mut self, axes: impl Into<Vec<u32>>) -> &mut Self {
        self.concrete_dimensions = axes.into();
        self
    }

    /// Set placement specialization.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{
    ///     InputSpecializationRequirementsBuilder, PlacementSpecialization,
    /// };
    ///
    /// let mut builder = InputSpecializationRequirementsBuilder::new();
    /// builder.placement(PlacementSpecialization::StorageClass);
    /// assert_eq!(builder.build()?.placement(), PlacementSpecialization::StorageClass);
    /// # Ok(())
    /// # }
    /// ```
    pub fn placement(&mut self, placement: PlacementSpecialization) -> &mut Self {
        self.placement = placement;
        self
    }

    /// Set layout specialization.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSpecializationRequirementsBuilder, LayoutSpecialization};
    ///
    /// let mut builder = InputSpecializationRequirementsBuilder::new();
    /// builder.rank(true).layout(LayoutSpecialization::ExactStrides);
    /// assert_eq!(builder.build()?.layout(), LayoutSpecialization::ExactStrides);
    /// # Ok(())
    /// # }
    /// ```
    pub fn layout(&mut self, layout: LayoutSpecialization) -> &mut Self {
        self.layout = layout;
        self
    }

    /// Set a required alignment class.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirementsBuilder;
    ///
    /// let mut builder = InputSpecializationRequirementsBuilder::new();
    /// builder.alignment_log2(Some(1));
    /// assert_eq!(builder.build()?.alignment_log2(), Some(1));
    /// # Ok(())
    /// # }
    /// ```
    pub fn alignment_log2(&mut self, alignment_log2: Option<u8>) -> &mut Self {
        self.alignment_log2 = alignment_log2;
        self
    }

    /// Validate and build per-input requirements.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::InputSpecializationRequirementsBuilder;
    ///
    /// let requirements = InputSpecializationRequirementsBuilder::new().build()?;
    /// assert!(requirements.concrete_dimensions().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`InputSpecializationRequirementsError`] when the requested
    /// fields are internally inconsistent or outside finite lattices.
    pub fn build(
        self,
    ) -> Result<InputSpecializationRequirements, InputSpecializationRequirementsError> {
        validate_unique_axes(&self.concrete_dimensions)?;
        if !self.rank {
            if let Some(&axis) = self.concrete_dimensions.first() {
                return Err(InputSpecializationRequirementsError::RankRequired {
                    reason: RankRequirement::ConcreteAxis { axis },
                });
            }
            if self.layout == LayoutSpecialization::ExactStrides {
                return Err(InputSpecializationRequirementsError::RankRequired {
                    reason: RankRequirement::ExactStrides,
                });
            }
        }
        if let Some(alignment_log2) = self.alignment_log2 {
            if u32::from(alignment_log2) >= usize::BITS {
                return Err(
                    InputSpecializationRequirementsError::InvalidAlignmentClass { alignment_log2 },
                );
            }
        }
        Ok(InputSpecializationRequirements {
            dtype: self.dtype,
            rank: self.rank,
            concrete_dimensions: self.concrete_dimensions,
            placement: self.placement,
            layout: self.layout,
            alignment_log2: self.alignment_log2,
        })
    }
}

/// Aggregate specialization requirements for all inputs.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::SpecializationRequirements;
///
/// assert_eq!(SpecializationRequirements::polymorphic(2).inputs().len(), 2);
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpecializationRequirements {
    inputs: Vec<InputSpecializationRequirements>,
}

impl SpecializationRequirements {
    /// Build aggregate requirements from validated input requirements.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSpecializationRequirements, SpecializationRequirements};
    ///
    /// let input = InputSpecializationRequirements::builder().build()?;
    /// assert_eq!(SpecializationRequirements::new(vec![input]).inputs().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(inputs: impl Into<Vec<InputSpecializationRequirements>>) -> Self {
        Self {
            inputs: inputs.into(),
        }
    }

    /// Build fully polymorphic requirements for `input_count` inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::SpecializationRequirements;
    ///
    /// assert_eq!(SpecializationRequirements::polymorphic(3).inputs().len(), 3);
    /// ```
    pub fn polymorphic(input_count: usize) -> Self {
        let input = InputSpecializationRequirements {
            dtype: false,
            rank: false,
            concrete_dimensions: Vec::new(),
            placement: PlacementSpecialization::None,
            layout: LayoutSpecialization::None,
            alignment_log2: None,
        };
        Self {
            inputs: vec![input; input_count],
        }
    }

    /// Return per-input requirements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::SpecializationRequirements;
    ///
    /// assert!(SpecializationRequirements::polymorphic(0).inputs().is_empty());
    /// ```
    pub fn inputs(&self) -> &[InputSpecializationRequirements] {
        &self.inputs
    }

    /// Project a concrete input signature through these requirements.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError::Specialization`] when the signature arity does
    /// not match, a concrete axis is outside the actual rank, or a required
    /// alignment class is unavailable.
    pub fn project(
        &self,
        signature: &InputSignature,
    ) -> Result<SpecializationProjection, PrepareError> {
        if self.inputs.len() != signature.entries().len() {
            return Err(PrepareError::Specialization {
                source: SpecializationError::WrongInputCount {
                    expected: self.inputs.len(),
                    actual: signature.entries().len(),
                },
            });
        }

        let mut inputs = Vec::with_capacity(self.inputs.len());
        for (input, (requirements, entry)) in
            self.inputs.iter().zip(signature.entries()).enumerate()
        {
            let concrete_dimensions =
                project_concrete_dimensions(input, requirements, entry.shape())?;
            let alignment_log2 = project_alignment(input, requirements, entry.alignment_log2())?;
            inputs.push(InputSpecializationProjection {
                dtype: requirements.specializes_dtype().then_some(entry.dtype()),
                rank: requirements
                    .specializes_rank()
                    .then_some(entry.shape().len()),
                concrete_dimensions,
                placement: project_placement(requirements, entry.placement()),
                layout: project_layout(requirements, entry.layout_class().clone(), entry.strides()),
                alignment_log2,
            });
        }

        Ok(SpecializationProjection {
            requirements: self.clone(),
            inputs,
        })
    }

    /// Return whether `self` is strictly wider than `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSpecializationRequirements, SpecializationRequirements};
    ///
    /// let polymorphic = SpecializationRequirements::new(vec![
    ///     InputSpecializationRequirements::builder().build()?,
    /// ]);
    /// let mut builder = InputSpecializationRequirements::builder();
    /// builder.dtype(true);
    /// let dtype = SpecializationRequirements::new(vec![builder.build()?]);
    /// assert!(polymorphic.strictly_widens(&dtype));
    /// # Ok(())
    /// # }
    /// ```
    pub fn strictly_widens(&self, other: &Self) -> bool {
        self.inputs.len() == other.inputs.len()
            && self
                .inputs
                .iter()
                .zip(&other.inputs)
                .try_fold(false, |strict, (left, right)| {
                    input_widening(left, right).map(|input_strict| strict || input_strict)
                })
                .unwrap_or(false)
    }
}

/// Aggregate projection of a concrete signature.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{InputSignature, SpecializationRequirements};
///
/// let projection = SpecializationRequirements::polymorphic(0)
///     .project(&InputSignature::new(Vec::new()))
///     ?;
/// assert!(projection.inputs().is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpecializationProjection {
    requirements: SpecializationRequirements,
    inputs: Vec<InputSpecializationProjection>,
}

impl SpecializationProjection {
    /// Return the requirements that produced this projection.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let requirements = SpecializationRequirements::polymorphic(0);
    /// let projection = requirements.project(&InputSignature::new(Vec::new()))?;
    /// assert_eq!(projection.requirements(), &requirements);
    /// # Ok(())
    /// # }
    /// ```
    pub fn requirements(&self) -> &SpecializationRequirements {
        &self.requirements
    }

    /// Return per-input projections.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn inputs(&self) -> &[InputSpecializationProjection] {
        &self.inputs
    }
}

/// Projected specialization fields for one input.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{InputSignature, SpecializationRequirements};
///
/// let projection = SpecializationRequirements::polymorphic(0)
///     .project(&InputSignature::new(Vec::new()))
///     ?;
/// assert!(projection.inputs().is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSpecializationProjection {
    dtype: Option<DType>,
    rank: Option<usize>,
    concrete_dimensions: Vec<(u32, usize)>,
    placement: Option<PlacementProjection>,
    layout: Option<LayoutProjection>,
    alignment_log2: Option<u8>,
}

impl InputSpecializationProjection {
    /// Return projected dtype, if specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn dtype(&self) -> Option<DType> {
        self.dtype
    }

    /// Return projected rank, if specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn rank(&self) -> Option<usize> {
        self.rank
    }

    /// Return projected concrete dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn concrete_dimensions(&self) -> &[(u32, usize)] {
        &self.concrete_dimensions
    }

    /// Return projected placement, if specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn placement(&self) -> Option<&PlacementProjection> {
        self.placement.as_ref()
    }

    /// Return projected layout, if specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn layout(&self) -> Option<&LayoutProjection> {
        self.layout.as_ref()
    }

    /// Return projected alignment class, if specialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, SpecializationRequirements};
    ///
    /// let projection = SpecializationRequirements::polymorphic(0)
    ///     .project(&InputSignature::new(Vec::new()))
    ///     ?;
    /// assert!(projection.inputs().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn alignment_log2(&self) -> Option<u8> {
        self.alignment_log2
    }
}

/// Projected placement specialization.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{PlacementProjection, StorageClass};
///
/// let storage = StorageClass::new("tenferro.storage.host")?;
/// let projection = PlacementProjection::StorageClass(storage.clone());
/// assert_eq!(projection, PlacementProjection::StorageClass(storage));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PlacementProjection {
    /// Full device placement metadata.
    Device(Placement),
    /// Runtime-created storage class.
    StorageClass(StorageClass),
}

/// Projected layout specialization.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{LayoutClass, LayoutProjection};
///
/// let layout = LayoutClass::new("tenferro.layout.strided")?;
/// let projection = LayoutProjection::Class(layout.clone());
/// assert_eq!(projection, LayoutProjection::Class(layout));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LayoutProjection {
    /// Runtime-created layout class.
    Class(super::LayoutClass),
    /// Exact element strides.
    ExactStrides(StrideVec),
}

fn validate_unique_axes(axes: &[u32]) -> Result<(), InputSpecializationRequirementsError> {
    for duplicate_index in 0..axes.len() {
        if let Some(first_index) =
            (0..duplicate_index).find(|&first_index| axes[first_index] == axes[duplicate_index])
        {
            return Err(InputSpecializationRequirementsError::DuplicateAxis {
                axis: axes[duplicate_index],
                first_index,
                duplicate_index,
            });
        }
    }
    Ok(())
}

fn project_concrete_dimensions(
    input: usize,
    requirements: &InputSpecializationRequirements,
    shape: &[usize],
) -> Result<Vec<(u32, usize)>, PrepareError> {
    let mut projected = Vec::with_capacity(requirements.concrete_dimensions().len());
    for &axis in requirements.concrete_dimensions() {
        let axis_index = axis as usize;
        let Some(&dimension) = shape.get(axis_index) else {
            return Err(PrepareError::Specialization {
                source: SpecializationError::AxisOutOfRange {
                    input,
                    axis,
                    rank: shape.len(),
                },
            });
        };
        projected.push((axis, dimension));
    }
    Ok(projected)
}

fn project_alignment(
    input: usize,
    requirements: &InputSpecializationRequirements,
    actual: Option<u8>,
) -> Result<Option<u8>, PrepareError> {
    match (requirements.alignment_log2(), actual) {
        (None, _) => Ok(None),
        (Some(required_alignment_log2), None) => Err(PrepareError::Specialization {
            source: SpecializationError::AlignmentUnavailable {
                input,
                required_alignment_log2,
            },
        }),
        (Some(required), Some(actual)) => Ok(Some(required.min(actual))),
    }
}

fn project_placement(
    requirements: &InputSpecializationRequirements,
    placement: &Placement,
) -> Option<PlacementProjection> {
    match requirements.placement() {
        PlacementSpecialization::None => None,
        PlacementSpecialization::Device => Some(PlacementProjection::Device(placement.clone())),
        PlacementSpecialization::StorageClass => Some(PlacementProjection::StorageClass(
            storage_class(&placement.memory_kind),
        )),
    }
}

fn project_layout(
    requirements: &InputSpecializationRequirements,
    layout_class: super::LayoutClass,
    strides: &[isize],
) -> Option<LayoutProjection> {
    match requirements.layout() {
        LayoutSpecialization::None => None,
        LayoutSpecialization::Class => Some(LayoutProjection::Class(layout_class)),
        LayoutSpecialization::ExactStrides => Some(LayoutProjection::ExactStrides(
            strides.iter().copied().collect(),
        )),
    }
}

fn storage_class(memory_kind: &MemoryKind) -> StorageClass {
    match memory_kind {
        MemoryKind::Device => StorageClass::runtime_created("tenferro.storage.device.v1"),
        MemoryKind::PinnedHost => StorageClass::runtime_created("tenferro.storage.pinned-host.v1"),
        MemoryKind::UnpinnedHost => {
            StorageClass::runtime_created("tenferro.storage.unpinned-host.v1")
        }
        MemoryKind::Managed => StorageClass::runtime_created("tenferro.storage.managed.v1"),
        MemoryKind::Other(payload) if payload.is_empty() => {
            StorageClass::runtime_created("tenferro.storage.other-empty.v1")
        }
        MemoryKind::Other(payload) => StorageClass::runtime_created(other_utf8_storage_id(payload)),
    }
}

fn other_utf8_storage_id(payload: &str) -> String {
    let mut value =
        String::with_capacity("tenferro.storage.other-utf8-.v1".len() + payload.len() * 2);
    value.push_str("tenferro.storage.other-utf8-");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in payload.as_bytes() {
        value.push(HEX[(byte >> 4) as usize] as char);
        value.push(HEX[(byte & 0x0f) as usize] as char);
    }
    value.push_str(".v1");
    value
}

fn input_widening(
    left: &InputSpecializationRequirements,
    right: &InputSpecializationRequirements,
) -> Option<bool> {
    let mut strict = false;
    strict |= bool_widens(left.specializes_dtype(), right.specializes_dtype())?;
    strict |= bool_widens(left.specializes_rank(), right.specializes_rank())?;
    strict |= axes_widen(left.concrete_dimensions(), right.concrete_dimensions())?;
    strict |= placement_widens(left.placement(), right.placement())?;
    strict |= layout_widens(left.layout(), right.layout())?;
    strict |= alignment_widens(left.alignment_log2(), right.alignment_log2())?;
    Some(strict)
}

fn bool_widens(left: bool, right: bool) -> Option<bool> {
    match (left, right) {
        (false, true) => Some(true),
        (left, right) if left == right => Some(false),
        _ => None,
    }
}

fn axes_widen(left: &[u32], right: &[u32]) -> Option<bool> {
    if left.iter().all(|axis| right.contains(axis)) {
        Some(left.len() < right.len())
    } else {
        None
    }
}

fn placement_widens(left: PlacementSpecialization, right: PlacementSpecialization) -> Option<bool> {
    (left <= right).then_some(left < right)
}

fn layout_widens(left: LayoutSpecialization, right: LayoutSpecialization) -> Option<bool> {
    let left_rank = layout_rank(left);
    let right_rank = layout_rank(right);
    (left_rank <= right_rank).then_some(left_rank < right_rank)
}

fn layout_rank(layout: LayoutSpecialization) -> u8 {
    match layout {
        LayoutSpecialization::None => 0,
        LayoutSpecialization::Class => 1,
        LayoutSpecialization::ExactStrides => 2,
    }
}

fn alignment_widens(left: Option<u8>, right: Option<u8>) -> Option<bool> {
    match (left, right) {
        (None, None) => Some(false),
        (None, Some(_)) => Some(true),
        (Some(_), None) => None,
        (Some(left), Some(right)) if left <= right => Some(left < right),
        _ => None,
    }
}
