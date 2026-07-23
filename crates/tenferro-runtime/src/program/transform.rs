use std::collections::HashMap;

use super::{
    FrozenProgram, ImportedProgramValues, ProgramImport, ProgramValue, SemanticFingerprint,
    SemanticProgramBuilder, SemanticTransformError,
};

/// Opaque fixed-size identity of one semantic transform implementation/configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransformIdentity([u8; 16]);

impl TransformIdentity {
    /// Construct a caller-stable transform identity.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Borrow the fixed-size identity bytes.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// Validation-preserving destination transaction exposed to transforms.
pub struct SemanticTransformContext<'a> {
    builder: &'a mut SemanticProgramBuilder,
}

impl SemanticTransformContext<'_> {
    /// Import source structure, bindings, and ordered roots atomically.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticTransformError::Build`] when the underlying
    /// [`SemanticProgramBuilder`](super::SemanticProgramBuilder) rejects
    /// foreign roots/bindings, invalid structure, or an unrepresentable value
    /// count.
    pub fn import_program(
        &mut self,
        input: &FrozenProgram,
        roots: &[ProgramValue],
    ) -> Result<ImportedProgramValues, SemanticTransformError> {
        self.builder
            .import(ProgramImport {
                program: input.program.as_ref(),
                bindings: &input.bindings,
                roots,
            })
            .map_err(SemanticTransformError::from)
    }

    /// Borrow the destination builder for validated semantic additions.
    pub fn builder(&mut self) -> &mut SemanticProgramBuilder {
        self.builder
    }
}

/// Object-safe transformation over immutable semantic programs and bindings.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::program::{
///     FrozenProgram, ProgramValue, SemanticTransform, SemanticTransformContext,
///     SemanticTransformError, TransformIdentity,
/// };
///
/// struct Identity;
/// impl SemanticTransform for Identity {
///     fn identity(&self) -> TransformIdentity {
///         TransformIdentity::from_bytes([7; 16])
///     }
///
///     fn apply(
///         &self,
///         context: &mut SemanticTransformContext<'_>,
///         input: &FrozenProgram,
///     ) -> Result<Box<[ProgramValue]>, SemanticTransformError> {
///         Ok(context
///             .import_program(input, input.program.outputs())?
///             .roots()
///             .into())
///     }
/// }
///
/// fn accepts_object_safe(_: &dyn SemanticTransform) {}
/// accepts_object_safe(&Identity);
/// ```
pub trait SemanticTransform: Send + Sync {
    /// Return caller-stable identity including transform configuration.
    fn identity(&self) -> TransformIdentity;

    /// Import/build into the supplied fresh destination and return local roots.
    ///
    /// # Errors
    ///
    /// Implementations return [`SemanticTransformError::Build`] for validated
    /// builder/import failures or [`SemanticTransformError::Rejected`] when
    /// their semantic policy does not support the input. The compiler driver
    /// additionally reports [`SemanticTransformError::ForeignReturnedValue`],
    /// [`SemanticTransformError::DroppedBindings`], or
    /// [`SemanticTransformError::Finish`] after this callback returns.
    fn apply(
        &self,
        context: &mut SemanticTransformContext<'_>,
        input: &FrozenProgram,
    ) -> Result<Box<[ProgramValue]>, SemanticTransformError>;
}

#[allow(dead_code, reason = "wired into GraphCompiler in Phase 3 A3")]
pub(crate) fn apply_semantic_transform(
    input: &FrozenProgram,
    transform: &dyn SemanticTransform,
) -> Result<FrozenProgram, SemanticTransformError> {
    let mut builder = SemanticProgramBuilder::new();
    let roots = {
        let mut context = SemanticTransformContext {
            builder: &mut builder,
        };
        transform.apply(&mut context, input)?
    };
    if roots
        .iter()
        .any(|root| builder.validate_value(*root).is_err())
    {
        return Err(SemanticTransformError::ForeignReturnedValue);
    }
    let output = builder.finish(&roots)?;
    if !output.bindings.preserves_all(&input.bindings) {
        return Err(SemanticTransformError::DroppedBindings);
    }
    Ok(output)
}

#[allow(dead_code, reason = "wired into GraphCompiler in Phase 3 A3")]
pub(crate) fn apply_semantic_transforms(
    input: &FrozenProgram,
    transforms: &[&dyn SemanticTransform],
) -> Result<FrozenProgram, SemanticTransformError> {
    let mut current = input.clone();
    for transform in transforms {
        current = apply_semantic_transform(&current, *transform)?;
    }
    Ok(current)
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[allow(dead_code, reason = "wired into GraphCompiler in Phase 3 A3")]
struct TransformCacheKey {
    input: SemanticFingerprint,
    ordered_transforms: Box<[TransformIdentity]>,
}

#[allow(dead_code, reason = "wired into GraphCompiler in Phase 3 A3")]
struct TransformCacheEntry {
    input: FrozenProgram,
    output: FrozenProgram,
}

/// Collision-safe cache used by the later graph compiler transform pipeline.
#[allow(dead_code, reason = "wired into GraphCompiler in Phase 3 A3")]
pub(crate) struct SemanticTransformCache {
    buckets: HashMap<TransformCacheKey, Vec<TransformCacheEntry>>,
    len: usize,
}

#[allow(dead_code, reason = "wired into GraphCompiler in Phase 3 A3")]
impl SemanticTransformCache {
    pub(crate) fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            len: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn apply(
        &mut self,
        input: &FrozenProgram,
        transforms: &[&dyn SemanticTransform],
    ) -> Result<FrozenProgram, SemanticTransformError> {
        let key = TransformCacheKey {
            input: input.program.semantic_fingerprint(),
            ordered_transforms: transforms
                .iter()
                .map(|transform| transform.identity())
                .collect(),
        };
        if let Some(output) = self.buckets.get(&key).and_then(|bucket| {
            bucket.iter().find_map(|entry| {
                (entry.input.program.semantic_eq(input.program.as_ref())
                    && entry.input.bindings.cache_exact_eq(&input.bindings))
                .then(|| entry.output.clone())
            })
        }) {
            return Ok(output);
        }

        let output = apply_semantic_transforms(input, transforms)?;
        self.buckets
            .entry(key)
            .or_default()
            .push(TransformCacheEntry {
                input: input.clone(),
                output: output.clone(),
            });
        self.len += 1;
        Ok(output)
    }
}
