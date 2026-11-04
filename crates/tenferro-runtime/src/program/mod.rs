//! Immutable backend-neutral semantic programs.
//!
//! Semantic structure freezes separately from tensor defaults and large
//! constants. Builder-issued values and binding keys are opaque and scoped to
//! one builder/program.
//!
//! # Build, bind, freeze, query, and import
//!
//! ```
//! use std::sync::Arc;
//! use tenferro_ops::dim_expr::DimExpr;
//! use tenferro_runtime::program::{
//!     CoreSemanticOp, ProgramInputSpec, ProgramQueryError, ProgramImport,
//!     SemanticProgramBuilder,
//! };
//! use tenferro_tensor::{DType, Tensor};
//!
//! let mut builder = SemanticProgramBuilder::new();
//! let input = builder.input(ProgramInputSpec::new(
//!     DType::F64,
//!     [DimExpr::Const(2)],
//! ))?;
//! let key = builder.bind_input(
//!     input,
//!     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?,
//! )?;
//! let output = builder.add_op(CoreSemanticOp::Neg, &[input])?[0];
//! let frozen = builder.finish(&[output])?;
//! assert_eq!(frozen.bindings.iter().count(), 1);
//! assert!(frozen.bindings.get(key).is_some());
//! assert_eq!(frozen.program.semantic_fingerprint().as_bytes().len(), 16);
//!
//! let mut other = SemanticProgramBuilder::new();
//! let foreign = other.input(ProgramInputSpec::new(
//!     DType::F64,
//!     [DimExpr::Const(2)],
//! ))?;
//! assert_eq!(
//!     frozen.program.value_metadata(foreign),
//!     Err(ProgramQueryError::ForeignValue),
//! );
//!
//! let mut destination = SemanticProgramBuilder::new();
//! let imported = destination.import(ProgramImport {
//!     program: frozen.program.as_ref(),
//!     bindings: &frozen.bindings,
//!     roots: frozen.program.outputs(),
//! })?;
//! let roundtrip = destination.finish(imported.roots())?;
//! assert!(frozen.program.semantic_eq(roundtrip.program.as_ref()));
//! assert_eq!(roundtrip.bindings.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Opaque identities
//!
//! Raw slots and builder nonces are deliberately inaccessible.
//!
//! ```compile_fail
//! use tenferro_runtime::program::ProgramValue;
//! fn raw_slot(value: ProgramValue) -> u32 {
//!     value.slot
//! }
//! ```
//!
//! ```compile_fail
//! use tenferro_runtime::program::BindingKey;
//! fn raw_slot(key: BindingKey) -> u32 {
//!     key.slot
//! }
//! ```
//!
//! Frozen operation views expose immutable slices only.
//!
//! ```compile_fail
//! use tenferro_runtime::program::SemanticOperationView;
//! fn mutate(view: SemanticOperationView<'_>) {
//!     view.outputs()[0] = view.inputs()[0];
//! }
//! ```
//!
//! Structured control flow is a typed unsupported case until the reserved
//! region/block model is implemented.
//!
//! ```
//! use tenferro_runtime::program::ProgramBuildError;
//! let error = ProgramBuildError::UnsupportedControlFlow { construct: "while" };
//! assert!(matches!(
//!     error,
//!     ProgramBuildError::UnsupportedControlFlow { construct: "while" }
//! ));
//! ```

mod bindings;
mod builder;
mod error;
mod identity;
mod import;
mod metadata;
mod op;
mod semantic;
mod transform;
mod value;

pub use bindings::ProgramBindings;
pub use builder::SemanticProgramBuilder;
pub use error::{
    EffectResourceError, ProgramBindingError, ProgramBuildError, ProgramFinishError,
    ProgramQueryError, ProgramStructuralError, SemanticTransformError,
};
pub use identity::SemanticFingerprint;
pub use import::{ImportedProgramValues, ProgramImport};
pub use metadata::{
    Alias, AliasKind, Effect, EffectAccess, EffectResource, ProgramInputSpec, ProgramShapeRelation,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticPlacementKind,
    SemanticProvenanceKind, SemanticProvenanceView, ShapeGuard,
};
pub use op::{CoreSemanticOp, CoreSemanticOpConversionError, SemanticOpRef, SemanticOperationView};
pub use semantic::{FrozenProgram, SemanticProgram};
pub use transform::{SemanticTransform, SemanticTransformContext, TransformIdentity};
pub use value::{BindingKey, ProgramValue};

#[cfg(test)]
mod tests;
