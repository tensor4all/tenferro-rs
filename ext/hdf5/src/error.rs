//! Errors of the tenferro HDF5 adapter.
//!
//! Every variant names the operation or the schema rule that failed, so a caller
//! can tell a corrupt file from a wrong call without parsing messages.

use tenferro_tensor::DType;

/// A schema rule the dataset violates.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::SchemaViolation;
///
/// let violation = SchemaViolation::MissingAttribute("type");
/// assert!(violation.to_string().contains("type"));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SchemaViolation {
    /// A required attribute is absent.
    #[error("missing required attribute {0:?}")]
    MissingAttribute(&'static str),
    /// The `type` attribute names a different dataset kind.
    #[error("dataset is not a tenferro tensor (type = {found:?})")]
    NotTenferroTensor {
        /// Value found in the `type` attribute.
        found: String,
    },
    /// The `dtype` attribute is not one of the schema's names.
    #[error("unknown dtype name {0:?}")]
    UnknownDTypeName(String),
    /// The HDF5 datatype disagrees with the `dtype` attribute.
    #[error("HDF5 datatype {found} does not encode {dtype:?}")]
    DatatypeMismatch {
        /// Dtype named by the `dtype` attribute.
        dtype: DType,
        /// HDF5 datatype found on the dataset.
        found: String,
    },
    /// The dataset has a null dataspace, which carries no shape at all.
    #[error("dataset has a null dataspace")]
    NullDataspace,
    /// The named attribute is not a one-element scalar.
    #[error("attribute {0:?} is not a scalar attribute")]
    AttributeNotScalar(&'static str),
    /// The named attribute holds bytes that are not valid UTF-8.
    #[error("attribute {0:?} does not hold valid UTF-8")]
    AttributeNotUtf8(&'static str),
    /// The named attribute is not an integer attribute.
    #[error("attribute {name:?} has datatype {found}, expected an integer")]
    AttributeNotInteger {
        /// Attribute name.
        name: &'static str,
        /// Datatype found on the attribute.
        found: String,
    },
}

/// Failure of a tensor payload read or write.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::Error;
///
/// let error = Error::RankTooLarge { rank: 33 };
/// assert!(error.to_string().contains("33"));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An HDF5 call failed.
    #[error("{op}: {source}")]
    Hdf5 {
        /// Adapter operation that was running.
        op: &'static str,
        /// Underlying HDF5 error.
        #[source]
        source: hdf5_rt::Error,
    },
    /// The tensor's payload lives in a backend buffer, and this adapter never
    /// transfers or materializes one.
    #[error("{op}: tensor is not host-resident: {source}")]
    NotHostResident {
        /// Adapter operation that was running.
        op: &'static str,
        /// Error the tensor raised while refusing host access.
        #[source]
        source: tenferro_tensor::Error,
    },
    /// The tensor's payload is not compact column-major, and this adapter never
    /// compacts a view implicitly.
    #[error("tensor layout is not compact column-major: shape {shape:?}, strides {strides:?}")]
    NotCompact {
        /// Logical shape of the rejected view.
        shape: Vec<usize>,
        /// Strides of the rejected view.
        strides: Vec<isize>,
    },
    /// The dtype has no v1 encoding. Only [`DType::External`] payloads reach this.
    #[error("dtype {dtype:?} has no HDF5 encoding in schema version 1")]
    UnsupportedDType {
        /// Rejected dtype.
        dtype: DType,
    },
    /// HDF5 dataspaces cannot exceed rank 32.
    #[error(
        "tensor rank {rank} exceeds the HDF5 maximum rank {}",
        crate::schema::MAX_RANK
    )]
    RankTooLarge {
        /// Rank of the rejected tensor.
        rank: usize,
    },
    /// The link name is not a single ordinary link name.
    #[error(
        "invalid dataset name {name:?}: expected one non-empty link name without '/', '.' or '..'"
    )]
    InvalidName {
        /// Rejected name.
        name: String,
    },
    /// A link with this name already exists. Replacing is the caller's decision.
    #[error("a link named {name:?} already exists; unlink it first to replace it")]
    AlreadyExists {
        /// Name of the existing link.
        name: String,
    },
    /// The dataset declares a schema version this adapter does not read.
    #[error("unsupported schema version {found}; this adapter reads version {supported}")]
    UnsupportedVersion {
        /// Version found in the dataset.
        found: i64,
        /// Version this adapter reads.
        supported: i64,
    },
    /// The logical element count exceeds the caller's limit.
    #[error("dataset holds {elements} elements, above the configured limit {limit}")]
    ElementLimit {
        /// Element count the dataset declares.
        elements: usize,
        /// Limit supplied through `ReadOptions`.
        limit: usize,
    },
    /// A typed read was asked for a dtype the dataset does not store.
    #[error("dataset stores {found:?}, this read expects {expected:?}")]
    DTypeMismatch {
        /// Dtype the caller asked for.
        expected: DType,
        /// Dtype the dataset stores.
        found: DType,
    },
    /// A statically ranked read was asked for a rank the dataset does not have.
    #[error("dataset has rank {found}, this read expects rank {expected}")]
    RankMismatch {
        /// Rank the caller asked for.
        expected: usize,
        /// Rank the dataset has.
        found: usize,
    },
    /// The declared shape and element size do not fit in a host allocation.
    #[error("shape {dims:?} does not fit in a host allocation")]
    ShapeOverflow {
        /// Rejected logical shape.
        dims: Vec<usize>,
    },
    /// The dataset violates the v1 schema.
    #[error("dataset {name:?} violates the tenferro schema: {violation}")]
    Schema {
        /// Dataset link name.
        name: String,
        /// Violated rule.
        #[source]
        violation: SchemaViolation,
    },
    /// A text attribute cannot be encoded as an HDF5 string.
    #[error("attribute {name:?} value {value:?} cannot be encoded as an HDF5 string")]
    InvalidAttributeValue {
        /// Attribute name.
        name: &'static str,
        /// Rejected value.
        value: String,
    },
    /// A stored `bool` element is neither 0 nor 1.
    #[error("stored bool element {flat_index} has byte {byte}, expected 0 or 1")]
    InvalidBool {
        /// Column-major flat index of the offending element.
        flat_index: usize,
        /// Stored byte.
        byte: u8,
    },
    /// The tensor constructor rejected the read shape or length.
    #[error("cannot build the tensor: {source}")]
    Tensor {
        /// Error the tensor constructor raised.
        #[source]
        source: tenferro_tensor::Error,
    },
}
