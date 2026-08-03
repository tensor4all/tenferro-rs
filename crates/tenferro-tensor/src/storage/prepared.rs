use std::fmt;
use std::ops::{Deref, DerefMut};

use crate::DType;

/// Typed failure at the private prepared-access boundary.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum AccessError {
    #[error("invalid checked layout: {message}")]
    InvalidLayout { message: String },
    #[error("dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("mapped byte length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("mapped bytes are not aligned for {required}-byte alignment")]
    Misaligned { required: usize },
    #[error("provider `{backend}` does not support the requested mapping")]
    Unsupported { backend: &'static str },
    #[error("provider mapping failed: {message}")]
    Provider { message: String },
    #[error("provider completion cannot be proven: {message}")]
    CompletionUnproven { message: String },
}

trait ReadMappingAccess {
    fn bytes(&self) -> &[u8];
}

trait WriteMappingAccess {
    fn len(&self) -> usize;
    fn bytes_mut(&mut self) -> &mut [u8];
}

struct BorrowedRead<'a>(&'a [u8]);

impl ReadMappingAccess for BorrowedRead<'_> {
    fn bytes(&self) -> &[u8] {
        self.0
    }
}

struct GuardRead<G>(G);

impl<G> ReadMappingAccess for GuardRead<G>
where
    G: Deref<Target = [u8]>,
{
    fn bytes(&self) -> &[u8] {
        self.0.deref()
    }
}

struct BorrowedWrite<'a>(&'a mut [u8]);

impl WriteMappingAccess for BorrowedWrite<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self.0
    }
}

struct GuardWrite<G>(G);

impl<G> WriteMappingAccess for GuardWrite<G>
where
    G: DerefMut<Target = [u8]>,
{
    fn len(&self) -> usize {
        self.0.deref().len()
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self.0.deref_mut()
    }
}

/// Borrowed provider mapping retained by a prepared read.
pub(crate) struct ProviderReadMapping<'a> {
    access: Box<dyn ReadMappingAccess + 'a>,
}

impl fmt::Debug for ProviderReadMapping<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderReadMapping")
            .field("byte_len", &self.bytes().len())
            .finish_non_exhaustive()
    }
}

impl<'a> ProviderReadMapping<'a> {
    pub(crate) fn from_slice(bytes: &'a [u8]) -> Self {
        Self {
            access: Box::new(BorrowedRead(bytes)),
        }
    }

    pub(crate) fn from_guard<G>(guard: G) -> Self
    where
        G: Deref<Target = [u8]> + 'a,
    {
        Self {
            access: Box::new(GuardRead(guard)),
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        self.access.bytes()
    }
}

/// Borrowed provider mapping retained by a prepared write.
pub(crate) struct ProviderWriteMapping<'a> {
    access: Box<dyn WriteMappingAccess + 'a>,
}

impl fmt::Debug for ProviderWriteMapping<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderWriteMapping")
            .field("byte_len", &self.access.len())
            .finish_non_exhaustive()
    }
}

impl<'a> ProviderWriteMapping<'a> {
    pub(crate) fn from_slice(bytes: &'a mut [u8]) -> Self {
        Self {
            access: Box::new(BorrowedWrite(bytes)),
        }
    }

    pub(crate) fn from_guard<G>(guard: G) -> Self
    where
        G: DerefMut<Target = [u8]> + 'a,
    {
        Self {
            access: Box::new(GuardWrite(guard)),
        }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        self.access.bytes_mut()
    }
}

// The remaining checked/prepared types are introduced in the next TDD step.
