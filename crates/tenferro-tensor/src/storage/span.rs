use std::num::NonZeroUsize;

use super::{AllocationKey, RootResourceIdentity};

/// A half-open byte range with checked end arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ByteRange {
    byte_offset: usize,
    byte_len: usize,
}

impl ByteRange {
    pub(crate) const fn new(byte_offset: usize, byte_len: usize) -> Self {
        Self {
            byte_offset,
            byte_len,
        }
    }

    pub(crate) const fn byte_offset(self) -> usize {
        self.byte_offset
    }

    pub(crate) const fn byte_len(self) -> usize {
        self.byte_len
    }

    pub(crate) fn checked_end(self) -> Result<usize, SpanValidationError> {
        self.byte_offset
            .checked_add(self.byte_len)
            .ok_or(SpanValidationError::RangeOverflow {
                byte_offset: self.byte_offset,
                byte_len: self.byte_len,
            })
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.byte_len == 0
    }

    pub(crate) fn overlaps(self, other: Self) -> Result<bool, SpanValidationError> {
        let self_end = self.checked_end()?;
        let other_end = other.checked_end()?;
        if self.is_empty() || other.is_empty() {
            return Ok(false);
        }

        Ok(self.byte_offset < other_end && other.byte_offset < self_end)
    }
}

/// Errors returned by checked root/span metadata operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum SpanValidationError {
    #[error("byte range end overflows usize: offset {byte_offset}, length {byte_len}")]
    RangeOverflow { byte_offset: usize, byte_len: usize },
    #[error(
        "base plus relative byte offset overflows usize: base {base_byte_offset}, relative {relative_byte_offset}"
    )]
    OffsetOverflow {
        base_byte_offset: usize,
        relative_byte_offset: usize,
    },
    #[error("alignment must be a nonzero power of two: {alignment}")]
    InvalidAlignment { alignment: usize },
    #[error("byte offset {byte_offset} is not aligned to {alignment}")]
    MisalignedOffset {
        byte_offset: usize,
        alignment: usize,
    },
    #[error("root-resource identity does not match the bound span")]
    DifferentRoot {
        expected: super::RootResourceId,
        actual: super::RootResourceId,
    },
    #[error("requested span lies outside the root resource extent")]
    OutsideRootExtent {
        key: AllocationKey,
        root_byte_offset: usize,
        root_byte_len: usize,
        requested_byte_offset: usize,
        requested_byte_len: usize,
    },
}

/// The complete checked byte extent reported for a provider root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RootResourceExtent {
    key: AllocationKey,
    byte_offset: usize,
    byte_len: usize,
    guaranteed_alignment: NonZeroUsize,
}

impl RootResourceExtent {
    pub(crate) fn try_new(
        key: AllocationKey,
        byte_offset: usize,
        byte_len: usize,
        guaranteed_alignment: usize,
    ) -> Result<Self, SpanValidationError> {
        // Range arithmetic is checked before alignment so overflow has stable
        // precedence even when multiple fields are invalid.
        ByteRange::new(byte_offset, byte_len).checked_end()?;
        let guaranteed_alignment = checked_alignment(byte_offset, guaranteed_alignment)?;

        Ok(Self {
            key,
            byte_offset,
            byte_len,
            guaranteed_alignment,
        })
    }

    pub(crate) const fn key(self) -> AllocationKey {
        self.key
    }

    pub(crate) const fn byte_offset(self) -> usize {
        self.byte_offset
    }

    pub(crate) const fn byte_len(self) -> usize {
        self.byte_len
    }

    pub(crate) const fn guaranteed_alignment(self) -> NonZeroUsize {
        self.guaranteed_alignment
    }

    pub(crate) fn validate(&self) -> Result<(), SpanValidationError> {
        ByteRange::new(self.byte_offset, self.byte_len).checked_end()?;
        checked_alignment(self.byte_offset, self.guaranteed_alignment.get())?;
        Ok(())
    }

    /// Validate a relative range in the canonical order without creating an
    /// unbound span value.
    pub(crate) fn validate_relative_range(
        &self,
        relative: ByteRange,
    ) -> Result<(), SpanValidationError> {
        self.relative_parts(relative).map(|_| ())
    }

    pub(crate) fn relative_parts(
        &self,
        relative: ByteRange,
    ) -> Result<(usize, usize, NonZeroUsize), SpanValidationError> {
        // Check every participating end before alignment or containment. This
        // preserves the documented overflow precedence for compound inputs.
        let root_end = ByteRange::new(self.byte_offset, self.byte_len).checked_end()?;
        let relative_end = relative.checked_end()?;
        let byte_offset = self.byte_offset.checked_add(relative.byte_offset).ok_or(
            SpanValidationError::OffsetOverflow {
                base_byte_offset: self.byte_offset,
                relative_byte_offset: relative.byte_offset,
            },
        )?;
        let child_end = ByteRange::new(byte_offset, relative.byte_len).checked_end()?;

        if relative_end > self.byte_len || child_end > root_end {
            return Err(SpanValidationError::OutsideRootExtent {
                key: self.key,
                root_byte_offset: self.byte_offset,
                root_byte_len: self.byte_len,
                requested_byte_offset: byte_offset,
                requested_byte_len: relative.byte_len,
            });
        }

        let root_alignment = checked_alignment(self.byte_offset, self.guaranteed_alignment.get())?;
        let child_alignment = conservative_child_alignment(root_alignment, relative.byte_offset);
        let child_alignment = checked_alignment(byte_offset, child_alignment)?;

        Ok((byte_offset, relative.byte_len, child_alignment))
    }

    #[cfg(test)]
    pub(crate) const fn test_corrupt(
        key: AllocationKey,
        byte_offset: usize,
        byte_len: usize,
        guaranteed_alignment: NonZeroUsize,
    ) -> Self {
        Self {
            key,
            byte_offset,
            byte_len,
            guaranteed_alignment,
        }
    }
}

/// A checked span bound to the exact root identity that created it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RootBoundSpan {
    root_identity: RootResourceIdentity,
    byte_offset: usize,
    byte_len: usize,
    guaranteed_alignment: NonZeroUsize,
}

impl RootBoundSpan {
    pub(crate) const fn root_identity(self) -> RootResourceIdentity {
        self.root_identity
    }

    pub(crate) const fn byte_offset(self) -> usize {
        self.byte_offset
    }

    pub(crate) const fn byte_len(self) -> usize {
        self.byte_len
    }

    pub(crate) const fn guaranteed_alignment(self) -> NonZeroUsize {
        self.guaranteed_alignment
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.byte_len == 0
    }

    pub(crate) fn overlaps(&self, other: &Self) -> Result<bool, SpanValidationError> {
        if self.root_identity.root_resource() != other.root_identity.root_resource() {
            return Ok(false);
        }

        ByteRange::new(self.byte_offset, self.byte_len)
            .overlaps(ByteRange::new(other.byte_offset, other.byte_len))
    }

    pub(crate) fn contains(&self, other: &Self) -> Result<bool, SpanValidationError> {
        if self.root_identity.root_resource() != other.root_identity.root_resource() {
            return Ok(false);
        }

        let self_end = ByteRange::new(self.byte_offset, self.byte_len).checked_end()?;
        let other_end = ByteRange::new(other.byte_offset, other.byte_len).checked_end()?;
        Ok(self.byte_offset <= other.byte_offset && other_end <= self_end)
    }

    pub(crate) fn from_parts(
        root_identity: RootResourceIdentity,
        byte_offset: usize,
        byte_len: usize,
        guaranteed_alignment: NonZeroUsize,
    ) -> Self {
        Self {
            root_identity,
            byte_offset,
            byte_len,
            guaranteed_alignment,
        }
    }
}

fn checked_alignment(
    byte_offset: usize,
    guaranteed_alignment: usize,
) -> Result<NonZeroUsize, SpanValidationError> {
    let alignment =
        NonZeroUsize::new(guaranteed_alignment).ok_or(SpanValidationError::InvalidAlignment {
            alignment: guaranteed_alignment,
        })?;
    if !alignment.is_power_of_two() {
        return Err(SpanValidationError::InvalidAlignment {
            alignment: guaranteed_alignment,
        });
    }
    if byte_offset % alignment.get() != 0 {
        return Err(SpanValidationError::MisalignedOffset {
            byte_offset,
            alignment: guaranteed_alignment,
        });
    }
    Ok(alignment)
}

fn conservative_child_alignment(
    root_guaranteed_alignment: NonZeroUsize,
    relative_byte_offset: usize,
) -> usize {
    if relative_byte_offset == 0 {
        return root_guaranteed_alignment.get();
    }

    let relative_alignment = 1usize << relative_byte_offset.trailing_zeros();
    root_guaranteed_alignment.get().min(relative_alignment)
}
