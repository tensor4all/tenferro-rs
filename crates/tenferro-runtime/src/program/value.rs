use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_BUILDER_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ProgramBuilderNonce(u64);

impl ProgramBuilderNonce {
    pub(crate) fn fresh() -> Self {
        loop {
            let nonce = NEXT_BUILDER_NONCE.fetch_add(1, Ordering::Relaxed);
            if nonce != 0 {
                return Self(nonce);
            }
        }
    }
}

/// An opaque SSA value owned by one semantic-program builder.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramValue {
    pub(crate) slot: u32,
    pub(crate) owner: ProgramBuilderNonce,
}

impl ProgramValue {
    pub(crate) fn new(slot: u32, owner: ProgramBuilderNonce) -> Self {
        Self { slot, owner }
    }
}

impl fmt::Debug for ProgramValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProgramValue(<opaque>)")
    }
}
