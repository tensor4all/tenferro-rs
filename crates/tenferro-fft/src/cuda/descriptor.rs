use std::hash::{Hash, Hasher};

use tenferro_gpu::cuda::CudaRuntimeIdentity;

use super::error::CudaFftError;

/// The cuFFT precision and real/complex transform family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CufftTransformKind {
    C2c32,
    C2c64,
    R2c32,
    R2c64,
    C2r32,
    C2r64,
}

/// Direction passed to cuFFT for a transform plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CufftDirection {
    Forward,
    Inverse,
}

/// Checked rank-one arguments for cuFFT's `PlanMany64` interface.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CufftPlanDescriptor {
    pub(crate) kind: CufftTransformKind,
    pub(crate) direction: CufftDirection,
    pub(crate) rank: i32,
    pub(crate) n: [i64; 1],
    pub(crate) inembed: [i64; 1],
    pub(crate) onembed: [i64; 1],
    pub(crate) istride: i64,
    pub(crate) idist: i64,
    pub(crate) ostride: i64,
    pub(crate) odist: i64,
    pub(crate) batch: i64,
}

impl CufftPlanDescriptor {
    pub(crate) fn new(
        kind: CufftTransformKind,
        direction: CufftDirection,
        n: usize,
        batch: usize,
    ) -> Result<Self, CudaFftError> {
        if n == 0 {
            return Err(CudaFftError::InvalidConfiguration { field: "n" });
        }
        if batch == 0 {
            return Err(CudaFftError::InvalidConfiguration { field: "batch" });
        }

        let n_i64 = checked_cufft_i64(n, "n")?;
        let batch_i64 = checked_cufft_i64(batch, "batch")?;
        let element_count = n
            .checked_mul(batch)
            .ok_or(CudaFftError::InvalidConfiguration {
                field: "element_count",
            })?;
        checked_cufft_i64(element_count, "element_count")?;
        let half = n
            .checked_div(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(CudaFftError::InvalidConfiguration {
                field: "half_spectrum_len",
            })?;
        let half_i64 = checked_cufft_i64(half, "half_spectrum_len")?;

        let (inembed, onembed) = match kind {
            CufftTransformKind::C2c32 | CufftTransformKind::C2c64 => ([n_i64], [n_i64]),
            CufftTransformKind::R2c32 | CufftTransformKind::R2c64 => ([n_i64], [half_i64]),
            CufftTransformKind::C2r32 | CufftTransformKind::C2r64 => ([half_i64], [n_i64]),
        };

        Ok(Self {
            kind,
            direction,
            rank: 1,
            n: [n_i64],
            inembed,
            onembed,
            istride: batch_i64,
            idist: 1,
            ostride: batch_i64,
            odist: 1,
            batch: batch_i64,
        })
    }
}

fn checked_cufft_i64(value: usize, field: &'static str) -> Result<i64, CudaFftError> {
    i64::try_from(value).map_err(|_| CudaFftError::InvalidConfiguration { field })
}

/// Hashable structural arguments used to discriminate one cuFFT plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CufftPlanStructuralKey {
    pub(crate) device_ordinal: usize,
    pub(crate) kind: CufftTransformKind,
    pub(crate) direction: CufftDirection,
    pub(crate) n: usize,
    pub(crate) batch: usize,
    pub(crate) istride: i64,
    pub(crate) idist: i64,
    pub(crate) ostride: i64,
    pub(crate) odist: i64,
}

/// Exact runtime identity and structural arguments used to cache one cuFFT plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CufftPlanKey<I = CudaRuntimeIdentity> {
    pub(crate) runtime_identity: I,
    pub(crate) device_ordinal: usize,
    pub(crate) kind: CufftTransformKind,
    pub(crate) direction: CufftDirection,
    pub(crate) n: usize,
    pub(crate) batch: usize,
    pub(crate) istride: i64,
    pub(crate) idist: i64,
    pub(crate) ostride: i64,
    pub(crate) odist: i64,
}

impl<I> CufftPlanKey<I> {
    pub(crate) fn structural_key(&self) -> CufftPlanStructuralKey {
        CufftPlanStructuralKey {
            device_ordinal: self.device_ordinal,
            kind: self.kind,
            direction: self.direction,
            n: self.n,
            batch: self.batch,
            istride: self.istride,
            idist: self.idist,
            ostride: self.ostride,
            odist: self.odist,
        }
    }
}

impl<I> Hash for CufftPlanKey<I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // The runtime identity intentionally need only expose equality, not a
        // hash. The cache discriminator therefore hashes every structural
        // field and relies on exact key equality to reject collisions,
        // including keys from distinct runtime instances on one device ordinal.
        self.structural_key().hash(state);
    }
}
