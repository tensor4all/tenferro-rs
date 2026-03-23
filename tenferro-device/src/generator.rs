use std::convert::TryFrom;
use std::f64::consts::TAU;

use rand_core::Rng;

use crate::{Error, Result};

#[cfg(feature = "cuda")]
fn cuda_device_zero_based_is_available(device_id: usize) -> bool {
    std::panic::catch_unwind(|| {
        cudarc::runtime::result::device::get_count()
            .map(|count| device_id < count as usize)
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

#[derive(Debug)]
struct GeneratorState {
    engine: mt19937::MT19937,
    cached_normal: Option<f64>,
}

impl GeneratorState {
    fn from_seed(seed: u64) -> Self {
        let low = seed as u32;
        let high = (seed >> 32) as u32;
        let seed_words = if high == 0 {
            vec![low]
        } else {
            vec![low, high]
        };
        Self {
            engine: mt19937::MT19937::new_with_slice_seed(&seed_words),
            cached_normal: None,
        }
    }
}

/// Pseudo-random number generator used across the tenferro workspace.
///
/// The CPU half uses an MT19937 engine seeded from a `u64`. CUDA execution
/// uses the same public `Generator` surface but advances an internal
/// seed/offset pair that device kernels consume through a Philox-style
/// counter-based scheme.
///
/// # Examples
///
/// ```
/// use tenferro_device::Generator;
///
/// let mut generator = Generator::cpu(1234);
/// let sample = generator.sample_uniform_f64();
/// assert!(sample >= 0.0 && sample < 1.0);
/// ```
#[derive(Debug)]
pub struct Generator {
    state: GeneratorState,
    #[cfg(feature = "cuda")]
    device_id: Option<usize>,
    #[cfg(feature = "cuda")]
    seed: u64,
    #[cfg(feature = "cuda")]
    offset: u64,
}

impl Generator {
    /// Create a CPU generator from a seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_device::Generator;
    ///
    /// let _generator = Generator::cpu(42);
    /// ```
    pub fn cpu(seed: u64) -> Self {
        Self {
            state: GeneratorState::from_seed(seed),
            #[cfg(feature = "cuda")]
            device_id: None,
            #[cfg(feature = "cuda")]
            seed,
            #[cfg(feature = "cuda")]
            offset: 0,
        }
    }

    /// Create a CUDA generator from a seed and device ordinal.
    ///
    /// The CPU half of the RNG phase only records the metadata so the public
    /// API is in place for the later CUDA implementation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::Generator;
    ///
    /// let _generator = Generator::cuda(0, 1234).unwrap();
    /// ```
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: usize, seed: u64) -> Result<Self> {
        if !cuda_device_zero_based_is_available(device_id) {
            return Err(Error::DeviceError(format!(
                "CUDA generator requires available device {device_id}"
            )));
        }
        Ok(Self {
            state: GeneratorState::from_seed(seed),
            device_id: Some(device_id),
            seed,
            offset: 0,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_seed_and_offset(&self, expected_device_id: usize) -> Result<(u64, u64)> {
        match self.device_id {
            Some(device_id) if device_id == expected_device_id => Ok((self.seed, self.offset)),
            Some(device_id) => Err(Error::DeviceError(format!(
                "CUDA generator is bound to device {device_id}, expected device {expected_device_id}"
            ))),
            None => Err(Error::InvalidArgument(
                "CPU generator cannot drive CUDA RNG execution".into(),
            )),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn advance_cuda_offset(
        &mut self,
        expected_device_id: usize,
        delta: u64,
    ) -> Result<()> {
        match self.device_id {
            Some(device_id) if device_id == expected_device_id => {
                self.offset = self
                    .offset
                    .checked_add(delta)
                    .ok_or_else(|| Error::DeviceError("CUDA generator offset overflow".into()))?;
                Ok(())
            }
            Some(device_id) => Err(Error::DeviceError(format!(
                "CUDA generator is bound to device {device_id}, expected device {expected_device_id}"
            ))),
            None => Err(Error::InvalidArgument(
                "CPU generator cannot drive CUDA RNG execution".into(),
            )),
        }
    }

    /// Draw a floating-point sample from the half-open interval `[0, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_device::Generator;
    ///
    /// let mut generator = Generator::cpu(7);
    /// let x = generator.sample_uniform_f64();
    /// assert!(x >= 0.0 && x < 1.0);
    /// ```
    pub fn sample_uniform_f64(&mut self) -> f64 {
        mt19937::gen_res53(&mut self.state.engine)
    }

    /// Draw a standard normal sample using Box-Muller sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_device::Generator;
    ///
    /// let mut generator = Generator::cpu(7);
    /// let _z = generator.sample_standard_normal_f64();
    /// ```
    pub fn sample_standard_normal_f64(&mut self) -> f64 {
        if let Some(sample) = self.state.cached_normal.take() {
            return sample;
        }

        let u1 = self.sample_uniform_f64().max(f64::MIN_POSITIVE);
        let u2 = self.sample_uniform_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.state.cached_normal = Some(z1);
        z0
    }

    /// Draw an integer sample from the half-open interval `[low, high)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `low >= high`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_device::Generator;
    ///
    /// let mut generator = Generator::cpu(7);
    /// let x = generator.sample_integer_i32(-3, 7).unwrap();
    /// assert!((-3..7).contains(&x));
    /// ```
    pub fn sample_integer_i32(&mut self, low: i32, high: i32) -> Result<i32> {
        if low >= high {
            return Err(Error::InvalidArgument(format!(
                "invalid integer sample range [{low}, {high})"
            )));
        }

        let span = i64::from(high) - i64::from(low);
        let span_u64 = u64::try_from(span).map_err(|_| {
            Error::InvalidArgument(format!("integer sample span {span} does not fit into u64"))
        })?;
        let threshold = u64::MAX - (u64::MAX % span_u64);

        loop {
            let candidate = self.state.engine.next_u64();
            if candidate < threshold {
                let value = i64::from(low)
                    + i64::try_from(candidate % span_u64).map_err(|_| {
                        Error::InvalidArgument("integer sample conversion overflow".into())
                    })?;
                return i32::try_from(value).map_err(|_| {
                    Error::InvalidArgument(format!("integer sample {value} does not fit into i32"))
                });
            }
        }
    }
}
