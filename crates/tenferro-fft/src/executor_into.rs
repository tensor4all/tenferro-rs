//! Caller-owned output execution and reusable multi-axis schedules.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use tenferro_runtime::ExtensionCacheKey;
use tenferro_tensor::{
    AllocationDomainId, AsView, DType, Dest, DestMode, Engine, Placement, Tensor, TensorRead,
    TensorWrite,
};

use crate::{
    output_shape_c2c, output_shape_c2r, output_shape_r2c, validated_fft_op, with_fft_exec_session,
    FftBackend, FftExecutionCache, FftExecutor, FftOperation, FftOptions, FftPlanCache,
    FftPlanSpec, FftnOptions, FFT_EXTENSION_FAMILY_ID,
};

type Result<T> = tenferro_tensor::Result<T>;
type OperationOf = fn(&'static str, DType) -> Result<FftOperation>;

#[derive(Clone, Copy)]
enum Options<'a> {
    One(&'a FftOptions),
    Many(&'a FftnOptions),
}

enum SavedOptions {
    One(FftOptions),
    Many(FftnOptions),
}

impl Options<'_> {
    fn hash(self, state: &mut impl Hasher) {
        match self {
            Self::One(o) => {
                0_u8.hash(state);
                o.n.hash(state);
                o.axis.hash(state);
                (o.norm as u8).hash(state);
            }
            Self::Many(o) => {
                1_u8.hash(state);
                o.shape.hash(state);
                o.axes.hash(state);
                (o.norm as u8).hash(state);
            }
        }
    }

    fn saved(self) -> SavedOptions {
        match self {
            Self::One(o) => SavedOptions::One(*o),
            Self::Many(o) => SavedOptions::Many(o.clone()),
        }
    }

    fn matches(self, saved: &SavedOptions) -> bool {
        match (self, saved) {
            (Self::One(a), SavedOptions::One(b)) => a == b,
            (Self::Many(a), SavedOptions::Many(b)) => a == b,
            _ => false,
        }
    }
}

struct Schedule {
    operation: FftOperation,
    dtype: DType,
    shape: Vec<usize>,
    placement: Placement,
    allocation_domain: Option<AllocationDomainId>,
    options: SavedOptions,
    specs: Vec<FftPlanSpec>,
    output_shape: Vec<usize>,
    output_dtype: DType,
    intermediates: Vec<Tensor>,
}

impl Schedule {
    fn new(
        input: &TensorRead<'_>,
        name: &'static str,
        operation: FftOperation,
        options: Options<'_>,
    ) -> Result<Self> {
        let (axes, lengths, norm) = match options {
            Options::One(o) => (vec![o.axis], o.n.map(|n| vec![n]), o.norm),
            Options::Many(o) => (
                o.resolved_axes(input.shape().len()),
                o.shape.clone(),
                o.norm,
            ),
        };
        let op = validated_fft_op(
            name,
            operation,
            Some(input.dtype()),
            input.shape().len(),
            Some(input.shape()),
            lengths.as_deref(),
            "shape",
            &axes,
            norm,
        )?;
        let mut shape = input.shape().to_vec();
        let mut dtype = input.dtype();
        let mut specs = Vec::with_capacity(op.axes.len());
        for index in op.execution_order() {
            let spec = op.step_spec(index, dtype, &shape);
            let axis = spec.normalized_axis();
            let len = spec.requested_len();
            shape = match spec.operation() {
                FftOperation::C2cForward | FftOperation::C2cInverse => {
                    output_shape_c2c(&shape, axis, len)?
                }
                FftOperation::R2cFull | FftOperation::R2cOnesided => {
                    output_shape_r2c(&shape, axis, len, spec.operation().is_onesided())?
                }
                FftOperation::C2r => output_shape_c2r(&shape, axis, len)?,
            };
            dtype = match (spec.operation(), dtype) {
                (FftOperation::C2r, DType::C32) => DType::F32,
                (FftOperation::C2r, DType::C64) => DType::F64,
                (_, DType::F32) => DType::C32,
                (_, DType::F64) => DType::C64,
                (_, dtype) => dtype,
            };
            specs.push(spec);
        }
        Ok(Self {
            operation,
            dtype: input.dtype(),
            shape: input.shape().to_vec(),
            placement: input.placement().clone(),
            allocation_domain: input.allocation_domain(),
            options: options.saved(),
            specs,
            output_shape: shape,
            output_dtype: dtype,
            intermediates: Vec::with_capacity(op.axes.len().saturating_sub(1)),
        })
    }

    fn matches(
        &self,
        input: &TensorRead<'_>,
        operation: FftOperation,
        options: Options<'_>,
    ) -> bool {
        self.operation == operation
            && self.dtype == input.dtype()
            && self.shape == input.shape()
            && self.placement == *input.placement()
            && options.matches(&self.options)
    }

    fn validate_destination(&self, name: &'static str, dest: &TensorWrite<'_>) -> Result<()> {
        if dest.dtype() != self.output_dtype {
            return Err(tenferro_tensor::Error::dtype_mismatch(
                name,
                self.output_dtype,
                dest.dtype(),
            ));
        }
        if dest.shape() != self.output_shape {
            return Err(tenferro_tensor::Error::shape_mismatch(
                name,
                self.output_shape.clone(),
                dest.shape().to_vec(),
            ));
        }
        let output = dest.as_read();
        if output.placement() != &self.placement {
            return Err(tenferro_tensor::Error::invalid_argument(
                name,
                "dest",
                "FFT destination placement must match the input placement",
            ));
        }
        if output.allocation_domain() != self.allocation_domain {
            return Err(tenferro_tensor::Error::runtime_state(
                name,
                "FFT destination allocation domain must match the input allocation domain",
            ));
        }
        // INVARIANT: Owned tensors are compact by construction; querying their synthetic
        // layout would allocate metadata for ranks above INLINE_RANK.
        if matches!(dest, TensorWrite::View(_)) && !dest.is_col_major_contiguous()? {
            return Err(tenferro_tensor::Error::unsupported(
                name,
                "FFT destinations must be compact column-major",
            ));
        }
        Ok(())
    }

    fn execute(
        &mut self,
        input: TensorRead<'_>,
        name: &'static str,
        backend: &mut dyn FftBackend,
        plans: &mut FftPlanCache,
        dest: TensorWrite<'_>,
    ) -> Result<()> {
        self.validate_destination(name, &dest)?;
        backend.validate_fft_read_input(name, &input)?;
        let count = self.specs.len();
        for index in 0..count {
            let spec = &self.specs[index];
            if index + 1 == count {
                let source = if index == 0 {
                    input.clone()
                } else {
                    TensorRead::Tensor(&self.intermediates[index - 1])
                };
                return backend.execute_fft_read_into(
                    source,
                    spec,
                    FftExecutionCache::caller_owned(plans),
                    dest,
                );
            }
            if index == self.intermediates.len() {
                let source = if index == 0 {
                    input.clone()
                } else {
                    TensorRead::Tensor(&self.intermediates[index - 1])
                };
                self.intermediates.push(backend.execute_fft_read(
                    source,
                    spec,
                    FftExecutionCache::caller_owned(plans),
                )?);
            } else {
                let (previous, next) = self.intermediates.split_at_mut(index);
                let source = if index == 0 {
                    input.clone()
                } else {
                    TensorRead::Tensor(&previous[index - 1])
                };
                backend.execute_fft_read_into(
                    source,
                    spec,
                    FftExecutionCache::caller_owned(plans),
                    TensorWrite::from_tensor(&mut next[0]),
                )?;
            }
        }
        Err(tenferro_tensor::Error::invalid_argument(
            name,
            "axes",
            "FFT requires at least one axis",
        ))
    }

    fn retained_bytes(&self) -> usize {
        let metadata = self
            .shape
            .capacity()
            .saturating_add(self.output_shape.capacity())
            .saturating_add(
                self.specs
                    .iter()
                    .map(|s| {
                        s.input_shape().len()
                            + s.axes().len()
                            + s.lengths().map_or(0, <[usize]>::len)
                    })
                    .sum::<usize>(),
            )
            .saturating_mul(std::mem::size_of::<usize>());
        let options = match &self.options {
            SavedOptions::One(_) => 0,
            SavedOptions::Many(o) => o
                .axes
                .capacity()
                .saturating_mul(std::mem::size_of::<isize>())
                .saturating_add(o.shape.as_ref().map_or(0, |s| {
                    s.capacity().saturating_mul(std::mem::size_of::<usize>())
                })),
        };
        let payload = self
            .intermediates
            .iter()
            .map(|t| {
                let bytes: usize = match t.dtype() {
                    DType::C32 | DType::F64 => 8,
                    DType::C64 => 16,
                    _ => 4,
                };
                t.shape()
                    .iter()
                    .fold(bytes, |total, &dim| total.saturating_mul(dim))
            })
            .fold(0usize, usize::saturating_add);
        std::mem::size_of::<Self>()
            .saturating_add(metadata)
            .saturating_add(options)
            .saturating_add(payload)
            .saturating_add(
                self.specs
                    .capacity()
                    .saturating_mul(std::mem::size_of::<FftPlanSpec>()),
            )
            .saturating_add(
                self.intermediates
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Tensor>()),
            )
    }
}

fn execute_into(
    input: TensorRead<'_>,
    name: &'static str,
    operation_of: OperationOf,
    options: Options<'_>,
    plans: &mut FftPlanCache,
    engine: &mut dyn Engine,
    dest: Dest<TensorWrite<'_>>,
) -> Result<()> {
    let dest = overwrite(name, dest)?;
    let operation = operation_of(name, input.dtype())?;
    let mut hash = DefaultHasher::new();
    operation.hash(&mut hash);
    input.dtype().hash(&mut hash);
    input.shape().hash(&mut hash);
    input.placement().hash(&mut hash);
    options.hash(&mut hash);
    let key = ExtensionCacheKey::new(FFT_EXTENSION_FAMILY_ID, "into-schedules", hash.finish());
    // Plain host tensors own portable storage; managed/device intermediates must
    // die with their engine, so those schedules are intentionally per-call.
    let reusable = input.placement().device.is_none()
        && input.is_host_memory()
        && input.allocation_domain().is_none();
    if reusable {
        if let Some(schedule) = plans.store_mut().get::<Arc<Mutex<Schedule>>>(&key).cloned() {
            let mut schedule = schedule.lock().map_err(|_| {
                tenferro_tensor::Error::runtime_state(name, "FFT schedule cache lock poisoned")
            })?;
            if schedule.matches(&input, operation, options) {
                return with_fft_exec_session(engine, name, |backend| {
                    schedule.execute(input, name, backend, plans, dest)
                });
            }
        }
    }
    let mut schedule = Schedule::new(&input, name, operation, options)?;
    schedule.validate_destination(name, &dest)?;
    with_fft_exec_session(engine, name, |backend| {
        schedule.execute(input, name, backend, plans, dest)
    })?;
    if reusable {
        let bytes = schedule.retained_bytes();
        plans
            .store_mut()
            .put(key, Arc::new(Mutex::new(schedule)), bytes);
    }
    Ok(())
}

pub(crate) fn overwrite<W>(name: &'static str, dest: Dest<W>) -> Result<W> {
    if dest.mode() != DestMode::Overwrite {
        return Err(tenferro_tensor::Error::unsupported(
            name,
            "FFT supports only Dest::Overwrite",
        ));
    }
    Ok(dest.into_target())
}

pub(crate) fn one_axis_into(
    input: TensorRead<'_>,
    name: &'static str,
    operation: OperationOf,
    options: &FftOptions,
    plans: &mut FftPlanCache,
    engine: &mut dyn Engine,
    dest: Dest<TensorWrite<'_>>,
) -> Result<()> {
    execute_into(
        input,
        name,
        operation,
        Options::One(options),
        plans,
        engine,
        dest,
    )
}

pub(crate) fn n_axes_into(
    input: TensorRead<'_>,
    name: &'static str,
    operation: OperationOf,
    options: &FftnOptions,
    plans: &mut FftPlanCache,
    engine: &mut dyn Engine,
    dest: Dest<TensorWrite<'_>>,
) -> Result<()> {
    execute_into(
        input,
        name,
        operation,
        Options::Many(options),
        plans,
        engine,
        dest,
    )
}

impl FftExecutor {
    /// Write [`FftOps::fft_into`](crate::FftOps::fft_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2], vec![1.0_f64; 2])?;
    /// let mut output = Tensor::from_vec_col_major([2], vec![Complex64::new(0.0, 0.0); 2])?;
    /// executor.fft_into(&input, &FftOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn fft_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        one_axis_into(
            input.as_read(),
            "FftExecutor::fft_into",
            crate::concrete_fft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }

    /// Write [`FftOps::ifft_into`](crate::FftOps::ifft_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2], vec![Complex64::new(1.0, 0.0); 2])?;
    /// let mut output = Tensor::from_vec_col_major([2], vec![Complex64::new(0.0, 0.0); 2])?;
    /// executor.ifft_into(&input, &FftOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn ifft_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        one_axis_into(
            input.as_read(),
            "FftExecutor::ifft_into",
            crate::concrete_ifft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }

    /// Write [`FftOps::rfft_into`](crate::FftOps::rfft_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2], vec![1.0_f64; 2])?;
    /// let mut output = Tensor::from_vec_col_major([2], vec![Complex64::new(0.0, 0.0); 2])?;
    /// executor.rfft_into(&input, &FftOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rfft_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        one_axis_into(
            input.as_read(),
            "FftExecutor::rfft_into",
            crate::concrete_rfft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }

    /// Write [`FftOps::irfft_into`](crate::FftOps::irfft_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2], vec![Complex64::new(1.0, 0.0); 2])?;
    /// let mut output = Tensor::from_vec_col_major([2], vec![0.0_f64; 2])?;
    /// executor.irfft_into(&input, &FftOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn irfft_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        one_axis_into(
            input.as_read(),
            "FftExecutor::irfft_into",
            crate::concrete_irfft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }

    /// Write [`FftOps::fftn_into`](crate::FftOps::fftn_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftnOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2, 2], vec![1.0_f64; 4])?;
    /// let mut output = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(0.0, 0.0); 4])?;
    /// executor.fftn_into(&input, &FftnOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn fftn_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftnOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        n_axes_into(
            input.as_read(),
            "FftExecutor::fftn_into",
            crate::concrete_fft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }

    /// Write [`FftOps::ifftn_into`](crate::FftOps::ifftn_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftnOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(1.0, 0.0); 4])?;
    /// let mut output = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(0.0, 0.0); 4])?;
    /// executor.ifftn_into(&input, &FftnOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn ifftn_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftnOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        n_axes_into(
            input.as_read(),
            "FftExecutor::ifftn_into",
            crate::concrete_ifft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }

    /// Write [`FftOps::rfftn_into`](crate::FftOps::rfftn_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftnOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2, 2], vec![1.0_f64; 4])?;
    /// let mut output = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(0.0, 0.0); 4])?;
    /// executor.rfftn_into(&input, &FftnOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rfftn_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftnOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        n_axes_into(
            input.as_read(),
            "FftExecutor::rfftn_into",
            crate::concrete_rfft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }

    /// Write [`FftOps::irfftn_into`](crate::FftOps::irfftn_into) using this executor's reusable cache.
    ///
    /// Repeated CPU calls over owned [`Tensor`] inputs and destinations in plain host
    /// storage allocate no Rust heap memory once plans, scratch and intermediates
    /// fit the warmed cache. Typed-input and compact-view adapters also avoid
    /// allocation up to the inline metadata rank (8); higher-rank adapters may
    /// allocate shape/stride metadata.
    /// Clearing or reducing cache limits requires warming it again.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` for modes other than `Dest::Overwrite`, unsupported
    /// engines, dtypes or destination layouts; `Validation` for invalid axes,
    /// lengths, shape, dtype or placement; and backend storage/execution errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftnOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    /// let mut engine = CpuEngine::serial();
    /// let mut executor = FftExecutor::default();
    /// let input = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(1.0, 0.0); 4])?;
    /// let mut output = Tensor::from_vec_col_major([2, 2], vec![0.0_f64; 4])?;
    /// executor.irfftn_into(&input, &FftnOptions::default(), &mut engine, Dest::overwrite(&mut output))?;
    /// assert_eq!(output.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn irfftn_into<X: AsView + ?Sized>(
        &mut self,
        input: &X,
        options: &FftnOptions,
        engine: &mut dyn Engine,
        dest: Dest<TensorWrite<'_>>,
    ) -> Result<()> {
        n_axes_into(
            input.as_read(),
            "FftExecutor::irfftn_into",
            crate::concrete_irfft_operation,
            options,
            &mut self.plans,
            engine,
            dest,
        )
    }
}
