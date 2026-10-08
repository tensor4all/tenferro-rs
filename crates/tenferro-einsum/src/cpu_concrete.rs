//! Concrete CPU numerical execution. Notation and order search stay in tenferro.

use crate::ContractionTree;
use std::{
    any::Any,
    sync::{Arc, Mutex},
};

#[cfg(feature = "native")]
type NumericalPlan<T> = EinsumPlan<T>;
#[cfg(feature = "blas")]
type NumericalPlan<T> = EinsumPlan<T, cpueinsum_blas::Blas>;

#[derive(Default)]
pub(crate) struct Cache(Mutex<CacheState>);
#[derive(Default)]
struct CacheState {
    entry: Option<Entry>,
    events: tenferro_tensor::CacheStats,
}
impl Cache {
    pub(crate) fn clear(&mut self) {
        let state = self
            .0
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.entry = None;
        state.events.clears = state.events.clears.saturating_add(1);
        self.0.clear_poison();
    }
    pub(crate) fn stats(&self) -> tenferro_tensor::Result<tenferro_tensor::CacheStats> {
        let state = self.0.lock().map_err(|_| {
            tenferro_tensor::Error::runtime_state("CPU einsum cache", "poisoned metadata lock")
        })?;
        let mut stats = state.events;
        if let Some(entry) = &state.entry {
            stats.entries = 1;
            // Known host descriptors and lower plan header; lower-private
            // metadata heap accounting remains a lower-library responsibility.
            stats.retained_bytes = std::mem::size_of::<Entry>()
                .saturating_add(std::mem::size_of_val(entry.plan.as_ref()))
                .saturating_add(
                    entry.layouts.capacity() * std::mem::size_of::<(Vec<usize>, Vec<isize>)>(),
                );
            for (dims, strides) in &entry.layouts {
                stats.retained_bytes = stats
                    .retained_bytes
                    .saturating_add(dims.capacity() * std::mem::size_of::<usize>())
                    .saturating_add(strides.capacity() * std::mem::size_of::<isize>());
            }
        }
        Ok(stats)
    }
}
struct Entry {
    layouts: Vec<(Vec<usize>, Vec<isize>)>,
    plan: Arc<dyn Any + Send + Sync>,
}
impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuNumericalPlanCache")
            .finish_non_exhaustive()
    }
}

use cpueinsum::{EinsumPlan, EinsumSpec, Layout};

/// Translate a lower einsum failure, keeping a typed route decline distinct.
///
/// The lower library owns route selection, so a declined or unsupported step is
/// `Unsupported` rather than a tenferro backend failure, and its diagnostic
/// stays in the message.
fn map_lower_einsum_error(op: &'static str, error: cpueinsum::Error) -> tenferro_tensor::Error {
    if let cpueinsum::Error::Contract { source, .. } = &error {
        if source.is_unsupported() {
            return tenferro_tensor::Error::unsupported(
                op,
                format!("lower einsum declined this route: {source}"),
            );
        }
    }
    tenferro_tensor::Error::backend_source(op, error)
}
use tenferro_cpu::with_cpu_exec_session;
use tenferro_tensor::{
    BackendSession, DType, Tensor, TensorRead, TensorScalar, TensorView, TensorViewMut,
    TensorWrite, TypedTensorView,
};

#[cfg(feature = "native")]
trait Scalar: TensorScalar + cpueinsum::Scalar {}
#[cfg(feature = "native")]
impl<T: TensorScalar + cpueinsum::Scalar> Scalar for T {}
#[cfg(feature = "blas")]
trait Scalar: TensorScalar + cpueinsum_blas::BlasScalar {}
#[cfg(feature = "blas")]
impl<T: TensorScalar + cpueinsum_blas::BlasScalar> Scalar for T {}

pub(crate) fn execute(
    session: &mut dyn BackendSession,
    inputs: &[TensorRead<'_>],
    tree: &ContractionTree,
) -> Option<tenferro_tensor::Result<Tensor>> {
    execute_cached(session, inputs, tree, None)
}

pub(crate) fn execute_cached(
    session: &mut dyn BackendSession,
    inputs: &[TensorRead<'_>],
    tree: &ContractionTree,
    cache: Option<&Cache>,
) -> Option<tenferro_tensor::Result<Tensor>> {
    // Binary operations already use the session's prepared binary cache.
    if inputs.len() < 3
        || !matches!(
            inputs.first()?.dtype(),
            DType::F32 | DType::F64 | DType::C32 | DType::C64
        )
    {
        return None;
    }
    for input in inputs {
        tenferro_cpu::validate_cpu_host_read("CPU einsum", "input", input).ok()?;
    }
    with_cpu_exec_session(session, |cpu| {
        let domain = cpu.domain_id();
        cpu.with_contraction_exec(|exec, buffers, workspaces| {
            macro_rules! dispatch {
                ($($variant:ident),*) => { match inputs[0].dtype() {
                    $(DType::$variant => {
                        let views = inputs.iter().map(|input| match input.clone().tensor_view() {
                            TensorView::$variant(view) => Ok(view),
                            _ => Err(tenferro_tensor::Error::dtype_mismatch("CPU einsum", inputs[0].dtype(), input.dtype())),
                        }).collect::<tenferro_tensor::Result<Vec<_>>>()?;
                        run(&Execution { exec, workspaces, max_retained_bytes: buffers.max_retained_capacity_bytes() }, tree, cache, domain, |shape| {
                            let mut output = tenferro_cpu_basic::PooledUninitOutput::new(buffers, shape)?;
                            output.as_uninit_slice_mut().fill(std::mem::MaybeUninit::new(Default::default()));
                            // SAFETY: every physical element of the compact output
                            // has just been initialized, before lower N-ary execution.
                            unsafe { output.assume_init() }
                        }, &views, tree.output_shape())
                    },)*
                    _ => unreachable!("supported scalar dispatch checked above"),
                } };
            }
            dispatch!(F32, F64, C32, C64)
        })
    })
}

struct Execution<'a, 'pool> {
    exec: &'a cpueinsum::Exec<'pool>,
    workspaces: &'a tenferro_cpu::ContractionWorkspaces,
    max_retained_bytes: usize,
}

fn run<T: Scalar>(
    execution: &Execution<'_, '_>,
    tree: &ContractionTree,
    cache: Option<&Cache>,
    domain: tenferro_tensor::CpuDomainId,
    allocate: impl FnOnce(Vec<usize>) -> tenferro_tensor::Result<tenferro_tensor::TypedTensor<T>>,
    inputs: &[TypedTensorView<'_, T>],
    shape: Vec<usize>,
) -> tenferro_tensor::Result<Tensor> {
    let strides = tenferro_tensor::col_major_strides(&shape)?;
    let mut output = allocate(shape.clone())?;
    let mut output_view = output.as_view_mut();
    let mut out = cpueinsum::strided_view::StridedViewMut::new(
        output_view.host_storage_mut()?,
        &shape,
        &strides,
        0,
    )
    .map_err(|error| tenferro_tensor::Error::backend_source("CPU einsum view", error))?;
    run_into(execution, tree, cache, inputs, &mut out)?;
    output.set_cpu_affinity(Some(domain));
    Ok(Tensor::from_typed(output))
}

fn run_into<T: Scalar>(
    execution: &Execution<'_, '_>,
    tree: &ContractionTree,
    cache: Option<&Cache>,
    inputs: &[TypedTensorView<'_, T>],
    out: &mut cpueinsum::strided_view::StridedViewMut<'_, T>,
) -> tenferro_tensor::Result<()> {
    let shape = out.dims();
    let strides = out.strides();
    let views = inputs
        .iter()
        .map(|view| {
            cpueinsum::strided_view::StridedView::new(
                view.host_storage()?,
                view.shape(),
                view.strides(),
                view.offset(),
            )
            .map_err(|error| tenferro_tensor::Error::backend_source("CPU einsum view", error))
        })
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    let prepare = || -> tenferro_tensor::Result<NumericalPlan<T>> {
        let labels: Vec<Vec<i64>> = tree
            .subscripts
            .inputs
            .iter()
            .map(|axes| axes.iter().map(|&axis| i64::from(axis)).collect())
            .collect();
        let label_refs: Vec<_> = labels.iter().map(Vec::as_slice).collect();
        let output: Vec<_> = tree
            .subscripts
            .output
            .iter()
            .map(|&axis| i64::from(axis))
            .collect();
        let order: Vec<_> = tree
            .steps
            .iter()
            .map(|step| [step.left, step.right])
            .collect();
        let spec = EinsumSpec::new(&label_refs, &output, &order)
            .map_err(|error| map_lower_einsum_error("CPU einsum", error))?;
        let layouts = views
            .iter()
            .map(|v| Layout::new(v.dims(), v.strides()))
            .collect::<cpueinsum::Result<Vec<_>>>()
            .map_err(|error| map_lower_einsum_error("CPU einsum", error))?;
        let output_layout = Layout::new(shape, strides)
            .map_err(|error| map_lower_einsum_error("CPU einsum", error))?;
        #[cfg(feature = "native")]
        let plan = EinsumPlan::<T>::new(&spec, &layouts, output_layout);
        #[cfg(feature = "blas")]
        let plan = EinsumPlan::<T, _>::with_backend(
            cpueinsum_blas::Blas::default(),
            &spec,
            &layouts,
            output_layout,
        );
        plan.map_err(|error| map_lower_einsum_error("CPU einsum", error))
    };
    let plan = if let Some(cache) = cache {
        // Release the metadata lock before any numerical execution or user callback.
        let mut state = cache.0.lock().map_err(|_| {
            tenferro_tensor::Error::runtime_state("CPU einsum cache", "poisoned metadata lock")
        })?;
        let matched = state.entry.as_ref().is_some_and(|e| {
            e.plan.is::<NumericalPlan<T>>()
                && e.layouts.len() == views.len() + 1
                && e.layouts.iter().zip(&views).all(|((dims, strides), v)| {
                    dims.as_slice() == v.dims() && strides.as_slice() == v.strides()
                })
                && e.layouts
                    .last()
                    .is_some_and(|(dims, ss)| dims.as_slice() == shape && ss.as_slice() == strides)
        });
        if matched {
            state.events.hits = state.events.hits.saturating_add(1);
        } else {
            state.events.misses = state.events.misses.saturating_add(1);
            let plan = Arc::new(prepare()?);
            if state.entry.is_some() {
                state.events.evictions = state.events.evictions.saturating_add(1);
            }
            state.entry = Some(Entry {
                layouts: views
                    .iter()
                    .map(|v| (v.dims().to_vec(), v.strides().to_vec()))
                    .chain(std::iter::once((shape.to_vec(), strides.to_vec())))
                    .collect(),
                plan,
            });
        }
        Arc::clone(&state.entry.as_ref().expect("prepared cache entry").plan)
            .downcast::<NumericalPlan<T>>()
            .map_err(|_| {
                tenferro_tensor::Error::runtime_state(
                    "CPU einsum cache",
                    "incompatible numerical plan",
                )
            })?
    } else {
        Arc::new(prepare()?)
    };
    // Destination and scratch are initialized, including caller-owned gaps.
    execution.workspaces.with_scratch::<T, _>(
        plan.scratch_len(),
        execution.max_retained_bytes,
        |scratch| {
            plan.execute_into(execution.exec, &views, out, scratch)
                .map_err(|error| map_lower_einsum_error("CPU einsum", error))
        },
    )
}

pub(crate) fn supports(session: &mut dyn BackendSession, inputs: &[TensorRead<'_>]) -> bool {
    inputs.len() >= 3
        && inputs.first().is_some_and(|input| {
            matches!(
                input.dtype(),
                DType::F32 | DType::F64 | DType::C32 | DType::C64
            )
        })
        && with_cpu_exec_session(session, |_| ()).is_some()
}

pub(crate) fn execute_into(
    session: &mut dyn BackendSession,
    inputs: &[TensorRead<'_>],
    tree: &ContractionTree,
    cache: Option<&Cache>,
    out: TensorWrite<'_>,
) -> tenferro_tensor::Result<()> {
    let dtype = out.dtype();
    tenferro_cpu::validate_cpu_host_write("CPU einsum", "output", &out)?;
    for input in inputs {
        tenferro_cpu::validate_cpu_host_read("CPU einsum", "input", input)?;
        if input.dtype() != dtype {
            return Err(tenferro_tensor::Error::dtype_mismatch(
                "CPU einsum",
                dtype,
                input.dtype(),
            ));
        }
    }
    if out.shape() != tree.output_shape() {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "CPU einsum",
            tree.output_shape(),
            out.shape().to_vec(),
        ));
    }
    with_cpu_exec_session(session, |cpu| cpu.with_contraction_exec(|exec, buffers, workspaces| {
        macro_rules! dispatch {
            ($($variant:ident => $ty:ty),*) => {
                match dtype {
                    $(DType::$variant => {
                        let output = match out {
                            TensorWrite::View(TensorViewMut::$variant(view)) => view,
                            TensorWrite::Tensor(tensor) => tensor.as_typed_mut::<$ty>()
                                .ok_or_else(|| tenferro_tensor::Error::runtime_state("CPU einsum", "incompatible typed output"))?.as_view_mut(),
                            _ => return Err(tenferro_tensor::Error::runtime_state("CPU einsum", "incompatible output view")),
                        };
                        let dims = output.shape().to_vec();
                        let strides = output.strides().to_vec();
                        let offset = output.offset();
                        let mut output = output;
                        let mut lower_out = cpueinsum::strided_view::StridedViewMut::new(output.host_storage_mut()?, &dims, &strides, offset)
                            .map_err(|error| tenferro_tensor::Error::backend_source("CPU einsum view", error))?;
                        let views = inputs.iter().map(|input| match input.clone().tensor_view() {
                            TensorView::$variant(view) => Ok(view),
                            _ => Err(tenferro_tensor::Error::runtime_state("CPU einsum", "incompatible input view")),
                        }).collect::<tenferro_tensor::Result<Vec<_>>>()?;
                        run_into(&Execution { exec, workspaces, max_retained_bytes: buffers.max_retained_capacity_bytes() }, tree, cache, &views, &mut lower_out)
                    },)*
                    _ => Err(tenferro_tensor::Error::unsupported_dtype("CPU einsum", dtype, "unsupported scalar")),
                }
            }
        }
        dispatch!(F32 => f32, F64 => f64, C32 => tenferro_tensor::Complex<f32>, C64 => tenferro_tensor::Complex<f64>)
    })).ok_or_else(|| tenferro_tensor::Error::runtime_state("CPU einsum", "CPU session required"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::BackendSessionHost;
    #[test]
    fn nary_delegation_covers_four_scalar_types() {
        macro_rules! check {
            ($ty:ty, $values:expr, $one:expr, $zero:expr) => {{
                let values: Vec<$ty> = $values;
                let a = Tensor::from_typed(
                    tenferro_tensor::TypedTensor::from_vec_col_major(vec![2, 2], values.clone())
                        .unwrap(),
                );
                let id = Tensor::from_typed(
                    tenferro_tensor::TypedTensor::<$ty>::from_vec_col_major(
                        vec![2, 2],
                        vec![$one, $zero, $zero, $one],
                    )
                    .unwrap(),
                );
                let subs = crate::Subscripts::parse("ij,jk,kl->il").unwrap();
                let tree =
                    ContractionTree::optimize(&subs, &[a.shape(), id.shape(), id.shape()]).unwrap();
                let mut backend = tenferro_cpu::CpuBackend::with_threads(1).unwrap();
                let mut cache = Cache::default();
                let out = backend
                    .with_backend_session(|session| {
                        execute_cached(
                            session,
                            &[
                                TensorRead::from_tensor(&a),
                                TensorRead::from_tensor(&id),
                                TensorRead::from_tensor(&id),
                            ],
                            &tree,
                            Some(&cache),
                        )
                        .expect("concrete CPU route must be selected")
                    })
                    .unwrap()
                    .unwrap();
                assert_eq!(out.as_slice::<$ty>().unwrap(), values);
                assert!(out
                    .as_typed::<$ty>()
                    .unwrap()
                    .placement()
                    .cpu_affinity
                    .is_some());
                let first = Arc::clone(&cache.0.lock().unwrap().entry.as_ref().unwrap().plan);
                backend
                    .with_backend_session(|session| {
                        execute_cached(
                            session,
                            &[
                                TensorRead::from_tensor(&a),
                                TensorRead::from_tensor(&id),
                                TensorRead::from_tensor(&id),
                            ],
                            &tree,
                            Some(&cache),
                        )
                        .unwrap()
                    })
                    .unwrap()
                    .unwrap();
                let second = Arc::clone(&cache.0.lock().unwrap().entry.as_ref().unwrap().plan);
                assert!(
                    Arc::ptr_eq(&first, &second),
                    "warm execution must reuse the numerical plan"
                );
                let stats = cache.stats().unwrap();
                assert_eq!((stats.entries, stats.hits, stats.misses), (1, 1, 1));
                assert!(stats.retained_bytes > 0);
                cache.clear();
                let cleared = cache.stats().unwrap();
                assert_eq!(
                    (cleared.entries, cleared.retained_bytes, cleared.clears),
                    (0, 0, 1)
                );
            }};
        }
        check!(f32, vec![1., 2., 3., 4.], 1., 0.);
        check!(f64, vec![1., 2., 3., 4.], 1., 0.);
        use num_complex::{Complex32, Complex64};
        check!(
            Complex32,
            vec![
                Complex32::new(1., 2.),
                Complex32::new(3., 4.),
                Complex32::new(5., 6.),
                Complex32::new(7., 8.)
            ],
            Complex32::new(1., 0.),
            Complex32::new(0., 0.)
        );
        check!(
            Complex64,
            vec![
                Complex64::new(1., 2.),
                Complex64::new(3., 4.),
                Complex64::new(5., 6.),
                Complex64::new(7., 8.)
            ],
            Complex64::new(1., 0.),
            Complex64::new(0., 0.)
        );
    }
}
