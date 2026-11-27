//! Running a sampled einsum program on the concrete layer (any engine) or the
//! eager layer (recorded for reverse mode).
//!
//! Provenance: adapted from `src/exec.rs` of Fanerst/sampled-einsum at commit
//! c482b79a0f1e407390ef7822a89e7706220021d1 (Feng Pan, Apache-2.0): the slot
//! table with live-element accounting, the gather of one axis through a
//! `[prefix, extent, suffix]` view (tenferro's CUDA gather has fast kernels up
//! to rank 8 only; the prototype measured the circuit network's 30-axis gathers
//! 30 times slower than their memory traffic otherwise), and the upload of
//! host indices for a device engine. The gathered-pairs step is new: on the
//! concrete layer it runs through the chunk loop shared with the sliced einsum
//! (`crate::chunked`), writing each chunk into a window of the preallocated
//! step result instead of concatenating pieces.

use std::cell::Cell;

use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, Engine, GatherConfig, MemoryKind, Tensor,
    TensorRead, TensorView, TensorViewMut, TensorWrite,
};

use super::plan::{Instr, PairStep, Program};
use crate::chunked::{run_chunks, ChunkSource};
use crate::{ConcreteEinsumPlan, Error, Result};

/// What one run measured.
#[derive(Clone, Debug, Default)]
pub(crate) struct ExecStats {
    pub(crate) peak_elements: usize,
    pub(crate) chunks: usize,
    /// Rows passed through `index_select`, counted at the call (leaf
    /// restrictions, one-sided steps and gathered pairs alike).
    pub(crate) gathered_rows: usize,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// `shape` with the axes before and after `axis` merged into one each.
fn around(shape: &[usize], axis: usize) -> [usize; 3] {
    [
        shape[..axis].iter().product(),
        shape[axis],
        shape[axis + 1..].iter().product(),
    ]
}

/// `read` seen as `[prefix, extent, suffix]` around `axis`: a relabeling of a
/// compact column-major tensor, not a copy.
fn merged_read<'a>(read: &TensorRead<'a>, axis: usize) -> Result<TensorRead<'a>> {
    let shape = around(read.shape(), axis);
    let dtype = read.dtype();
    let view = read.clone().tensor_view().ok_or_else(|| {
        Error::Tensor(tenferro_tensor::Error::unsupported_dtype(
            super::OP,
            dtype,
            "an operand without a borrowed view cannot be gathered",
        ))
    })?;
    let view = match view {
        TensorView::F32(v) => TensorView::F32(v.reshape_view(&shape)?),
        TensorView::F64(v) => TensorView::F64(v.reshape_view(&shape)?),
        TensorView::I32(v) => TensorView::I32(v.reshape_view(&shape)?),
        TensorView::I64(v) => TensorView::I64(v.reshape_view(&shape)?),
        TensorView::Bool(v) => TensorView::Bool(v.reshape_view(&shape)?),
        TensorView::C32(v) => TensorView::C32(v.reshape_view(&shape)?),
        TensorView::C64(v) => TensorView::C64(v.reshape_view(&shape)?),
    };
    Ok(TensorRead::from_view(view))
}

/// `index_select` of `positions` along `axis` (positions may repeat).
fn index_select(
    exec: &mut dyn Engine,
    read: &TensorRead<'_>,
    axis: usize,
    positions: &[usize],
    gathered: &Cell<usize>,
) -> Result<Tensor> {
    gathered.set(gathered.get() + positions.len());
    let mut out_shape = read.shape().to_vec();
    out_shape[axis] = positions.len();
    let [prefix, _, suffix] = around(read.shape(), axis);
    let config = GatherConfig {
        offset_dims: vec![0, 2],
        collapsed_slice_dims: vec![1],
        start_index_map: vec![1],
        index_vector_dim: 1,
        slice_sizes: vec![prefix, 1, suffix],
    };
    let index: Vec<i64> = positions
        .iter()
        .map(|&p| {
            i64::try_from(p).map_err(|_| {
                Error::invalid_argument(super::OP, "samples", format!("position {p} exceeds i64"))
            })
        })
        .collect::<Result<_>>()?;
    let host = Tensor::from_vec_col_major(vec![positions.len(), 1], index)?;
    // The merged view is a relabeling only of a compact layout; a strided
    // operand (a transposed or reversed caller view) is gathered at full rank.
    let (operand, config) = match merged_read(read, axis) {
        Ok(merged) => (merged, config),
        Err(_) => {
            let mut slice_sizes = read.shape().to_vec();
            slice_sizes[axis] = 1;
            let full = GatherConfig {
                offset_dims: (0..read.shape().len()).filter(|&d| d != axis).collect(),
                collapsed_slice_dims: vec![axis],
                start_index_map: vec![axis],
                index_vector_dim: 1,
                slice_sizes,
            };
            (read.clone(), full)
        }
    };
    // A device engine refuses host indices rather than copy them implicitly.
    let uploaded = match operand.placement().memory_kind {
        MemoryKind::Device => Some(exec.upload_host_tensor(TensorRead::from_tensor(&host))?),
        _ => None,
    };
    let indices = TensorRead::from_tensor(uploaded.as_ref().unwrap_or(&host));
    let out = exec.gather_read(operand, indices, &config)?;
    Ok(exec.reshape_read(TensorRead::from_tensor(&out), &out_shape)?)
}

/// A gathered-pairs step as chunks of the shared chunk loop: chunk `id`
/// gathers its pairs' rows of both children and writes its window of the
/// step result.
struct PairChunks<'p, 'a> {
    step: &'p PairStep,
    plans: &'p [ConcreteEinsumPlan],
    left: TensorRead<'a>,
    right: TensorRead<'a>,
    gathered: &'p Cell<usize>,
}

impl ChunkSource for PairChunks<'_, '_> {
    type Held = (Tensor, Tensor);

    fn hold(&self, id: usize, exec: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        let range = self.step.range(id);
        let err = |e: Error| e.into_tensor_error(super::OP);
        let a = index_select(
            exec,
            &self.left,
            self.step.axis_left,
            &self.step.left_rows[range.clone()],
            self.gathered,
        )
        .map_err(err)?;
        let b = index_select(
            exec,
            &self.right,
            self.step.axis_right,
            &self.step.right_rows[range],
            self.gathered,
        )
        .map_err(err)?;
        Ok((a, b))
    }

    fn reads<'s>(
        &'s self,
        held: &'s (Tensor, Tensor),
    ) -> tenferro_tensor::Result<Vec<TensorRead<'s>>> {
        Ok(vec![
            TensorRead::from_tensor(&held.0),
            TensorRead::from_tensor(&held.1),
        ])
    }

    fn plan(&self, id: usize) -> &ConcreteEinsumPlan {
        &self.plans[self.step.contraction(id)]
    }

    fn window(&self, id: usize) -> Option<(usize, std::ops::Range<usize>)> {
        Some((self.step.out_axis, self.step.range(id)))
    }

    fn held_elements(held: &(Tensor, Tensor)) -> usize {
        numel(held.0.shape()) + numel(held.1.shape())
    }
}

fn slot<'a>(
    operands: &[TensorRead<'a>],
    temps: &'a [Option<Tensor>],
    s: usize,
) -> Result<TensorRead<'a>> {
    if s < operands.len() {
        Ok(operands[s].clone())
    } else {
        temps[s]
            .as_ref()
            .map(TensorRead::from_tensor)
            .ok_or_else(|| Error::planning(format!("slot {s} read before it was written")))
    }
}

/// Erased mutable view of an owned tensor.
fn view_mut(tensor: &mut Tensor) -> Result<TensorViewMut<'_>> {
    use tenferro_tensor::{Complex32, Complex64, DType};
    let dtype = tensor.dtype();
    let missing = || {
        Error::Tensor(tenferro_tensor::Error::unsupported_dtype(
            super::OP,
            dtype,
            "a step result without a typed payload",
        ))
    };
    Ok(match dtype {
        DType::F32 => TensorViewMut::F32(
            tensor
                .as_typed_mut::<f32>()
                .ok_or_else(missing)?
                .as_view_mut(),
        ),
        DType::F64 => TensorViewMut::F64(
            tensor
                .as_typed_mut::<f64>()
                .ok_or_else(missing)?
                .as_view_mut(),
        ),
        DType::I32 => TensorViewMut::I32(
            tensor
                .as_typed_mut::<i32>()
                .ok_or_else(missing)?
                .as_view_mut(),
        ),
        DType::I64 => TensorViewMut::I64(
            tensor
                .as_typed_mut::<i64>()
                .ok_or_else(missing)?
                .as_view_mut(),
        ),
        DType::Bool => TensorViewMut::Bool(
            tensor
                .as_typed_mut::<bool>()
                .ok_or_else(missing)?
                .as_view_mut(),
        ),
        DType::C32 => TensorViewMut::C32(
            tensor
                .as_typed_mut::<Complex32>()
                .ok_or_else(missing)?
                .as_view_mut(),
        ),
        DType::C64 => TensorViewMut::C64(
            tensor
                .as_typed_mut::<Complex64>()
                .ok_or_else(missing)?
                .as_view_mut(),
        ),
        DType::External(_) => return Err(missing()),
    })
}

/// Run `program` on concrete operands.
pub(crate) fn execute_concrete(
    program: &Program,
    plans: &[ConcreteEinsumPlan],
    operands: &[TensorRead<'_>],
    exec: &mut dyn Engine,
) -> Result<(Tensor, ExecStats)> {
    let mut temps: Vec<Option<Tensor>> = (0..program.n_slots).map(|_| None).collect();
    let mut live = 0usize;
    let mut stats = ExecStats::default();
    let gathered = Cell::new(0usize);
    for instr in &program.instrs {
        let (out, value) = match instr {
            Instr::Free { slot } => {
                if let Some(t) = temps[*slot].take() {
                    live -= numel(t.shape());
                }
                continue;
            }
            Instr::Einsum {
                inputs,
                contraction,
                out,
            } => {
                let reads = inputs
                    .iter()
                    .map(|&s| slot(operands, &temps, s))
                    .collect::<Result<Vec<_>>>()?;
                (*out, plans[*contraction].run_reads(&reads[..], exec)?)
            }
            Instr::IndexSelect {
                input,
                axis,
                positions,
                out,
            } => {
                let read = slot(operands, &temps, *input)?;
                (
                    *out,
                    index_select(exec, &read, *axis, positions, &gathered)?,
                )
            }
            Instr::Reshape { input, shape, out } => {
                let read = slot(operands, &temps, *input)?;
                (*out, exec.reshape_read(read, shape)?)
            }
            Instr::Pairs(step) => {
                let dtype = slot(operands, &temps, step.left)?.dtype();
                let mut result =
                    exec.fill_scalar(&step.out_shape, ContractionScalar::zero(dtype)?)?;
                live += numel(result.shape());
                stats.peak_elements = stats.peak_elements.max(live);
                let held = {
                    let source = PairChunks {
                        step,
                        plans,
                        left: slot(operands, &temps, step.left)?,
                        right: slot(operands, &temps, step.right)?,
                        gathered: &gathered,
                    };
                    let overwrite = DotGeneralAccumulation::overwrite(dtype)?;
                    let mut out = TensorWrite::from_view(view_mut(&mut result)?);
                    run_chunks(
                        &source,
                        0..step.chunks(),
                        exec,
                        &mut out,
                        overwrite,
                        overwrite,
                    )?
                };
                stats.peak_elements = stats.peak_elements.max(live + held);
                stats.chunks += step.chunks();
                temps[step.out] = Some(result);
                continue;
            }
        };
        live += numel(value.shape());
        stats.peak_elements = stats.peak_elements.max(live);
        temps[out] = Some(value);
    }
    stats.gathered_rows = gathered.get();
    let result = temps[program.result]
        .take()
        .ok_or_else(|| Error::planning("the plan produced no result"))?;
    if result.shape() != program.output_shape.as_slice() {
        return Err(Error::shape_mismatch(
            super::OP,
            program.output_shape.clone(),
            result.shape().to_vec(),
        ));
    }
    Ok((result, stats))
}

/// Run `program` on eager tensors: every operation is recorded on the tape.
#[cfg(feature = "autodiff")]
pub(crate) fn execute_eager(
    program: &Program,
    session: &mut tenferro_ad::EagerSession<'_>,
    operands: &[&tenferro_ad::EagerTensor],
) -> tenferro_ad::Result<(tenferro_ad::EagerTensor, ExecStats)> {
    use crate::EagerEinsumOps;
    use tenferro_ad::EagerTensor;

    let internal = |message: &str| {
        crate::api::traced::runtime_err(Error::planning(message.to_string()), super::OP)
    };
    let mut temps: Vec<Option<EagerTensor>> = (0..program.n_slots).map(|_| None).collect();
    let mut live = 0usize;
    let mut stats = ExecStats::default();
    macro_rules! get {
        ($s:expr) => {{
            let s = $s;
            if s < operands.len() {
                operands[s]
            } else {
                temps[s]
                    .as_ref()
                    .ok_or_else(|| internal("a slot was read before it was written"))?
            }
        }};
    }
    for instr in &program.instrs {
        let (out, value) = match instr {
            Instr::Free { slot } => {
                if let Some(t) = temps[*slot].take() {
                    live -= numel(t.shape());
                }
                continue;
            }
            Instr::Einsum {
                inputs,
                contraction,
                out,
            } => {
                let refs = inputs
                    .iter()
                    .map(|&s| Ok(get!(s)))
                    .collect::<tenferro_ad::Result<Vec<_>>>()?;
                let subscripts = &program.contractions[*contraction].subscripts;
                let spec = crate::EinsumSubscripts::from(subscripts);
                (*out, session.einsum(&refs, &spec)?)
            }
            Instr::IndexSelect {
                input,
                axis,
                positions,
                out,
            } => {
                let t = get!(*input);
                stats.gathered_rows += positions.len();
                (*out, session.index_select(t, *axis as isize, positions)?)
            }
            Instr::Reshape { input, shape, out } => {
                let t = get!(*input);
                (*out, session.reshape(t, shape.clone())?)
            }
            Instr::Pairs(step) => {
                let spec_full =
                    crate::EinsumSubscripts::from(&program.contractions[step.full].subscripts);
                let spec_tail = step
                    .tail
                    .map(|t| crate::EinsumSubscripts::from(&program.contractions[t].subscripts));
                let mut pieces = Vec::with_capacity(step.chunks());
                for id in 0..step.chunks() {
                    let range = step.range(id);
                    stats.gathered_rows += 2 * range.len();
                    let a = session.index_select(
                        get!(step.left),
                        step.axis_left as isize,
                        &step.left_rows[range.clone()],
                    )?;
                    let b = session.index_select(
                        get!(step.right),
                        step.axis_right as isize,
                        &step.right_rows[range],
                    )?;
                    let held = numel(a.shape()) + numel(b.shape());
                    let spec = match (&spec_tail, step.contraction(id) == step.full) {
                        (Some(tail), false) => tail,
                        _ => &spec_full,
                    };
                    let piece = session.einsum(&[&a, &b], spec)?;
                    live += numel(piece.shape());
                    stats.peak_elements = stats.peak_elements.max(live + held);
                    pieces.push(piece);
                }
                stats.chunks += pieces.len();
                if pieces.len() == 1 {
                    let piece = pieces.pop().expect("one piece");
                    temps[step.out] = Some(piece);
                } else {
                    let refs: Vec<&EagerTensor> = pieces.iter().collect();
                    let joined = session.concatenate(&refs, step.out_axis as isize)?;
                    stats.peak_elements = stats.peak_elements.max(live + numel(joined.shape()));
                    // The pieces are dropped; `joined` takes their count.
                    temps[step.out] = Some(joined);
                }
                continue;
            }
        };
        live += numel(value.shape());
        stats.peak_elements = stats.peak_elements.max(live);
        temps[out] = Some(value);
    }
    let result = temps[program.result]
        .take()
        .ok_or_else(|| internal("the plan produced no result"))?;
    if result.shape() != program.output_shape.as_slice() {
        return Err(internal("the result has another shape than planned"));
    }
    Ok((result, stats))
}
