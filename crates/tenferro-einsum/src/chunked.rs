//! The chunk loop shared by the sliced einsum (#2676) and the pair steps of
//! the sampled einsum (#2675).
//!
//! Both run one contraction many times over pieces of the same operands: a
//! chunk source hands out each chunk's operand reads, and one
//! [`ConcreteEinsumPlan`] prepared from the first chunk's dtypes and shapes
//! (a second one at most, for a shorter last chunk) runs every chunk into the
//! destination. Every chunk presents the same layout, so neither that plan
//! nor the engine's binary-plan cache is prepared again per chunk.
//!
//! The two callers differ only in what a chunk is and where it writes:
//!
//! | | operands of a chunk | destination of a chunk |
//! |---|---|---|
//! | sliced (#2676) | metadata-only views, each sliced axis narrowed to one value | the whole output, accumulated (`beta = 1` after the first) |
//! | sampled pairs (#2675) | the configuration rows of the chunk's pairs, gathered | a metadata-only window of the step result on its configuration axis, overwritten |

use std::ops::Range;

use tenferro_tensor::{
    DotGeneralAccumulation, Engine, Placement, StridedSliceSpec, TensorRead, TensorViewMut,
    TensorWrite, TypedTensorViewMut,
};

use crate::ConcreteEinsumPlan;

type Res<T> = tenferro_tensor::Result<T>;

/// What one chunk of a chunked contraction reads and where it writes.
pub(crate) trait ChunkSource {
    /// Per-chunk owned data the reads borrow from: gathered copies for the
    /// sampled pairs, the slice assignment for the sliced einsum.
    type Held;

    /// Build chunk `id`'s owned data.
    fn hold(&self, id: usize, exec: &mut dyn Engine) -> Res<Self::Held>;

    /// The operand reads of a chunk, borrowing from `held` and the source.
    fn reads<'s>(&'s self, held: &'s Self::Held) -> Res<Vec<TensorRead<'s>>>;

    /// The prepared plan for chunk `id`.
    fn plan(&self, id: usize) -> &ConcreteEinsumPlan;

    /// The range on `axis` of the destination that chunk `id` writes, or
    /// `None` when it writes the whole destination.
    fn window(&self, id: usize) -> Option<(usize, Range<usize>)>;

    /// Elements of a chunk's owned data, counted as live temporaries.
    fn held_elements(_held: &Self::Held) -> usize {
        0
    }
}

/// Run chunks `ids` of `source` into `out`: the first with `first`, every
/// later one with `rest`. Returns the most elements one chunk held.
pub(crate) fn run_chunks<S: ChunkSource>(
    source: &S,
    ids: Range<usize>,
    exec: &mut dyn Engine,
    out: &mut TensorWrite<'_>,
    first: DotGeneralAccumulation,
    rest: DotGeneralAccumulation,
) -> crate::Result<usize> {
    let mut held_peak = 0;
    for (position, id) in ids.enumerate() {
        let held = source.hold(id, exec)?;
        held_peak = held_peak.max(S::held_elements(&held));
        let reads = source.reads(&held)?;
        let accumulation = if position == 0 { first } else { rest };
        let target = match source.window(id) {
            None => reborrow(out)?,
            Some((axis, range)) => window(out, axis, range)?,
        };
        source
            .plan(id)
            .run_reads_into_accum(&reads[..], exec, accumulation, target)?;
    }
    Ok(held_peak)
}

/// A shorter-lived write target to the same destination.
fn reborrow<'b>(out: &'b mut TensorWrite<'_>) -> Res<TensorWrite<'b>> {
    Ok(match out {
        TensorWrite::Tensor(tensor) => TensorWrite::Tensor(tensor),
        TensorWrite::View(view) => {
            let all = vec![StridedSliceSpec::all(); view.shape().len()];
            TensorWrite::View(slice_view_mut(view, &all)?)
        }
    })
}

/// A metadata-only write target over `range` of `axis` of `out`, which must
/// be a view (the sampled executor passes its step result as one).
///
/// A host window with the default placement is rebased onto its storage from
/// the window's first element (offset 0), as the sliced loop rebases its
/// operand views: the CPU binary-plan cache keys on the destination's offset
/// too, so equal-size windows then share one engine plan. Other windows keep
/// their offset.
fn window<'b>(
    out: &'b mut TensorWrite<'_>,
    axis: usize,
    range: Range<usize>,
) -> Res<TensorWrite<'b>> {
    let TensorWrite::View(view) = out else {
        return Err(tenferro_tensor::Error::runtime_state(
            "chunked einsum",
            "a windowed destination must be passed as a view",
        ));
    };
    let specs: Vec<StridedSliceSpec> = (0..view.shape().len())
        .map(|a| {
            if a == axis {
                StridedSliceSpec::new(range.start as isize, Some(range.end as isize), 1)
            } else {
                StridedSliceSpec::all()
            }
        })
        .collect();
    Ok(TensorWrite::View(match view {
        TensorViewMut::F32(v) => TensorViewMut::F32(typed_window(v, axis, &range, &specs)?),
        TensorViewMut::F64(v) => TensorViewMut::F64(typed_window(v, axis, &range, &specs)?),
        TensorViewMut::I32(v) => TensorViewMut::I32(typed_window(v, axis, &range, &specs)?),
        TensorViewMut::I64(v) => TensorViewMut::I64(typed_window(v, axis, &range, &specs)?),
        TensorViewMut::Bool(v) => TensorViewMut::Bool(typed_window(v, axis, &range, &specs)?),
        TensorViewMut::C32(v) => TensorViewMut::C32(typed_window(v, axis, &range, &specs)?),
        TensorViewMut::C64(v) => TensorViewMut::C64(typed_window(v, axis, &range, &specs)?),
    }))
}

/// `range` of `axis` of `view`, rebased to offset 0 when it is a host view
/// with the default placement and non-negative strides.
fn typed_window<'b, T: 'static>(
    view: &'b mut TypedTensorViewMut<'_, T>,
    axis: usize,
    range: &Range<usize>,
    specs: &[StridedSliceSpec],
) -> Res<TypedTensorViewMut<'b, T>> {
    let mut shape = view.shape().to_vec();
    let strides = view.strides().to_vec();
    shape[axis] = range.len();
    let start = view.offset() + range.start as isize * strides[axis];
    let rebase = view.placement() == &Placement::default()
        && strides.iter().all(|&stride| stride >= 0)
        && start >= 0
        && view.host_storage_mut().is_ok();
    if rebase {
        let data = view.host_storage_mut()?;
        let rest = data.get_mut(start as usize..).ok_or_else(|| {
            tenferro_tensor::Error::runtime_state(
                "chunked einsum",
                "a destination window starts past its storage",
            )
        })?;
        return TypedTensorViewMut::from_slice(&shape, &strides, 0, rest);
    }
    view.slice_view(specs)
}

fn slice_view_mut<'b>(
    view: &'b mut TensorViewMut<'_>,
    specs: &[StridedSliceSpec],
) -> Res<TensorViewMut<'b>> {
    Ok(match view {
        TensorViewMut::F32(v) => TensorViewMut::F32(v.slice_view(specs)?),
        TensorViewMut::F64(v) => TensorViewMut::F64(v.slice_view(specs)?),
        TensorViewMut::I32(v) => TensorViewMut::I32(v.slice_view(specs)?),
        TensorViewMut::I64(v) => TensorViewMut::I64(v.slice_view(specs)?),
        TensorViewMut::Bool(v) => TensorViewMut::Bool(v.slice_view(specs)?),
        TensorViewMut::C32(v) => TensorViewMut::C32(v.slice_view(specs)?),
        TensorViewMut::C64(v) => TensorViewMut::C64(v.slice_view(specs)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::{Complex32, Complex64, Tensor, TensorScalar, TypedTensor};

    /// A window `1..3` of axis 1 of a `[2, 4]` tensor, and its reborrow.
    fn windowed<T: TensorScalar + Default + Clone>() {
        let mut t =
            TypedTensor::<T>::from_vec_col_major(vec![2, 4], vec![T::default(); 8]).unwrap();
        let mut out = TensorWrite::from_view(T::tensor_view_mut(t.as_view_mut()));
        {
            let mut w = window(&mut out, 1, 1..3).unwrap();
            let TensorWrite::View(view) = &mut w else {
                panic!("a window is a view");
            };
            assert_eq!(view.shape(), &[2, 2]);
            // Rebased: the window starts at offset 0 of its own storage.
            let mut again = reborrow(&mut w).unwrap();
            let TensorWrite::View(v) = &mut again else {
                panic!("a reborrowed view stays a view");
            };
            assert_eq!(v.shape(), &[2, 2]);
        }
        drop(out);
        assert_eq!(t.host_data().unwrap().len(), 8);
    }

    #[test]
    fn windows_of_every_dtype_are_rebased_views() {
        windowed::<f32>();
        windowed::<f64>();
        windowed::<i32>();
        windowed::<i64>();
        windowed::<bool>();
        windowed::<Complex32>();
        windowed::<Complex64>();
    }

    #[test]
    fn a_reversed_view_is_windowed_without_rebasing() {
        let mut t =
            TypedTensor::<f64>::from_vec_col_major(vec![4], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let mut full = t.as_view_mut();
        let reversed = full.slice_view(&[StridedSliceSpec::reverse()]).unwrap();
        let mut out = TensorWrite::from_view(f64::tensor_view_mut(reversed));
        let w = window(&mut out, 0, 1..3).unwrap();
        let TensorWrite::View(TensorViewMut::F64(v)) = w else {
            panic!("an f64 view");
        };
        assert_eq!(v.shape(), &[2]);
        assert!(v.strides()[0] < 0, "the reversed window keeps its stride");
    }

    #[test]
    fn an_owned_destination_cannot_be_windowed() {
        let mut t = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
        let mut out = TensorWrite::from_tensor(&mut t);
        assert!(window(&mut out, 0, 0..1).is_err());
        assert!(matches!(
            reborrow(&mut out).unwrap(),
            TensorWrite::Tensor(_)
        ));
    }
}
