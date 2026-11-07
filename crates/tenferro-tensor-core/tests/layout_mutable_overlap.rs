//! One injectivity rule for mutable layouts.
//!
//! `TensorLayout::validate_mutable_no_overlap` must answer exactly what
//! `strided_view::is_injective_layout` answers: the analysis that guards
//! strided-rs' own mutable kernels is the analysis tenferro uses, so a layout
//! the kernels accept for a write is never refused by tenferro metadata and a
//! layout they refuse is never accepted here.
//!
//! INVARIANT: the two entry points are compared, not two copies of a proof —
//! this file fails if tenferro reintroduces a local shortcut, an element-count
//! ceiling, or a looser test than the shared rule.

use tenferro_tensor_core::{DynRank, TensorLayout, ValidationError};

fn layout(shape: &[usize], strides: &[isize]) -> TensorLayout<DynRank> {
    let span = shape
        .iter()
        .zip(strides)
        .map(|(&extent, &stride)| {
            (extent.saturating_sub(1) as isize)
                .checked_mul(stride)
                .unwrap_or(0)
        })
        .sum::<isize>();
    let buffer_len = usize::try_from(span).unwrap_or(0) + 1;
    TensorLayout::<DynRank>::from_parts(
        shape.iter().copied().collect(),
        strides.iter().copied().collect(),
        0,
        buffer_len,
    )
    .expect("the reachable span of the fixture fits its buffer")
}

/// The audit's interleaved layout, pinned verbatim.
///
/// `[3, 1500]` with strides `[2000, 3]` is injective: `2000 = 2 (mod 3)`, so
/// the three rows never collide. tenferro used to refuse it because the layout
/// has more than 4096 elements and the local exact fallback gave up there,
/// which rejected a legal layout on the performance-critical write path.
#[test]
fn audit_interleaved_layout_is_accepted_as_injective() {
    assert!(strided_view::is_injective_layout(&[3, 1500], &[2000, 3]));
    let audited = layout(&[3, 1500], &[2000, 3]);
    audited
        .validate_mutable_no_overlap()
        .expect("an injective interleaved layout is writable");
}

/// Interleaved layouts above the retired 4096-element ceiling stay accepted.
#[test]
fn large_interleaved_layouts_agree_with_the_shared_rule() {
    for (shape, strides) in [
        (vec![4097usize, 2usize], vec![2isize, 4097]),
        (vec![3, 6_000_000], vec![6_000_001, 3]),
        (vec![2, 2, 2000], vec![4, 5, 6]),
    ] {
        let shared = strided_view::is_injective_layout(&shape, &strides);
        let tenferro = layout(&shape, &strides).validate_mutable_no_overlap();
        assert_eq!(
            shared,
            tenferro.is_ok(),
            "{shape:?}/{strides:?}: the shared injectivity rule and the mutable-layout check disagree"
        );
        assert!(shared, "{shape:?}/{strides:?} is injective");
    }
}

/// Overlapping layouts are still refused: removing the ceiling must not turn
/// the check into an accept-everything.
#[test]
fn genuinely_overlapping_layouts_are_still_rejected() {
    for (shape, strides) in [
        (vec![2usize, 2usize], vec![1isize, 1]),
        (vec![3, 3], vec![1, 0]),
        (vec![3, 1500], vec![2000, 2]),
        (vec![256, 256], vec![256, 256]),
    ] {
        assert!(!strided_view::is_injective_layout(&shape, &strides));
        let tenferro = layout(&shape, &strides);
        assert!(
            matches!(
                tenferro.validate_mutable_no_overlap(),
                Err(ValidationError::OverlappingMutableLayout)
            ),
            "{shape:?}/{strides:?} aliases and must keep its overlap error"
        );
    }
}

/// Empty layouts stay writable before any element-count or injectivity work.
#[test]
fn empty_layouts_are_accepted_before_injectivity() {
    let empty = TensorLayout::<DynRank>::from_parts(
        vec![2, usize::MAX, 0].into_iter().collect(),
        vec![1, 1, 1].into_iter().collect(),
        0,
        0,
    )
    .expect("an empty view needs no elements");
    empty
        .validate_mutable_no_overlap()
        .expect("an empty view is trivially writable");
}
