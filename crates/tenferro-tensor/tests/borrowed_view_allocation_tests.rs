use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use num_complex::Complex64;
use tenferro_tensor::{
    Rank, SliceConfig, Tensor, TensorRead, TensorValue, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        // SAFETY: this allocator forwards the unchanged layout to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout came from the corresponding System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        // SAFETY: ptr and layout came from System, and new_size is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations(op: impl FnOnce()) -> usize {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    op();
    COUNTING.set(false);
    ALLOCATIONS.get()
}

#[test]
fn small_dynamic_borrowed_view_metadata_stays_inline() {
    let data = [0_i32; 6];
    let read_allocations = count_allocations(|| {
        let view = TypedTensorView::from_slice([2, 3], [1, 2], 0, &data).unwrap();
        black_box(view);
    });

    let mut data = [0_i32; 6];
    let write_allocations = count_allocations(|| {
        let view = TypedTensorViewMut::from_slice([2, 3], [1, 2], 0, &mut data).unwrap();
        black_box(view);
    });

    assert_eq!(read_allocations, 0);
    assert_eq!(write_allocations, 0);
}

#[test]
fn plain_static_rank_owner_adopts_vec_without_metadata_allocation() {
    let data = vec![String::from("a"), String::from("b")];
    let pointer = data.as_ptr();
    let mut owner = None;
    let ctor_allocations = count_allocations(|| {
        owner = Some(TypedTensor::<String, Rank<1>>::from_vec_col_major([2], data).unwrap());
    });
    let owner = owner.unwrap();
    let view_allocations = count_allocations(|| {
        assert_eq!(owner.host_data().unwrap().as_ptr(), pointer);
        assert_eq!(owner.as_view().get(&[1]).unwrap(), "b");
    });
    assert_eq!(ctor_allocations, 0);
    assert_eq!(view_allocations, 0);
    let original = owner.into_host_vec().unwrap();
    assert_eq!(original.as_ptr(), pointer);
}

#[test]
fn erased_plain_owner_and_read_borrow_keep_original_allocation() {
    let data = vec![1.0_f64, 2.0];
    let pointer = data.as_ptr();
    let allocations = count_allocations(|| {
        let owner = TypedTensor::<f64>::from_vec_col_major([2], data).unwrap();
        let tensor = Tensor::from_typed(owner);
        let read = TensorRead::from_tensor(&tensor);
        assert_eq!(read.as_slice::<f64>().unwrap().as_ptr(), pointer);
        assert_eq!(
            tensor
                .as_typed::<f64>()
                .unwrap()
                .host_data()
                .unwrap()
                .as_ptr(),
            pointer
        );
        black_box(tensor);
    });
    assert_eq!(allocations, 0);
}

#[test]
fn static_rank_borrowed_erasure_does_not_allocate_or_promote_storage() {
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let allocations = count_allocations(|| {
        let view =
            TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data).unwrap();
        let read = view.into_tensor_read().unwrap();
        assert_eq!(read.as_slice::<f64>().unwrap(), &data);
        black_box(read);
    });
    assert_eq!(allocations, 0);
}

#[test]
fn ranked_owner_view_erasure_keeps_its_root_borrow() {
    let owner =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let read = owner.as_view().into_tensor_read().unwrap();
    assert_eq!(read.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn small_complex_real_views_do_not_allocate_metadata() {
    let data = [Complex64::new(1.0, 2.0); 6];
    let immutable = count_allocations(|| {
        let view =
            TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 3], [1, 2], 0, &data).unwrap();
        let real = view.as_real_view().unwrap();
        assert_eq!(real.shape(), &[2, 2, 3]);
        assert_eq!(real.as_slice().unwrap()[1], 2.0);
        black_box(real);
    });
    let mut data = [Complex64::new(1.0, 2.0); 6];
    let mutable = count_allocations(|| {
        let mut view =
            TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 3], [1, 2], 0, &mut data)
                .unwrap();
        let mut real = view.as_real_view_mut().unwrap();
        real.host_storage_mut().unwrap()[1] = 4.0;
        black_box(real);
    });
    let dynamic = count_allocations(|| {
        let view = TypedTensorView::from_slice([2, 3], [1, 2], 0, &data).unwrap();
        black_box(view.as_real_view().unwrap());
    });
    let dynamic_mut = count_allocations(|| {
        let mut view = TypedTensorViewMut::from_slice([2, 3], [1, 2], 0, &mut data).unwrap();
        black_box(view.as_real_view_mut().unwrap());
    });
    assert_eq!(data[0].im, 4.0);
    assert_eq!(immutable, 0);
    assert_eq!(mutable, 0);
    assert_eq!(dynamic, 0);
    assert_eq!(dynamic_mut, 0);

    // Arbitrary rank may spill, but its layout must still be valid.
    let high_rank = TypedTensorView::from_slice([1; 8], [1; 8], 0, &data[..1]).unwrap();
    assert_eq!(
        high_rank.as_real_view().unwrap().shape(),
        &[2, 1, 1, 1, 1, 1, 1, 1, 1]
    );
}

/// #2143 item 3 / `metadata_view_alloc_counter`: the public metadata-only view
/// constructors must not allocate per call at rank <= 8. This is the acceptance
/// signal for the #2040 rows; the timing rows are secondary evidence.
#[test]
fn metadata_only_view_construction_allocates_nothing() {
    fn square() -> TensorValue {
        TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1.0_f64; 4]).unwrap())
    }

    fn flat() -> TensorValue {
        TensorValue::from_tensor(
            Tensor::from_vec_col_major([1024], (0..1024).map(|i| i as f64).collect()).unwrap(),
        )
    }

    fn rank8() -> TensorValue {
        TensorValue::from_tensor(Tensor::from_vec_col_major([2; 8], vec![0.0_f64; 256]).unwrap())
    }

    for _ in 0..4 {
        let value = square();
        let allocations = count_allocations(|| {
            black_box(value.transpose_view([1, 0]).unwrap());
        });
        assert_eq!(allocations, 0, "transpose_view allocated");
    }

    for _ in 0..4 {
        let value = TensorValue::from_tensor(
            Tensor::from_vec_col_major([4096], (0..4096).map(|i| i as f64).collect()).unwrap(),
        );
        let config = SliceConfig {
            starts: vec![128],
            limits: vec![3968],
            strides: vec![2],
        };
        let allocations = count_allocations(|| {
            black_box(value.slice_view(&config).unwrap());
        });
        assert_eq!(allocations, 0, "slice_view allocated");
    }

    // Rank 8 keeps the per-axis slice specs inline.
    for _ in 0..4 {
        let value = rank8();
        let config = SliceConfig {
            starts: vec![0; 8],
            limits: vec![2; 8],
            strides: vec![1; 8],
        };
        let allocations = count_allocations(|| {
            black_box(value.slice_view(&config).unwrap());
        });
        assert_eq!(allocations, 0, "rank-8 slice_view allocated");
    }

    for _ in 0..4 {
        let value = flat();
        let allocations = count_allocations(|| {
            black_box(value.reshape_view([32, 32]).unwrap());
        });
        assert_eq!(allocations, 0, "reshape_view allocated");
    }

    // A singleton axis whose stored stride is not canonical takes the relaxed
    // col-major-contiguous path instead of the exact compact check.
    for _ in 0..4 {
        let value = TensorValue::from_tensor(
            Tensor::from_vec_col_major([1, 3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        )
        .transpose_view([1, 0])
        .unwrap();
        assert_eq!(value.strides(), &[1, 1]);
        let allocations = count_allocations(|| {
            black_box(value.reshape_view([3]).unwrap());
        });
        assert_eq!(allocations, 0, "relaxed reshape_view allocated");
    }
}
