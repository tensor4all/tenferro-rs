use std::fs;
use std::path::PathBuf;

use tenferro_tensor::{Rank, Tensor, TensorRead, TensorView, TypedTensor};

fn source(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(path)).expect("source contract file")
}

#[test]
fn canonical_owner_view_and_mutable_view_surface_is_available() {
    let mut owner =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).expect("owner");
    {
        let view = owner.as_view();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[1, 2]);
        let duplicate = view.duplicate().expect("view duplicate");
        assert_eq!(duplicate.as_slice().expect("duplicate data"), &[1.0; 4]);
    }
    {
        let mut view = owner.as_view_mut();
        assert_eq!(view.shape(), &[2, 2]);
        view.get_mut(&[1, 0])
            .expect("mutable element")
            .clone_from(&3.0);
        let duplicate = view.duplicate().expect("mutable view duplicate");
        assert_eq!(
            duplicate.as_slice().expect("duplicate data"),
            &[1.0, 3.0, 1.0, 1.0]
        );
    }
    let owner = Tensor::F64(
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0; 4]).expect("dynamic owner"),
    );
    let read = TensorRead::from_tensor(&owner);
    assert_eq!(read.shape(), &[2, 2]);
}

#[test]
fn dtype_erased_views_have_explicit_duplicate_boundaries() {
    let tensor = Tensor::from_vec_col_major([2], vec![2.0_f64, 4.0]).expect("tensor");
    let view = match &tensor {
        Tensor::F64(tensor) => TensorView::F64(tensor.as_view()),
        _ => unreachable!("constructed an f64 tensor"),
    };
    let duplicate = view.duplicate().expect("duplicate");
    assert_eq!(duplicate.as_slice::<f64>().expect("data"), &[2.0, 4.0]);
}

#[test]
fn provider_exports_and_transfer_defaults_are_normalized() {
    let gpu_lib = source("../tenferro-gpu/src/lib.rs");
    assert!(gpu_lib.contains("pub mod cuda"));
    assert!(gpu_lib.contains("pub mod apple"));
    assert!(!gpu_lib.contains("pub mod cuda_interop"));
    assert!(!gpu_lib.contains("pub use cubecl::{"));

    let webgpu_mod = source("../tenferro-gpu/src/webgpu/mod.rs");
    assert!(webgpu_mod.contains("#[doc(hidden)]\npub mod interop"));
    let webgpu_interop = source("../tenferro-gpu/src/webgpu/interop.rs");
    assert!(webgpu_interop.contains("WebGpuFftOutput"));
    assert!(!webgpu_interop.contains("pub fn allocate_raw"));
    assert!(!webgpu_interop.contains("pub fn finish_"));
    assert!(!webgpu_interop.contains("pub fn client"));
    assert!(!webgpu_interop.contains("pub fn c32_input_parts"));

    let backend = source("src/backend.rs");
    let transfer = &backend[backend
        .find("pub trait TensorDeviceTransfer")
        .expect("transfer trait")..];
    assert!(!transfer.contains("tensor.duplicate()"));
    assert!(transfer.contains("TensorRead<'_>"));

    let types = source("src/types.rs");
    assert!(!types.contains("impl<T: Clone"));
    assert!(!types.contains("pub struct ArcTensor"));
}
