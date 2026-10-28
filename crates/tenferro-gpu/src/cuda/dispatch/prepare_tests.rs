//! The engine's prepared dispatcher grows with the auxiliary raw-scope launches it serves: a
//! module is prepared once and a function bound once, however the launches interleave (#2296).

use crate::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{GatherConfig, Tensor, TensorIndexing, TensorRead, TensorStructural};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

/// The engine's prepared set: entries held, modules prepared and functions bound.
fn prepared(engine: &CudaEngine) -> (usize, usize, usize) {
    let state = engine.prepared.borrow();
    let dispatcher = state.dispatcher.as_ref();
    (
        state.entries.len(),
        dispatcher.map_or(0, |dispatcher| dispatcher.modules.modules()),
        dispatcher.map_or(0, |dispatcher| dispatcher.modules.bound_functions()),
    )
}

/// A rank-9 transpose materializes through the `_any` symbol with its geometry in device
/// memory, a rank-9 gather does the same for its packed plan, and a compact copy binds
/// `CopyExact` in between. Each launch extends the engine's prepared set; repeating the first
/// one neither prepares a module nor binds a function again.
#[test]
#[ignore = "requires CUDA"]
fn interleaved_auxiliary_launches_extend_the_prepared_set() {
    let Some(mut engine) = engine() else { return };
    let runtime = engine.runtime().clone();

    // Nine axes exceed the by-value ABI's rank, so the transpose takes the general path.
    let shape = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 3];
    let len: usize = shape.iter().product();
    let host =
        Tensor::from_vec_col_major(shape.clone(), (0..len).map(|index| index as f64).collect())
            .expect("host tensor");
    let source = upload_tensor(&runtime, &host).expect("upload");
    let perm: Vec<usize> = (0..shape.len()).rev().collect();
    let transposed = engine
        .transpose_read(TensorRead::from_tensor(&source), &perm)
        .expect("rank-9 transpose");
    let after_transpose = prepared(&engine);
    assert_eq!(
        after_transpose,
        (1, after_transpose.1, 1),
        "the wide materialization adds one entry and binds its one symbol"
    );
    assert!(after_transpose.1 >= 1, "its module is prepared");

    // A rank-9 operand and a rank-9 plan make the gather take the general path as well.
    let operand_shape = vec![3usize, 1, 1, 1, 1, 1, 1, 2, 4];
    let operand_data: Vec<f64> = (0..24).map(|index| index as f64).collect();
    let operand_host =
        Tensor::from_vec_col_major(operand_shape.clone(), operand_data).expect("operand");
    let operand = upload_tensor(&runtime, &operand_host).expect("upload operand");
    let indices_host = Tensor::from_vec_col_major(vec![2, 1], vec![0i64, 1]).expect("indices");
    let indices = upload_tensor(&runtime, &indices_host).expect("upload indices");
    let gathered = engine
        .gather(
            &operand,
            &indices,
            &GatherConfig {
                offset_dims: vec![1],
                collapsed_slice_dims: vec![0, 1, 2, 3, 4, 5, 6, 7],
                start_index_map: vec![7],
                index_vector_dim: 1,
                slice_sizes: vec![1, 1, 1, 1, 1, 1, 1, 1, 2],
            },
        )
        .expect("rank-9 gather");
    let after_gather = prepared(&engine);
    assert_eq!(after_gather.0, 2, "the gather adds its entry");
    assert_eq!(
        after_gather.2,
        after_transpose.2 + 1,
        "and binds one more function"
    );
    assert!(
        after_gather.1 >= after_transpose.1,
        "no prepared module is dropped"
    );

    // A compact copy binds `CopyExact` beside the two wide entries.
    let copied = engine
        .to_contiguous_read(TensorRead::from_tensor(&gathered))
        .expect("copy");
    let after_copy = prepared(&engine);
    assert_eq!(after_copy.0, 3, "the copy adds its entry");
    assert_eq!(
        after_copy.2,
        after_gather.2 + 1,
        "and binds one more function"
    );
    assert!(
        after_copy.1 >= after_gather.1,
        "no prepared module is dropped"
    );
    let compiled = runtime.compile_attempts();

    // The first launch again: its entry, module and function are all still held.
    let repeated = engine
        .transpose_read(TensorRead::from_tensor(&source), &perm)
        .expect("rank-9 transpose again");
    assert_eq!(
        prepared(&engine),
        after_copy,
        "a repeated auxiliary launch neither prepares nor binds again"
    );
    assert_eq!(
        runtime.compile_attempts(),
        compiled,
        "a repeated auxiliary launch performs no NVRTC compilation"
    );

    // The launches above did the work: the copy carries the gathered window values.
    assert_eq!(transposed.shape(), &[3, 1, 1, 1, 1, 1, 1, 1, 2][..]);
    assert_eq!(repeated.shape(), transposed.shape());
    let observed = download_tensor(&runtime, &copied).expect("download");
    assert_eq!(
        observed.as_slice::<f64>().expect("f64 payload"),
        &[0.0, 3.0, 6.0, 9.0][..]
    );
}
