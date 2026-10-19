//! Layout values reach the CUDA kernels at run time (#2010, #1885 §3.4).
//!
//! Each test launches one kernel over several layouts that differ in every
//! value the kernel used to receive as `#[comptime]` (extents, strides,
//! offsets, lengths, starts, paddings, window sizes, axis offsets) at a fixed
//! rank and dtype. Every layout's result is compared with an exact host
//! reference bit for bit, and the launch itself would fail if a layout value
//! were still a compile-time argument of the wrong kind.
//!
//! The compile-count half of this check is not run: it read
//! `CudaServer::compiled_kernel_ids`, a fork-only accessor, and upstream 0.11
//! exposes no way to enumerate compiled kernels or ask whether a kernel type
//! was loaded. See docs/design/cubecl-011-compile-count-hook.md; the assertion
//! returns when that hook exists upstream.

use num_complex::Complex64;
use tenferro_tensor::{
    backend::BackendSessionHost, ContractionScalar, TensorRead, TensorView, TensorViewMut,
    TensorWrite,
};

use super::super::interop::fill_zero_write;
use super::super::CudaEngine;
use super::{cpu_backend, download, gpu_backend, tensor_f64, tensor_i64, upload};
use crate::config::{GatherConfig, PadConfig, SliceConfig};
use crate::{Tensor, TypedTensor};

/// Run `launch(layout)` for every layout.
///
/// The compile-count assertion this helper used to make needs the upstream test
/// hook described in docs/design/cubecl-011-compile-count-hook.md; until it
/// exists, the launches still run and each test checks its own result.
fn assert_one_module_for_all_layouts(
    backend: &mut CudaEngine,
    kernel: &str,
    layouts: usize,
    mut launch: impl FnMut(&mut CudaEngine, usize),
) {
    assert!(layouts >= 4, "use several distinct layouts");
    let _ = kernel;
    for layout in 0..layouts {
        launch(backend, layout);
    }
}

/// Index-coded payload: every element is distinct and exactly representable.
fn coded(len: usize, salt: f64) -> Vec<f64> {
    (0..len).map(|i| salt + i as f64).collect()
}

/// Column-major gather of a strided view out of `data`.
fn strided_reference(data: &[f64], shape: &[usize], strides: &[isize], offset: isize) -> Vec<f64> {
    let len: usize = shape.iter().product();
    (0..len)
        .map(|mut flat| {
            let mut index = offset;
            for (&dim, &stride) in shape.iter().zip(strides) {
                index += (flat % dim) as isize * stride;
                flat /= dim;
            }
            data[usize::try_from(index).unwrap()]
        })
        .collect()
}

/// Allocation positions a strided view addresses, in column-major order.
fn strided_positions(shape: &[usize], strides: &[isize], offset: isize) -> Vec<usize> {
    let len: usize = shape.iter().product();
    (0..len)
        .map(|mut flat| {
            let mut index = offset;
            for (&dim, &stride) in shape.iter().zip(strides) {
                index += (flat % dim) as isize * stride;
                flat /= dim;
            }
            usize::try_from(index).unwrap()
        })
        .collect()
}

fn assert_bits_eq(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (lhs, rhs)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "{what}: element {index}: {lhs} != {rhs}"
        );
    }
}

fn host_f64(backend: &CudaEngine, tensor: &Tensor) -> Vec<f64> {
    download(backend, tensor)
        .as_slice::<f64>()
        .unwrap()
        .to_vec()
}

fn typed_f64(tensor: &Tensor) -> &TypedTensor<f64> {
    tensor.as_typed::<f64>().expect("f64 tensor")
}

/// Rank-3 views that no axis fusion can collapse and that are not a tiled
/// transpose, with a different extent, stride and offset tuple each.
const STRIDED_VIEWS: [([usize; 3], [isize; 3], isize); 5] = [
    ([2, 3, 4], [2, 7, 23], 1),
    ([3, 2, 5], [3, 11, 25], 4),
    ([4, 3, 2], [2, 9, 29], 0),
    ([2, 5, 3], [5, 12, 61], 7),
    ([5, 2, 2], [2, 13, 31], 3),
];

const STRIDED_ALLOCATION: usize = 512;

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn materialize_strided_kernel_compiles_once_per_rank() {
    let mut gpu = gpu_backend();
    let data = coded(STRIDED_ALLOCATION, 0.25);
    let source = upload(&gpu, &tensor_f64(vec![STRIDED_ALLOCATION], data.clone()));
    assert_one_module_for_all_layouts(
        &mut gpu,
        "materialize_strided_kernel",
        STRIDED_VIEWS.len(),
        |gpu, layout| {
            let (shape, strides, offset) = STRIDED_VIEWS[layout];
            let view = typed_f64(&source)
                .backend_region_view(shape.to_vec(), strides.to_vec(), offset)
                .unwrap();
            let out = gpu
                .to_contiguous_view_typed(&view, "kernel_specialization_test")
                .unwrap();
            let actual = host_f64(gpu, &Tensor::from_typed(out));
            assert_bits_eq(
                &actual,
                &strided_reference(&data, &shape, &strides, offset),
                "materialize",
            );
        },
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn tiled_transpose_kernel_compiles_once_per_tile_configuration() {
    // Row-major [rows, cols] views of compact allocations are the tiled 2D
    // transpose; the batch stride and both fast extents differ per layout.
    let shapes = [(3usize, 5usize), (17, 4), (8, 33), (21, 19), (2, 40)];
    let mut gpu = gpu_backend();
    assert_one_module_for_all_layouts(
        &mut gpu,
        "tiled_transpose_kernel",
        shapes.len(),
        |gpu, layout| {
            let (rows, cols) = shapes[layout];
            let data = coded(rows * cols, 1.5);
            let source = upload(gpu, &tensor_f64(vec![rows * cols], data.clone()));
            let strides = [cols as isize, 1];
            let view = typed_f64(&source)
                .backend_region_view(vec![rows, cols], strides.to_vec(), 0)
                .unwrap();
            let out = gpu
                .to_contiguous_view_typed(&view, "kernel_specialization_test")
                .unwrap();
            let actual = host_f64(gpu, &Tensor::from_typed(out));
            assert_bits_eq(
                &actual,
                &strided_reference(&data, &[rows, cols], &strides, 0),
                "tiled transpose",
            );
        },
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn strided_copy_kernels_compile_once_per_rank() {
    let mut gpu = gpu_backend();
    let source_data = coded(STRIDED_ALLOCATION, 0.5);
    let source = upload(
        &gpu,
        &tensor_f64(vec![STRIDED_ALLOCATION], source_data.clone()),
    );

    // Strided source into a strided destination: `strided_to_strided_kernel`.
    assert_one_module_for_all_layouts(
        &mut gpu,
        "strided_to_strided_kernel",
        STRIDED_VIEWS.len(),
        |gpu, layout| {
            let (shape, src_strides, src_offset) = STRIDED_VIEWS[layout];
            // A second gapped, non-overlapping layout for the destination.
            let dst_strides = [
                3,
                3 * shape[0] as isize + 1,
                (3 * shape[0] as isize + 1) * shape[1] as isize + 2,
            ];
            let dst_offset = layout as isize + 5;
            let dst_init = coded(STRIDED_ALLOCATION, -1000.0);
            let mut destination =
                upload(gpu, &tensor_f64(vec![STRIDED_ALLOCATION], dst_init.clone()));
            {
                let src = typed_f64(&source)
                    .backend_region_view(shape.to_vec(), src_strides.to_vec(), src_offset)
                    .unwrap();
                let mut dst = destination
                    .as_typed_mut::<f64>()
                    .unwrap()
                    .backend_region_view_mut(shape.to_vec(), dst_strides.to_vec(), dst_offset)
                    .unwrap();
                gpu.copy_view_to_view_typed(&src, &mut dst, "kernel_specialization_test")
                    .unwrap();
            }
            let mut expected = dst_init;
            let values = strided_reference(&source_data, &shape, &src_strides, src_offset);
            for (position, value) in strided_positions(&shape, &dst_strides, dst_offset)
                .into_iter()
                .zip(values)
            {
                expected[position] = value;
            }
            assert_bits_eq(&host_f64(gpu, &destination), &expected, "strided copy");
        },
    );

    // Compact zero-offset source into a strided destination at an offset:
    // `contiguous_to_view_kernel` (the offset keeps it off the tiled route).
    assert_one_module_for_all_layouts(
        &mut gpu,
        "contiguous_to_view_kernel",
        STRIDED_VIEWS.len(),
        |gpu, layout| {
            let (shape, dst_strides, dst_offset) = STRIDED_VIEWS[layout];
            let dst_offset = dst_offset + 1;
            let len: usize = shape.iter().product();
            let values = coded(len, 3.0);
            let compact = upload(gpu, &tensor_f64(shape.to_vec(), values.clone()));
            let dst_init = coded(STRIDED_ALLOCATION, -500.0);
            let mut destination =
                upload(gpu, &tensor_f64(vec![STRIDED_ALLOCATION], dst_init.clone()));
            {
                let src = typed_f64(&compact).as_view();
                let mut dst = destination
                    .as_typed_mut::<f64>()
                    .unwrap()
                    .backend_region_view_mut(shape.to_vec(), dst_strides.to_vec(), dst_offset)
                    .unwrap();
                gpu.copy_view_to_view_typed(&src, &mut dst, "kernel_specialization_test")
                    .unwrap();
            }
            let mut expected = dst_init;
            for (position, value) in strided_positions(&shape, &dst_strides, dst_offset)
                .into_iter()
                .zip(values)
            {
                expected[position] = value;
            }
            assert_bits_eq(
                &host_f64(gpu, &destination),
                &expected,
                "contiguous to view",
            );
        },
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn fill_zero_view_kernel_compiles_once_per_rank() {
    let mut gpu = gpu_backend();
    assert_one_module_for_all_layouts(
        &mut gpu,
        "fill_zero_view_kernel",
        STRIDED_VIEWS.len(),
        |gpu, layout| {
            let (shape, strides, offset) = STRIDED_VIEWS[layout];
            let init = coded(STRIDED_ALLOCATION, 9.0);
            let mut output = upload(gpu, &tensor_f64(vec![STRIDED_ALLOCATION], init.clone()));
            {
                let region = output
                    .as_typed_mut::<f64>()
                    .unwrap()
                    .backend_region_view_mut(shape.to_vec(), strides.to_vec(), offset)
                    .unwrap();
                fill_zero_write(
                    gpu.runtime(),
                    TensorWrite::from_view(TensorViewMut::F64(region)),
                )
                .unwrap();
            }
            let mut expected = init;
            for position in strided_positions(&shape, &strides, offset) {
                expected[position] = 0.0;
            }
            assert_bits_eq(&host_f64(gpu, &output), &expected, "fill zero view");
        },
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn axpby_strided_source_kernel_compiles_once_per_rank() {
    let mut gpu = gpu_backend();
    let source_data = coded(STRIDED_ALLOCATION, 1.0);
    let source = upload(
        &gpu,
        &tensor_f64(vec![STRIDED_ALLOCATION], source_data.clone()),
    );
    assert_one_module_for_all_layouts(
        &mut gpu,
        "axpby_strided_source_float",
        STRIDED_VIEWS.len(),
        |gpu, layout| {
            let (shape, strides, offset) = STRIDED_VIEWS[layout];
            let len: usize = shape.iter().product();
            let y_init = coded(len, -20.0);
            let mut y = upload(gpu, &tensor_f64(shape.to_vec(), y_init.clone()));
            let x = typed_f64(&source)
                .backend_region_view(shape.to_vec(), strides.to_vec(), offset)
                .unwrap();
            gpu.with_backend_session(|session| {
                session.axpby_read_into_accum(
                    ContractionScalar::F64(2.0),
                    TensorRead::from_view(TensorView::F64(x)),
                    ContractionScalar::F64(1.0),
                    TensorWrite::from_tensor(&mut y),
                )
            })
            .unwrap()
            .unwrap();
            // Small integers plus 0.5 / 0.25 offsets: every product and sum is
            // exact with or without FMA contraction.
            let expected: Vec<f64> = strided_reference(&source_data, &shape, &strides, offset)
                .into_iter()
                .zip(y_init)
                .map(|(x, y)| 2.0 * x + y)
                .collect();
            assert_bits_eq(&host_f64(gpu, &y), &expected, "axpby strided source");
        },
    );
    // The complex twin shares the same runtime layout arguments.
    let complex_source: Vec<Complex64> = (0..STRIDED_ALLOCATION)
        .map(|i| Complex64::new(i as f64, 0.5 - i as f64))
        .collect();
    let complex = upload(
        &gpu,
        &Tensor::from_vec_col_major(vec![STRIDED_ALLOCATION], complex_source.clone()).unwrap(),
    );
    assert_one_module_for_all_layouts(
        &mut gpu,
        "axpby_strided_source_complex",
        STRIDED_VIEWS.len(),
        |gpu, layout| {
            let (shape, strides, offset) = STRIDED_VIEWS[layout];
            let len: usize = shape.iter().product();
            let mut y = upload(
                gpu,
                &Tensor::from_vec_col_major(shape.to_vec(), vec![Complex64::new(1.0, -1.0); len])
                    .unwrap(),
            );
            let x = complex
                .as_typed::<Complex64>()
                .unwrap()
                .backend_region_view(shape.to_vec(), strides.to_vec(), offset)
                .unwrap();
            gpu.with_backend_session(|session| {
                session.axpby_read_into_accum(
                    ContractionScalar::C64(Complex64::new(1.0, 0.0)),
                    TensorRead::from_view(TensorView::C64(x)),
                    ContractionScalar::C64(Complex64::new(1.0, 0.0)),
                    TensorWrite::from_tensor(&mut y),
                )
            })
            .unwrap()
            .unwrap();
            let actual = download(gpu, &y);
            let actual = actual.as_slice::<Complex64>().unwrap();
            let positions = strided_positions(&shape, &strides, offset);
            for (index, (value, position)) in actual.iter().zip(positions).enumerate() {
                let expected = complex_source[position] + Complex64::new(1.0, -1.0);
                assert_eq!(
                    (value.re.to_bits(), value.im.to_bits()),
                    (expected.re.to_bits(), expected.im.to_bits()),
                    "complex axpby element {index}"
                );
            }
        },
    );
}

/// Run the same session op on the CPU and CUDA backends and compare bit for bit.
fn assert_cpu_parity(
    gpu: &mut CudaEngine,
    inputs: &[&Tensor],
    op: impl Fn(&mut dyn tenferro_tensor::backend::Engine, &[&Tensor]) -> Tensor + Sync,
) {
    let expected = cpu_backend()
        .with_backend_session(|session| op(session, inputs))
        .unwrap();
    let device_inputs: Vec<Tensor> = inputs.iter().map(|input| upload(gpu, input)).collect();
    let device_refs: Vec<&Tensor> = device_inputs.iter().collect();
    let actual = gpu
        .with_backend_session(|session| op(session, &device_refs))
        .unwrap();
    let actual = download(gpu, &actual);
    assert_eq!(actual.shape(), expected.shape());
    assert_bits_eq(
        actual.as_slice::<f64>().unwrap(),
        expected.as_slice::<f64>().unwrap(),
        "CPU parity",
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn indexing_kernels_compile_once_per_rank() {
    let mut gpu = gpu_backend();
    let input = tensor_f64(vec![8, 9], coded(72, 0.75));

    // slice: the starts are runtime; the step stays a compile-time attribute.
    let starts = [[0usize, 0], [1, 2], [3, 1], [2, 4], [5, 0], [4, 3]];
    assert_one_module_for_all_layouts(&mut gpu, "slice_kernel", starts.len(), |gpu, layout| {
        let start = starts[layout];
        let config = SliceConfig {
            starts: start.to_vec(),
            limits: vec![start[0] + 3, start[1] + 5],
            strides: vec![1, 2],
        };
        assert_cpu_parity(gpu, &[&input], |session, inputs| {
            session.slice(inputs[0], &config).unwrap()
        });
    });

    // dynamic_slice: the slice sizes are runtime.
    let sizes = [[2usize, 3], [3, 2], [4, 4], [1, 5], [5, 1], [2, 2]];
    let dynamic_starts = tensor_i64(vec![2], vec![2, 3]);
    assert_one_module_for_all_layouts(
        &mut gpu,
        "dynamic_slice_kernel",
        sizes.len(),
        |gpu, layout| {
            let sizes = sizes[layout];
            assert_cpu_parity(gpu, &[&input, &dynamic_starts], |session, inputs| {
                session.dynamic_slice(inputs[0], inputs[1], &sizes).unwrap()
            });
        },
    );

    // pad: both edge paddings (including negative, cropping ones) and the
    // interior padding are runtime.
    let pads: [([i64; 2], [i64; 2], [i64; 2]); 6] = [
        ([1, 0], [0, 1], [0, 1]),
        ([0, 2], [1, 0], [1, 0]),
        ([-1, 1], [2, -1], [2, 0]),
        ([2, -2], [-1, 2], [0, 0]),
        ([0, 0], [0, 0], [1, 1]),
        ([-2, -1], [1, 1], [0, 2]),
    ];
    assert_one_module_for_all_layouts(&mut gpu, "pad_kernel", pads.len(), |gpu, layout| {
        let (low, high, interior) = pads[layout];
        let config = PadConfig {
            edge_padding_low: low.to_vec(),
            edge_padding_high: high.to_vec(),
            interior_padding: interior.to_vec(),
        };
        assert_cpu_parity(gpu, &[&input], |session, inputs| {
            session.pad(inputs[0], &config).unwrap()
        });
    });

    // gather: the window sizes are runtime; the dimension numbers stay
    // compile-time axis mappings.
    let windows = [[1usize, 1], [2, 3], [3, 2], [4, 1], [1, 4], [2, 2]];
    let indices = tensor_i64(vec![3, 2], vec![0, 4, 7, 1, 3, 5]);
    assert_one_module_for_all_layouts(&mut gpu, "gather_kernel", windows.len(), |gpu, layout| {
        let config = GatherConfig {
            offset_dims: vec![1, 2],
            collapsed_slice_dims: vec![],
            start_index_map: vec![0, 1],
            index_vector_dim: 1,
            slice_sizes: windows[layout].to_vec(),
        };
        assert_cpu_parity(gpu, &[&input, &indices], |session, inputs| {
            session.gather(inputs[0], inputs[1], &config).unwrap()
        });
    });

    // concatenate: each input is copied at a different running axis offset.
    let widths = [[1usize, 2, 3], [2, 5, 1], [4, 1, 2], [3, 3, 3], [1, 1, 6]];
    assert_one_module_for_all_layouts(
        &mut gpu,
        "concatenate_copy_kernel",
        widths.len(),
        |gpu, layout| {
            let parts: Vec<Tensor> = widths[layout]
                .iter()
                .enumerate()
                .map(|(part, &width)| {
                    tensor_f64(vec![3, width], coded(3 * width, 100.0 * part as f64))
                })
                .collect();
            let refs: Vec<&Tensor> = parts.iter().collect();
            assert_cpu_parity(gpu, &refs, |session, inputs| {
                session.concatenate(inputs, 1).unwrap()
            });
        },
    );
}
