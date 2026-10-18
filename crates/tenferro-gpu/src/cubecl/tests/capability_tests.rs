use num_complex::{Complex32, Complex64};
use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};
use tenferro_cpu::cpu_capabilities;
use tenferro_tensor::{
    capability_output_dtype, BackendId, BackendSessionHost, DType, OperationCapability,
    SupportLevel, Tensor, TensorRead,
};

use crate::config::CompareDir;
use crate::cubecl::gpu_available;
use crate::cuda::{cuda_capabilities, CudaBackend};
use crate::DotGeneralConfig;

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_bool, tensor_c32, tensor_c64,
    tensor_f32, tensor_f64, tensor_i32, tensor_i64, upload,
};

#[test]
fn cuda_capability_table_reports_current_core_coverage() {
    let capabilities = cuda_capabilities();

    let add_i32 = capabilities
        .iter()
        .find(|entry| entry.op == PrimitiveOpKind::Add && entry.dtype == DType::I32)
        .expect("CUDA add/i32 should be described");
    assert_eq!(add_i32.backend, BackendId::Cuda);
    assert_eq!(add_i32.result, SupportLevel::Native);
    assert_eq!(add_i32.read_inputs, SupportLevel::FallbackCopy);

    let compare_i64 = capabilities
        .iter()
        .find(|entry| entry.op == PrimitiveOpKind::Compare && entry.dtype == DType::I64)
        .expect("CUDA compare/i64 should be described");
    assert_eq!(compare_i64.output_dtype, DType::Bool);

    let exp_c64 = capabilities
        .iter()
        .find(|entry| entry.op == PrimitiveOpKind::Exp && entry.dtype == DType::C64)
        .expect("CUDA exp/c64 unsupported entry should be described");
    assert_eq!(exp_c64.result, SupportLevel::Unsupported);

    let dot_c64 = capabilities
        .iter()
        .find(|entry| entry.op == PrimitiveOpKind::DotGeneral && entry.dtype == DType::C64)
        .expect("CUDA dot_general/c64 should be described");
    assert_eq!(dot_c64.accumulation, SupportLevel::Native);

    let div_i32 = capabilities
        .iter()
        .find(|entry| entry.op == PrimitiveOpKind::Div && entry.dtype == DType::I32)
        .expect("CUDA div/i32 should be described after #1320");
    assert_eq!(div_i32.result, SupportLevel::Native);

    let rem_i64 = capabilities
        .iter()
        .find(|entry| entry.op == PrimitiveOpKind::Rem && entry.dtype == DType::I64)
        .expect("CUDA rem/i64 should be described after #1320");
    assert_eq!(rem_i64.result, SupportLevel::Native);
}

#[test]
fn cuda_capability_entries_match_core_catalog_dtype_policy() {
    for entry in cuda_capabilities() {
        assert_eq!(
            capability_output_dtype(entry.op, entry.dtype),
            Some(entry.output_dtype),
            "descriptor output dtype drift for {:?}/{:?}",
            entry.op,
            entry.dtype
        );
    }
}

#[test]
fn cuda_unsupported_entries_have_cpu_descriptor_counterparts() {
    for entry in cuda_capabilities()
        .iter()
        .filter(|entry| !entry.result.is_supported())
    {
        let cpu_entry = cpu_capabilities()
            .iter()
            .find(|cpu| cpu.op == entry.op && cpu.dtype == entry.dtype)
            .unwrap_or_else(|| {
                panic!(
                    "CPU descriptor missing counterpart for CUDA unsupported {:?}/{:?}",
                    entry.op, entry.dtype
                )
            });
        assert_eq!(cpu_entry.output_dtype, entry.output_dtype);
    }
}

#[test]
fn cuda_core_capability_table_in_guide_matches_descriptor() {
    let guide = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/guides/devices-and-gpu.md"
    ));
    let start = "<!-- cuda-core-capability:start -->\n";
    let end = "<!-- cuda-core-capability:end -->";
    let actual = guide
        .split_once(start)
        .and_then(|(_, rest)| rest.split_once(end).map(|(table, _)| table))
        .expect("CUDA core capability table markers should exist");

    assert_eq!(actual, generated_cuda_core_capability_table());
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_descriptor_supported_entries_match_cpu_smoke_cases() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for entry in cuda_capabilities()
        .iter()
        .copied()
        .filter(|entry| entry.result.is_supported())
    {
        let cpu_entry = cpu_capabilities()
            .iter()
            .find(|cpu_entry| cpu_entry.op == entry.op && cpu_entry.dtype == entry.dtype)
            .unwrap_or_else(|| {
                panic!(
                    "CPU descriptor missing supported CUDA counterpart {:?}/{:?}",
                    entry.op, entry.dtype
                )
            });
        assert!(
            cpu_entry.result.is_supported(),
            "CPU descriptor marks CUDA-supported {:?}/{:?} unsupported",
            entry.op,
            entry.dtype
        );

        run_supported_case(&mut cpu, &mut gpu, entry);
    }
}

fn generated_cuda_core_capability_table() -> String {
    let mut table = String::from(
        "| Primitive op | Native CUDA dtypes | Unsupported descriptor dtypes | Output dtype | Native axes |\n\
         | --- | --- | --- | --- | --- |\n",
    );
    for op in all_primitive_descriptors() {
        let entries: Vec<_> = cuda_capabilities()
            .iter()
            .copied()
            .filter(|entry| entry.op == op.kind)
            .collect();
        if entries.is_empty() {
            continue;
        }
        table.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |\n",
            op.name,
            dtype_list(&entries, true),
            dtype_list(&entries, false),
            output_dtype_summary(&entries),
            axis_summary(&entries),
        ));
    }
    table
}

fn dtype_list(entries: &[OperationCapability], supported: bool) -> String {
    let dtypes: Vec<_> = entries
        .iter()
        .filter(|entry| entry.result.is_supported() == supported)
        .map(|entry| dtype_name(entry.dtype))
        .collect();
    if dtypes.is_empty() {
        "none".to_string()
    } else {
        dtypes.join(", ")
    }
}

fn output_dtype_summary(entries: &[OperationCapability]) -> String {
    if entries
        .iter()
        .all(|entry| entry.output_dtype == entry.dtype)
    {
        "same as input".to_string()
    } else if entries
        .iter()
        .all(|entry| entry.output_dtype == entries[0].output_dtype)
    {
        dtype_name(entries[0].output_dtype)
    } else {
        entries
            .iter()
            .map(|entry| format!("`{:?}->{:?}`", entry.dtype, entry.output_dtype))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn axis_summary(entries: &[OperationCapability]) -> String {
    let entry = entries
        .iter()
        .find(|entry| entry.result.is_supported())
        .unwrap_or_else(|| entries.first().expect("non-empty capability entries"));
    format!(
        "result {}; read {}; write {}; strided {}; accumulation {}",
        level_name(entry.result),
        level_name(entry.read_inputs),
        level_name(entry.write_output),
        level_name(entry.strided_output),
        level_name(entry.accumulation)
    )
}

fn level_name(level: SupportLevel) -> &'static str {
    match level {
        SupportLevel::Unsupported => "Unsupported",
        SupportLevel::FallbackCopy => "FallbackCopy",
        SupportLevel::Native => "Native",
    }
}

fn dtype_name(dtype: DType) -> String {
    format!("`{dtype:?}`")
}

fn run_supported_case(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
) {
    match entry.op {
        PrimitiveOpKind::Add => assert_binary_matches(cpu, gpu, entry, |b, l, r| {
            b.with_backend_session(|__s| {
                __s.add_read(TensorRead::from_tensor(l), TensorRead::from_tensor(r))
            })
            .unwrap()
        }),
        PrimitiveOpKind::Sub => assert_binary_matches(cpu, gpu, entry, |b, l, r| {
            b.with_backend_session(|__s| {
                __s.sub_read(TensorRead::from_tensor(l), TensorRead::from_tensor(r))
            })
            .unwrap()
        }),
        PrimitiveOpKind::Mul => assert_binary_matches(cpu, gpu, entry, |b, l, r| {
            b.with_backend_session(|__s| {
                __s.mul_read(TensorRead::from_tensor(l), TensorRead::from_tensor(r))
            })
            .unwrap()
        }),
        PrimitiveOpKind::Neg => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.neg_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Conj => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.conj_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Div => assert_binary_matches(cpu, gpu, entry, |b, l, r| {
            b.with_backend_session(|__s| {
                __s.div_read(TensorRead::from_tensor(l), TensorRead::from_tensor(r))
            })
            .unwrap()
        }),
        PrimitiveOpKind::Rem => assert_binary_matches(cpu, gpu, entry, |b, l, r| {
            b.with_backend_session(|__s| __s.rem(l, r)).unwrap()
        }),
        PrimitiveOpKind::Abs => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.abs_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Sign => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.sign_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Maximum => {
            assert_binary_matches(cpu, gpu, entry, |b, l, r| {
                b.with_backend_session(|__s| {
                    __s.maximum_read(TensorRead::from_tensor(l), TensorRead::from_tensor(r))
                })
                .unwrap()
            });
        }
        PrimitiveOpKind::Minimum => {
            assert_binary_matches(cpu, gpu, entry, |b, l, r| {
                b.with_backend_session(|__s| {
                    __s.minimum_read(TensorRead::from_tensor(l), TensorRead::from_tensor(r))
                })
                .unwrap()
            });
        }
        PrimitiveOpKind::Compare => assert_compare_matches(cpu, gpu, entry),
        PrimitiveOpKind::Select => assert_select_matches(cpu, gpu, entry),
        PrimitiveOpKind::Clamp => assert_clamp_matches(cpu, gpu, entry),
        PrimitiveOpKind::Exp => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.exp_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Log => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.log_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Sin => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.sin_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Cos => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.cos_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Tanh => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.tanh_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Sqrt => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.sqrt_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Rsqrt => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.rsqrt_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Pow => assert_binary_matches(cpu, gpu, entry, |b, l, r| {
            b.with_backend_session(|__s| {
                __s.pow_read(TensorRead::from_tensor(l), TensorRead::from_tensor(r))
            })
            .unwrap()
        }),
        PrimitiveOpKind::Expm1 => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| {
                __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(x))
            })
            .unwrap()
        }),
        PrimitiveOpKind::Log1p => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.log1p_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::Erf => assert_unary_matches(cpu, gpu, entry, |b, x| {
            b.with_backend_session(|__s| __s.erf_read(TensorRead::from_tensor(x)))
                .unwrap()
        }),
        PrimitiveOpKind::ReduceSum => assert_reduction_matches(cpu, gpu, entry, |b, x, axes| {
            b.with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(x), axes))
                .unwrap()
        }),
        PrimitiveOpKind::ReduceSumSquares => {
            assert_reduction_matches(cpu, gpu, entry, |b, x, axes| {
                b.with_backend_session(|__s| {
                    __s.reduce_sum_squares_read(TensorRead::from_tensor(x), axes)
                })
                .unwrap()
            })
        }
        PrimitiveOpKind::ReduceProd => assert_reduction_matches(cpu, gpu, entry, |b, x, axes| {
            b.with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(x), axes))
                .unwrap()
        }),
        PrimitiveOpKind::ReduceMax => assert_reduction_matches(cpu, gpu, entry, |b, x, axes| {
            b.with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(x), axes))
                .unwrap()
        }),
        PrimitiveOpKind::ReduceMin => assert_reduction_matches(cpu, gpu, entry, |b, x, axes| {
            b.with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(x), axes))
                .unwrap()
        }),
        PrimitiveOpKind::DotGeneral => assert_dot_matches(cpu, gpu, entry),
        _ => panic!(
            "unsupported first-scope descriptor smoke op {:?}; descriptor={:?}",
            entry.op, entry
        ),
    }
}

fn assert_unary_matches(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
    op: impl Fn(&mut CudaBackend, &Tensor) -> tenferro_tensor::Result<Tensor>,
) {
    let input = sample_tensor(entry.dtype);
    let gpu_input = upload(gpu, &input);
    let expected = run_cpu_unary(cpu, entry.op, &input);
    let gpu_output = op(gpu, &gpu_input).unwrap();
    let actual = download(gpu, &gpu_output);
    assert_tensor_close(&actual, &expected, tolerance(entry.dtype));
}

fn assert_binary_matches(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
    op: impl Fn(&mut CudaBackend, &Tensor, &Tensor) -> tenferro_tensor::Result<Tensor>,
) {
    let lhs = sample_tensor(entry.dtype);
    let rhs = sample_rhs_tensor(entry.dtype);
    let gpu_lhs = upload(gpu, &lhs);
    let gpu_rhs = upload(gpu, &rhs);
    let expected = run_cpu_binary(cpu, entry.op, &lhs, &rhs);
    let gpu_output = op(gpu, &gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(gpu, &gpu_output);
    assert_tensor_close(&actual, &expected, tolerance(entry.dtype));
}

fn assert_reduction_matches(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
    op: impl Fn(&mut CudaBackend, &Tensor, &[usize]) -> tenferro_tensor::Result<Tensor>,
) {
    let input = reduction_sample_tensor(entry.dtype);
    let gpu_input = upload(gpu, &input);
    let axes = [0];
    let expected = run_cpu_reduction(cpu, entry.op, &input, &axes);
    let gpu_output = op(gpu, &gpu_input, &axes).unwrap();
    let actual = download(gpu, &gpu_output);
    assert_tensor_close(&actual, &expected, tolerance(entry.dtype));
}

fn assert_compare_matches(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
) {
    let lhs = sample_tensor(entry.dtype);
    let rhs = sample_rhs_tensor(entry.dtype);
    let gpu_lhs = upload(gpu, &lhs);
    let gpu_rhs = upload(gpu, &rhs);
    let expected = cpu
        .with_backend_session(|__s| {
            __s.compare_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &CompareDir::Ge,
            )
        })
        .unwrap()
        .unwrap();
    let gpu_output = gpu
        .with_backend_session(|__s| {
            __s.compare_read(
                TensorRead::from_tensor(&gpu_lhs),
                TensorRead::from_tensor(&gpu_rhs),
                &CompareDir::Ge,
            )
        })
        .unwrap()
        .unwrap();
    let actual = download(gpu, &gpu_output);
    assert_tensor_close(&actual, &expected, 0.0);
}

fn assert_select_matches(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
) {
    let pred = tensor_bool(vec![3], vec![true, false, true]);
    let lhs = sample_tensor(entry.dtype);
    let rhs = sample_rhs_tensor(entry.dtype);
    let gpu_pred = upload(gpu, &pred);
    let gpu_lhs = upload(gpu, &lhs);
    let gpu_rhs = upload(gpu, &rhs);
    let expected = cpu
        .with_backend_session(|__s| {
            __s.select_read(
                TensorRead::from_tensor(&pred),
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
            )
        })
        .unwrap()
        .unwrap();
    let gpu_output = gpu
        .with_backend_session(|__s| {
            __s.select_read(
                TensorRead::from_tensor(&gpu_pred),
                TensorRead::from_tensor(&gpu_lhs),
                TensorRead::from_tensor(&gpu_rhs),
            )
        })
        .unwrap()
        .unwrap();
    let actual = download(gpu, &gpu_output);
    assert_tensor_close(&actual, &expected, tolerance(entry.dtype));
}

fn assert_clamp_matches(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
) {
    let input = sample_tensor(entry.dtype);
    let lower = clamp_lower_tensor(entry.dtype);
    let upper = clamp_upper_tensor(entry.dtype);
    let gpu_input = upload(gpu, &input);
    let gpu_lower = upload(gpu, &lower);
    let gpu_upper = upload(gpu, &upper);
    let expected = cpu
        .with_backend_session(|__s| {
            __s.clamp_read(
                TensorRead::from_tensor(&input),
                TensorRead::from_tensor(&lower),
                TensorRead::from_tensor(&upper),
            )
        })
        .unwrap()
        .unwrap();
    let gpu_output = gpu
        .with_backend_session(|__s| {
            __s.clamp_read(
                TensorRead::from_tensor(&gpu_input),
                TensorRead::from_tensor(&gpu_lower),
                TensorRead::from_tensor(&gpu_upper),
            )
        })
        .unwrap()
        .unwrap();
    let actual = download(gpu, &gpu_output);
    assert_tensor_close(&actual, &expected, tolerance(entry.dtype));
}

fn assert_dot_matches(
    cpu: &mut tenferro_cpu::CpuBackend,
    gpu: &mut CudaBackend,
    entry: OperationCapability,
) {
    let (lhs, rhs) = dot_inputs(entry.dtype);
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let expected = cpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();
    let gpu_lhs = upload(gpu, &lhs);
    let gpu_rhs = upload(gpu, &rhs);
    let gpu_output = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&gpu_lhs),
                TensorRead::from_tensor(&gpu_rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();
    let actual = download(gpu, &gpu_output);
    assert_tensor_close(&actual, &expected, tolerance(entry.dtype));
}

fn run_cpu_unary(
    cpu: &mut tenferro_cpu::CpuBackend,
    op: PrimitiveOpKind,
    input: &Tensor,
) -> Tensor {
    match op {
        PrimitiveOpKind::Neg => cpu
            .with_backend_session(|__s| __s.neg_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Conj => cpu
            .with_backend_session(|__s| __s.conj_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Abs => cpu
            .with_backend_session(|__s| __s.abs_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Sign => cpu
            .with_backend_session(|__s| __s.sign_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Exp => cpu
            .with_backend_session(|__s| __s.exp_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Log => cpu
            .with_backend_session(|__s| __s.log_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Sin => cpu
            .with_backend_session(|__s| __s.sin_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Cos => cpu
            .with_backend_session(|__s| __s.cos_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Tanh => cpu
            .with_backend_session(|__s| __s.tanh_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Sqrt => cpu
            .with_backend_session(|__s| __s.sqrt_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Rsqrt => cpu
            .with_backend_session(|__s| __s.rsqrt_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Expm1 => cpu
            .with_backend_session(|__s| {
                __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(input))
            })
            .unwrap(),
        PrimitiveOpKind::Log1p => cpu
            .with_backend_session(|__s| __s.log1p_read(TensorRead::from_tensor(input)))
            .unwrap(),
        PrimitiveOpKind::Erf => cpu
            .with_backend_session(|__s| __s.erf_read(TensorRead::from_tensor(input)))
            .unwrap(),
        _ => panic!("not a unary smoke op: {op:?}"),
    }
    .unwrap()
}

fn run_cpu_binary(
    cpu: &mut tenferro_cpu::CpuBackend,
    op: PrimitiveOpKind,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Tensor {
    match op {
        PrimitiveOpKind::Add => cpu
            .with_backend_session(|__s| {
                __s.add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
            })
            .unwrap(),
        PrimitiveOpKind::Sub => cpu
            .with_backend_session(|__s| {
                __s.sub_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
            })
            .unwrap(),
        PrimitiveOpKind::Mul => cpu
            .with_backend_session(|__s| {
                __s.mul_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
            })
            .unwrap(),
        PrimitiveOpKind::Div => cpu
            .with_backend_session(|__s| {
                __s.div_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
            })
            .unwrap(),
        PrimitiveOpKind::Rem => cpu.with_backend_session(|__s| __s.rem(lhs, rhs)).unwrap(),
        PrimitiveOpKind::Maximum => cpu
            .with_backend_session(|__s| {
                __s.maximum_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
            })
            .unwrap(),
        PrimitiveOpKind::Minimum => cpu
            .with_backend_session(|__s| {
                __s.minimum_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
            })
            .unwrap(),
        PrimitiveOpKind::Pow => cpu
            .with_backend_session(|__s| {
                __s.pow_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
            })
            .unwrap(),
        _ => panic!("not a binary smoke op: {op:?}"),
    }
    .unwrap()
}

fn run_cpu_reduction(
    cpu: &mut tenferro_cpu::CpuBackend,
    op: PrimitiveOpKind,
    input: &Tensor,
    axes: &[usize],
) -> Tensor {
    match op {
        PrimitiveOpKind::ReduceSum => cpu
            .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(input), axes))
            .unwrap(),
        PrimitiveOpKind::ReduceSumSquares => cpu
            .with_backend_session(|__s| {
                __s.reduce_sum_squares_read(TensorRead::from_tensor(input), axes)
            })
            .unwrap(),
        PrimitiveOpKind::ReduceProd => cpu
            .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(input), axes))
            .unwrap(),
        PrimitiveOpKind::ReduceMax => cpu
            .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(input), axes))
            .unwrap(),
        PrimitiveOpKind::ReduceMin => cpu
            .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(input), axes))
            .unwrap(),
        _ => panic!("not a reduction smoke op: {op:?}"),
    }
    .unwrap()
}

fn sample_tensor(dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => tensor_f32(vec![3], vec![1.25, 2.0, 3.5]),
        DType::F64 => tensor_f64(vec![3], vec![1.25, 2.0, 3.5]),
        DType::I32 => tensor_i32(vec![3], vec![1, -2, 3]),
        DType::I64 => tensor_i64(vec![3], vec![1, -2, 3]),
        DType::Bool => tensor_bool(vec![3], vec![true, false, true]),
        DType::C32 => tensor_c32(
            vec![3],
            vec![
                Complex32::new(1.0, 0.5),
                Complex32::new(2.0, -0.25),
                Complex32::new(3.0, 0.75),
            ],
        ),
        DType::C64 => tensor_c64(
            vec![3],
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -0.25),
                Complex64::new(3.0, 0.75),
            ],
        ),
        // Test fixtures cover the preset dtypes; an externally defined scalar has no
        // fixture and would change what these tests assert.
        DType::External(_) => unreachable!("test fixtures cover the preset dtypes"),
    }
}

fn sample_rhs_tensor(dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => tensor_f32(vec![3], vec![2.0, 3.0, 4.0]),
        DType::F64 => tensor_f64(vec![3], vec![2.0, 3.0, 4.0]),
        DType::I32 => tensor_i32(vec![3], vec![4, 5, 6]),
        DType::I64 => tensor_i64(vec![3], vec![4, 5, 6]),
        DType::Bool => tensor_bool(vec![3], vec![false, false, true]),
        DType::C32 => tensor_c32(
            vec![3],
            vec![
                Complex32::new(2.0, -0.5),
                Complex32::new(3.0, 0.25),
                Complex32::new(4.0, -0.75),
            ],
        ),
        DType::C64 => tensor_c64(
            vec![3],
            vec![
                Complex64::new(2.0, -0.5),
                Complex64::new(3.0, 0.25),
                Complex64::new(4.0, -0.75),
            ],
        ),
        // Test fixtures cover the preset dtypes; an externally defined scalar has no
        // fixture and would change what these tests assert.
        DType::External(_) => unreachable!("test fixtures cover the preset dtypes"),
    }
}

fn reduction_sample_tensor(dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => tensor_f32(vec![2, 3], vec![1.25, 2.0, 3.5, 4.0, 5.0, 6.0]),
        DType::F64 => tensor_f64(vec![2, 3], vec![1.25, 2.0, 3.5, 4.0, 5.0, 6.0]),
        DType::I32 => tensor_i32(vec![2, 3], vec![1, -2, 3, 4, -5, 6]),
        DType::I64 => tensor_i64(vec![2, 3], vec![1, -2, 3, 4, -5, 6]),
        DType::Bool => tensor_bool(vec![2, 3], vec![true, false, true, false, true, false]),
        DType::C32 => tensor_c32(
            vec![2, 3],
            vec![
                Complex32::new(1.0, 0.5),
                Complex32::new(2.0, -0.25),
                Complex32::new(3.0, 0.75),
                Complex32::new(4.0, -0.5),
                Complex32::new(5.0, 0.25),
                Complex32::new(6.0, -0.75),
            ],
        ),
        DType::C64 => tensor_c64(
            vec![2, 3],
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -0.25),
                Complex64::new(3.0, 0.75),
                Complex64::new(4.0, -0.5),
                Complex64::new(5.0, 0.25),
                Complex64::new(6.0, -0.75),
            ],
        ),
        // Test fixtures cover the preset dtypes; an externally defined scalar has no
        // fixture and would change what these tests assert.
        DType::External(_) => unreachable!("test fixtures cover the preset dtypes"),
    }
}

fn clamp_lower_tensor(dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => tensor_f32(vec![3], vec![1.5, 1.5, 1.5]),
        DType::F64 => tensor_f64(vec![3], vec![1.5, 1.5, 1.5]),
        _ => panic!("clamp smoke case only covers float dtypes"),
    }
}

fn clamp_upper_tensor(dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => tensor_f32(vec![3], vec![3.0, 3.0, 3.0]),
        DType::F64 => tensor_f64(vec![3], vec![3.0, 3.0, 3.0]),
        _ => panic!("clamp smoke case only covers float dtypes"),
    }
}

fn dot_inputs(dtype: DType) -> (Tensor, Tensor) {
    match dtype {
        DType::F32 => (
            tensor_f32(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]),
            tensor_f32(vec![2, 2], vec![5.0, 7.0, 6.0, 8.0]),
        ),
        DType::F64 => (
            tensor_f64(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]),
            tensor_f64(vec![2, 2], vec![5.0, 7.0, 6.0, 8.0]),
        ),
        DType::C32 => (
            tensor_c32(
                vec![2, 2],
                vec![
                    Complex32::new(1.0, 0.5),
                    Complex32::new(3.0, -0.5),
                    Complex32::new(2.0, 0.25),
                    Complex32::new(4.0, 0.75),
                ],
            ),
            tensor_c32(
                vec![2, 2],
                vec![
                    Complex32::new(5.0, -0.25),
                    Complex32::new(7.0, 0.5),
                    Complex32::new(6.0, 0.75),
                    Complex32::new(8.0, -0.5),
                ],
            ),
        ),
        DType::C64 => (
            tensor_c64(
                vec![2, 2],
                vec![
                    Complex64::new(1.0, 0.5),
                    Complex64::new(3.0, -0.5),
                    Complex64::new(2.0, 0.25),
                    Complex64::new(4.0, 0.75),
                ],
            ),
            tensor_c64(
                vec![2, 2],
                vec![
                    Complex64::new(5.0, -0.25),
                    Complex64::new(7.0, 0.5),
                    Complex64::new(6.0, 0.75),
                    Complex64::new(8.0, -0.5),
                ],
            ),
        ),
        _ => panic!("dot_general smoke case only covers float and complex dtypes"),
    }
}

fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => 1e-4,
        DType::F64 | DType::C64 => 1e-9,
        DType::I32 | DType::I64 | DType::Bool => 0.0,
        // Test fixtures cover the preset dtypes; an externally defined scalar has no
        // fixture and would change what these tests assert.
        DType::External(_) => unreachable!("test fixtures cover the preset dtypes"),
    }
}
