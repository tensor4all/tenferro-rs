mod organization;

use num_traits::{One, Zero};
use tenferro_algebra::Scalar;
use tenferro_prims::{SemiringCoreDescriptor, TensorSemiringCore};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;

fn tensor_from_slice<T: Scalar>(data: &[T], dims: &[usize]) -> Tensor<T> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn helper_paths_cover_zero_dim_iteration_and_scale_output_branches() {
    let mut seen = 0usize;
    for_each_index(&[0, 3], |_| {
        seen += 1;
    });
    assert_eq!(seen, 0);

    assert_eq!(unflatten_index(5, &[2, 3]), vec![1, 2]);

    let mut output = tensor_from_slice(&[MaxPlus(1.0), MaxPlus(2.0)], &[2]);
    {
        let mut view = tensor_to_view_mut(&mut output).unwrap();
        scale_output(&mut view, MaxPlus::<f64>::zero());
    }
    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[MaxPlus::<f64>::zero(); 2]
    );

    {
        let mut view = tensor_to_view_mut(&mut output).unwrap();
        scale_output(&mut view, MaxPlus(3.0));
    }
    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[MaxPlus::<f64>::zero(); 2]
    );
}

#[test]
fn fallback_batched_gemm_and_make_contiguous_slow_path_are_covered() {
    let a = tensor_from_slice(
        &[
            MaxPlus(1.0),
            MaxPlus(2.0),
            MaxPlus(3.0),
            MaxPlus(4.0),
            MaxPlus(5.0),
            MaxPlus(6.0),
        ],
        &[2, 3],
    );
    let b = tensor_from_slice(
        &[
            MaxPlus(1.0),
            MaxPlus(0.0),
            MaxPlus(2.0),
            MaxPlus(3.0),
            MaxPlus(4.0),
            MaxPlus(5.0),
        ],
        &[3, 2],
    );
    let mut c = Tensor::zeros(
        &[2, 2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    {
        let a_view = tensor_to_view(&a).unwrap();
        let b_view = tensor_to_view(&b).unwrap();
        let mut c_view = tensor_to_view_mut(&mut c).unwrap();
        execute_batched_gemm_fallback(
            MaxPlus::one(),
            &[&a_view, &b_view],
            MaxPlus::zero(),
            &mut c_view,
            &[],
            2,
            2,
            3,
        )
        .unwrap();
    }
    let got = c.buffer().as_slice().unwrap();
    assert_eq!(got[0], MaxPlus(7.0));
    assert_eq!(got[1], MaxPlus(8.0));
    assert_eq!(got[2], MaxPlus(10.0));
    assert_eq!(got[3], MaxPlus(11.0));

    let input = a.permute(&[1, 0]).unwrap();
    let mut output = tensor_from_slice(
        &[
            MaxPlus(10.0),
            MaxPlus(20.0),
            MaxPlus(30.0),
            MaxPlus(40.0),
            MaxPlus(50.0),
            MaxPlus(60.0),
        ],
        &[3, 2],
    );
    {
        let in_view = tensor_to_view(&input).unwrap();
        let mut out_view = tensor_to_view_mut(&mut output).unwrap();
        execute_make_contiguous(MaxPlus(2.0), &in_view, MaxPlus(3.0), &mut out_view).unwrap();
    }
    let got = output.buffer().as_slice().unwrap();
    assert_eq!(
        got,
        &[
            MaxPlus(13.0),
            MaxPlus(23.0),
            MaxPlus(33.0),
            MaxPlus(43.0),
            MaxPlus(53.0),
            MaxPlus(63.0),
        ]
    );
}

#[test]
fn validation_helpers_reject_duplicate_modes_and_rank_mismatches() {
    let dup_reduce = tropical_plan::<MaxPlus<f64>>(
        &SemiringCoreDescriptor::ReduceAdd {
            modes_a: vec![0, 0],
            modes_c: vec![0],
        },
        &[&[2, 2], &[2]],
    );
    assert!(matches!(dup_reduce, Err(Error::InvalidArgument(_))));

    let bad_pair = tropical_plan::<MaxPlus<f64>>(
        &SemiringCoreDescriptor::Trace {
            modes_a: vec![0, 1],
            modes_c: vec![],
            paired: vec![(0, 0)],
        },
        &[&[2, 2], &[]],
    );
    assert!(matches!(bad_pair, Err(Error::InvalidArgument(_))));

    let bad_gemm = tropical_plan::<MaxPlus<f64>>(
        &SemiringCoreDescriptor::BatchedGemm {
            batch_dims: vec![],
            m: 2,
            n: 2,
            k: 2,
        },
        &[&[2, 3], &[2, 2], &[2, 2]],
    );
    assert!(matches!(bad_gemm, Err(Error::InvalidArgument(_))));
}

#[test]
fn execute_helpers_report_invalid_axis_contracts() {
    let input = tensor_from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
    );
    let mut scalar_out = tensor_from_slice(&[MaxPlus::<f64>::zero()], &[]);
    let input_view = tensor_to_view(&input).unwrap();

    {
        let mut out_view = tensor_to_view_mut(&mut scalar_out).unwrap();
        let err = execute_trace(
            MaxPlus::one(),
            &input_view,
            MaxPlus::zero(),
            &mut out_view,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    let mut matrix_out = Tensor::zeros(
        &[2, 2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    {
        let mut out_view = tensor_to_view_mut(&mut matrix_out).unwrap();
        let err = execute_anti_trace(
            MaxPlus::one(),
            &input_view,
            MaxPlus::zero(),
            &mut out_view,
            &[(0, 1)],
            &[2],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
    {
        let mut out_view = tensor_to_view_mut(&mut matrix_out).unwrap();
        let err = execute_anti_diag(
            MaxPlus::one(),
            &input_view,
            MaxPlus::zero(),
            &mut out_view,
            &[(0, 3)],
            &[0, 1],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}

#[test]
fn tropical_execute_and_family_impls_cover_error_and_fast_path_contracts() {
    let plan = TropicalPlan::<MaxPlus<f64>>::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
        _marker: PhantomData,
    };
    let input = tensor_from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
    );
    let input_view = tensor_to_view(&input).unwrap();
    let mut output = Tensor::zeros(
        &[2, 2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    {
        let mut out_view = tensor_to_view_mut(&mut output).unwrap();
        let err = tropical_execute(
            &plan,
            MaxPlus::one(),
            &[&input_view],
            MaxPlus::zero(),
            &mut out_view,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    let mut ctx = CpuContext::new(1);
    let fast_err = <CpuBackend as TensorSemiringFastPath<MaxPlusAlgebra<f64>>>::plan(
        &mut ctx,
        &SemiringFastPathDescriptor::Contract {
            modes_a: vec![0],
            modes_b: vec![0],
            modes_c: vec![0],
        },
        &[&[2], &[2], &[2]],
    )
    .unwrap_err();
    assert!(matches!(fast_err, Error::InvalidArgument(_)));
    assert!(!<CpuBackend as TensorSemiringFastPath<
        MaxPlusAlgebra<f64>,
    >>::has_fast_path(
        SemiringFastPathDescriptor::ElementwiseBinary {
            op: tenferro_prims::SemiringBinaryOp::Mul,
        },
    ));

    let make_plan = <CpuBackend as TensorSemiringCore<MaxPlusAlgebra<f64>>>::plan(
        &mut ctx,
        &SemiringCoreDescriptor::MakeContiguous,
        &[&[2, 2], &[2, 2]],
    )
    .unwrap();
    <CpuBackend as TensorSemiringCore<MaxPlusAlgebra<f64>>>::execute(
        &mut ctx,
        &make_plan,
        MaxPlus::one(),
        &[&input],
        MaxPlus::zero(),
        &mut output,
    )
    .unwrap();
}
