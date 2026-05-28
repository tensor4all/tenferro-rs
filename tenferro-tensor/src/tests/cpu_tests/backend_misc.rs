use super::*;

#[test]
fn test_reclaim_buffer_returns_host_buffer_to_pool() {
    let mut backend = CpuBackend::new();
    assert_eq!(backend.buffer_pool_len(), 0);
    let t = TensorElementwise::add(
        &mut backend,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0])),
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0])),
    )
    .unwrap();
    backend.reclaim_buffer(t);
    assert!(backend.buffer_pool_len() > 0);
}

#[test]
fn test_elementwise_add_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![4.0, 3.0, 2.0, 1.0],
    ));
    let out = backend.add(&lhs, &rhs).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 5.0);
    assert_eq!(get_f64(&out, &[3]), 5.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_structural_transpose_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let out = backend.transpose(&input, &[1, 0]).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0, 0]), 1.0);
    assert_eq!(get_f64(&out, &[1, 0]), 3.0);
    assert_eq!(get_f64(&out, &[0, 1]), 2.0);
    assert_eq!(get_f64(&out, &[1, 1]), 4.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_convert_acquires_output_from_dtype_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F32(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![1.25, 2.5, 3.75, 4.0],
    ));
    let out = backend.convert(&input, DType::F32).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f32(&out, &[0]), 1.25);
    assert_eq!(get_f32(&out, &[3]), 4.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_slice_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![0.0; 2],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let config = SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    };
    let out = backend.slice(&input, &config).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 2.0);
    assert_eq!(get_f64(&out, &[1]), 3.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_pad_acquires_and_zeroes_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![9.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let config = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    };
    let out = backend.pad(&input, &config).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 1.0);
    assert_eq!(get_f64(&out, &[2]), 2.0);
    assert_eq!(get_f64(&out, &[3]), 0.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_dynamic_update_slice_acquires_clone_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![9.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0, 1.0, 2.0, 3.0],
    ));
    let update = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![7.0, 8.0]));
    let starts = Tensor::I64(TypedTensor::from_vec_col_major(vec![1], vec![1]));
    let out = backend
        .dynamic_update_slice(&operand, &update, &starts)
        .unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 7.0);
    assert_eq!(get_f64(&out, &[2]), 8.0);
    assert_eq!(get_f64(&out, &[3]), 3.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_reclaim_buffer_covers_all_dtypes() {
    let mut backend = CpuBackend::new();
    let f32_t = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]));
    backend.reclaim_buffer(f32_t);
    let c32_t = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![1],
        vec![Complex32::new(1.0, 0.0)],
    ));
    backend.reclaim_buffer(c32_t);
    let c64_t = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![1],
        vec![Complex64::new(1.0, 0.0)],
    ));
    backend.reclaim_buffer(c64_t);
    assert!(backend.buffer_pool_len() >= 3);
}

#[test]
fn test_install_with_pool_preserves_buffers() {
    let mut backend = CpuBackend::with_threads(1);
    let t = TensorElementwise::add(
        &mut backend,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0])),
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0])),
    )
    .unwrap();
    assert_eq!(get_f64(&t, &[0]), 4.0);
    assert_eq!(get_f64(&t, &[1]), 6.0);
    assert_eq!(backend.buffer_pool_len(), 0);
}

#[test]
fn test_exec_session_read_reductions_and_reclaim_cover_typed_paths() {
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|exec| {
        let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
        let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
        let added = exec
            .add_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            .unwrap();
        assert_eq!(added.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

        let view_data = [2.0_f64, 3.0];
        let view_shape = [2usize];
        assert_eq!(
            exec.reduce_sum_read(
                TensorRead::from_view(TensorView::f64(&view_shape, &view_data).unwrap()),
                &[0],
            )
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
            &[5.0]
        );
        assert_eq!(
            exec.reduce_prod_read(TensorRead::from_tensor(&lhs), &[0])
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[2.0]
        );
        assert_eq!(
            exec.reduce_max_read(TensorRead::from_tensor(&rhs), &[0])
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[4.0]
        );
        assert_eq!(
            exec.reduce_min_read(TensorRead::from_tensor(&rhs), &[0])
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[3.0]
        );

        exec.reclaim_buffer(Tensor::F32(TypedTensor::from_vec_col_major(
            vec![1],
            vec![0.0_f32],
        )));
        exec.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
            vec![1],
            vec![0.0_f64],
        )));
        exec.reclaim_buffer(Tensor::I32(TypedTensor::from_vec_col_major(
            vec![1],
            vec![0_i32],
        )));
        exec.reclaim_buffer(Tensor::I64(TypedTensor::from_vec_col_major(
            vec![1],
            vec![0_i64],
        )));
        exec.reclaim_buffer(Tensor::Bool(TypedTensor::from_vec_col_major(
            vec![1],
            vec![false],
        )));
        exec.reclaim_buffer(Tensor::C32(TypedTensor::from_vec_col_major(
            vec![1],
            vec![Complex32::new(0.0, 0.0)],
        )));
        exec.reclaim_buffer(Tensor::C64(TypedTensor::from_vec_col_major(
            vec![1],
            vec![Complex64::new(0.0, 0.0)],
        )));
    });

    assert!(backend.buffer_pool_len() >= 7);
}

#[test]
fn test_default_backend_session_methods_cover_cache_fallbacks() {
    struct DefaultOnlyBackend;

    macro_rules! panic_backend_methods {
        ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
            $(
                fn $name(&mut self, $($arg: $argty),*) -> $ret {
                    $(let _ = &$arg;)*
                    panic!(concat!(stringify!($name), " should not be called by this test"))
                }
            )+
        };
    }

    impl BackendRuntimeCache for DefaultOnlyBackend {
        type RuntimeCache = ();
    }

    impl TensorElementwise for DefaultOnlyBackend {
        panic_backend_methods! {
        mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        neg(input: &Tensor) -> crate::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        abs(input: &Tensor) -> crate::Result<Tensor>;
        sign(input: &Tensor) -> crate::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
        }

        fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().add(lhs, rhs)
        }

        fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().conj(input)
        }
    }

    impl TensorAnalytic for DefaultOnlyBackend {
        panic_backend_methods! {
        exp(input: &Tensor) -> crate::Result<Tensor>;
        log(input: &Tensor) -> crate::Result<Tensor>;
        sin(input: &Tensor) -> crate::Result<Tensor>;
        cos(input: &Tensor) -> crate::Result<Tensor>;
        tanh(input: &Tensor) -> crate::Result<Tensor>;
        sqrt(input: &Tensor) -> crate::Result<Tensor>;
        rsqrt(input: &Tensor) -> crate::Result<Tensor>;
        pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        expm1(input: &Tensor) -> crate::Result<Tensor>;
        log1p(input: &Tensor) -> crate::Result<Tensor>;
        }
    }

    impl TensorStructural for DefaultOnlyBackend {
        panic_backend_methods! {
        transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
        convert(input: &Tensor, to: DType) -> crate::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        }
    }

    impl TensorReduction for DefaultOnlyBackend {
        fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_sum(input, axes)
        }

        fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_prod(input, axes)
        }

        fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_max(input, axes)
        }

        fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_min(input, axes)
        }
    }

    impl TensorIndexing for DefaultOnlyBackend {
        panic_backend_methods! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> crate::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> crate::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> crate::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> crate::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        }
    }

    impl TensorDot for DefaultOnlyBackend {
        fn dot_general(
            &mut self,
            lhs: &Tensor,
            rhs: &Tensor,
            config: &DotGeneralConfig,
        ) -> crate::Result<Tensor> {
            CpuBackend::new().dot_general(lhs, rhs, config)
        }
    }

    impl BackendCachedDot for DefaultOnlyBackend {}

    impl BackendSessionHost for DefaultOnlyBackend {}

    impl TensorDeviceTransfer for DefaultOnlyBackend {}

    impl TensorBuffer for DefaultOnlyBackend {}

    impl TensorFusion for DefaultOnlyBackend {}

    impl TensorBackend for DefaultOnlyBackend {}

    struct DefaultOnlyExec;

    impl TensorElementwise for DefaultOnlyExec {
        panic_backend_methods! {
        mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        neg(input: &Tensor) -> crate::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        abs(input: &Tensor) -> crate::Result<Tensor>;
        sign(input: &Tensor) -> crate::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
        }

        fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().add(lhs, rhs)
        }

        fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().conj(input)
        }
    }

    impl TensorAnalytic for DefaultOnlyExec {
        panic_backend_methods! {
        exp(input: &Tensor) -> crate::Result<Tensor>;
        log(input: &Tensor) -> crate::Result<Tensor>;
        sin(input: &Tensor) -> crate::Result<Tensor>;
        cos(input: &Tensor) -> crate::Result<Tensor>;
        tanh(input: &Tensor) -> crate::Result<Tensor>;
        sqrt(input: &Tensor) -> crate::Result<Tensor>;
        rsqrt(input: &Tensor) -> crate::Result<Tensor>;
        pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        expm1(input: &Tensor) -> crate::Result<Tensor>;
        log1p(input: &Tensor) -> crate::Result<Tensor>;
        }
    }

    impl TensorStructural for DefaultOnlyExec {
        panic_backend_methods! {
        transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
        convert(input: &Tensor, to: DType) -> crate::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        }
    }

    impl TensorReduction for DefaultOnlyExec {
        fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_sum(input, axes)
        }

        fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_prod(input, axes)
        }

        fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_max(input, axes)
        }

        fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_min(input, axes)
        }
    }

    impl TensorIndexing for DefaultOnlyExec {
        panic_backend_methods! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> crate::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> crate::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> crate::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> crate::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        }
    }

    impl TensorDot for DefaultOnlyExec {
        fn dot_general(
            &mut self,
            lhs: &Tensor,
            rhs: &Tensor,
            config: &DotGeneralConfig,
        ) -> crate::Result<Tensor> {
            CpuBackend::new().dot_general(lhs, rhs, config)
        }
    }

    impl SessionCachedDot for DefaultOnlyExec {}

    impl TensorBuffer for DefaultOnlyExec {
        fn reclaim_buffer(&mut self, _tensor: Tensor) {}
    }

    impl TensorFusion for DefaultOnlyExec {}

    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]);
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]);
    let one_shape = [1usize, 1];
    let lhs_data = [2.0_f64];
    let rhs_data = [3.0_f64];
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut backend = DefaultOnlyBackend;
    let mut cache = ();

    let add_read_tensor = TensorElementwise::add_read(
        &mut backend,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
    )
    .unwrap();
    assert_eq!(add_read_tensor.as_slice::<f64>().unwrap(), &[5.0]);
    let add_view_err = TensorElementwise::add_read(
        &mut backend,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_tensor(&rhs),
    )
    .unwrap_err();
    assert!(matches!(
        add_view_err,
        crate::Error::BackendFailure {
            op: "add",
            ref message,
        } if message.contains("borrowed tensor views")
    ));

    let reduce_input = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]);
    let reduce_view_shape = [2usize];
    let reduce_view_data = [2.0_f64, 3.0];
    assert_eq!(
        TensorReduction::reduce_sum_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[5.0]
    );
    assert_eq!(
        TensorReduction::reduce_prod_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[6.0]
    );
    assert_eq!(
        TensorReduction::reduce_max_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[3.0]
    );
    assert_eq!(
        TensorReduction::reduce_min_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[2.0]
    );
    for (op, err) in [
        (
            "reduce_sum",
            TensorReduction::reduce_sum_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
        (
            "reduce_prod",
            TensorReduction::reduce_prod_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
        (
            "reduce_max",
            TensorReduction::reduce_max_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
        (
            "reduce_min",
            TensorReduction::reduce_min_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
    ] {
        assert!(matches!(
            err,
            crate::Error::BackendFailure {
                op: actual_op,
                ref message,
            } if actual_op == op && message.contains("borrowed tensor views")
        ));
    }

    let direct = BackendCachedDot::dot_general_cached(
        &mut backend,
        &mut cache,
        Some(0),
        &lhs,
        &rhs,
        &config,
    )
    .unwrap();
    assert_eq!(direct.as_slice::<f64>().unwrap(), &[6.0]);

    let lhs_folded =
        TensorDot::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, false).unwrap();
    assert_eq!(lhs_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let both_folded =
        TensorDot::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, true).unwrap();
    assert_eq!(both_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let read_views = TensorDot::dot_general_read(
        &mut backend,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&one_shape, &rhs_data).unwrap()),
        &config,
    )
    .unwrap();
    assert_eq!(read_views.as_slice::<f64>().unwrap(), &[6.0]);

    let rhs_folded = BackendCachedDot::dot_general_with_conj_cached(
        &mut backend,
        &mut cache,
        Some(1),
        &lhs,
        &rhs,
        &config,
        false,
        true,
    )
    .unwrap();
    assert_eq!(rhs_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let uploaded = backend.upload_host_tensor(&lhs).unwrap();
    assert_eq!(uploaded.shape(), &[1, 1]);
    let downloaded = backend.download_to_host(&uploaded).unwrap();
    assert_eq!(downloaded.as_slice::<f64>().unwrap(), &[2.0]);
    backend.reclaim_buffer(downloaded);

    let fusion_plan = crate::backend::ElementwiseFusionPlan {
        dtype: DType::F64,
        n_inputs: 0,
        outputs: vec![],
        ops: vec![],
    };
    assert!(backend
        .execute_elementwise_fusion(&[], &fusion_plan)
        .unwrap()
        .is_none());

    let session_value =
        BackendSessionHost::with_backend_session_cached(&mut backend, &mut cache, |exec| {
            let cached = exec
                .dot_general_cached(Some(2), &lhs, &rhs, &config)
                .unwrap();
            let folded = exec
                .dot_general_with_conj_cached(Some(3), &lhs, &rhs, &config, true, false)
                .unwrap();
            cached.as_slice::<f64>().unwrap()[0] + folded.as_slice::<f64>().unwrap()[0]
        });
    assert_eq!(session_value, 12.0);

    let mut exec = DefaultOnlyExec;
    let exec_read_tensor = TensorDot::dot_general_read(
        &mut exec,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
    )
    .unwrap();
    assert_eq!(exec_read_tensor.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_read_views = TensorDot::dot_general_read(
        &mut exec,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&one_shape, &rhs_data).unwrap()),
        &config,
    )
    .unwrap();
    assert_eq!(exec_read_views.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_no_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, false, false).unwrap();
    assert_eq!(exec_no_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_lhs_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, true, false).unwrap();
    assert_eq!(exec_lhs_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_rhs_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, false, true).unwrap();
    assert_eq!(exec_rhs_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_both_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, true, true).unwrap();
    assert_eq!(exec_both_conj.as_slice::<f64>().unwrap(), &[6.0]);
}

#[test]
fn test_pool_backed_elementwise_public_paths_cover_dtypes_and_scalars() {
    let f32_scalar = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![2.0]));
    let c32_vec = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 1.0), Complex32::new(-3.0, 0.5)],
    ));
    assert_eq!(
        add(&f32_scalar, &c32_vec)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(3.0, 1.0)
    );
    assert_eq!(
        add(&c32_vec, &f32_scalar)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[1],
        Complex32::new(-1.0, 0.5)
    );
    assert_eq!(
        div(&f32_scalar, &c32_vec)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(1.0, -1.0)
    );
    assert_eq!(
        mul(&c32_vec, &f32_scalar)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(2.0, 2.0)
    );

    let f64_scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![4.0]));
    let c64_vec = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 2.0)],
    ));
    assert_c64_close(
        div(&c64_vec, &f64_scalar)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()[1],
        Complex64::new(0.0, 0.5),
    );

    assert!(neg(&Tensor::from_vec_col_major(vec![1], vec![1_i64]))
        .unwrap_err()
        .to_string()
        .contains("I64"));
    assert!(conj(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());
    assert!(abs(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());
    assert!(sign(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());

    let a = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
    ));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.0, 2.0), Complex64::new(5.0, 0.0)],
    ));
    assert_c64_close(
        get_c64(&maximum(&a, &b).unwrap(), &[0]),
        Complex64::new(3.0, 4.0),
    );
    assert_c64_close(
        get_c64(&minimum(&a, &b).unwrap(), &[0]),
        Complex64::new(0.0, 2.0),
    );
    assert!(get_bool(&compare(&a, &b, &CompareDir::Ge).unwrap(), &[0]));
    let pred = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, true]));
    assert_c64_close(
        get_c64(&select(&pred, &a, &b).unwrap(), &[1]),
        Complex64::new(1.0, 0.0),
    );
    assert_c64_close(
        get_c64(&clamp(&a, &b, &a).unwrap(), &[1]),
        Complex64::new(5.0, 0.0),
    );
}

#[test]
fn test_pool_backed_analytic_public_paths_cover_supported_dtypes() {
    let real = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 4.0]);
    assert_f64_close(
        crate::cpu::analytic::exp(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        1.0,
    );
    assert_f64_close(
        crate::cpu::analytic::sqrt(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[1],
        2.0,
    );
    assert_f64_close(
        crate::cpu::analytic::rsqrt(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[1],
        0.5,
    );
    assert_f64_close(
        crate::cpu::analytic::log1p(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        0.0,
    );
    assert_f64_close(
        crate::cpu::analytic::expm1(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        0.0,
    );

    let complex = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![1],
        vec![Complex64::new(1.0, 0.0)],
    ));
    assert_c64_close(
        crate::cpu::analytic::log(&complex)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()[0],
        Complex64::new(0.0, 0.0),
    );
    assert!(crate::cpu::analytic::sin(&complex).is_ok());
    assert!(crate::cpu::analytic::cos(&complex).is_ok());
    assert!(crate::cpu::analytic::tanh(&complex).is_ok());

    let base = Tensor::from_vec_col_major(vec![2], vec![2.0_f32, 3.0]);
    let exponent = Tensor::from_vec_col_major(vec![2], vec![3.0_f32, 2.0]);
    assert_eq!(
        crate::cpu::analytic::pow(&base, &exponent)
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[8.0, 9.0]
    );
    assert!(crate::cpu::analytic::exp(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());
    assert!(crate::cpu::analytic::pow(&real, &base).is_err());
}

#[test]
fn test_pool_backed_structural_public_paths_cover_dispatch_and_helpers() {
    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let transposed = transpose(&matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);

    let typed = TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]);
    let typed_t = crate::cpu::structural::typed_transpose(&typed, &[1, 0]).unwrap();
    assert_eq!(typed_t.host_data(), &[1, 3, 2, 4]);

    let row = TypedTensor::from_vec_col_major(vec![1, 2], vec![5.0_f32, 6.0]);
    let typed_b = crate::cpu::structural::typed_broadcast_in_dim(&row, &[2, 2], &[0, 1]).unwrap();
    assert_eq!(typed_b.host_data(), &[5.0, 5.0, 6.0, 6.0]);

    let scalar = Tensor::from_vec_col_major(vec![], vec![7.0_f64]);
    let broadcasted = broadcast_in_dim(&scalar, &[2, 2], &[]).unwrap();
    assert_eq!(
        broadcasted.as_slice::<f64>().unwrap(),
        &[7.0, 7.0, 7.0, 7.0]
    );

    let i64_matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]);
    let as_c64 = crate::cpu::structural::convert(&i64_matrix, DType::C64).unwrap();
    assert_eq!(as_c64.dtype(), DType::C64);
    let as_f32 = crate::cpu::structural::convert(&as_c64, DType::F32).unwrap();
    assert_eq!(as_f32.dtype(), DType::F32);
    let as_c32 = crate::cpu::structural::convert(&matrix, DType::C32).unwrap();
    assert_eq!(as_c32.dtype(), DType::C32);
    let as_i64 = crate::cpu::structural::convert(&as_c32, DType::I64).unwrap();
    assert_eq!(as_i64.as_slice::<i64>().unwrap(), &[1, 2, 3, 4]);

    let diag = extract_diagonal(&matrix, 0, 1).unwrap();
    assert_eq!(diag.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    let embedded = embed_diagonal(&diag, 0, 1).unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(embedded.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 4.0]);

    let typed_diag = crate::cpu::structural::typed_extract_diagonal(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
        0,
        1,
    )
    .unwrap();
    assert_eq!(typed_diag.host_data(), &[1.0, 4.0]);
    let typed_embedded = crate::cpu::structural::typed_embed_diagonal(&typed_diag, 0, 1).unwrap();
    assert_eq!(typed_embedded.host_data(), &[1.0, 0.0, 0.0, 4.0]);

    let lower = tril(&matrix, 0).unwrap();
    assert_eq!(lower.as_slice::<f64>().unwrap(), &[1.0, 2.0, 0.0, 4.0]);
    let upper = triu(&matrix, 0).unwrap();
    assert_eq!(upper.as_slice::<f64>().unwrap(), &[1.0, 0.0, 3.0, 4.0]);
    let typed_lower = crate::cpu::structural::typed_tril(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]),
        0,
    )
    .unwrap();
    assert_eq!(typed_lower.host_data(), &[1, 2, 0, 4]);
    let typed_upper = crate::cpu::structural::typed_triu(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]),
        0,
    )
    .unwrap();
    assert_eq!(typed_upper.host_data(), &[1, 0, 3, 4]);
    assert!(crate::cpu::structural::typed_tril(
        &TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        0
    )
    .is_err());

    let c32_matrix = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
    ));
    assert_eq!(transpose(&c32_matrix, &[1, 0]).unwrap().dtype(), DType::C32);
    assert_eq!(tril(&c32_matrix, 0).unwrap().dtype(), DType::C32);
}
