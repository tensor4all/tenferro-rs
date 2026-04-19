// Test file exercises the deprecated legacy `eval_semiring_ir` alongside the
// mainline `eval_exec_ir`; suppress the deprecation warnings at module scope
// rather than annotating each test.
#![allow(deprecated)]

use num_complex::Complex64;
use tenferro::error::Error;
use tenferro::exec::{eval_exec_ir, eval_semiring_ir, ExecInstruction, ExecOp, ExecProgram};
use tenferro_algebra::Standard;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor, TensorBackend, TypedTensor,
};

fn dim_shape(shape: &[usize]) -> Vec<DimExpr> {
    DimExpr::from_concrete(shape)
}

fn scalar_tensor(value: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(vec![], vec![value]))
}

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn scalar_value(tensor: &Tensor) -> f64 {
    match tensor {
        Tensor::F64(inner) => inner.host_data()[0],
        other => panic!("expected scalar f64 tensor, got {other:?}"),
    }
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        other => panic!("expected f64 tensor, got {other:?}"),
    }
}

fn scalar_c64_value(tensor: &Tensor) -> Complex64 {
    match tensor {
        Tensor::C64(inner) => inner.host_data()[0],
        other => panic!("expected scalar c64 tensor, got {other:?}"),
    }
}

fn typed_scalar(value: f64) -> TypedTensor<f64> {
    TypedTensor::from_vec(vec![], vec![value])
}

fn gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}

fn pad_config() -> PadConfig {
    PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    }
}

fn single_instruction_program(op: ExecOp, n_inputs: usize) -> ExecProgram {
    ExecProgram {
        instructions: vec![ExecInstruction {
            op,
            input_slots: (0..n_inputs).collect(),
            output_slots: vec![n_inputs],
            dtype: DType::F64,
            output_shapes: vec![Vec::new()],
            last_use: vec![false; n_inputs],
        }],
        input_slots: (0..n_inputs).collect(),
        output_slots: vec![n_inputs],
        n_slots: n_inputs + 1,
    }
}

#[derive(Default)]
struct FakeTensorBackend {
    calls: Vec<&'static str>,
    error_on: Option<&'static str>,
    reclaimed: usize,
}

impl FakeTensorBackend {
    fn result(&mut self, name: &'static str, value: f64) -> tenferro_tensor::Result<Tensor> {
        self.calls.push(name);
        if self.error_on == Some(name) {
            return Err(tenferro_tensor::Error::BackendFailure {
                op: name,
                message: "injected failure".into(),
            });
        }
        Ok(scalar_tensor(value))
    }
}

impl TensorBackend for FakeTensorBackend {
    fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("add", 1.0)
    }
    fn mul(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("mul", 2.0)
    }
    fn neg(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("neg", 3.0)
    }
    fn conj(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("conj", 4.0)
    }
    fn div(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("div", 5.0)
    }
    fn abs(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("abs", 6.0)
    }
    fn sign(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("sign", 7.0)
    }
    fn maximum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("maximum", 8.0)
    }
    fn minimum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("minimum", 9.0)
    }
    fn compare(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _dir: &CompareDir,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("compare", 10.0)
    }
    fn select(
        &mut self,
        _pred: &Tensor,
        _on_true: &Tensor,
        _on_false: &Tensor,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("select", 11.0)
    }
    fn clamp(
        &mut self,
        _input: &Tensor,
        _lower: &Tensor,
        _upper: &Tensor,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("clamp", 12.0)
    }
    fn exp(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("exp", 13.0)
    }
    fn log(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("log", 14.0)
    }
    fn sin(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("sin", 15.0)
    }
    fn cos(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("cos", 16.0)
    }
    fn tanh(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("tanh", 17.0)
    }
    fn sqrt(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("sqrt", 18.0)
    }
    fn rsqrt(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("rsqrt", 19.0)
    }
    fn pow(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("pow", 20.0)
    }
    fn expm1(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("expm1", 21.0)
    }
    fn log1p(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("log1p", 22.0)
    }
    fn transpose(&mut self, _input: &Tensor, _perm: &[usize]) -> tenferro_tensor::Result<Tensor> {
        self.result("transpose", 23.0)
    }
    fn reshape(&mut self, _input: &Tensor, _shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        self.result("reshape", 24.0)
    }
    fn broadcast_in_dim(
        &mut self,
        _input: &Tensor,
        _shape: &[usize],
        _dims: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("broadcast_in_dim", 25.0)
    }
    fn convert(&mut self, _input: &Tensor, _to: DType) -> tenferro_tensor::Result<Tensor> {
        self.result("convert", 25.5)
    }
    fn extract_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("extract_diagonal", 26.0)
    }
    fn embed_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("embed_diagonal", 27.0)
    }
    fn tril(&mut self, _input: &Tensor, _k: i64) -> tenferro_tensor::Result<Tensor> {
        self.result("tril", 27.5)
    }
    fn triu(&mut self, _input: &Tensor, _k: i64) -> tenferro_tensor::Result<Tensor> {
        self.result("triu", 27.75)
    }
    fn reduce_sum(&mut self, _input: &Tensor, _axes: &[usize]) -> tenferro_tensor::Result<Tensor> {
        self.result("reduce_sum", 28.0)
    }
    fn reduce_prod(&mut self, _input: &Tensor, _axes: &[usize]) -> tenferro_tensor::Result<Tensor> {
        self.result("reduce_prod", 29.0)
    }
    fn reduce_max(&mut self, _input: &Tensor, _axes: &[usize]) -> tenferro_tensor::Result<Tensor> {
        self.result("reduce_max", 30.0)
    }
    fn reduce_min(&mut self, _input: &Tensor, _axes: &[usize]) -> tenferro_tensor::Result<Tensor> {
        self.result("reduce_min", 31.0)
    }
    fn dot_general(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("dot_general", 32.0)
    }
    fn gather(
        &mut self,
        _operand: &Tensor,
        _start_indices: &Tensor,
        _config: &GatherConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("gather", 33.0)
    }
    fn scatter(
        &mut self,
        _operand: &Tensor,
        _scatter_indices: &Tensor,
        _updates: &Tensor,
        _config: &ScatterConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("scatter", 34.0)
    }
    fn slice(&mut self, _input: &Tensor, _config: &SliceConfig) -> tenferro_tensor::Result<Tensor> {
        self.result("slice", 35.0)
    }
    fn dynamic_slice(
        &mut self,
        _input: &Tensor,
        _starts: &Tensor,
        _slice_sizes: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("dynamic_slice", 36.0)
    }
    fn pad(&mut self, _input: &Tensor, _config: &PadConfig) -> tenferro_tensor::Result<Tensor> {
        self.result("pad", 37.0)
    }
    fn concatenate(
        &mut self,
        _inputs: &[&Tensor],
        _axis: usize,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("concatenate", 38.0)
    }
    fn reverse(&mut self, _input: &Tensor, _axes: &[usize]) -> tenferro_tensor::Result<Tensor> {
        self.result("reverse", 39.0)
    }
    fn cholesky(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("cholesky", 40.0)
    }
    fn triangular_solve(
        &mut self,
        _a: &Tensor,
        _b: &Tensor,
        _left_side: bool,
        _lower: bool,
        _transpose_a: bool,
        _unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        self.result("triangular_solve", 40.5)
    }
    fn lu(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        self.calls.push("lu");
        Ok(vec![
            scalar_tensor(40.75),
            scalar_tensor(41.0),
            scalar_tensor(41.25),
            scalar_tensor(41.5),
        ])
    }
    fn svd(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        self.calls.push("svd");
        Ok(vec![scalar_tensor(42.0), scalar_tensor(42.5)])
    }
    fn qr(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        self.calls.push("qr");
        Ok(vec![scalar_tensor(43.0), scalar_tensor(43.5)])
    }
    fn eigh(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        self.calls.push("eigh");
        Ok(vec![scalar_tensor(44.0), scalar_tensor(44.5)])
    }
    fn eig(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        self.calls.push("eig");
        Ok(vec![
            Tensor::C64(TypedTensor::from_vec(
                vec![],
                vec![Complex64::new(45.0, 0.5)],
            )),
            Tensor::C64(TypedTensor::from_vec(
                vec![],
                vec![Complex64::new(45.5, -0.5)],
            )),
        ])
    }
    fn solve(&mut self, _a: &Tensor, _b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.result("solve", 44.0)
    }

    fn reclaim_buffer(&mut self, _tensor: Tensor) {
        self.reclaimed += 1;
    }
}

#[test]
fn eval_exec_ir_dispatches_tensor_ops_to_backend_methods() {
    let cases = vec![
        (ExecOp::Transpose { perm: vec![0] }, 1, "transpose", 23.0),
        (
            ExecOp::Reshape {
                shape: dim_shape(&[1]),
            },
            1,
            "reshape",
            24.0,
        ),
        (
            ExecOp::BroadcastInDim {
                shape: dim_shape(&[1]),
                dims: vec![0],
            },
            1,
            "broadcast_in_dim",
            25.0,
        ),
        (ExecOp::Convert { to: DType::C64 }, 1, "convert", 25.5),
        (
            ExecOp::DotGeneral(DotGeneralConfig {
                lhs_contracting_dims: vec![0],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            }),
            2,
            "dot_general",
            32.0,
        ),
        (ExecOp::ReduceSum { axes: vec![0] }, 1, "reduce_sum", 28.0),
        (
            ExecOp::ExtractDiag {
                axis_a: 0,
                axis_b: 1,
            },
            1,
            "extract_diagonal",
            26.0,
        ),
        (
            ExecOp::EmbedDiag {
                axis_a: 0,
                axis_b: 1,
            },
            1,
            "embed_diagonal",
            27.0,
        ),
        (ExecOp::Tril { k: -1 }, 1, "tril", 27.5),
        (ExecOp::Triu { k: 1 }, 1, "triu", 27.75),
        (ExecOp::Add, 2, "add", 1.0),
        (ExecOp::Multiply, 2, "mul", 2.0),
        (ExecOp::Negate, 1, "neg", 3.0),
        (ExecOp::Conj, 1, "conj", 4.0),
        (ExecOp::Divide, 2, "div", 5.0),
        (ExecOp::Abs, 1, "abs", 6.0),
        (ExecOp::Sign, 1, "sign", 7.0),
        (ExecOp::Maximum, 2, "maximum", 8.0),
        (ExecOp::Minimum, 2, "minimum", 9.0),
        (ExecOp::Compare(CompareDir::Eq), 2, "compare", 10.0),
        (ExecOp::Select, 3, "select", 11.0),
        (ExecOp::Clamp, 3, "clamp", 12.0),
        (ExecOp::Exp, 1, "exp", 13.0),
        (ExecOp::Log, 1, "log", 14.0),
        (ExecOp::Sin, 1, "sin", 15.0),
        (ExecOp::Cos, 1, "cos", 16.0),
        (ExecOp::Tanh, 1, "tanh", 17.0),
        (ExecOp::Sqrt, 1, "sqrt", 18.0),
        (ExecOp::Rsqrt, 1, "rsqrt", 19.0),
        (ExecOp::Pow, 2, "pow", 20.0),
        (ExecOp::Expm1, 1, "expm1", 21.0),
        (ExecOp::Log1p, 1, "log1p", 22.0),
        (ExecOp::Gather(gather_config()), 2, "gather", 33.0),
        (ExecOp::Scatter(scatter_config()), 3, "scatter", 34.0),
        (
            ExecOp::Slice(SliceConfig {
                starts: vec![0],
                limits: vec![1],
                strides: vec![1],
            }),
            1,
            "slice",
            35.0,
        ),
        (
            ExecOp::DynamicSlice {
                slice_sizes: vec![1],
            },
            2,
            "dynamic_slice",
            36.0,
        ),
        (ExecOp::Pad(pad_config()), 1, "pad", 37.0),
        (ExecOp::Concatenate { axis: 0 }, 2, "concatenate", 38.0),
        (ExecOp::Reverse { axes: vec![0] }, 1, "reverse", 39.0),
        (ExecOp::ReduceProd { axes: vec![0] }, 1, "reduce_prod", 29.0),
        (ExecOp::ReduceMax { axes: vec![0] }, 1, "reduce_max", 30.0),
        (ExecOp::ReduceMin { axes: vec![0] }, 1, "reduce_min", 31.0),
        (ExecOp::Cholesky, 1, "cholesky", 40.0),
        (
            ExecOp::TriangularSolve {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            },
            2,
            "triangular_solve",
            40.5,
        ),
    ];

    for (op, n_inputs, expected_call, expected_value) in cases {
        let mut backend = FakeTensorBackend::default();
        let program = single_instruction_program(op, n_inputs);
        let inputs = (0..n_inputs)
            .map(|idx| scalar_tensor(idx as f64 + 1.0))
            .collect();
        let outputs = eval_exec_ir(&mut backend, &program, inputs).unwrap();

        assert_eq!(backend.calls, vec![expected_call]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(scalar_value(&outputs[0]), expected_value);
    }
}

#[test]
fn eval_exec_ir_executes_nary_einsum_via_nested_program() {
    let mut backend = CpuBackend::new();
    let program = single_instruction_program(
        ExecOp::NaryEinsum {
            subscripts: "ij,jk->ik".into(),
        },
        2,
    );
    let inputs = vec![
        f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ];

    let outputs = eval_exec_ir(&mut backend, &program, inputs).unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(f64_data(&outputs[0]), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn eval_exec_ir_materializes_constant_scalars_without_backend_dispatch() {
    let mut backend = FakeTensorBackend::default();
    let program = ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Constant {
                dtype: DType::F64,
                bytes: 2.5_f64.to_le_bytes().to_vec(),
            },
            input_slots: vec![],
            output_slots: vec![0],
            dtype: DType::F64,
            output_shapes: vec![Vec::new()],
            last_use: vec![],
        }],
        input_slots: vec![],
        output_slots: vec![0],
        n_slots: 1,
    };

    let outputs = eval_exec_ir(&mut backend, &program, vec![]).unwrap();

    assert!(backend.calls.is_empty());
    assert_eq!(outputs.len(), 1);
    assert_eq!(scalar_value(&outputs[0]), 2.5);
}

#[test]
fn eval_exec_ir_materializes_complex_constants() {
    let mut backend = FakeTensorBackend::default();
    let value = Complex64::new(1.5, -2.0);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&value.re.to_le_bytes());
    bytes.extend_from_slice(&value.im.to_le_bytes());
    let program = ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Constant {
                dtype: DType::C64,
                bytes,
            },
            input_slots: vec![],
            output_slots: vec![0],
            dtype: DType::C64,
            output_shapes: vec![Vec::new()],
            last_use: vec![],
        }],
        input_slots: vec![],
        output_slots: vec![0],
        n_slots: 1,
    };

    let outputs = eval_exec_ir(&mut backend, &program, vec![]).unwrap();

    assert!(backend.calls.is_empty());
    assert_eq!(outputs.len(), 1);
    assert_eq!(scalar_c64_value(&outputs[0]), value);
}

#[test]
fn eval_exec_ir_propagates_backend_errors() {
    let mut backend = FakeTensorBackend {
        calls: Vec::new(),
        error_on: Some("add"),
        reclaimed: 0,
    };
    let err = eval_exec_ir(
        &mut backend,
        &single_instruction_program(ExecOp::Add, 2),
        vec![scalar_tensor(1.0), scalar_tensor(2.0)],
    )
    .unwrap_err();

    assert_eq!(backend.calls, vec!["add"]);
    assert!(matches!(
        err,
        Error::TensorRuntime(tenferro_tensor::Error::BackendFailure { op: "add", .. })
    ));
}

#[test]
fn eval_exec_ir_reports_missing_slots_as_runtime_errors() {
    let mut backend = FakeTensorBackend::default();
    let program = ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Add,
            input_slots: vec![0, 1],
            output_slots: vec![2],
            dtype: DType::F64,
            output_shapes: vec![Vec::new()],
            last_use: vec![false, false],
        }],
        input_slots: vec![0],
        output_slots: vec![2],
        n_slots: 3,
    };

    let err = eval_exec_ir(&mut backend, &program, vec![scalar_tensor(1.0)]).unwrap_err();

    assert!(backend.calls.is_empty());
    assert!(matches!(
        err,
        Error::TensorRuntime(tenferro_tensor::Error::MissingValue { slot: 1 })
    ));
}

fn multi_output_program(op: ExecOp, n_inputs: usize, n_outputs: usize) -> ExecProgram {
    let output_slots: Vec<usize> = (n_inputs..n_inputs + n_outputs).collect();
    ExecProgram {
        instructions: vec![ExecInstruction {
            op,
            input_slots: (0..n_inputs).collect(),
            output_slots: output_slots.clone(),
            dtype: DType::F64,
            output_shapes: vec![Vec::new(); n_outputs],
            last_use: vec![false; n_inputs],
        }],
        input_slots: (0..n_inputs).collect(),
        output_slots,
        n_slots: n_inputs + n_outputs,
    }
}

#[test]
fn eval_exec_ir_dispatches_multi_output_linalg_ops() {
    let mut backend = FakeTensorBackend::default();

    // LU: 1 input, 4 outputs
    let program = multi_output_program(ExecOp::Lu, 1, 4);
    let outputs = eval_exec_ir(&mut backend, &program, vec![scalar_tensor(1.0)]).unwrap();
    assert_eq!(backend.calls, vec!["lu"]);
    assert_eq!(outputs.len(), 4);
    assert_eq!(scalar_value(&outputs[0]), 40.75);
    assert_eq!(scalar_value(&outputs[1]), 41.0);
    assert_eq!(scalar_value(&outputs[2]), 41.25);
    assert_eq!(scalar_value(&outputs[3]), 41.5);

    backend.calls.clear();

    // SVD: 1 input, 2 outputs (fake returns 2)
    let program = multi_output_program(ExecOp::Svd { eps: 1e-10 }, 1, 2);
    let outputs = eval_exec_ir(&mut backend, &program, vec![scalar_tensor(1.0)]).unwrap();
    assert_eq!(backend.calls, vec!["svd"]);
    assert_eq!(outputs.len(), 2);
    assert_eq!(scalar_value(&outputs[0]), 42.0);
    assert_eq!(scalar_value(&outputs[1]), 42.5);

    backend.calls.clear();

    // QR: 1 input, 2 outputs
    let program = multi_output_program(ExecOp::Qr, 1, 2);
    let outputs = eval_exec_ir(&mut backend, &program, vec![scalar_tensor(1.0)]).unwrap();
    assert_eq!(backend.calls, vec!["qr"]);
    assert_eq!(outputs.len(), 2);
    assert_eq!(scalar_value(&outputs[0]), 43.0);
    assert_eq!(scalar_value(&outputs[1]), 43.5);

    backend.calls.clear();

    // Eigh: 1 input, 2 outputs
    let program = multi_output_program(ExecOp::Eigh { eps: 1e-10 }, 1, 2);
    let outputs = eval_exec_ir(&mut backend, &program, vec![scalar_tensor(1.0)]).unwrap();
    assert_eq!(backend.calls, vec!["eigh"]);
    assert_eq!(outputs.len(), 2);
    assert_eq!(scalar_value(&outputs[0]), 44.0);
    assert_eq!(scalar_value(&outputs[1]), 44.5);

    backend.calls.clear();

    // Eig: 1 input, 2 outputs
    let program = multi_output_program(ExecOp::Eig, 1, 2);
    let outputs = eval_exec_ir(&mut backend, &program, vec![scalar_tensor(1.0)]).unwrap();
    assert_eq!(backend.calls, vec!["eig"]);
    assert_eq!(outputs.len(), 2);
    assert_eq!(scalar_c64_value(&outputs[0]), Complex64::new(45.0, 0.5));
    assert_eq!(scalar_c64_value(&outputs[1]), Complex64::new(45.5, -0.5));
}

#[test]
fn eval_exec_ir_reclaims_last_use_host_buffers() {
    let program = ExecProgram {
        instructions: vec![
            ExecInstruction {
                op: ExecOp::Add,
                input_slots: vec![0, 1],
                output_slots: vec![2],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::Negate,
                input_slots: vec![2],
                output_slots: vec![3],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                last_use: vec![true],
            },
        ],
        input_slots: vec![0, 1],
        output_slots: vec![3],
        n_slots: 4,
    };

    let mut backend = FakeTensorBackend::default();
    let outputs = eval_exec_ir(
        &mut backend,
        &program,
        vec![scalar_tensor(1.0), scalar_tensor(2.0)],
    )
    .unwrap();

    assert_eq!(backend.calls, vec!["add", "neg"]);
    assert_eq!(outputs.len(), 1);
    assert_eq!(scalar_value(&outputs[0]), 3.0);
    assert_eq!(backend.reclaimed, 3);
}

#[test]
fn eval_semiring_ir_executes_semiring_structural_and_gemm_ops() {
    let mut backend = CpuBackend::new();

    let add_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(ExecOp::Add, 2),
        vec![typed_scalar(2.0), typed_scalar(3.0)],
    )
    .unwrap();
    assert_eq!(add_out[0].host_data(), &[5.0]);

    let mul_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(ExecOp::Multiply, 2),
        vec![typed_scalar(2.0), typed_scalar(3.0)],
    )
    .unwrap();
    assert_eq!(mul_out[0].host_data(), &[6.0]);

    let reduce_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(ExecOp::ReduceSum { axes: vec![0] }, 1),
        vec![TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])],
    )
    .unwrap();
    assert_eq!(reduce_out[0].shape, vec![2]);
    assert_eq!(reduce_out[0].host_data(), &[3.0, 7.0]);

    let permute_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(ExecOp::Transpose { perm: vec![1, 0] }, 1),
        vec![TypedTensor::from_vec(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
    )
    .unwrap();
    assert_eq!(permute_out[0].shape, vec![3, 2]);
    assert_eq!(permute_out[0].host_data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);

    let reshape_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(
            ExecOp::Reshape {
                shape: dim_shape(&[3, 2]),
            },
            1,
        ),
        vec![TypedTensor::from_vec(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
    )
    .unwrap();
    assert_eq!(reshape_out[0].shape, vec![3, 2]);
    assert_eq!(reshape_out[0].host_data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let broadcast_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(
            ExecOp::BroadcastInDim {
                shape: dim_shape(&[3, 2]),
                dims: vec![0],
            },
            1,
        ),
        vec![TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0])],
    )
    .unwrap();
    assert_eq!(broadcast_out[0].shape, vec![3, 2]);
    assert_eq!(
        broadcast_out[0].host_data(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );

    let extract_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(
            ExecOp::ExtractDiag {
                axis_a: 0,
                axis_b: 1,
            },
            1,
        ),
        vec![TypedTensor::from_vec(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )],
    )
    .unwrap();
    assert_eq!(extract_out[0].shape, vec![3]);
    assert_eq!(extract_out[0].host_data(), &[1.0, 5.0, 9.0]);

    let embed_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(
            ExecOp::EmbedDiag {
                axis_a: 0,
                axis_b: 1,
            },
            1,
        ),
        vec![TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0])],
    )
    .unwrap();
    assert_eq!(embed_out[0].shape, vec![3, 3]);
    assert_eq!(
        embed_out[0].host_data(),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    );

    let gemm_out = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(
            ExecOp::DotGeneral(DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            }),
            2,
        ),
        vec![
            TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]),
        ],
    )
    .unwrap();
    assert_eq!(gemm_out[0].shape, vec![2, 2]);
    assert_eq!(gemm_out[0].host_data(), &[23.0, 34.0, 31.0, 46.0]);
}

#[test]
#[should_panic(expected = "non-semiring op in semiring program")]
fn eval_semiring_ir_panics_on_non_semiring_ops() {
    let mut backend = CpuBackend::new();
    let _ = eval_semiring_ir::<_, Standard<f64>>(
        &mut backend,
        &single_instruction_program(ExecOp::Negate, 1),
        vec![typed_scalar(1.0)],
    );
}
