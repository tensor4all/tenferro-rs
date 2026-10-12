use cubecl::ir::{
    Builtin, OpInserter, Scope,
    dialect::{
        branch::IfOp,
        cmp::{FClampNanOp, FMaxOp, FMinOp, ULessThanOp},
        general::{BufferLenOp, ReadBuiltinOp, SelectOp},
        math::{
            CConjOp, CosOp, ErfOp, ExpOp, Expm1Op, FAbsOp, FAddOp, FDivOp, FMulOp, FNegOp, IsNanOp,
            Log1pOp, LogOp, PowfOp, RsqrtOp, SinOp, SqrtOp, TanhOp,
        },
        memory::{IndexOp, LoadOp, StoreOp},
    },
    pliron::value::Value,
    settings::ExecutionMode,
};
use cubecl::prelude::{
    AddressType, CubeDim, CubePrimitive, KernelBuilder, KernelDefinition, KernelSettings,
};

use crate::backend::{ElementwiseFusionOp, ElementwiseFusionPlan};

const KERNEL_NAME: &str = "tenferro_fused_elementwise";

pub(crate) fn build_kernel_definition<T>(
    plan: &ElementwiseFusionPlan,
    address_type: AddressType,
    cube_dim: CubeDim,
) -> KernelDefinition
where
    T: CubePrimitive + Clone,
{
    let mut builder = KernelBuilder::new(KernelSettings::new(
        cube_dim.into(),
        ExecutionMode::Unchecked,
        address_type,
    ));

    let item = T::__expand_as_type(&builder.scope);
    let mut input_arrays = Vec::with_capacity(plan.input_count());
    for _ in 0..plan.input_count() {
        input_arrays.push(builder.buffer(item.clone()));
    }
    let mut output_arrays = Vec::with_capacity(plan.outputs().len());
    for _ in plan.outputs() {
        output_arrays.push(builder.buffer(item.clone()));
    }

    let absolute_pos = builder.scope.register_with_result(&ReadBuiltinOp::new(
        builder.scope.ctx_mut(),
        usize::__expand_as_type(&builder.scope),
        Builtin::AbsolutePos,
    ));
    // Output 0 was registered after the inputs, so its buffer binding position is
    // `plan.input_count()`.
    let output_len = builder.scope.register_with_result(&BufferLenOp::new(
        builder.scope.ctx_mut(),
        plan.input_count(),
    ));
    let in_bounds = builder.scope.register_with_result(&ULessThanOp::new(
        builder.scope.ctx_mut(),
        absolute_pos,
        output_len,
    ));

    let branch = IfOp::new(builder.scope.ctx_mut(), in_bounds);
    let then_block = branch.then_block(builder.scope.ctx());
    let then_scope = builder
        .scope
        .child(OpInserter::new_at_block_end(then_block));
    build_body(
        plan,
        &then_scope,
        &input_arrays,
        &output_arrays,
        absolute_pos,
    );
    then_scope.terminate_yield();
    let else_block = branch.else_block(builder.scope.ctx());
    let else_scope = builder
        .scope
        .child(OpInserter::new_at_block_end(else_block));
    else_scope.terminate_yield();
    builder.scope.register(&branch);

    builder.build()
}

fn build_body(
    plan: &ElementwiseFusionPlan,
    scope: &Scope,
    input_arrays: &[Value],
    output_arrays: &[Value],
    absolute_pos: Value,
) {
    let mut values = Vec::with_capacity(plan.input_count() + plan.ops().len());
    for &array in input_arrays {
        values.push(load_array_element(scope, array, absolute_pos));
    }
    for inst in plan.ops() {
        let inputs = inst
            .inputs()
            .iter()
            .map(|&value| values[value])
            .collect::<Vec<_>>();
        values.push(emit_op(scope, &inst.op(), &inputs));
    }
    for (&array, &value_id) in output_arrays.iter().zip(plan.outputs().iter()) {
        store_array_element(scope, array, absolute_pos, values[value_id]);
    }
}

fn emit_op(scope: &Scope, op: &ElementwiseFusionOp, inputs: &[Value]) -> Value {
    macro_rules! unary {
        ($ctor:path) => {
            scope.register_with_result(&$ctor(scope.ctx_mut(), inputs[0]))
        };
    }
    macro_rules! binary {
        ($ctor:path) => {
            scope.register_with_result(&$ctor(scope.ctx_mut(), inputs[0], inputs[1]))
        };
    }
    // NaN-propagating extrema: `if lhs != lhs { lhs } else if rhs != rhs { rhs } else { max/min }`,
    // the sequence the CUDA kernels use. CubeCL's own `FMaxNanOp`/`FMinNanOp` leave the
    // opposite-zero sign and the NaN payload unspecified, so they cannot stand in for it.
    macro_rules! nan_extrema {
        ($ctor:path) => {{
            let extrema = scope.register_with_result(&$ctor(scope.ctx_mut(), inputs[0], inputs[1]));
            let rhs_is_nan = scope.register_with_result(&IsNanOp::new(scope.ctx_mut(), inputs[1]));
            let rhs_or_extrema =
                scope.register_with_result(&SelectOp::new(scope.ctx_mut(), rhs_is_nan, inputs[1], extrema));
            let lhs_is_nan = scope.register_with_result(&IsNanOp::new(scope.ctx_mut(), inputs[0]));
            scope.register_with_result(&SelectOp::new(scope.ctx_mut(), lhs_is_nan, inputs[0], rhs_or_extrema))
        }};
    }

    match op {
        ElementwiseFusionOp::Add => binary!(FAddOp::new),
        ElementwiseFusionOp::Multiply => binary!(FMulOp::new),
        ElementwiseFusionOp::Negate => unary!(FNegOp::new),
        ElementwiseFusionOp::Conj => unary!(CConjOp::new),
        ElementwiseFusionOp::Divide => binary!(FDivOp::new),
        ElementwiseFusionOp::Abs => unary!(FAbsOp::new),
        ElementwiseFusionOp::Maximum => nan_extrema!(FMaxOp::new),
        ElementwiseFusionOp::Minimum => nan_extrema!(FMinOp::new),
        ElementwiseFusionOp::Clamp => scope.register_with_result(&FClampNanOp::new(
            scope.ctx_mut(),
            inputs[0],
            inputs[1],
            inputs[2],
        )),
        ElementwiseFusionOp::Exp => unary!(ExpOp::new),
        ElementwiseFusionOp::Log => unary!(LogOp::new),
        ElementwiseFusionOp::Sin => unary!(SinOp::new),
        ElementwiseFusionOp::Cos => unary!(CosOp::new),
        ElementwiseFusionOp::Tanh => unary!(TanhOp::new),
        ElementwiseFusionOp::Sqrt => unary!(SqrtOp::new),
        ElementwiseFusionOp::Rsqrt => unary!(RsqrtOp::new),
        ElementwiseFusionOp::Pow => binary!(PowfOp::new),
        ElementwiseFusionOp::Expm1 => unary!(Expm1Op::new),
        ElementwiseFusionOp::Log1p => unary!(Log1pOp::new),
        ElementwiseFusionOp::Erf => unary!(ErfOp::new),
        ElementwiseFusionOp::Remainder => {
            unreachable!("remainder fusion is rejected by classify before codegen")
        }
    }
}

fn load_array_element(scope: &Scope, array: Value, index: Value) -> Value {
    let ptr = scope.register_with_result(&IndexOp::new(scope.ctx_mut(), array, index, None));
    scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))
}

fn store_array_element(scope: &Scope, array: Value, index: Value, value: Value) {
    let ptr = scope.register_with_result(&IndexOp::new(scope.ctx_mut(), array, index, None));
    scope.register(&StoreOp::new(scope.ctx_mut(), ptr, value));
}
