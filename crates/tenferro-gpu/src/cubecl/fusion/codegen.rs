use cubecl::ir::{
    Arithmetic, BinaryOperator, Branch, Builtin, ClampOperator, Comparison, ElemType, If,
    IndexAssignOperator, IndexOperator, Instruction, ManagedVariable, Metadata, Operator, Type,
    UnaryOperator, Variable,
};
use cubecl::prelude::{
    AddressType, CubeDim, CubeElement, CubePrimitive, KernelBuilder, KernelDefinition,
    KernelSettings,
};

use crate::backend::{ElementwiseFusionOp, ElementwiseFusionPlan};

const KERNEL_NAME: &str = "tenferro_fused_elementwise";

pub(crate) fn build_kernel_definition<T>(
    plan: &ElementwiseFusionPlan,
    address_type: AddressType,
    cube_dim: CubeDim,
) -> KernelDefinition
where
    T: CubeElement + CubePrimitive + Clone,
{
    let mut builder = KernelBuilder::new();
    address_type.register(&mut builder.scope);

    let item = T::as_type(&builder.scope);
    let mut input_arrays = Vec::with_capacity(plan.input_count());
    for _ in 0..plan.input_count() {
        input_arrays.push(builder.input_array(item.clone()));
    }
    let mut output_arrays = Vec::with_capacity(plan.outputs().len());
    for _ in plan.outputs() {
        output_arrays.push(builder.output_array(item.clone()));
    }

    let absolute_pos = ManagedVariable::Plain(Variable::builtin(
        Builtin::AbsolutePos,
        usize::as_type(&builder.scope).storage_type(),
    ));
    let output_len = builder.scope.create_local(usize::as_type(&builder.scope));
    builder.scope.register(Instruction::new(
        Metadata::Length {
            var: *output_arrays[0],
        },
        *output_len,
    ));

    let in_bounds = emit_bool_binary(
        &mut builder.scope,
        absolute_pos.clone(),
        output_len.clone(),
        Comparison::Lower,
    );
    let mut body = builder.scope.child();
    let body = build_body::<T>(plan, &mut body, &input_arrays, &output_arrays, absolute_pos);
    builder.scope.register(Branch::If(Box::new(If {
        cond: *in_bounds,
        scope: body,
    })));

    builder.build(
        KernelSettings::default()
            .kernel_name(KERNEL_NAME)
            .address_type(address_type)
            .cube_dim(cube_dim),
    )
}

fn build_body<T>(
    plan: &ElementwiseFusionPlan,
    scope: &mut cubecl::prelude::Scope,
    input_arrays: &[ManagedVariable],
    output_arrays: &[ManagedVariable],
    absolute_pos: ManagedVariable,
) -> cubecl::ir::Scope
where
    T: CubeElement + CubePrimitive + Clone,
{
    let mut values = Vec::with_capacity(plan.input_count() + plan.ops().len());
    for array in input_arrays {
        values.push(load_array_element(
            scope,
            array.clone(),
            absolute_pos.clone(),
        ));
    }
    for inst in plan.ops() {
        let inputs = inst
            .inputs()
            .iter()
            .map(|&value| values[value].clone())
            .collect::<Vec<_>>();
        values.push(emit_op::<T>(scope, &inst.op(), &inputs));
    }
    for (array, &value_id) in output_arrays.iter().zip(plan.outputs().iter()) {
        store_array_element(
            scope,
            array.clone(),
            absolute_pos.clone(),
            values[value_id].clone(),
        );
    }
    scope.clone()
}

fn emit_op<T>(
    scope: &mut cubecl::prelude::Scope,
    op: &ElementwiseFusionOp,
    inputs: &[ManagedVariable],
) -> ManagedVariable
where
    T: CubeElement + CubePrimitive + Clone,
{
    match op {
        ElementwiseFusionOp::Add => {
            emit_binary_arithmetic(scope, &inputs[0], &inputs[1], Arithmetic::Add)
        }
        ElementwiseFusionOp::Multiply => {
            emit_binary_arithmetic(scope, &inputs[0], &inputs[1], Arithmetic::Mul)
        }
        ElementwiseFusionOp::Negate => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Neg),
        ElementwiseFusionOp::Conj => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Conj),
        ElementwiseFusionOp::Divide => {
            emit_binary_arithmetic(scope, &inputs[0], &inputs[1], Arithmetic::Div)
        }
        ElementwiseFusionOp::Abs => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Abs),
        ElementwiseFusionOp::Maximum => {
            emit_binary_arithmetic(scope, &inputs[0], &inputs[1], Arithmetic::Max)
        }
        ElementwiseFusionOp::Minimum => {
            emit_binary_arithmetic(scope, &inputs[0], &inputs[1], Arithmetic::Min)
        }
        ElementwiseFusionOp::Clamp => emit_clamp(scope, &inputs[0], &inputs[1], &inputs[2]),
        ElementwiseFusionOp::Exp => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Exp),
        ElementwiseFusionOp::Log => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Log),
        ElementwiseFusionOp::Sin => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Sin),
        ElementwiseFusionOp::Cos => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Cos),
        ElementwiseFusionOp::Tanh => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Tanh),
        ElementwiseFusionOp::Sqrt => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Sqrt),
        ElementwiseFusionOp::Rsqrt => {
            emit_unary_arithmetic(scope, &inputs[0], Arithmetic::InverseSqrt)
        }
        ElementwiseFusionOp::Pow => {
            emit_binary_arithmetic(scope, &inputs[0], &inputs[1], Arithmetic::Powf)
        }
        ElementwiseFusionOp::Expm1 => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Expm1),
        ElementwiseFusionOp::Log1p => emit_unary_arithmetic(scope, &inputs[0], Arithmetic::Log1p),
    }
}

fn emit_unary_arithmetic(
    scope: &mut cubecl::prelude::Scope,
    input: &ManagedVariable,
    op: fn(UnaryOperator) -> Arithmetic,
) -> ManagedVariable {
    let input = input.clone().consume();
    let out = scope.create_local(input.ty);
    scope.register(Instruction::new(op(UnaryOperator { input }), *out));
    out
}

fn emit_binary_arithmetic(
    scope: &mut cubecl::prelude::Scope,
    lhs: &ManagedVariable,
    rhs: &ManagedVariable,
    op: fn(BinaryOperator) -> Arithmetic,
) -> ManagedVariable {
    let lhs = lhs.clone().consume();
    let rhs = rhs.clone().consume();
    let out = scope.create_local(lhs.ty);
    scope.register(Instruction::new(op(BinaryOperator { lhs, rhs }), *out));
    out
}

fn emit_bool_binary(
    scope: &mut cubecl::prelude::Scope,
    lhs: ManagedVariable,
    rhs: ManagedVariable,
    op: fn(BinaryOperator) -> Comparison,
) -> ManagedVariable {
    let lhs = lhs.consume();
    let rhs = rhs.consume();
    let out = scope.create_local(Type::scalar(ElemType::Bool));
    scope.register(Instruction::new(op(BinaryOperator { lhs, rhs }), *out));
    out
}

fn emit_clamp(
    scope: &mut cubecl::prelude::Scope,
    input: &ManagedVariable,
    lower: &ManagedVariable,
    upper: &ManagedVariable,
) -> ManagedVariable {
    let input = input.clone().consume();
    let min_value = lower.clone().consume();
    let max_value = upper.clone().consume();
    let out = scope.create_local(input.ty);
    scope.register(Instruction::new(
        Arithmetic::Clamp(ClampOperator {
            input,
            min_value,
            max_value,
        }),
        *out,
    ));
    out
}

fn load_array_element(
    scope: &mut cubecl::prelude::Scope,
    array: ManagedVariable,
    index: ManagedVariable,
) -> ManagedVariable {
    let array = array.consume();
    let index = index.consume();
    let out = scope.create_local(array.ty);
    scope.register(Instruction::new(
        Operator::UncheckedIndex(IndexOperator {
            list: array,
            index,
            vector_size: 0,
            unroll_factor: 1,
        }),
        *out,
    ));
    out
}

fn store_array_element(
    scope: &mut cubecl::prelude::Scope,
    array: ManagedVariable,
    index: ManagedVariable,
    value: ManagedVariable,
) {
    scope.register(Instruction::new(
        Operator::UncheckedIndexAssign(IndexAssignOperator {
            index: index.consume(),
            value: value.consume(),
            vector_size: 0,
            unroll_factor: 1,
        }),
        array.consume(),
    ));
}
