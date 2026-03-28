use tenferro_dynamic_compute::{
    set_default_runtime, with_default_runtime, RuntimeContext, ScalarType, Tensor,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

#[test]
fn typed_tensor_converts_into_dynamic_tensor_surface() {
    let dense =
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let value: Tensor = dense.into();

    assert_eq!(value.scalar_type(), ScalarType::F64);
    assert_eq!(value.dims(), &[2]);
}

#[test]
fn runtime_reexports_install_default_runtime() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let runtime_name = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();

    assert_eq!(runtime_name, "cpu");
}
