use tenferro_algebra::Standard;
use tenferro_einsum::einsum;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

#[test]
fn empty_vector_dot_product_returns_zero_scalar() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    let b = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "i,i->", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[] as &[usize]);
    let scalar = result.buffer().as_slice().unwrap()[result.offset() as usize];
    assert_eq!(scalar, 0.0);
}
