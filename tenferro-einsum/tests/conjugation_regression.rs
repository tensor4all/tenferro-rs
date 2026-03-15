use num_complex::Complex64;
use tenferro_algebra::Standard;
use tenferro_einsum::einsum;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

#[test]
fn einsum_resolves_lazy_conjugated_operands() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let x_conj = x.conj();

    let out =
        einsum::<Standard<Complex64>, CpuBackend>(&mut ctx, "i,i->", &[&x_conj, &x], None).unwrap();
    let scalar = out.buffer().as_slice().unwrap()[out.offset() as usize];
    assert_eq!(scalar, Complex64::new(30.0, 0.0));
}
