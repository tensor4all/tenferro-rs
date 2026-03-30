use super::{reverse_leaf_f64, tensor_from_vec_f64 as tensor_from_slice, tensor_to_vec_f64};
use tenferro_prims::CpuContext;
use tidu::expert::Tape;

use crate::ops::std_ad;
use crate::{set_default_runtime, AdTensor, RuntimeContext};

#[test]
fn std_ad_reverse_pullback_returns_zero_for_constant_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(tensor_from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]), &tape);
    let out = std_ad(&x).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0], &[]));

    let grads = crate::ops::ad::pullback_wrt(&out, &cotangent, &[&x]).unwrap();
    let actual = tensor_to_vec_f64(grads[0].as_ref().unwrap().payload());
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "std_ad gradient should stay finite for constant input, got {actual:?}"
    );
    assert_eq!(actual, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn std_ad_forward_tangent_returns_zero_for_constant_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = tensor_from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    let dx = tensor_from_slice(&[1.0, -1.0, 0.5, -0.5], &[2, 2]);
    let out = std_ad(&AdTensor::new_forward(x, dx).unwrap())
        .run()
        .unwrap();

    let tangent = tensor_to_vec_f64(out.tangent().unwrap());
    assert!(
        tangent.iter().all(|value| value.is_finite()),
        "std_ad tangent should stay finite for constant input, got {tangent:?}"
    );
    assert_eq!(tangent, vec![0.0]);
}
