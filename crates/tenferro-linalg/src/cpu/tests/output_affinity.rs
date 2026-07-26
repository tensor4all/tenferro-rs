use super::*;

use tenferro_tensor::{CpuDomainId, MemoryKind, Placement};

fn remote_domain(selected: CpuDomainId) -> CpuDomainId {
    let candidate = CpuDomainId::new(selected.as_u64().wrapping_add(1));
    if candidate == selected {
        CpuDomainId::new(selected.as_u64().wrapping_sub(1))
    } else {
        candidate
    }
}

fn placed_matrix(values: Vec<f64>, domain: CpuDomainId) -> Tensor {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], values).unwrap();
    tensor.set_placement(Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: Some(domain),
    });
    Tensor::F64(tensor)
}

fn assert_selected(outputs: &[Tensor], selected: CpuDomainId) {
    assert!(!outputs.is_empty());
    assert!(outputs
        .iter()
        .all(|output| output.placement().cpu_affinity == Some(selected)));
}

#[test]
fn linalg_fresh_tagging_is_field_only_and_allocation_free() {
    let backend = include_str!("../backend.rs");
    let tagging = backend
        .split_once("trait FreshLinalgOutput")
        .expect("CPU linalg should define fresh-output tagging")
        .1
        .split_once("impl LinalgBackend for CpuBackend")
        .expect("fresh-output tagging should precede the linalg implementation")
        .0;

    assert!(tagging.contains("set_cpu_affinity(Some(domain))"));
    for forbidden in ["placement().clone", "format!", "HashMap", ".hash("] {
        assert!(
            !tagging.contains(forbidden),
            "fresh linalg tagging must not contain `{forbidden}`"
        );
    }
}

#[test]
fn decomposition_vectors_tag_every_fresh_output_with_the_selected_domain() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let general = placed_matrix(vec![4.0, 2.0, 1.0, 3.0], remote);
    let symmetric = placed_matrix(vec![4.0, 1.0, 1.0, 3.0], remote);

    for outputs in [
        backend.svd(&general).unwrap(),
        backend.qr(&general).unwrap(),
        backend.lu(&general).unwrap(),
        backend.full_piv_lu(&general).unwrap(),
        backend.eigh(&symmetric).unwrap(),
        backend.eig(&general).unwrap(),
    ] {
        assert_selected(&outputs, selected);
    }

    assert_eq!(general.placement().cpu_affinity, Some(remote));
    assert_eq!(symmetric.placement().cpu_affinity, Some(remote));
}

#[test]
fn linalg_single_outputs_tag_the_selected_domain() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let general = placed_matrix(vec![4.0, 2.0, 1.0, 3.0], remote);
    let symmetric = placed_matrix(vec![4.0, 1.0, 1.0, 3.0], remote);

    for output in [
        backend.svd_values(&general).unwrap(),
        backend.eigh_values(&symmetric).unwrap(),
        backend.eig_values(&general).unwrap(),
        backend.cholesky(&symmetric).unwrap(),
    ] {
        assert_eq!(output.placement().cpu_affinity, Some(selected));
    }
}

#[test]
fn zero_extent_solve_output_is_still_tagged_as_a_fresh_allocation() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let mut a = TypedTensor::<f64>::from_vec_col_major(vec![0, 0], vec![]).unwrap();
    let mut b = TypedTensor::<f64>::from_vec_col_major(vec![0], vec![]).unwrap();
    for tensor in [&mut a, &mut b] {
        tensor.set_placement(Placement {
            memory_kind: MemoryKind::UnpinnedHost,
            device: None,
            cpu_affinity: Some(remote),
        });
    }
    let a = Tensor::F64(a);
    let b = Tensor::F64(b);

    let output = backend.full_piv_lu_solve(&a, &b, false).unwrap();

    assert_eq!(output.shape(), &[0]);
    assert_eq!(output.placement().cpu_affinity, Some(selected));
    assert_eq!(a.placement().cpu_affinity, Some(remote));
    assert_eq!(b.placement().cpu_affinity, Some(remote));
}

#[test]
fn prepared_lu_composition_tags_its_final_permutation_output() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let a = placed_matrix(vec![4.0, 2.0, 1.0, 3.0], remote);
    let b = placed_matrix(vec![1.0, 2.0, 3.0, 4.0], remote);
    let factor = backend.lu_factor(&a).unwrap();

    let output = backend
        .lu_solve_prepared(&a, &factor[0], &factor[1], &b, true, false)
        .unwrap();

    assert_eq!(output.placement().cpu_affinity, Some(selected));
    assert_eq!(a.placement().cpu_affinity, Some(remote));
    assert_eq!(b.placement().cpu_affinity, Some(remote));
}
