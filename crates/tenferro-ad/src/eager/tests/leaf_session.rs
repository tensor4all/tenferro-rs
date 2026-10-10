//! Eager leaf construction must not pay a backend session for host tensors
//! (#1704).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use num_complex::Complex64;

use crate::eager_backend::EagerBackend;
use crate::{EagerRuntime, Error};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSessionHost, MemoryKind, Placement, Tensor, TensorRead, TensorView, TypedTensor,
    TypedTensorView,
};
use tenferro_tensor::{DynRank, ErasedHostTensor, Host};

use super::super::EagerTensor;

#[test]
fn eager_admission_callbacks_and_results_need_not_be_send() -> Result<(), Error> {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(2).unwrap())?;
    let caller = std::thread::current().id();
    let value = std::rc::Rc::new(7);
    let result = runtime.with_execution_session(|_| {
        assert_eq!(std::thread::current().id(), caller);
        std::rc::Rc::clone(&value)
    })?;
    assert!(std::rc::Rc::ptr_eq(&value, &result));
    let result = runtime.with_eager_session(|_| {
        assert_eq!(std::thread::current().id(), caller);
        Ok::<_, Error>(std::rc::Rc::clone(&value))
    })?;
    assert!(std::rc::Rc::ptr_eq(&value, &result));
    Ok(())
}

#[test]
fn eager_global_and_foreign_pool_children_do_not_wait_for_parent_owner() -> Result<(), Error> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for foreign in [false, true] {
        let runtime =
            EagerRuntime::with_cpu_backend(CpuBackend::with_threads_isolated_arbiter_for_test(1))?;
        let child = Arc::clone(&runtime);
        let (tx, rx) = std::sync::mpsc::channel();
        let received = runtime.with_execution_session(|_| {
            let job = move || {
                let _ = tx.send(child.with_execution_session(|_| ()));
            };
            if foreign {
                pool.spawn(job);
            } else {
                rayon::spawn(job);
            }
            rx.recv_timeout(std::time::Duration::from_secs(5))
        })?;
        assert!(matches!(
            received,
            Ok(Err(Error::SessionEntry(
                tenferro_tensor::SessionEntryError::Contended { .. }
            )))
        ));
        pool.install(|| runtime.with_execution_session(|_| ()))?;
    }
    Ok(())
}

#[test]
fn runtime_bound_eager_session_reuses_entry_and_rejects_foreign_tensors() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64])?)?;
    let foreign = other.variable_from(Tensor::from_vec_col_major(vec![], vec![3.0_f64])?)?;

    let y = ctx.with_eager_session(|session| {
        let negated = session.neg(&x)?;
        let restored = session.neg(&negated)?;
        let magnitude = session.abs(&negated)?;
        let exponential = session.exp(&negated)?;
        let conjugate = session.conj(&restored)?;
        assert!(matches!(
            session.neg(&foreign),
            Err(Error::ContextMismatch { .. })
        ));
        Ok::<_, Error>((negated, restored, magnitude, exponential, conjugate))
    })?;
    assert_eq!(y.0.value()?.as_slice::<f64>()?, &[-2.0]);
    assert_eq!(y.1.value()?.as_slice::<f64>()?, &[2.0]);
    assert_eq!(y.2.value()?.as_slice::<f64>()?, &[2.0]);
    assert!((y.3.value()?.as_slice::<f64>()?[0] - (-2.0_f64).exp()).abs() < 1e-12);
    assert_eq!(y.4.value()?.as_slice::<f64>()?, &[2.0]);
    assert_eq!(ctx.grad(&y.0, &x)?.value()?.as_slice::<f64>()?, &[-1.0]);
    assert_eq!(ctx.grad(&y.2, &x)?.value()?.as_slice::<f64>()?, &[1.0]);
    Ok(())
}

#[test]
fn runtime_bound_unary_family_preserves_values_and_ad() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64])?)?;
    let values = ctx.with_eager_session(|session| {
        let sign = session.sign(&x)?;
        let log = session.log(&x)?;
        let sqrt = session.sqrt(&x)?;
        let rsqrt = session.rsqrt(&x)?;
        let sin = session.sin(&x)?;
        let cos = session.cos(&x)?;
        let tanh = session.tanh(&x)?;
        let expm1 = session.expm1(&x)?;
        let log1p = session.log1p(&x)?;
        let sum = session.add(&sin, &log)?;
        Ok::<_, Error>((sign, sqrt, rsqrt, cos, tanh, expm1, log1p, sum))
    })?;
    let scalar =
        |tensor: &EagerTensor| -> Result<f64, Error> { Ok(tensor.value()?.as_slice::<f64>()?[0]) };
    assert_eq!(scalar(&values.0)?, 1.0);
    for (actual, expected) in [
        (scalar(&values.1)?, 2.0_f64.sqrt()),
        (scalar(&values.2)?, 2.0_f64.sqrt().recip()),
        (scalar(&values.3)?, 2.0_f64.cos()),
        (scalar(&values.4)?, 2.0_f64.tanh()),
        (scalar(&values.5)?, 2.0_f64.exp_m1()),
        (scalar(&values.6)?, 2.0_f64.ln_1p()),
        (scalar(&ctx.grad(&values.7, &x)?)?, 2.0_f64.cos() + 0.5),
    ] {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
    Ok(())
}

#[test]
fn runtime_bound_diagonals_preserve_values_ad_and_runtime_identity() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let foreign = other.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2])?)?;
    let (diagonal, restored, total) = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.embed_diag(&foreign, 0, 1),
            Err(Error::ContextMismatch { .. })
        ));
        let diagonal = session.extract_diag(&x, 0, 1)?;
        let restored = session.embed_diag(&diagonal, 0, 1)?;
        let total = session.reduce_sum(&diagonal, None)?;
        Ok::<_, Error>((diagonal, restored, total))
    })?;
    assert_eq!(diagonal.value()?.as_slice::<f64>()?, &[1.0, 4.0]);
    assert_eq!(restored.value()?.as_slice::<f64>()?, &[1.0, 0.0, 0.0, 4.0]);
    assert_eq!(
        ctx.grad(&total, &x)?.value()?.as_slice::<f64>()?,
        &[1.0, 0.0, 0.0, 1.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_gather_and_concatenate_preserve_ad_and_identity() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![3],
        vec![10.0_f64, 20.0, 30.0],
    )?)?;
    let indices = ctx.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![2_i64, 0])?)?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?)?;
    let foreign_index =
        other.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 1])?)?;
    let foreign_stack =
        other.constant_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3])?)?;
    let config = tenferro_tensor::GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let (joined, total) = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.concatenate(&[&x, &foreign], 0),
            Err(Error::ContextMismatch { .. })
        ));
        assert!(session.concatenate(&[], 0).is_err());
        assert!(matches!(
            session.gather(&x, &foreign_index, config.clone()),
            Err(Error::ContextMismatch { .. })
        ));
        assert!(matches!(
            session.stack(&[&x, &foreign_stack], 0),
            Err(Error::ContextMismatch { .. })
        ));
        let selected = session.gather(&x, &indices, config)?;
        let joined = session.concatenate(&[&selected, &selected], 0)?;
        let total = session.reduce_sum(&joined, None)?;
        Ok::<_, Error>((joined, total))
    })?;
    assert_eq!(
        joined.value()?.as_slice::<f64>()?,
        &[30.0, 10.0, 30.0, 10.0]
    );
    assert_eq!(
        ctx.grad(&total, &x)?.value()?.as_slice::<f64>()?,
        &[2.0, 0.0, 2.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_scaling_preserves_errors_and_gradient() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?)?;
    let integer = ctx.constant_from(Tensor::from_vec_col_major(vec![1], vec![1_i64])?)?;
    let total = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.scale_real(&foreign, 2.0),
            Err(Error::ContextMismatch { .. })
        ));
        assert!(matches!(
            session.scale_real(&integer, f64::NAN),
            Err(Error::TensorRuntime(
                tenferro_tensor::Error::Validation { .. }
            ))
        ));
        let scaled = session.scale_real(&x, 2.5)?;
        session.reduce_sum(&scaled, None)
    })?;
    assert_eq!(
        ctx.grad(&total, &x)?.value()?.as_slice::<f64>()?,
        &[2.5, 2.5]
    );
    Ok(())
}

#[test]
fn runtime_bound_matmul_preserves_values_ad_and_validation() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let lhs = ctx.variable_from(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let rhs = ctx.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0])?)?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64; 2])?)?;
    let (product, total) = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.matmul(&lhs, &foreign),
            Err(Error::ContextMismatch { .. })
        ));
        assert!(matches!(
            session.matmul(&rhs, &lhs),
            Err(Error::TensorRuntime(tenferro_tensor::Error::Validation {
                source: tenferro_tensor::ValidationError::ShapeMismatch { .. },
                ..
            }))
        ));
        let product = session.matmul(&lhs, &rhs)?;
        let total = session.reduce_sum(&product, None)?;
        Ok::<_, Error>((product, total))
    })?;
    assert_eq!(product.value()?.as_slice::<f64>()?, &[23.0, 34.0]);
    assert_eq!(
        ctx.grad(&total, &lhs)?.value()?.as_slice::<f64>()?,
        &[5.0, 5.0, 6.0, 6.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_reductions_preserve_values_ad_and_axis_errors() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![3],
        vec![2.0_f64, 3.0, 5.0],
    )?)?;
    let (product, maximum, minimum) = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.reduce_prod(&x, Some(&[0, 0])),
            Err(Error::TensorRuntime(tenferro_tensor::Error::Validation {
                source: tenferro_tensor::ValidationError::DuplicateAxis { .. },
                ..
            }))
        ));
        Ok::<_, Error>((
            session.reduce_prod(&x, None)?,
            session.reduce_max(&x, None)?,
            session.reduce_min(&x, None)?,
        ))
    })?;
    assert_eq!(product.value()?.as_slice::<f64>()?, &[30.0]);
    assert_eq!(maximum.value()?.as_slice::<f64>()?, &[5.0]);
    assert_eq!(minimum.value()?.as_slice::<f64>()?, &[2.0]);
    assert_eq!(
        ctx.grad(&product, &x)?.value()?.as_slice::<f64>()?,
        &[15.0, 10.0, 6.0]
    );
    assert_eq!(
        ctx.grad(&maximum, &x)?.value()?.as_slice::<f64>()?,
        &[0.0, 0.0, 1.0]
    );
    assert_eq!(
        ctx.grad(&minimum, &x)?.value()?.as_slice::<f64>()?,
        &[1.0, 0.0, 0.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_triangles_preserve_values_gradients_and_runtime_identity() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?)?;
    let (lower, upper, sum) = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.tril(&foreign, 0),
            Err(Error::ContextMismatch { .. })
        ));
        let lower = session.tril(&x, 0)?;
        let upper = session.triu(&x, 0)?;
        let sum = session.reduce_sum(&lower, None)?;
        Ok::<_, Error>((lower, upper, sum))
    })?;
    assert_eq!(lower.value()?.as_slice::<f64>()?, &[1.0, 2.0, 0.0, 4.0]);
    assert_eq!(upper.value()?.as_slice::<f64>()?, &[1.0, 0.0, 3.0, 4.0]);
    assert_eq!(
        ctx.grad(&sum, &x)?.value()?.as_slice::<f64>()?,
        &[1.0, 1.0, 0.0, 1.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_padding_and_reverse_preserve_gradients() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let (reversed, sum) = ctx.with_eager_session(|session| {
        let padded = session.pad(
            &x,
            tenferro_tensor::PadConfig {
                edge_padding_low: vec![1],
                edge_padding_high: vec![1],
                interior_padding: vec![1],
            },
        )?;
        let reversed = session.reverse(&padded, &[0])?;
        let sum = session.reduce_sum(&reversed, None)?;
        Ok::<_, Error>((reversed, sum))
    })?;
    assert_eq!(
        reversed.value()?.as_slice::<f64>()?,
        &[0.0, 2.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(ctx.grad(&sum, &x)?.value()?.as_slice::<f64>()?, &[1.0, 1.0]);
    Ok(())
}

#[test]
fn runtime_bound_dynamic_slice_preserves_ad_and_rejects_foreign_indices() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![4],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![1], vec![1_i64])?)?;
    let (selected, sum) = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.dynamic_slice(&x, &foreign, &[2]),
            Err(Error::ContextMismatch { .. })
        ));
        let starts = session.constant_from(Tensor::from_vec_col_major(vec![1], vec![1_i64])?)?;
        let selected = session.dynamic_slice(&x, &starts, &[2])?;
        let sum = session.reduce_sum(&selected, None)?;
        Ok::<_, Error>((selected, sum))
    })?;
    assert_eq!(selected.value()?.as_slice::<f64>()?, &[2.0, 3.0]);
    assert_eq!(
        ctx.grad(&sum, &x)?.value()?.as_slice::<f64>()?,
        &[0.0, 1.0, 1.0, 0.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_scatter_preserves_values_ad_and_foreign_checks() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2])?)?;
    let updates = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0])?)?;
    let config = tenferro_tensor::ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let (result, sum) = ctx.with_eager_session(|session| {
        let input =
            session.constant_from(Tensor::from_vec_col_major(vec![4], vec![0.0_f64; 4])?)?;
        let indices =
            session.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![1_i64, 3])?)?;
        assert!(matches!(
            session.scatter(&input, &indices, &foreign, config.clone()),
            Err(Error::ContextMismatch { .. })
        ));
        let result = session.scatter(&input, &indices, &updates, config)?;
        let sum = session.reduce_sum(&result, None)?;
        Ok::<_, Error>((result, sum))
    })?;
    assert_eq!(result.value()?.as_slice::<f64>()?, &[0.0, 5.0, 0.0, 7.0]);
    assert_eq!(
        ctx.grad(&sum, &updates)?.value()?.as_slice::<f64>()?,
        &[1.0, 1.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_conjugating_dot_preserves_fast_and_tracked_paths() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let lhs = ctx.constant_from(Tensor::from_vec_col_major(
        vec![1, 1],
        vec![Complex64::new(1.0, 2.0)],
    )?)?;
    let rhs = ctx.constant_from(Tensor::from_vec_col_major(
        vec![1, 1],
        vec![Complex64::new(3.0, 4.0)],
    )?)?;
    let tracked = ctx.variable_from(Tensor::from_vec_col_major(
        vec![1, 1],
        vec![Complex64::new(1.0, 2.0)],
    )?)?;
    let config = tenferro_tensor::DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let (plain, lhs_conj, both, tracked_out) = ctx.with_eager_session(|session| {
        Ok::<_, Error>((
            session.dot_general_with_conj(&lhs, &rhs, config.clone(), false, false)?,
            session.dot_general_with_conj(&lhs, &rhs, config.clone(), true, false)?,
            session.dot_general_with_conj(&lhs, &rhs, config.clone(), true, true)?,
            session.dot_general_with_conj(&tracked, &rhs, config.clone(), true, false)?,
        ))
    })?;
    assert_eq!(
        plain.value()?.as_slice::<Complex64>()?,
        &[Complex64::new(-5.0, 10.0)]
    );
    assert_eq!(
        lhs_conj.value()?.as_slice::<Complex64>()?,
        &[Complex64::new(11.0, -2.0)]
    );
    assert_eq!(
        both.value()?.as_slice::<Complex64>()?,
        &[Complex64::new(-5.0, -10.0)]
    );
    assert_eq!(
        tracked_out.value()?.as_slice::<Complex64>()?,
        &[Complex64::new(11.0, -2.0)]
    );
    assert!(tracked_out.tracks_grad());
    Ok(())
}

#[test]
fn runtime_bound_eager_dot_general_preserves_ad_and_validates_dimensions() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?)?;
    let (lhs, result, loss) = ctx.with_eager_session(|session| {
        let lhs = session.variable_from(Tensor::from_vec_col_major(
            vec![2, 2],
            vec![1.0_f64, 2.0, 3.0, 4.0],
        )?)?;
        let rhs = session.constant_from(Tensor::from_vec_col_major(
            vec![2, 2],
            vec![5.0_f64, 6.0, 7.0, 8.0],
        )?)?;
        let config = tenferro_tensor::DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        };
        let result = session.dot_general(&lhs, &rhs, config.clone())?;
        let loss = session.reduce_sum(&result, None)?;
        assert!(matches!(
            session.dot_general(&lhs, &foreign, config),
            Err(Error::ContextMismatch { .. })
        ));
        let invalid = tenferro_tensor::DotGeneralConfig {
            lhs_contracting_dims: [2].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        };
        assert!(matches!(
            session.dot_general(&lhs, &rhs, invalid),
            Err(Error::TensorRuntime(_))
        ));
        Ok::<_, Error>((lhs, result, loss))
    })?;
    assert_eq!(
        result.value()?.as_slice::<f64>()?,
        &[23.0, 34.0, 31.0, 46.0]
    );
    assert_eq!(
        ctx.grad(&loss, &lhs)?.value()?.as_slice::<f64>()?,
        &[12.0, 12.0, 14.0, 14.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_eager_binary_ops_preserve_broadcast_and_vjp() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let scalar = ctx.variable_from(Tensor::from_vec_col_major(vec![], vec![3.0_f64])?)?;
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0])?,
        ctx.clone(),
    )?;

    let (sum, difference, product, loss) = ctx.with_eager_session(|session| {
        let sum = session.add(&x, &scalar)?;
        let difference = session.sub(&x, &scalar)?;
        let product = session.mul(&x, &scalar)?;
        assert!(matches!(
            session.reduce_sum(&x, Some(&[1])),
            Err(Error::TensorRuntime(_))
        ));
        let loss = session.reduce_sum(&product, None)?;
        Ok::<_, Error>((sum, difference, product, loss))
    })?;
    assert_eq!(loss.value()?.as_slice::<f64>()?, &[9.0]);
    assert_eq!(
        ctx.grad(&loss, &x)?.value()?.as_slice::<f64>()?,
        &[3.0, 3.0]
    );
    assert_eq!(
        ctx.grad(&loss, &scalar)?.value()?.as_slice::<f64>()?,
        &[3.0]
    );
    assert_eq!(sum.value()?.as_slice::<f64>()?, &[4.0, 5.0]);
    assert_eq!(difference.value()?.as_slice::<f64>()?, &[-2.0, -1.0]);
    assert_eq!(product.value()?.as_slice::<f64>()?, &[3.0, 6.0]);
    assert_eq!(
        ctx.vjp(&sum, &x, &seed)?.value()?.as_slice::<f64>()?,
        &[1.0, 1.0]
    );
    assert_eq!(
        ctx.vjp(&sum, &scalar, &seed)?.value()?.as_slice::<f64>()?,
        &[2.0]
    );
    assert_eq!(
        ctx.vjp(&product, &scalar, &seed)?
            .value()?
            .as_slice::<f64>()?,
        &[3.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_binary_family_preserves_broadcast_ad_and_context() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![6.0_f64, 8.0])?)?;
    let scalar = ctx.variable_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64])?)?;
    let foreign = other.constant_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64])?)?;
    let (divided, remainder, power, maximum, minimum, loss) =
        ctx.with_eager_session(|session| {
            assert!(matches!(
                session.div(&x, &foreign),
                Err(Error::ContextMismatch { .. })
            ));
            let divided = session.div(&x, &scalar)?;
            let remainder = session.rem(&x, &scalar)?;
            let power = session.pow(&x, &scalar)?;
            let maximum = session.maximum(&x, &scalar)?;
            let minimum = session.minimum(&x, &scalar)?;
            let loss = session.reduce_sum(&divided, None)?;
            Ok::<_, Error>((divided, remainder, power, maximum, minimum, loss))
        })?;
    assert_eq!(divided.value()?.as_slice::<f64>()?, &[3.0, 4.0]);
    assert_eq!(remainder.value()?.as_slice::<f64>()?, &[0.0, 0.0]);
    assert_eq!(power.value()?.as_slice::<f64>()?, &[36.0, 64.0]);
    assert_eq!(maximum.value()?.as_slice::<f64>()?, &[6.0, 8.0]);
    assert_eq!(minimum.value()?.as_slice::<f64>()?, &[2.0, 2.0]);
    assert_eq!(
        ctx.grad(&loss, &x)?.value()?.as_slice::<f64>()?,
        &[0.5, 0.5]
    );
    assert_eq!(
        ctx.grad(&loss, &scalar)?.value()?.as_slice::<f64>()?,
        &[-3.5]
    );
    Ok(())
}

#[test]
fn runtime_bound_eager_select_and_clamp_preserve_broadcast_ad() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![3],
        vec![-2.0_f64, 0.5, 5.0],
    )?)?;
    let (selected, clamped, selected_loss, clamped_loss) = ctx.with_eager_session(|session| {
        let zero = session.constant_from(Tensor::from_vec_col_major(vec![], vec![0.0_f64])?)?;
        let lo = session.constant_from(Tensor::from_vec_col_major(vec![], vec![-1.0_f64])?)?;
        let hi = session.constant_from(Tensor::from_vec_col_major(vec![], vec![4.0_f64])?)?;
        let positive = session.compare(&x, &zero, tenferro_tensor::CompareDir::Gt)?;
        let selected = session.select(&positive, &x, &zero)?;
        let clamped = session.clamp(&x, &lo, &hi)?;
        let selected_loss = session.reduce_sum(&selected, None)?;
        let clamped_loss = session.reduce_sum(&clamped, None)?;
        Ok::<_, Error>((selected, clamped, selected_loss, clamped_loss))
    })?;
    assert_eq!(selected.value()?.as_slice::<f64>()?, &[0.0, 0.5, 5.0]);
    assert_eq!(clamped.value()?.as_slice::<f64>()?, &[-1.0, 0.5, 4.0]);
    assert_eq!(
        ctx.grad(&selected_loss, &x)?.value()?.as_slice::<f64>()?,
        &[0.0, 1.0, 1.0]
    );
    assert_eq!(
        ctx.grad(&clamped_loss, &x)?.value()?.as_slice::<f64>()?,
        &[0.0, 1.0, 0.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_eager_convert_preserves_checked_and_lossy_paths() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let (converted, casted) = ctx.with_eager_session(|session| {
        let x = session.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.2_f64, -2.8])?)?;
        assert!(matches!(
            session.convert(&x, tenferro_tensor::DType::I32),
            Err(Error::TensorRuntime(_))
        ));
        let converted = session.convert(&x, tenferro_tensor::DType::C64)?;
        let casted = session.cast(&x, tenferro_tensor::DType::I32)?;
        Ok::<_, Error>((converted, casted))
    })?;
    assert_eq!(converted.dtype(), tenferro_tensor::DType::C64);
    assert_eq!(casted.value()?.as_slice::<i32>()?, &[1, -2]);
    Ok(())
}

#[test]
fn runtime_bound_eager_transpose_and_slice_preserve_ad_and_ownership() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let (sliced, copy, loss) = ctx.with_eager_session(|session| {
        assert!(matches!(
            session.transpose(&x, &[0, 0]),
            Err(Error::TensorRuntime(_))
        ));
        let transposed = session.transpose(&x, &[1, 0])?;
        let sliced = session.slice(
            &transposed,
            tenferro_tensor::SliceConfig {
                starts: vec![0, 0],
                limits: vec![2, 1],
                strides: vec![1, 1],
            },
        )?;
        let copy = session.duplicate_value(&sliced)?;
        let loss = session.reduce_sum(&sliced, None)?;
        Ok::<_, Error>((sliced, copy, loss))
    })?;
    assert_eq!(sliced.shape(), &[2, 1]);
    assert_eq!(copy.as_slice::<f64>()?, &[1.0, 3.0]);
    assert_eq!(
        ctx.grad(&loss, &x)?.value()?.as_slice::<f64>()?,
        &[1.0, 0.0, 1.0, 0.0]
    );
    Ok(())
}

#[test]
fn runtime_bound_captured_views_materialize_untracked_inputs_in_session() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let constant = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let untracked = ctx.with_eager_session(|session| session.add(&constant, &constant))?;
    assert!(untracked.semantic_trace.is_none());
    let (reshaped, transposed, sliced, broadcast) = ctx.with_eager_session(|session| {
        let _capture = ctx.capture_trace();
        let reshaped = session.reshape(&untracked, [2, 1])?;
        let transposed = session.transpose(&reshaped, &[1, 0])?;
        let sliced = session.slice(
            &transposed,
            tenferro_tensor::SliceConfig {
                starts: vec![0, 0],
                limits: vec![1, 2],
                strides: vec![1, 1],
            },
        )?;
        let broadcast = session.broadcast_in_dim(&untracked, &[2, 2], &[0])?;
        Ok::<_, Error>((reshaped, transposed, sliced, broadcast))
    })?;
    for (index, value) in [&reshaped, &transposed, &sliced, &broadcast]
        .iter()
        .enumerate()
    {
        assert!(
            value.semantic_trace.is_some(),
            "missing trace for view {index}"
        );
    }
    assert_eq!(
        broadcast.duplicate_value()?.as_slice::<f64>()?,
        &[2.0, 4.0, 2.0, 4.0]
    );
    Ok(())
}

#[test]
fn borrowed_eager_no_grad_guard_is_scoped_inside_backend_callback() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let variable = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?)?;
    let output = ctx.with_eager_session(|session| {
        let _guard = ctx.no_grad();
        session.mul(&variable, &variable)
    })?;
    assert!(!output.tracks_grad());
    assert_eq!(output.value()?.as_slice::<f64>()?, &[4.0]);
    Ok(())
}

#[test]
fn runtime_bound_eager_views_preserve_values_and_validate_before_dispatch() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let other = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let foreign = other.variable_from(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?)?;

    let (reshaped, broadcast, reshaped_copy, broadcast_copy) =
        ctx.with_eager_session(|session| {
            assert!(matches!(
                session.reshape(&foreign, [1, 2]),
                Err(Error::ContextMismatch { .. })
            ));
            let reshaped = session.reshape(&x, [1, 2])?;
            let broadcast = session.broadcast_in_dim(&x, &[2, 2], &[0])?;
            let reshaped_copy = session.duplicate_value(&reshaped)?;
            let broadcast_copy = session.duplicate_value(&broadcast)?;
            Ok::<_, Error>((reshaped, broadcast, reshaped_copy, broadcast_copy))
        })?;
    assert_eq!(reshaped.shape(), &[1, 2]);
    assert_eq!(broadcast.shape(), &[2, 2]);
    assert_eq!(reshaped_copy.as_slice::<f64>()?, &[1.0, 2.0]);
    assert_eq!(broadcast_copy.as_slice::<f64>()?, &[1.0, 2.0, 1.0, 2.0]);
    Ok(())
}

#[test]
fn tracked_eager_operation_uses_the_borrowed_session_once() -> Result<(), Error> {
    let materializations = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(AtomicUsize::new(0));
    let ctx = Arc::new(EagerRuntime::from_backend(
        EagerBackend::recording_cpu_counting_sessions(materializations, Arc::clone(&sessions)),
    )?);
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64])?)?;
    assert_eq!(sessions.load(Ordering::Relaxed), 0);
    let y = ctx.with_eager_session(|session| session.neg(&x))?;
    assert_eq!(sessions.load(Ordering::Relaxed), 1);
    assert_eq!(y.value()?.as_slice::<f64>()?, &[-2.0]);
    Ok(())
}

#[test]
fn borrowed_leaf_constructors_share_one_runtime_session() -> Result<(), Error> {
    let materializations = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(AtomicUsize::new(0));
    let ctx = Arc::new(EagerRuntime::from_backend(
        EagerBackend::recording_cpu_counting_sessions(materializations, Arc::clone(&sessions)),
    )?);
    let (variable, constant) = ctx.with_eager_session(|session| {
        let variable =
            session.variable_from(Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?)?;
        let constant =
            session.constant_from(Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?)?;
        Ok::<_, Error>((variable, constant))
    })?;
    assert_eq!(sessions.load(Ordering::Relaxed), 1);
    assert!(variable.tracks_grad());
    assert!(!constant.tracks_grad());
    assert_eq!(variable.value()?.as_slice::<f64>()?, &[2.0]);
    assert_eq!(constant.value()?.as_slice::<f64>()?, &[3.0]);
    Ok(())
}

#[test]
fn host_leaf_construction_does_not_enter_a_backend_session() -> Result<(), Error> {
    let materializations = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(AtomicUsize::new(0));
    let ctx = Arc::new(EagerRuntime::from_backend(
        EagerBackend::recording_cpu_counting_sessions(
            Arc::clone(&materializations),
            Arc::clone(&sessions),
        ),
    )?);

    let native = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])
        .map_err(Error::from)?;
    let leaf = EagerTensor::from_tensor_in(native, Arc::clone(&ctx))?;

    assert_eq!(
        sessions.load(Ordering::Relaxed),
        0,
        "a host-placement leaf must not open a backend session"
    );
    assert_eq!(
        leaf.value()?.as_slice::<f64>().map_err(Error::from)?,
        &[1.0, 2.0, 3.0, 4.0],
        "the session-free path must materialize the same value"
    );

    // The counter observes real entries: the borrowed eager region enters a session.
    let doubled = leaf.runtime().with_eager_session(|s| s.mul(&leaf, &leaf))?;
    assert!(
        sessions.load(Ordering::Relaxed) > 0,
        "an eager operation must still enter a backend session"
    );
    assert_eq!(
        doubled.value()?.as_slice::<f64>().map_err(Error::from)?,
        &[1.0, 4.0, 9.0, 16.0]
    );
    Ok(())
}

#[test]
fn gradient_slots_share_the_callers_session() -> Result<(), Error> {
    let materializations = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(AtomicUsize::new(0));
    let ctx = Arc::new(EagerRuntime::from_backend(
        EagerBackend::recording_cpu_counting_sessions(materializations, Arc::clone(&sessions)),
    )?);
    let left = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?)?;
    let right = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?)?;
    let cotangents = HashMap::from([
        (
            left.key.clone(),
            Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
        ),
        (
            right.key.clone(),
            Tensor::from_vec_col_major(vec![1], vec![4.0_f64])?,
        ),
    ]);

    let before = sessions.load(Ordering::Relaxed);
    ctx.with_execution_session(|session| ctx.store_grads(&cotangents, session))??;
    assert_eq!(sessions.load(Ordering::Relaxed), before + 1);
    assert_eq!(
        left.grad()?.unwrap().to_tensor()?.as_slice::<f64>()?,
        &[3.0]
    );
    assert_eq!(
        right.grad()?.unwrap().to_tensor()?.as_slice::<f64>()?,
        &[4.0]
    );
    Ok(())
}

#[test]
fn gradient_slots_accumulate_on_borrowed_cpu_session() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let left = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?)?;
    let right = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?)?;
    let cotangents = HashMap::from([
        (
            left.key.clone(),
            Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
        ),
        (
            right.key.clone(),
            Tensor::from_vec_col_major(vec![1], vec![4.0_f64])?,
        ),
    ]);

    for expected in [[3.0, 4.0], [6.0, 8.0]] {
        ctx.with_execution_session(|session| ctx.store_grads(&cotangents, session))??;
        assert_eq!(
            left.grad()?.unwrap().to_tensor()?.as_slice::<f64>()?,
            &[expected[0]]
        );
        assert_eq!(
            right.grad()?.unwrap().to_tensor()?.as_slice::<f64>()?,
            &[expected[1]]
        );
    }
    Ok(())
}

/// The eager fast path must accept and decline exactly what the CPU backend's
/// session path does, so leaf construction cannot diverge from the backend.
#[test]
fn host_leaf_materialization_matches_the_cpu_backend_acceptance() -> Result<(), Error> {
    let backend = EagerBackend::cpu(CpuBackend::new());
    let host = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])
        .map_err(Error::from)?;

    // Accepted: an owned host tensor with a preset scalar, with the same bytes
    // the CPU backend's own entry produces.
    let fast = backend
        .to_contiguous_host_read(&TensorRead::from_tensor(&host))
        .expect("an owned host tensor has a session-free path")
        .map_err(Error::from)?;
    let mut cpu = CpuBackend::new();
    let session = cpu
        .with_backend_session(|__s| __s.to_contiguous_read(TensorRead::from_tensor(&host)))
        .unwrap()
        .map_err(Error::from)?;
    assert_eq!(
        fast.as_slice::<f64>().map_err(Error::from)?,
        session.as_slice::<f64>().map_err(Error::from)?
    );

    // Declined: a view read keeps the session path.
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let view = TensorView::F64(
        TypedTensorView::from_col_major(&[2, 2], &data)
            .map_err(Error::from)?
            .transpose_view([1, 0])
            .map_err(Error::from)?,
    );
    assert!(backend
        .to_contiguous_host_read(&TensorRead::from_view(view))
        .is_none());

    // Declined: a device placement is refused by the CPU backend, so the eager
    // path must decline and let the session report that typed error.
    let mut placed =
        TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).map_err(Error::from)?;
    placed.set_placement(Placement {
        memory_kind: MemoryKind::Device,
        device: None,
        cpu_affinity: None,
    });
    let placed = Tensor::from_typed(placed);
    assert!(backend
        .to_contiguous_host_read(&TensorRead::from_tensor(&placed))
        .is_none());
    assert!(CpuBackend::new()
        .with_backend_session(|__s| __s.to_contiguous_read(TensorRead::from_tensor(&placed)))
        .unwrap()
        .is_err());

    // Declined: a caller-owned external scalar keeps the session path.
    let payload =
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![7.0_f64])
            .expect("valid host tensor");
    let external = Tensor::external(ErasedHostTensor::new(payload));
    assert!(backend
        .to_contiguous_host_read(&TensorRead::from_tensor(&external))
        .is_none());
    Ok(())
}

#[test]
fn backward_enters_a_bounded_number_of_backend_sessions() -> Result<(), Error> {
    let materializations = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(AtomicUsize::new(0));
    let ctx = Arc::new(EagerRuntime::from_backend(
        EagerBackend::recording_cpu_counting_sessions_with_engine(
            materializations,
            Arc::clone(&sessions),
        ),
    )?);
    let x = ctx.variable_from(Tensor::from_vec_col_major(
        vec![3],
        vec![1.0_f64, 2.0, 3.0],
    )?)?;
    let w = ctx.variable_from(Tensor::from_vec_col_major(
        vec![3],
        vec![4.0_f64, 5.0, 6.0],
    )?)?;
    let loss = ctx.with_eager_session(|session| {
        let xw = session.mul(&x, &w)?;
        let e = session.exp(&xw)?;
        let s = session.mul(&e, &x)?;
        session.reduce_sum(&s, Some(&[0]))
    })?;

    let before = sessions.load(Ordering::Relaxed);
    let grads = loss.backward()?;
    let entered = sessions.load(Ordering::Relaxed) - before;
    // Seed creation, derivative input staging and gradient storage: one entry
    // each, independent of how many residuals, bindings or gradients exist.
    assert!(entered <= 3, "backward entered {entered} backend sessions");
    assert!(grads.grad(&x.key).is_some());
    assert!(grads.grad(&w.key).is_some());
    Ok(())
}

#[test]
fn calling_thread_no_grad_governs_a_session_callback() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;

    // Whichever thread the managed executor picks, the outer guard governs it.
    let y = {
        let _guard = ctx.no_grad();
        ctx.with_eager_session(|s| s.neg(&x))?
    };
    assert!(
        !y.tracks_grad(),
        "the outer no_grad must reach the callback"
    );

    // The worker's counters are restored: without the guard, recording resumes
    // for a callback that may land on the same worker.
    let z = ctx.with_eager_session(|s| s.neg(&x))?;
    assert!(z.tracks_grad());
    Ok(())
}

#[test]
fn a_guard_started_inside_the_callback_stays_local_to_it() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?)?;
    let (untracked, tracked) = ctx.with_eager_session(|s| {
        let untracked = {
            let _guard = ctx.no_grad();
            s.neg(&x)?
        };
        Ok::<_, Error>((untracked, s.neg(&x)?))
    })?;
    assert!(!untracked.tracks_grad());
    assert!(tracked.tracks_grad());
    // Nothing leaked back to the calling thread.
    let after = ctx.with_eager_session(|s| s.neg(&x))?;
    assert!(after.tracks_grad());
    Ok(())
}

#[test]
fn calling_thread_capture_trace_governs_a_session_callback() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?,
        ctx.clone(),
    )?;
    let y = {
        let _capture = ctx.capture_trace();
        ctx.with_eager_session(|s| s.mul(&x, &x))?
    };
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0])?,
        ctx.clone(),
    )?;
    // Differentiating with respect to the untracked leaf needs the semantic
    // trace that only the outer capture guard makes the callback record.
    let dx = ctx.vjp(&y, &x, &seed)?;
    assert_eq!(dx.value()?.as_slice::<f64>()?, &[2.0, 4.0]);
    Ok(())
}

#[test]
fn inherited_modes_apply_on_another_thread_and_are_restored() {
    use super::super::{eager_capture_active, eager_grad_recording_enabled, InheritedEagerModes};

    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let modes = {
        let _no_grad = ctx.no_grad();
        let _capture = ctx.capture_trace();
        InheritedEagerModes::capture()
    };
    // An explicitly spawned thread stands in for an executor worker, so the
    // cross-thread path is exercised independent of executor scheduling.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert!(eager_grad_recording_enabled());
                assert!(!eager_capture_active());
                {
                    let _inherited = modes.enter();
                    assert!(!eager_grad_recording_enabled());
                    assert!(eager_capture_active());
                }
                assert!(eager_grad_recording_enabled());
                assert!(!eager_capture_active());

                // Restored on unwind as well.
                let unwound = std::panic::catch_unwind(|| {
                    let _inherited = modes.enter();
                    panic!("callback panicked");
                });
                assert!(unwound.is_err());
                assert!(eager_grad_recording_enabled());
                assert!(!eager_capture_active());
            })
            .join()
            .unwrap();
    });
    // On the capturing thread itself entering adds nothing.
    {
        let _inherited = modes.enter();
        assert!(eager_grad_recording_enabled());
    }
}

#[test]
fn into_value_refuses_a_view_layout_instead_of_changing_values() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap())?;
    let x = ctx.constant_from(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let y = ctx.with_eager_session(|s| s.transpose(&x, &[1, 0]))?;
    let handle = match y.into_value() {
        Ok(tensor) => panic!("extracted {:?}", tensor.as_slice::<f64>()),
        Err(crate::IntoValueError::Extract { value, .. }) => value,
        Err(crate::IntoValueError::NotUnique(_)) => panic!("the handle is unique"),
    };
    // The handle comes back unchanged and still reads the transposed values;
    // an explicit duplicate is the compact copy.
    let copy = ctx.with_eager_session(|s| s.duplicate_value(&handle))?;
    assert_eq!(copy.as_slice::<f64>()?, &[1.0, 3.0, 2.0, 4.0]);
    Ok(())
}

#[test]
fn nested_entry_into_the_same_runtime_is_rejected_without_deadlock() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?)?;
    let (eager, execution) = ctx.with_eager_session(|_| {
        Ok::<_, Error>((
            ctx.with_eager_session(|s| s.neg(&x)),
            ctx.with_execution_session(|_| ()),
        ))
    })?;
    for nested in [eager.map(|_| ()), execution] {
        assert!(matches!(
            nested,
            Err(Error::SessionEntry(
                tenferro_tensor::SessionEntryError::Reentered { .. }
            ))
        ));
    }
    // The runtime is usable again once the outer session ends.
    let y = ctx.with_eager_session(|s| s.neg(&x))?;
    assert_eq!(y.value()?.as_slice::<f64>()?, &[-2.0]);
    Ok(())
}

/// A held backend session keeps its admission for its whole lifetime, so an eager entry on that
/// thread must be rejected before it waits on this runtime's owner lock (#1945 U3).
#[test]
fn a_held_backend_session_rejects_eager_entry_before_waiting() -> Result<(), Error> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap())?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?)?;
    let marker =
        tenferro_tensor::HeldSessionMarker::enter("held session under test").expect("marker");
    assert_reentered(ctx.with_eager_session(|s| s.neg(&x)).map(|_| ()));
    assert_reentered(ctx.with_execution_session(|_| ()).map(|_| ()));
    drop(marker);
    // The runtime is usable again once the held session ends.
    let y = ctx.with_eager_session(|s| s.neg(&x))?;
    assert_eq!(y.value()?.as_slice::<f64>()?, &[-2.0]);
    Ok(())
}

fn assert_reentered<T: std::fmt::Debug>(result: Result<T, Error>) {
    assert!(
        matches!(
            result,
            Err(Error::SessionEntry(
                tenferro_tensor::SessionEntryError::Reentered { .. }
            ))
        ),
        "expected a typed reentry rejection, got {result:?}"
    );
}

/// #1946 F1: entering a *different* runtime from a session callback must be
/// rejected before waiting on its owner lock. Otherwise this order deadlocks:
/// runtime A's callback holds the CPU permit; another thread holds runtime B's
/// owner lock and waits for that permit; A's callback waits for B's owner lock.
/// The scenario runs on a helper thread so a regression fails the test through
/// the watchdog instead of hanging the suite.
#[test]
fn nested_entry_into_another_runtime_is_rejected_before_its_owner_lock() {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let first = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
        let second = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let other = second.clone();
        let nested_target = second.clone();
        let nested = first
            .with_execution_session(move |_| {
                // Another thread takes runtime B's owner lock and then needs the
                // CPU permit that this callback holds.
                let worker = std::thread::spawn(move || {
                    let mut backend = other.lock_backend().unwrap();
                    locked_tx.send(()).unwrap();
                    let _ = release_rx.recv();
                    backend.with_backend_session(|_| ()).unwrap();
                });
                locked_rx.recv().unwrap();
                let nested = nested_target.with_execution_session(|_| ());
                release_tx.send(()).unwrap();
                (nested, worker)
            })
            .unwrap();
        let (nested, worker) = nested;
        // The worker gets the permit once the outer callback has returned.
        worker.join().unwrap();
        // Both runtimes recover for independent top-level calls.
        first.with_execution_session(|_| ()).unwrap();
        second.with_execution_session(|_| ()).unwrap();
        done_tx.send(nested.map(|_| ())).unwrap();
    });
    let nested = done_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("nested cross-runtime entry deadlocked (#1946 F1)");
    assert_reentered(nested);
}

/// #1946 F1: an eager runtime entered from inside a plain CPU backend session
/// must reject before its owner lock too. The deadlocking order: this thread's
/// CPU session holds the permit; another thread holds the runtime's owner lock
/// and waits for the permit; this thread waits for the owner lock.
#[test]
fn eager_entry_from_a_cpu_backend_session_is_rejected_before_its_owner_lock() {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
        let x = ctx
            .variable_from(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap())
            .unwrap();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let owner = ctx.clone();
        let nested_ctx = ctx.clone();
        let nested_x = x.clone();
        let mut backend = CpuBackend::with_threads(1).unwrap();
        let (nested, worker) = backend
            .with_backend_session(move |_| {
                let worker = std::thread::spawn(move || {
                    let mut locked = owner.lock_backend().unwrap();
                    locked_tx.send(()).unwrap();
                    let _ = release_rx.recv();
                    locked.with_backend_session(|_| ()).unwrap();
                });
                locked_rx.recv().unwrap();
                let nested = nested_ctx
                    .with_eager_session(|s| s.neg(&nested_x))
                    .map(|_| ());
                release_tx.send(()).unwrap();
                (nested, worker)
            })
            .unwrap();
        worker.join().unwrap();
        // The runtime recovers for an independent top-level call.
        let y = ctx.with_eager_session(|s| s.neg(&x)).unwrap();
        assert_eq!(y.value().unwrap().as_slice::<f64>().unwrap(), &[-3.0]);
        done_tx.send(nested).unwrap();
    });
    let nested = done_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("eager entry from a CPU session deadlocked (#1946 F1)");
    assert_reentered(nested);
}

/// #1946 F1: inside a shared CPU execution scope the permit is already held, so
/// a busy runtime owner is reported as contended instead of waited for; a free
/// owner is taken and the operation runs under the scope.
#[test]
fn eager_entry_in_an_execution_scope_does_not_wait_on_a_busy_owner() {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let backend = CpuBackend::with_threads(1).unwrap();
        let ctx = EagerRuntime::with_cpu_backend(backend.clone()).unwrap();
        let x = ctx
            .variable_from(Tensor::from_vec_col_major(vec![1], vec![5.0_f64]).unwrap())
            .unwrap();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let owner = ctx.clone();
        let scoped_ctx = ctx.clone();
        let scoped_x = x.clone();
        let (free, busy, worker) = backend
            .with_execution_scope(move || {
                let free = scoped_ctx.with_eager_session(|s| s.neg(&scoped_x)).unwrap();
                let worker = std::thread::spawn(move || {
                    let locked = owner.lock_backend().unwrap();
                    locked_tx.send(()).unwrap();
                    let _ = release_rx.recv();
                    drop(locked);
                });
                locked_rx.recv().unwrap();
                let busy = scoped_ctx
                    .with_eager_session(|s| s.neg(&scoped_x))
                    .map(|_| ());
                release_tx.send(()).unwrap();
                (free, busy, worker)
            })
            .unwrap();
        worker.join().unwrap();
        done_tx
            .send((
                free.value().unwrap().as_slice::<f64>().unwrap().to_vec(),
                busy,
            ))
            .unwrap();
    });
    let (free, busy) = done_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("eager entry in an execution scope waited on a busy owner (#1946 F1)");
    assert_eq!(free, vec![-5.0]);
    assert!(
        matches!(
            busy,
            Err(Error::SessionEntry(
                tenferro_tensor::SessionEntryError::Contended { .. }
            ))
        ),
        "expected a typed contention, got {busy:?}"
    );
}
