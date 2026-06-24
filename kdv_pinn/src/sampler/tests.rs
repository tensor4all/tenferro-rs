use super::*;

#[test]
fn collocation_has_correct_shape() {
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut rng = rand::thread_rng();
    let (x, t) = sampler.collocation(32, &mut rng);
    assert_eq!(x.shape(), &[32, 1]);
    assert_eq!(t.shape(), &[32, 1]);
}

#[test]
fn collocation_columns_are_x_and_t() {
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut rng = rand::thread_rng();
    let (x, t) = sampler.collocation(32, &mut rng);
    let x_vals = x.as_slice::<f64>().unwrap();
    let t_vals = t.as_slice::<f64>().unwrap();
    assert!(x_vals.iter().all(|&v| (-5.0..=5.0).contains(&v)));
    assert!(t_vals.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn initial_has_correct_shape_and_x_range() {
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut rng = rand::thread_rng();
    let (x, u) = sampler.initial(16, &mut rng);
    assert_eq!(x.shape(), &[16, 1]);
    assert_eq!(u.shape(), &[16, 1]);
    let x_vals = x.as_slice::<f64>().unwrap();
    assert!(x_vals.iter().all(|&v| (-5.0..=5.0).contains(&v)));
}

#[test]
fn boundary_has_correct_shape() {
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut rng = rand::thread_rng();
    let (x, t, u) = sampler.boundary(8, &mut rng);
    assert_eq!(x.shape(), &[8, 1]);
    assert_eq!(t.shape(), &[8, 1]);
    assert_eq!(u.shape(), &[8, 1]);
}

#[test]
fn boundary_is_stratified() {
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut rng = rand::thread_rng();
    let n = 8;
    let (x, _t, _u) = sampler.boundary(n, &mut rng);
    let x_vals = x.as_slice::<f64>().unwrap();
    let n_min = n / 2;
    assert!(x_vals[..n_min].iter().all(|&v| v == -5.0));
    assert!(x_vals[n_min..].iter().all(|&v| v == 5.0));
}
