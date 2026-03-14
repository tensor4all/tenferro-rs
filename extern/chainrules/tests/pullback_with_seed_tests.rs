use chainrules::{Differentiable, Tape};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    x: f64,
    y: f64,
}

impl Differentiable for Pair {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent {
        Self { x: 0.0, y: 0.0 }
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self {
            x: a.x + b.x,
            y: a.y + b.y,
        }
    }

    fn num_elements(&self) -> usize {
        2
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        Self { x: 1.0, y: 1.0 }
    }
}

#[test]
fn pullback_with_seed_accepts_tensor_like_outputs() {
    let tape = Tape::<Pair>::new();
    let x = tape.leaf(Pair { x: 1.0, y: -2.0 });
    let grads = tape
        .pullback_with_seed(&x, Pair { x: 3.0, y: 4.0 })
        .unwrap();
    assert_eq!(
        *grads.get(x.node_id().unwrap()).unwrap(),
        Pair { x: 3.0, y: 4.0 }
    );
}

#[test]
fn plain_pullback_still_requires_scalar_outputs() {
    let tape = Tape::<Pair>::new();
    let x = tape.leaf(Pair { x: 1.0, y: -2.0 });
    let err = match tape.pullback(&x) {
        Ok(_) => panic!("plain pullback should still reject non-scalar outputs"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        chainrules::AutodiffError::NonScalarLoss { num_elements: 2 }
    ));
}
