use std::collections::HashSet;

use crate::dim_expr::DimExpr;

#[test]
fn test_const_eval() {
    assert_eq!(DimExpr::Const(42).eval(&[]), 42);
}

#[test]
fn test_input_dim_eval() {
    let e = DimExpr::InputDim {
        input_idx: 0,
        axis: 1,
    };
    assert_eq!(e.eval(&[&[3, 7, 5]]), 7);
}

#[test]
fn test_arithmetic() {
    let shapes: &[&[usize]] = &[&[3, 4], &[5]];
    let e = DimExpr::mul(
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: 0,
            axis: 1,
        },
    );
    assert_eq!(e.eval(shapes), 12);
    assert_eq!(DimExpr::add(e.clone(), DimExpr::Const(3)).eval(shapes), 15);
    assert_eq!(DimExpr::floor_div(e, DimExpr::Const(4)).eval(shapes), 3);
}

#[test]
#[should_panic(expected = "DimExpr::Sub underflow")]
fn test_sub_underflow_panics_instead_of_wrapping() {
    let expr = DimExpr::sub(DimExpr::Const(2), DimExpr::Const(5));
    let _ = expr.eval(&[]);
}

#[test]
fn test_min_max() {
    let shapes: &[&[usize]] = &[&[3, 7]];
    let e_min = DimExpr::min(
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: 0,
            axis: 1,
        },
    );
    assert_eq!(e_min.eval(shapes), 3);
    let e_max = DimExpr::max(
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: 0,
            axis: 1,
        },
    );
    assert_eq!(e_max.eval(shapes), 7);
}

#[test]
fn test_max_input_idx() {
    assert_eq!(DimExpr::Const(5).max_input_idx(), None);
    assert_eq!(
        DimExpr::InputDim {
            input_idx: 2,
            axis: 0
        }
        .max_input_idx(),
        Some(2)
    );
    let e = DimExpr::add(
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: 3,
            axis: 1,
        },
    );
    assert_eq!(e.max_input_idx(), Some(3));
}

#[test]
fn test_remap() {
    let e = DimExpr::mul(
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: 0,
            axis: 1,
        },
    );
    let remapped = e.remap(0, 1);
    assert_eq!(
        remapped,
        DimExpr::mul(
            DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            },
            DimExpr::InputDim {
                input_idx: 1,
                axis: 1,
            },
        )
    );
}

#[test]
fn test_remap_selective() {
    let e = DimExpr::add(
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::InputDim {
            input_idx: 2,
            axis: 1,
        },
    );
    let remapped = e.remap(0, 1);
    assert_eq!(
        remapped,
        DimExpr::add(
            DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            },
            DimExpr::InputDim {
                input_idx: 2,
                axis: 1,
            },
        )
    );
}

#[test]
fn test_input_shape() {
    let exprs = DimExpr::input_shape(0, 3);
    assert_eq!(exprs.len(), 3);
    assert_eq!(
        exprs[0],
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0
        }
    );
    assert_eq!(
        exprs[2],
        DimExpr::InputDim {
            input_idx: 0,
            axis: 2
        }
    );
}

#[test]
fn test_from_concrete() {
    let exprs = DimExpr::from_concrete(&[3, 4, 5]);
    assert_eq!(
        exprs,
        vec![DimExpr::Const(3), DimExpr::Const(4), DimExpr::Const(5)]
    );
}

#[test]
fn test_hash_eq_structural() {
    let a = DimExpr::mul(DimExpr::Const(2), DimExpr::Const(3));
    let b = DimExpr::mul(DimExpr::Const(2), DimExpr::Const(3));
    let c = DimExpr::mul(DimExpr::Const(3), DimExpr::Const(2));
    assert_eq!(a, b);
    assert_ne!(a, c);
    let mut set = HashSet::new();
    set.insert(a.clone());
    assert!(set.contains(&b));
    assert!(!set.contains(&c));
}
