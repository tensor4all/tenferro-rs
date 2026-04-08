#!/usr/bin/env python3
"""Numerically validate the vector examples in ad-architecture.md.

This script checks the two 1D vector examples added to the design doc:

1. Elementwise `y = exp(a * x)` with `x, a in R^2`
2. Reduction `y = sum(exp(a * x))` with `x, a in R^2`

It prints one readable sample for manual inspection, then runs randomized
finite-difference and adjoint-identity checks.
"""

from __future__ import annotations

import math
import random
from typing import Callable


def elem_mul(xs: list[float], ys: list[float]) -> list[float]:
    return [x * y for x, y in zip(xs, ys)]


def elem_exp(xs: list[float]) -> list[float]:
    return [math.exp(x) for x in xs]


def dot(xs: list[float], ys: list[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))


def f_vec(x: list[float], a: list[float]) -> list[float]:
    return elem_exp(elem_mul(a, x))


def g_sum(x: list[float], a: list[float]) -> float:
    return sum(f_vec(x, a))


def jvp_vec(x: list[float], a: list[float], dx: list[float]) -> dict[str, list[float]]:
    p2 = elem_mul(a, x)
    p3 = elem_exp(p2)
    l1 = elem_mul(a, dx)
    l2 = elem_mul(p3, l1)
    return {"p2": p2, "p3": p3, "l1": l1, "l2": l2, "dy": l2}


def vjp_vec(x: list[float], a: list[float], ct_y: list[float]) -> dict[str, list[float]]:
    p2 = elem_mul(a, x)
    p3 = elem_exp(p2)
    t1 = elem_mul(p3, ct_y)
    t2 = elem_mul(a, t1)
    return {"p2": p2, "p3": p3, "t1": t1, "t2": t2, "ct_x": t2}


def jvp_sum(x: list[float], a: list[float], dx: list[float]) -> dict[str, object]:
    p2 = elem_mul(a, x)
    p3 = elem_exp(p2)
    l1 = elem_mul(a, dx)
    l2 = elem_mul(p3, l1)
    l3 = sum(l2)
    return {"p2": p2, "p3": p3, "l1": l1, "l2": l2, "l3": l3, "dy": l3}


def vjp_sum(x: list[float], a: list[float], ct_y: float) -> dict[str, object]:
    p2 = elem_mul(a, x)
    p3 = elem_exp(p2)
    t1 = [ct_y, ct_y]
    t2 = elem_mul(p3, t1)
    t3 = elem_mul(a, t2)
    return {"p2": p2, "p3": p3, "t1": t1, "t2": t2, "t3": t3, "ct_x": t3}


def central_diff_vec(
    fun: Callable[[list[float]], list[float]],
    x: list[float],
    dx: list[float],
    eps: float = 1e-6,
) -> list[float]:
    x_plus = [xi + eps * dxi for xi, dxi in zip(x, dx)]
    x_minus = [xi - eps * dxi for xi, dxi in zip(x, dx)]
    y_plus = fun(x_plus)
    y_minus = fun(x_minus)
    return [(yp - ym) / (2.0 * eps) for yp, ym in zip(y_plus, y_minus)]


def central_diff_scalar(
    fun: Callable[[list[float]], float],
    x: list[float],
    dx: list[float],
    eps: float = 1e-6,
) -> float:
    x_plus = [xi + eps * dxi for xi, dxi in zip(x, dx)]
    x_minus = [xi - eps * dxi for xi, dxi in zip(x, dx)]
    return (fun(x_plus) - fun(x_minus)) / (2.0 * eps)


def print_sample_examples() -> None:
    x = [0.2, -0.4]
    a = [1.5, -0.7]
    dx = [0.3, -0.1]
    ct_y_vec = [0.8, -1.2]
    ct_y_scalar = 0.5

    print("Example A: y = exp(a * x), x,a in R^2, wrt x")
    res_a_fwd = jvp_vec(x, a, dx)
    res_a_bwd = vjp_vec(x, a, ct_y_vec)
    fd_a = central_diff_vec(lambda x_: f_vec(x_, a), x, dx)
    print("  primal p2 = a*x        =", res_a_fwd["p2"])
    print("  primal p3 = exp(p2)    =", res_a_fwd["p3"])
    print("  linear l1 = a*dx       =", res_a_fwd["l1"])
    print("  linear l2 = p3*l1      =", res_a_fwd["l2"])
    print("  finite-diff dy         =", fd_a)
    print("  reverse t1 = p3*ct_y   =", res_a_bwd["t1"])
    print("  reverse t2 = a*t1      =", res_a_bwd["t2"])
    print("  adjoint check <ct_y,dy> == <ct_x,dx>")
    print("   lhs =", dot(ct_y_vec, res_a_fwd["dy"]))
    print("   rhs =", dot(res_a_bwd["ct_x"], dx))

    print()
    print("Example B: y = sum(exp(a * x)), x,a in R^2, wrt x")
    res_b_fwd = jvp_sum(x, a, dx)
    res_b_bwd = vjp_sum(x, a, ct_y_scalar)
    fd_b = central_diff_scalar(lambda x_: g_sum(x_, a), x, dx)
    print("  primal p2 = a*x              =", res_b_fwd["p2"])
    print("  primal p3 = exp(p2)          =", res_b_fwd["p3"])
    print("  linear l1 = a*dx             =", res_b_fwd["l1"])
    print("  linear l2 = p3*l1            =", res_b_fwd["l2"])
    print("  linear l3 = sum(l2)          =", res_b_fwd["l3"])
    print("  finite-diff dy               =", fd_b)
    print("  reverse t1 = broadcast(ct_y) =", res_b_bwd["t1"])
    print("  reverse t2 = p3*t1           =", res_b_bwd["t2"])
    print("  reverse t3 = a*t2            =", res_b_bwd["t3"])
    print("  adjoint check ct_y*dy == <ct_x,dx>")
    print("   lhs =", ct_y_scalar * res_b_fwd["dy"])
    print("   rhs =", dot(res_b_bwd["ct_x"], dx))


def verify_randomized_cases() -> None:
    random.seed(0)

    for _ in range(50):
        x = [random.uniform(-1.0, 1.0) for _ in range(2)]
        a = [random.uniform(-1.5, 1.5) for _ in range(2)]
        dx = [random.uniform(-0.5, 0.5) for _ in range(2)]
        ct_vec = [random.uniform(-1.0, 1.0) for _ in range(2)]

        fwd = jvp_vec(x, a, dx)["dy"]
        fd = central_diff_vec(lambda x_: f_vec(x_, a), x, dx)
        max_err = max(abs(u - v) for u, v in zip(fwd, fd))
        adjoint_err = abs(dot(ct_vec, fwd) - dot(vjp_vec(x, a, ct_vec)["ct_x"], dx))
        assert max_err < 1e-8, (x, a, dx, fwd, fd)
        assert adjoint_err < 1e-10, (x, a, dx, ct_vec, adjoint_err)

    for _ in range(50):
        x = [random.uniform(-1.0, 1.0) for _ in range(2)]
        a = [random.uniform(-1.5, 1.5) for _ in range(2)]
        dx = [random.uniform(-0.5, 0.5) for _ in range(2)]
        ct = random.uniform(-1.0, 1.0)

        fwd = jvp_sum(x, a, dx)["dy"]
        fd = central_diff_scalar(lambda x_: g_sum(x_, a), x, dx)
        adjoint_err = abs(ct * fwd - dot(vjp_sum(x, a, ct)["ct_x"], dx))
        assert abs(fwd - fd) < 1e-8, (x, a, dx, fwd, fd)
        assert adjoint_err < 1e-10, (x, a, dx, ct, adjoint_err)

    print()
    print("verified: 50 vector elementwise cases + 50 reduction cases")


def main() -> None:
    print_sample_examples()
    verify_randomized_cases()


if __name__ == "__main__":
    main()
