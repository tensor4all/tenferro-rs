"""Derive the constants of `src/math.rs` (tenferro-rs#2714).

Every constant is a truncated Taylor coefficient or a split of ln 2 / pi,
rounded to nearest by mpmath at 200 bits; nothing is taken from another
library. Run with any Python that has mpmath (recorded with mpmath 1.4.1 and numpy, which formats the shortest f32 literal):

    python3 reference/math_constants.py
"""

import mpmath as mp
import numpy as np

mp.mp.prec = 200


def rnd(x, bits):
    """Round `x` to `bits` significant bits (round to nearest even)."""
    with mp.workprec(bits):
        return +x


def lit(x, bits):
    """The shortest literal that parses to `x` rounded to `bits` (24 or 53)."""
    value = float(rnd(x, bits))
    if bits == 53:
        return repr(value)
    if abs(value) >= 1e-4:
        return np.format_float_positional(np.float32(value), unique=True)
    return np.format_float_scientific(np.float32(value), unique=True)


def split(x, hi_bits, bits):
    hi = rnd(x, hi_bits)
    lo = rnd(x - hi, bits)
    return hi, lo


for name, bits, log_terms, sin_terms, cos_terms, ln2_hi_bits in (
    ("f64", 53, 11, 9, 9, 42),
    ("f32", 24, 5, 5, 6, 16),
):
    print(f"// {name}")
    hi, lo = split(mp.log(2), ln2_hi_bits, bits)
    print(f"LN2_HI = {lit(hi, bits)}  ({ln2_hi_bits} significant bits)")
    print(f"LN2_LO = {lit(lo, bits)}")
    print(f"SQRT_2 = {lit(mp.sqrt(2), bits)}")
    # log(1+f) = f - hfsq + s*(hfsq + R(z)), z = s^2, R(z) = sum_k 2/(2k+1) z^k.
    print("LOG_R =", [lit(mp.mpf(2) / (2 * k + 1), bits) for k in range(1, log_terms + 1)])
    hi, lo = split(mp.pi, bits, bits)
    print(f"PI_HI = {lit(hi, bits)}")
    print(f"PI_LO = {lit(lo, bits)}")
    # sin(pi r) = pi r + r^3 * S(r^2), S_k = (-1)^k pi^(2k+1) / (2k+1)!, k >= 1.
    print("SIN =", [lit((-1) ** k * mp.pi ** (2 * k + 1) / mp.factorial(2 * k + 1), bits)
                    for k in range(1, sin_terms)])
    # cos(pi r) = 1 + r^2 * C(r^2), C_k = (-1)^k pi^(2k) / (2k)!, k >= 1.
    print("COS =", [lit((-1) ** k * mp.pi ** (2 * k) / mp.factorial(2 * k), bits)
                    for k in range(1, cos_terms)])
