"""Reference values for `tests/philox.rs`, `tests/samples.rs` and `tests/math.rs`.

An implementation of the layout documented in `src/lib.rs`, written
again in Python: the two block functions (checked here against the Random123
known-answer vectors), the word stream, the Philox key domains, and the
transforms. Exact transforms (bits, uniforms, integers) are computed exactly;
normal samples and `log`/`sincospi` are the exact mathematical values from
mpmath, rounded once to the output type, so a test can bound the distance of
tenferro's bits from them. Recorded with Python 3.12, mpmath 1.4.1, numpy 2.5.5:

    python3 reference/samples.py
"""

import struct

import mpmath as mp
import numpy as np

mp.mp.prec = 200
M32 = 0xFFFFFFFF


def philox(ctr, key):
    ctr, key = list(ctr), list(key)
    for rnd in range(10):
        if rnd:
            key = [(key[0] + 0x9E3779B9) & M32, (key[1] + 0xBB67AE85) & M32]
        p0, p1 = 0xD2511F53 * ctr[0], 0xCD9E8D57 * ctr[2]
        ctr = [(p1 >> 32) ^ ctr[1] ^ key[0], p1 & M32, (p0 >> 32) ^ ctr[3] ^ key[1], p0 & M32]
    return ctr


def threefry(ctr, key):
    rot = [13, 15, 26, 6, 17, 29, 16, 24]
    ks = [key[0], key[1], 0x1BD11BDA ^ key[0] ^ key[1]]
    x0, x1 = (ctr[0] + ks[0]) & M32, (ctr[1] + ks[1]) & M32
    for r in range(20):
        x0 = (x0 + x1) & M32
        x1 = ((x1 << rot[r % 8]) | (x1 >> (32 - rot[r % 8]))) & M32
        x1 ^= x0
        if r % 4 == 3:
            i = r // 4 + 1
            x0 = (x0 + ks[i % 3]) & M32
            x1 = (x1 + ks[(i + 1) % 3] + i) & M32
    return [x0, x1]


assert philox([0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344], [0xA4093822, 0x299F31D0]) == [
    0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]
assert threefry([0x243F6A88, 0x85A308D3], [0x13198A2E, 0x03707344]) == [0xC4923A9C, 0x483DF7A0]

ALGORITHMS = {
    "philox": dict(
        words=4,
        block=lambda key, c: philox([c & M32, c >> 32, 0, 0], key),
        split=lambda key, i: philox([i & M32, i >> 32, 0, 1], key)[:2],
        fold_in=lambda key, d: philox([d, 0, 0, 2], key)[:2],
    ),
    "threefry": dict(
        words=2,
        block=lambda key, c: threefry([c >> 32, c & M32], key),
        split=lambda key, i: threefry([i >> 32, i & M32], key),
        fold_in=lambda key, d: threefry([0, d], key),
    ),
}


def stream(alg, key, first, n):
    w = ALGORITHMS[alg]["words"]
    return [ALGORITHMS[alg]["block"](key, j // w)[j % w] for j in range(first, first + n)]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f64_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def to_float(x, t):
    """mpf -> nearest float of type t (exactly representable result)."""
    with mp.workprec(24 if t == "f32" else 53):
        y = +x
    return np.float32(float(y)) if t == "f32" else float(y)


def bits_of(x, t):
    return f"0x{f32_bits(x):08x}" if t == "f32" else f"0x{f64_bits(x):016x}"


def unit(words, t):
    """Mantissa fill: an exact mpf in [0, 1)."""
    if t == "f32":
        return mp.mpf(words[0] >> 9) / 2**23
    return mp.mpf(((words[0] << 32) | words[1]) >> 12) / 2**52


W = {"f32": 1, "f64": 2}


def uniform(alg, key, i, t):
    return unit(stream(alg, key, W[t] * i, W[t]), t)


def uniform_complex(alg, key, i, t):
    ws = stream(alg, key, 2 * W[t] * i, 2 * W[t])
    return unit(ws[: W[t]], t), unit(ws[W[t]:], t)


def box_muller(alg, key, d, t):
    a, b = uniform_complex(alg, key, d, t)
    return mp.log(1 - a), mp.sinpi(2 * b), mp.cospi(2 * b)


def normal(alg, key, i, t):
    log_u, s, c = box_muller(alg, key, i // 2, t)
    r = mp.sqrt(-2 * log_u)
    return r * (c if i % 2 == 0 else s)


def normal_complex(alg, key, i, t):
    log_u, s, c = box_muller(alg, key, i, t)
    r = mp.sqrt(-log_u)
    return r * c, r * s


def integer(alg, key, i, lo, hi):
    ws = stream(alg, key, 2 * i, 2)
    return lo + ((((ws[0] << 32) | ws[1]) * (hi - lo)) >> 64)


def uniform_range(alg, key, i, t, lo, hi):
    """`lo + (hi - lo) * u` in the IEEE arithmetic of t, below hi."""
    ty = np.float32 if t == "f32" else np.float64
    u = ty(float(uniform(alg, key, i, t)))
    x = ty(ty(lo) + (ty(hi) - ty(lo)) * u)
    return x if x < ty(hi) else np.nextafter(ty(hi), ty(-np.inf))


def row(name, values):
    print(f"{name}: [{', '.join(values)}]")


KEY_SEED = 2026
for alg in ("philox", "threefry"):
    key = [KEY_SEED >> 32, KEY_SEED & M32]
    print(f"== {alg}, key = seed({KEY_SEED}) = {key}")
    row("split(3)", [f"[0x{a:08x}, 0x{b:08x}]" for a, b in (ALGORITHMS[alg]["split"](key, i) for i in range(3))])
    row("fold_in(0, 1, 0xdeadbeef)", [f"[0x{a:08x}, 0x{b:08x}]" for a, b in
                                      (ALGORITHMS[alg]["fold_in"](key, d) for d in (0, 1, 0xDEADBEEF))])
    row("bits 0..8", [f"0x{w:08x}" for w in stream(alg, key, 0, 8)])
    for t in ("f32", "f64"):
        row(f"uniform {t} 0..8", [bits_of(to_float(uniform(alg, key, i, t), t), t) for i in range(8)])
        row(f"uniform_range {t} [-2, 3) 0..8", [bits_of(uniform_range(alg, key, i, t, -2.0, 3.0), t) for i in range(8)])
        row(f"uniform_complex {t} 0..4", [f"[{bits_of(to_float(a, t), t)}, {bits_of(to_float(b, t), t)}]"
                                          for a, b in (uniform_complex(alg, key, i, t) for i in range(4))])
        row(f"normal {t} 0..8 (mpmath, rounded once)", [bits_of(to_float(normal(alg, key, i, t), t), t) for i in range(8)])
        row(f"normal_complex {t} 0..4 (mpmath, rounded once)", [
            f"[{bits_of(to_float(a, t), t)}, {bits_of(to_float(b, t), t)}]"
            for a, b in (normal_complex(alg, key, i, t) for i in range(4))])
    row("integer i32 [-5, 5) 0..8", [str(integer(alg, key, i, -5, 5)) for i in range(8)])
    row("integer i32 [MIN, MAX) 0..8", [str(integer(alg, key, i, -2**31, 2**31 - 1)) for i in range(8)])
    row("integer i64 [-2^62, 2^62) 0..8", [str(integer(alg, key, i, -2**62, 2**62)) for i in range(8)])

print("== log and sincospi at fixed points (mpmath, rounded once)")
LOG_POINTS = {
    # Box-Muller extremes: the smallest 1 - a, the largest below 1, and 1.
    "f32": ["2^-23", "1 - 2^-23", "1", "0.5", "0.75", "0.7071067", "1.4142135", "10", "1e-30", "3e38", "1e-45"],
    "f64": ["2^-52", "1 - 2^-52", "1", "0.5", "0.75", "0.7071067811865476", "1.4142135623730951", "10",
            "1e-300", "1.7e308", "5e-324"],
}
SINCOSPI_POINTS = {
    # 2b covers [0, 2) in steps of 2^-22 (f32) or 2^-51 (f64); 0.25 and 0.5
    # with neighbours are where the quadrant changes.
    "f32": ["0", "2^-22", "0.25 - 2^-25", "0.25", "0.25 + 2^-24", "0.5 - 2^-24", "0.5", "0.5 + 2^-23",
            "1", "1.5", "2 - 2^-22", "0.1", "-0.75", "12345.5"],
    "f64": ["0", "2^-51", "0.25 - 2^-54", "0.25", "0.25 + 2^-53", "0.5 - 2^-53", "0.5", "0.5 + 2^-52",
            "1", "1.5", "2 - 2^-51", "0.1", "-0.75", "12345.5"],
}


def parse(expr, t):
    value = eval(expr.replace("^", "**"), {})
    return to_float(mp.mpf(value), t)


for t in ("f32", "f64"):
    for expr in LOG_POINTS[t]:
        x = parse(expr, t)
        print(f"log {t} x={bits_of(x, t)} ({expr}) -> {bits_of(to_float(mp.log(mp.mpf(float(x))), t), t)}")
    for expr in SINCOSPI_POINTS[t]:
        x = parse(expr, t)
        xm = mp.mpf(float(x))
        print(f"sincospi {t} x={bits_of(x, t)} ({expr}) -> "
              f"{bits_of(to_float(mp.sinpi(xm), t), t)}, {bits_of(to_float(mp.cospi(xm), t), t)}")
