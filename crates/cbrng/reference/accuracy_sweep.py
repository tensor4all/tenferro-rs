"""Accuracy sweep of `log` and `sincospi` against mpmath (tenferro-rs#2714).

Builds a throwaway driver crate in a temporary directory (a path dependency on
this crate) and feeds it about 140,000 inputs per run: the Box-Muller domains
(`1 - k 2^-p` for `log`, `k 2^(1-p)` for `sincospi`), a log-uniform spread
over the whole range, and points near 1, 0.25 and 0.5. It prints the largest
error in ulp of the exact value for each function and precision. Recorded with
Python 3.12, mpmath 1.4.1 and the default 20000 points per family:

    f32 cospi 0.9621   f32 log 0.7306   f32 sinpi 0.9411
    f64 cospi 0.9345   f64 log 0.7561   f64 sinpi 0.9840

The exhaustive f32 sweep over every Box-Muller input is the ignored test
`f32_functions_over_every_box_muller_input` in `tests/math.rs`.

    python3 reference/accuracy_sweep.py [points per family]
"""

import pathlib
import random
import struct
import subprocess
import sys
import tempfile

import mpmath as mp

mp.mp.prec = 160

# Reads "<f32|f64> <log|sincospi> <hex bits>" lines, writes the result bits.
DRIVER = r"""
use std::io::{BufRead, Write};
use cbrng::{log, sincospi};

fn main() {
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<&str> = line.split(' ').collect();
        let bits = u64::from_str_radix(fields[2], 16).unwrap();
        let result = match (fields[0], fields[1]) {
            ("f64", "log") => format!("{:016x}", log(f64::from_bits(bits)).to_bits()),
            ("f64", _) => {
                let (s, c) = sincospi(f64::from_bits(bits));
                format!("{:016x} {:016x}", s.to_bits(), c.to_bits())
            }
            (_, "log") => format!("{:08x}", log(f32::from_bits(bits as u32)).to_bits()),
            _ => {
                let (s, c) = sincospi(f32::from_bits(bits as u32));
                format!("{:08x} {:08x}", s.to_bits(), c.to_bits())
            }
        };
        writeln!(out, "{result}").unwrap();
    }
}
"""

PRECISION = {"f32": 24, "f64": 53}
MIN_EXPONENT = {"f32": -126, "f64": -1022}


def to_bits(x, t):
    if t == "f64":
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits(b, t):
    if t == "f64":
        return struct.unpack("<d", struct.pack("<Q", b))[0]
    return struct.unpack("<f", struct.pack("<I", b))[0]


def rounded(x, t):
    """`x` rounded to the nearest value of type t, as a Python float."""
    return from_bits(to_bits(float(x), t), t)


def ulp(y, t):
    """The ulp of the exact value y in type t."""
    p, emin = PRECISION[t], MIN_EXPONENT[t]
    y = abs(y)
    if y == 0:
        return mp.ldexp(1, emin - p + 1)
    e = max(int(mp.floor(mp.log(y, 2))), emin)
    while mp.ldexp(1, e) > y and e > emin:
        e -= 1
    while mp.ldexp(1, e + 1) <= y:
        e += 1
    return mp.ldexp(1, e - p + 1)


def ulp_error(got, exact, t):
    if exact == 0:
        return 0.0 if got == 0 else float("inf")
    return float(abs(mp.mpf(got) - exact) / ulp(exact, t))


def inputs(n):
    """(type, function, x) cases; the draw order fixes the recorded maxima."""
    random.seed(1)
    cases = []
    for t in ("f64", "f32"):
        p = PRECISION[t] - 1
        lo, hi = (-1070, 1023) if t == "f64" else (-140, 127)
        xs = [1 - random.randrange(0, 2**p) * 2.0**-p for _ in range(n)]
        xs += [2.0**-p, 1 - 2.0**-p, 0.5, 0.75, 1.0]
        xs += [mp.mpf(2) ** random.uniform(lo, hi) for _ in range(n)]
        xs += [1 + random.uniform(-1e-3, 1e-3) for _ in range(n // 4)]
        cases += [(t, "log", x) for x in (rounded(x, t) for x in xs) if x > 0]
        ys = [random.randrange(0, 2**p) * 2.0 ** (1 - p) for _ in range(n)]
        ys += [random.uniform(-100, 100) for _ in range(n // 4)]
        ys += [0.25 + k * 2.0**-p for k in range(-50, 50)]
        ys += [0.5 + k * 2.0**-p for k in range(-50, 50)]
        cases += [(t, "sincospi", rounded(y, t)) for y in ys]
    return cases


def run_driver(lines):
    crate = pathlib.Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "src").mkdir()
        (root / "Cargo.toml").write_text(
            '[package]\nname = "sweep-driver"\nversion = "0.0.0"\nedition = "2021"\n'
            f'[dependencies]\ncbrng = {{ path = "{crate}" }}\n[workspace]\n'
        )
        (root / "src" / "main.rs").write_text(DRIVER)
        subprocess.run(["cargo", "build", "-q", "--release"], cwd=root, check=True)
        return subprocess.run(
            [root / "target" / "release" / "sweep-driver"],
            input=lines, capture_output=True, text=True, check=True,
        ).stdout.splitlines()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    cases = inputs(n)
    results = run_driver("".join(f"{t} {f} {to_bits(x, t):x}\n" for t, f, x in cases))
    worst = {}
    for (t, f, x), line in zip(cases, results, strict=True):
        got = [from_bits(int(word, 16), t) for word in line.split()]
        xm = mp.mpf(x)
        if f == "log":
            errors = {"log": ulp_error(got[0], mp.log(xm), t)}
        else:
            errors = {
                "sinpi": ulp_error(got[0], mp.sinpi(xm), t),
                "cospi": ulp_error(got[1], mp.cospi(xm), t),
            }
        for name, error in errors.items():
            if error > worst.get((t, name), (-1.0,))[0]:
                worst[(t, name)] = (error, x)
    print(f"{len(cases)} inputs, mpmath {mp.__version__}")
    for (t, name), (error, x) in sorted(worst.items()):
        print(f"{t} {name}: max {error:.4f} ulp at x = {x!r}")


if __name__ == "__main__":
    main()
