#!/usr/bin/env python3
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = """[[package]]
name = "colmat"
version = "0.34.1"
[[package]]
name = "threadteam"
version = "0.6.0"
[[package]]
name = "strided-basic"
version = "0.15.0"
[[package]]
name = "strided-fused"
version = "0.15.0"
[[package]]
name = "strided-kernel"
version = "0.15.0"
[[package]]
name = "strided-nn"
version = "0.15.0"
[[package]]
name = "strided-perm"
version = "0.15.0"
[[package]]
name = "strided-traits"
version = "0.7.2"
[[package]]
name = "strided-view"
version = "0.8.1"
"""
MIXED = CLEAN + """[[package]]
name = "strided-view"
version = "0.7.1"
"""

def run(lock_text):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Cargo.lock"
        p.write_text(lock_text)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check-stack-versions.py"), "--lock", str(p)], capture_output=True, text=True)
        return r.returncode, r.stdout

def main():
    assert run(CLEAN)[0] == 0
    assert run(MIXED)[0] == 1
    forbidden = CLEAN + '[[package]]\nname = "t4a-faer-traits"\nversion = "0.24.3"\n'
    rc, out = run(forbidden)
    assert rc == 1 and "forbidden dependency t4a-faer-traits" in out
    unlisted = CLEAN + '[[package]]\nname = "strided-einsum"\nversion = "0.12.0"\n'
    rc, out = run(unlisted)
    assert rc == 1 and "unlisted strided-einsum" in out
    missing = '[[package]]\nname = "threadteam"\nversion = "0.6.0"\n'
    rc, out = run(missing)
    assert rc == 1 and "missing colmat" in out
    wrong = CLEAN.replace('version = "0.34.1"', 'version = "0.33.0"')
    rc, out = run(wrong)
    assert rc == 1 and "colmat 0.33.0" in out
    print("test-check-stack-versions: ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
