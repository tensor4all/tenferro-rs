import pathlib
p = pathlib.Path("docs/guides/external-linalg-interop.md")
t = p.read_text()
reps = [
("and a separately pinned `faer` cannot meet a confusing type mismatch (#1939).",
 "A separately pinned `faer` is safe for plain column-major slices, but not for the engine-token\nroute above: that faer build must be compiled against the same `threadteam` the engine hands\nout, or the `TeamExec` types do not meet (#1939)."),
("a direct dependency remains safe because only plain\ncolumn-major slices cross the boundary.",
 "a direct dependency is safe only for plain\ncolumn-major slices."),
("Each\n`t4a-faer` release line pins its own `threadteam` (0.30 needs `^0.4.0`, 0.33\nneeds `^0.5.1`), so a direct `faer` dependency on another line yields a second\n`Pool`/`TeamExec`/`Par` type that cannot be passed to or built from this one.",
 "Whether a direct `faer` dependency can share the engine team depends on the\n`threadteam` each line resolves, not on the faer version alone: `t4a-faer` 0.30\nrequires `threadteam ^0.4.0`, while 0.32 and 0.33 both require `^0.5.1`. A line\nthat resolves a different `threadteam` yields a second `Pool`/`TeamExec` that\ncannot be passed across. Even when the `threadteam` matches, the `Par` type of\neach faer build is distinct, so a `Par` cannot be moved between builds, though\none can be constructed from a compatible team."),
]
for old, new in reps:
    assert old in t, old[:40]
    t = t.replace(old, new, 1)
p.write_text(t)
print("ok")
