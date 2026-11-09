import pathlib
p = pathlib.Path("docs/guides/external-linalg-interop.md")
t = p.read_text()
old = "tenferro hands out no `faer::Par`, so no faer value crosses a tenferro boundary\nA separately pinned `faer` is safe for plain column-major slices, but not for the engine-token\nroute above: that faer build must be compiled against the same `threadteam` the engine hands\nout, or the `TeamExec` types do not meet (#1939)."
new = "tenferro hands out no `faer::Par`, so a faer value never has to cross a tenferro\nboundary. A separately pinned `faer` is safe for plain column-major slices, but\nnot for the engine-token route above: that faer build must be compiled against\nthe same `threadteam` the engine hands out, or the `TeamExec` types do not meet\n(#1939)."
assert old in t
p.write_text(t.replace(old, new, 1))
print("ok")
