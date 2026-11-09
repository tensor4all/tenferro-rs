import pathlib
p = pathlib.Path("docs/guides/external-linalg-interop.md")
t = p.read_text()
old = "carries a `threadteam::TeamExec` (`t4a-faer 0.33.0/src/lib.rs:1078`). Whether a direct `faer` dependency can share the engine team depends on the"
new = "carries a `threadteam::TeamExec` (`t4a-faer 0.33.0/src/lib.rs:1078`). Whether a\ndirect `faer` dependency can share the engine team depends on the"
assert old in t
p.write_text(t.replace(old, new, 1))
print("ok")
