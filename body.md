## What

Fixes the statements in `docs/guides/external-linalg-interop.md` that a review of #2542 found false (that PR is already merged, so this is a follow-up):

- "No faer value crosses a tenferro boundary ... a direct dependency remains safe because only plain column-major slices cross the boundary". The engine-token example above passes a `threadteam::TeamExec` into `faer::Par::Team`, so a `faer` build resolving another `threadteam` fails there (E0308). The text now says a direct dependency is safe for plain slices, and for the engine-token route only when it resolves the same `threadteam`.
- "Each `t4a-faer` release line pins its own `threadteam` ... another line yields a second `Pool`/`TeamExec`/`Par`". 0.32 and 0.33 both require `threadteam ^0.5.1`, so their `Pool`/`TeamExec` are the same type and one can construct the other `Par::Team`; only the `Par` types stay distinct. The text now states the rule in terms of the resolved `threadteam`.

Docs only: no code, build or test change. A previous revision of this branch accidentally added three helper scripts; they are deleted.

## Verification

```
# threadteam requirement of each published t4a-faer line (crates.io registry metadata)
$ for v in 0.30.0 0.32.0 0.33.0; do curl -s https://crates.io/api/v1/crates/t4a-faer/$v/dependencies | grep -o "\"crate_id\":\"threadteam\"[^}]*"; done
0.30.0 -> "req":"^0.4.0"
0.32.0 -> "req":"^0.5.1"
0.33.0 -> "req":"^0.5.1"

# the workspace this guide belongs to
$ gh api repos/tensor4all/tenferro-rs/contents/Cargo.toml --jq .content | base64 -d | grep -nE "^(threadteam|faer) ="
81:threadteam = "0.5.1"
104:faer = { package = "t4a-faer", version = "0.33.0", default-features = false, features = ["std", "rayon", "threadteam"] }

$ git diff --check
(clean)
```

The `docs-site` job renders the page; there is no code path to test.
