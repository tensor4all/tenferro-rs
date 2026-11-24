# Crate Rules

Short, crate-scoped rules, each seeded from a real incident. They sit on top of
the shared tensor4all rules and `REPOSITORY_RULES.md`; they do not replace
either. Every entry names the code it applies to and the check that catches a
violation. If you cannot name both, the entry does not belong here.

## When to read

Before you edit `src/`, `tests/`, or `benches/` of a crate listed below, read
its file. The root `AGENTS.md` (which `CLAUDE.md` imports) carries the same
index, so this is the path an agent is pointed at. Reviewers check a diff
against the entries of every listed crate it touches.

| Crate | File | Incidents it covers |
|---|---|---|
| `tenferro-linalg` | [`tenferro-linalg.md`](tenferro-linalg.md) | overflow before scaling; benchmark arms doing different work |
| `tenferro-fft` | [`tenferro-fft.md`](tenferro-fft.md) | test inventories that must move with the code |
| `tenferro-ad` | [`tenferro-ad.md`](tenferro-ad.md) | a test asserting a property the code does not have; a process-global counter in a test |

A crate without a file has no crate-specific entries yet. That is not a
statement that the shared rules do not apply to it.

## Entry template

Copy this block into `<crate>.md`. Keep each entry under about fifteen lines.

```markdown
### <The rule, as an imperative sentence>

- **Applies to:** <paths, functions, or code shapes in this crate; give a
  `grep` pattern when the shape is syntactic>
- **Check:** <the test, clamp, script, lint, or review question that fails when
  the rule is broken, with the exact command if there is one>
- **Incident:** <owner/repo#number, with a link to the comment or SHA-pinned
  line that shows it> -- <one line: what happened, in the incident's own terms>
```

## Adding an entry

- Add an entry only with an incident behind it: an issue, a review finding,
  or a CI failure that a person or agent actually hit. Link the comment or the
  SHA-pinned source line, not just the PR number, because the evidence is
  usually in a comment and line numbers move.
- Put the entry in the crate where the code it names lives. Do not copy one
  entry into every crate; when a rule applies to several crates, put it in
  the crate where you can name a concrete path and say "applies also to" there.
- When the rule is not specific to tenferro (it would hold in any Rust crate),
  the generic sentence belongs in `tensor4all-agent-rules`. Keep the crate
  entry for the tenferro paths and the check, and link the shared rule once it
  exists. Do not vendor shared rules here (see `ai/README.md`).
- New `##` sections in `REPOSITORY_RULES.md` and `PERFORMANCE_TIPS.md` are
  routing units for `scripts/repository-rules-review.py`. These files are not,
  so adding an entry needs no script change.
- When the code an entry names is removed or renamed, update or delete the
  entry in the same PR. A stale entry is a defect, not an instruction.
