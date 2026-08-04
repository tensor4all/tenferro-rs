# Review bot transport and guard hardening

## Summary

`scripts/repository-rules-review.py` was ported to two sibling repositories
(tensor4all/strided-rs#222, tensor4all/tensor4all-rs#568). Running it there
surfaced defects that all originate in this copy, plus a Codex review of the
port raised further findings against the same shared code. This change fixes
them here and keeps the three copies byte-identical in their shared machinery.

No production Rust code is touched. The only behavioural surface is the review
bot itself.

## Classification ledger

Findings come from two sources: a live run of the ported script, and the Codex
review of tensor4all/tensor4all-rs#568, which reviewed this same shared code.
Evidence below was re-derived against this worktree, not taken from the
comments.

| # | Source | Class | Evidence at this hash | Verification target | Outcome |
|---|--------|-------|----------------------|--------------------|---------|
| 1 | Live run (local, py3.9) | Auto Fix | `socket.timeout` escaped `(KeyError, ValueError, URLError, TimeoutError)`; reproduced by injecting it into `urlopen` | `test_transport_errors_cover_every_below_json_failure` | Fixed |
| 2 | Derived from #1 | Auto Fix | `ConnectionResetError`, `ssl.SSLError` reachable on every version; `http.client.IncompleteRead` is not an `OSError` (verified with `issubclass`) | same test | Fixed |
| 3 | Live run | Auto Fix | 108k single chunk exceeded the 120s default | manual: same commit now 2 chunks / 235s | Fixed |
| 4 | Codex #1603 P1 | Auto Fix | 300s x 2 retries x 3 chunks can exceed `timeout-minutes: 20`, killing the job before the report | `test_budget_is_smaller_than_the_workflow_timeout`, `test_call_deepseek_does_not_retry_past_the_deadline` | Fixed |
| 5 | Codex #1603 P1 | Auto Fix | No work log for a ~500-line AI-assisted change with design tradeoffs | this file | Fixed |
| 6 | Codex #568 P1 | Auto Fix | Typed declaration: redactor masked the type, literal survived | `test_contains_sensitive_text_flags_typed_declaration` | Fixed |
| 7 | Codex #568 P1 | Auto Fix | Quoted value with spaces neither detected nor fully redacted | `test_redact_sensitive_text_masks_whole_quoted_value` | Fixed |
| 8 | Derived from #7 | Verify First | Widening the value pattern broke this repository's existing `token_type` regression test — confirmed the false positive is real before choosing a fix | `test_metadata_names_are_not_credentials` | Fixed by moving discrimination to the name |
| 9 | Derived from #6 | Auto Fix | This file's own fixtures tripped the improved guard; self-scan showed 15 lines | self-scan now zero; `--worktree` review passes | Fixed |
| 10 | Codex #568 P2 | Auto Fix | Repeated `@@` header gave later chunks wrong line numbers | `test_split_oversized_hunk_renumbers_each_chunk` | Fixed |
| 11 | Codex #568 P2 | Auto Fix | `git diff <base>` omits untracked paths | manual: untracked file with a prohibited fence is now caught | Fixed |
| 12 | Codex #568 P2 | Auto Fix | `git` C-quotes non-ASCII pathnames by default | `test_run_git_disables_pathname_quoting` | Fixed |
| 13 | Codex #568 P2 | Auto Fix | Path-only routing never supplied unsafe rules for a generic filename | `test_select_rule_sections_routes_on_changed_content` | Fixed |
| 14 | Derived from #13 | Auto Fix | `HUMAN_ONLY_SECTIONS` was never subtracted in this copy; exclusion held only because no trigger named one | `test_content_triggers_never_select_human_only_sections` | Fixed |
| 15 | Neighborhood scan | Auto Fix | A stale duplicate `QUOTED_SECRET_ASSIGNMENT` after `SEVERITY_ALIASES` silently overrode the new definition | grep: one definition remains | Fixed |

Nothing was classified `Stale / Out of Scope` or `Design Gate`. Findings 6, 7,
10–13 were raised against the port but live in this copy, so they are fixed
here and the three copies are kept identical in their shared machinery.

## Decisions

### Transport failures: catch two roots, not a list of leaves

The handler caught `(KeyError, ValueError, URLError, TimeoutError)`. That
enumerates leaves and misses siblings:

| Failure | Was caught | Why |
|---|---|---|
| `socket.timeout` | Python 3.10+ only | Alias of `TimeoutError` only from 3.10 |
| `ConnectionResetError` | No | `OSError` subclass, never named |
| `ssl.SSLError` | No | `OSError` subclass, never named |
| `http.client.IncompleteRead` | No | Sibling of `OSError`, not a subclass |

Rejected: adding `socket.timeout` to the tuple. It fixes one row and leaves
the shape that produced the bug. Chosen: name the two roots,
`TRANSPORT_ERRORS = (OSError, http.client.HTTPException)`, with a test that
asserts all five failure types are subclasses of it.

CI runs `ubuntu-latest`, so the `socket.timeout` row is not currently
reachable there. The script is documented for local use with a repo-root
`.env`, and that is where it was observed on Python 3.9.

The severity of the miss is not the crash. The workflow tees only stdout into
the report, so a traceback goes to stderr and the PR comment reads *"Review
step did not produce a report"* — no diagnostic, and on this repository the
gate is wired into required checks.

### Retry budget bounded by the job deadline

Retrying once on a transient failure prevents one blocked PR per network blip.
But with a 300s per-attempt timeout, a retried chunk costs ~605s and a
three-chunk diff can exceed the workflow's 20-minute limit. The job is then
killed mid-request and the report is lost — reintroducing the failure the
retry was meant to remove.

Chosen: a cumulative `DEFAULT_BUDGET_SECONDS = 900` ceiling. Each request's
timeout is clamped to the remaining budget, a retry that would cross the
deadline is skipped, and exhausting the budget emits a `warn` (not `block`)
naming how many chunks went unreviewed. Deterministic checks still cover the
whole diff, so a partial LLM pass must not fail the gate.

Rejected: raising `timeout-minutes`. That moves the cliff without removing it.
A test asserts the budget stays below the workflow's own timeout.

### Secret guard: the value cannot be the discriminator

Two escapes, both allowing a credential to reach the external model:

- A typed declaration (`const API_KEY: &str = "..."`) let the redactor treat
  the type colon as the separator and mask the type, leaving the literal.
- A quoted value containing spaces was not detected at all, and the redactor
  masked only up to the first space — uploading most of a passphrase.

Widening the value pattern to allow spaces then broke a regression test this
repository already had: `token_type: "WebGPU event token from another queue"`
is prose, not a credential. A diceware passphrase is prose by construction, so
no value-shape heuristic can separate the two.

Chosen: move the discrimination to the name. `is_credential_name` rejects
identifiers ending in metadata suffixes (`_type`, `_name`, `_path`, `_id`, …)
in both detection and redaction. An assignment that opens a quote closing on a
later line is treated as disqualifying, since no single-line pattern can see
the value.

### The guard's own fixtures tripped the guard

Once detection improved, this file's secret-shaped test fixtures blocked the
LLM pass on any PR touching them — leaving a maintainer waiver as the only
route. Fixtures are now assembled at runtime from fragments, so the source
carries no contiguous secret-shaped literal while the tests still exercise the
real shapes. One explanatory comment had to be reworded for the same reason.

Verified by scanning both scripts with `contains_sensitive_text`: zero hits.

### Chunk headers renumbered

Splitting an oversized hunk repeated the original `@@` header on every chunk,
so the model derived line numbers thousands of lines too small. Those findings
were dropped by `filter_findings`, or worse retained against an unrelated added
line that happened to collide. Each chunk now carries a header rewritten to its
own offsets, counting context, removal, and addition lines separately. An
unparseable header falls back to the previous verbatim behaviour rather than
inventing offsets.

Two existing tests had pinned the repeat-verbatim behaviour as intended; they
now assert that chunk starts chain and that counts sum to the original hunk.

### Routing on content, and honouring human-only sections

Path-only routing never supplied `Unsafe Code Boundary` for an `unsafe` block
added under a generic filename, and the prompt forbids inventing requirements
that were not supplied — so the rule was unenforceable there. `CONTENT_TRIGGERS`
matches signals in the changed lines themselves.

Adding content routing exposed that `HUMAN_ONLY_SECTIONS` was never subtracted
in this copy. The guarantee in "Performance-Gated Experiment Protocol" —
*"intentionally not routed to the diff-scoped review bot"* — held only because
no trigger happened to name one. It is now subtracted explicitly.

### Pathname quoting

`git diff --name-only` C-quotes non-ASCII paths by default, and the quoted form
matches no real path, so such a file was reviewed by nothing. All git
invocations now pass `-c core.quotePath=false`.

## Verification

- Full `scripts/test-repository-rules-review.py` suite passes, with new tests
  for each decision above.
- The bot reviews its own branch cleanly in all three repositories.
- `scripts/test-doc-consistency.py` fails locally on Python 3.9 for an
  unrelated reason (`tomllib` needs 3.11+); confirmed identical on clean
  `main`.

## Residual risks

- `is_credential_name`'s metadata suffix list is a denylist. A field named
  `token_descriptor` would still be treated as a credential and its value
  redacted from the uploaded diff. Over-redaction degrades review quality but
  does not leak.
- The 900s budget is a fixed constant. A repository that raises
  `timeout-minutes` gains nothing until the constant moves with it; the test
  only checks the budget is smaller, not that it is proportionate.
- `CONTENT_TRIGGERS` is a heuristic keyed on source text. It will miss a rule
  signal expressed differently and will occasionally supply a section that
  turns out to be irrelevant, costing tokens rather than correctness.

## Review Follow-ups (Codex on PR #1604)

Three findings, each reproduced against the pre-fix script first.

- **P1 — expression continuations blocked the gate.** The new bare
  continuation alternative `[^\s"'#][^\s#]{7,}` matched a whole EXPRESSION,
  so `let api_key =` continued by `std::env::var("API_KEY")?;` was reported as
  a leaked credential and a valid credential-LOADING change could not pass the
  required review-bot gate. The alternative is now `BARE_SECRET_VALUE`,
  restricted to the characters a credential literal is made of
  (`[A-Za-z0-9][A-Za-z0-9._~+/=-]{7,}` — base64/hex/JWT alphabets plus URL-safe
  punctuation). Call, path and index syntax falls outside the class. The
  motivating unquoted-secret case and JWT-shaped values still match.
- **P2 — a fully deleted file was omitted.** `+++ /dev/null` cleared
  `current_file`, so a whole-file deletion never entered the set that exists to
  retain unanchored blocks about deletions. The parser now falls back to the
  old-side path from the preceding `--- a/...` header — the path a finding
  about the removal names.
- **P2 — the exception was too wide.** Any patch removing even one line put
  the file in the set, which disabled the anti-generalization filter for most
  modified files. The rule is now "removes lines and adds none": a replacement
  edit supplies the lines that took the deleted ones' place, so a real finding
  about it can and should name one of them. The helper is renamed
  `files_with_unanchorable_deletions` to state that condition.

Coverage: `test_sensitive_diff_ignores_an_expression_continuation`,
`test_files_with_unanchorable_deletions_keeps_a_fully_deleted_file`, and
`test_files_with_unanchorable_deletions_skips_replacement_edits`.

## Review Follow-ups, Round 2 (Codex on PR #1604)

Two findings, both reproduced against `f1c2cd37` first.

- **P1 — field accesses still read as credential values.** Narrowing the bare
  continuation class to `[A-Za-z0-9._~+/=-]` excluded call and path syntax but
  kept `.`, so `let api_key =` continued by `settings.api_key;` still matched
  and credential-LOADING code still could not pass the required gate. `.` is
  now out of the class. The cost is the unquoted-JWT continuation shape; a
  quoted JWT still matches the quoted alternatives and the `Bearer` form is
  covered by `SECRET_VALUE_PATTERNS`, whereas a dotted bare token on a
  continuation line is far more often a field access. Blocking valid code on
  the required gate is the worse of the two errors.
- **P2 — mixed-hunk deletions lost the exception.** Additions and deletions
  were aggregated per FILE, so an unrelated addition anywhere in the file
  removed a deletion-only hunk from the set and `filter_findings` dropped the
  real block. The unit of "nothing to anchor to" is the HUNK: a file now
  qualifies when at least one hunk removes lines and adds none. This keeps
  round 1's narrowing intact — a pure replacement edit has both in the same
  hunk, so it still does not qualify.

Coverage: `test_sensitive_diff_ignores_a_field_access_continuation` and
`test_files_with_unanchorable_deletions_keeps_a_mixed_hunk_deletion`, alongside
the existing replacement-edit and whole-file-deletion cases which pin that the
per-hunk rule did not widen the exception back out.
