# Review gate: hunk state and a head-bound waiver

Applies [tensor4all-meta#88](https://github.com/tensor4all/tensor4all-meta/issues/88)
and [#89](https://github.com/tensor4all/tensor4all-meta/issues/89) to this
repository's review gate. The reference implementation and its own fixtures live
in `tensor4all-meta` under `scripts/review-gate/`.

## Decisions

- **One parser, four call sites.** `added_lines_with_text`,
  `added_lines_by_file`, `files_with_unanchorable_deletions`,
  `split_large_file_diff` and `sensitive_diff_location` now all walk
  `parse_diff`, which spends the `@@ -a,b +c,d @@` line budget before deciding
  that any `+++ ` line is a header. Each helper keeps its previous signature and
  its previous result for well-formed input, so call sites did not change.
  Rejected: patching each loop separately. The four loops drifted into two
  different notions of "header" already, and a per-loop fix would let them drift
  again.
- **The `--name-status -z` cross-check is included** as a second, blocking
  layer. It costs one local `git diff` call and cannot be influenced by the diff
  text. This repository does run the gate on fork pull requests (see the gate
  comment in `review_bot.yml`), so the diff text is genuinely author-controlled
  and the untrusted boundary `AGENTS.md` asks a task to introduce is present
  here. Fail-closed is safe only because git's own vocabulary is accepted
  exactly: binary files, mode-only changes, renames, deletes and quoted
  non-ASCII paths are all admitted, and untracked worktree files are added to
  git's list because `unified_diff` synthesises their diff itself.
- **The waiver is a comment naming a 40-character sha**, not a label. Both
  inline verifiers in `review_bot.yml` (`rules-review:waive` and
  `rules-review:no-llm`) call the same `scripts/review-waiver.js`, so the two
  can no longer disagree. Consequence to expect: applying the label without a
  `waive-review <sha>` comment now fails the job, and the failure message names
  the sha to post.

## Verification conclusions and constraints

- The pre-fix parser is kept as `legacy_added_lines_with_text` in
  `scripts/test-repository-rules-review.py` and asserted to accept the forged
  path, so a regression is a failing test rather than a silent reopening.
- Over 349 per-file diffs from 40 consecutive commits on `main`, the five
  helpers return byte-identical results before and after the change, including
  `split_large_file_diff` with its chunk limit forced low enough to split in 181
  of them.
- The gate returns `pass` on 30 consecutive `main` commits and in `--worktree`
  mode with a modified, an added and a deleted path present.
- One deliberate difference from the pre-fix helpers: a file that names a path
  but adds no line (binary, rename-only) now gets an empty bucket instead of no
  entry. Every call site tests `if not touched`, so it is inert; the shared
  fixtures in `tensor4all-meta` assume the entry exists.
- The block path of the path cross-check is covered by unit tests, not
  end-to-end: the gate derives both the diff and the name-status list from git,
  so only a hand-written diff can make them disagree.
- Not verified here: the two GitHub Actions jobs are exercised by
  `node --test scripts/test-review-waiver.js` against fixture comment sets, not
  by a live pull request. `actionlint` v1.7.7 accepts the workflow.
