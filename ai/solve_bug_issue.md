# Solve One Bug Issue Headlessly

This document is a strict runbook for Codex running in headless mode on this repository.

Your job is to autonomously select and handle exactly one bug-fixing workstream from GitHub issues:
- inspect open bug and bug-like issues
- choose the highest-value issue that is practical to complete now
- close it only if it is clearly irrelevant
- otherwise fix it, open a PR, monitor CI, and continue until merge or a defined stop condition
- clean up the worktree and restore the local checkout to `origin/main` when done

Do not ask the user questions. Do not stop for confirmation. Resolve ambiguity from the repository, GitHub issues, current code, existing tests, recent commits, and existing PRs. If you still cannot proceed safely, leave a clear comment on the issue or PR and terminate cleanly.

## Hard Rules

1. Work from the latest `origin/main`.
2. Use a dedicated git worktree and a dedicated branch for this run.
3. Do not modify the user's existing checkout except to sync it back to `origin/main` during final cleanup.
4. If the user's existing checkout is dirty at start, do not stash, reset, or otherwise modify it. Preserve it as-is and do all work in the dedicated worktree only.
5. Only consider open issues that are either:
   - labeled `bug`, or
   - bug-like by title/body/comments.
6. Close an issue only when it is clearly one of:
   - already fixed on current `origin/main`
   - duplicate of another issue or PR
   - incorrect report, with concrete evidence
7. Do not close an issue merely because it is hard, large, unclear, or not reproducible yet.
8. If an issue is relevant but not a good candidate for this run because it depends on unresolved prerequisites or is too large for one PR, leave a comment explaining the blocker and skip it.
9. The limit of two fix attempts applies only to the issue's core code fix. CI follow-up fixes do not count toward this limit.
10. Use the repository-local PR workflow. Do not use raw `gh pr create` directly.
11. At the end, remove the dedicated worktree and delete run-specific large temporary artifacts. Only sync the user's existing checkout back to `origin/main` if that checkout was clean at the start of the run.

## What Counts As Bug-Like

Treat an open issue as bug-like if it reports a current mismatch between expected and actual behavior, even without a `bug` label. Examples:
- segfault, UB, memory safety, use-after-free, overflow
- silent wrong result, data corruption, shape/stride/layout bug
- panic or contract violation on accepted input
- incorrect error handling
- regression in existing functionality
- compatibility problems that amount to broken expected behavior for an existing API

Do not treat pure feature requests, design ideas, refactors, or roadmap items as bug-like unless they also document a present incorrect behavior.

## Candidate Discovery

1. Fetch the latest GitHub state and the latest `origin/main`.
2. List open issues with enough metadata to evaluate:
   - number
   - title
   - labels
   - author
   - body
   - comments
   - updated time
   - URL
3. Build a candidate set from `bug` and bug-like issues only.
4. For each candidate, check whether it is already addressed by:
   - a merged PR
   - an open PR
   - a newer umbrella issue
   - current code and tests on `origin/main`

## Prioritization

Choose one workstream using this order of precedence.

### 1. Severity

Highest priority:
- segfault
- UB or memory safety
- data corruption
- silent wrong result

Next:
- panic
- contract violation
- reproducible incorrect failure mode

Lower:
- performance bug
- compatibility gap
- low-impact bug with narrow scope

### 2. Dependency Blocking

Prefer issues that unblock other valuable bug work. Skip issues that require unresolved prerequisites before a correct one-PR fix is possible.

### 3. Executability In One PR

Prefer issues that can be fully handled in one reasonable PR:
- root cause identified
- focused code change
- targeted tests
- no major architecture split required

If the issue is important but not realistically finishable in one PR, leave a blocker comment and skip it for this run.

### 4. Bundle Opportunity

If multiple issues share the same root cause, same files, or same fix pattern, you may combine them into one PR. If you do, make sure the PR remains coherent and verify all bundled issues.

### 5. Evidence Quality

Prefer issues with concrete repro steps, code pointers, failing tests, or direct current-code evidence.

## Triage Outcomes

Each candidate should end in exactly one of these outcomes.

### Outcome A: Irrelevant, Close It

Allowed only for:
- already fixed
- duplicate
- incorrect report

Before closing, leave a comment with the concrete reason and evidence:
- relevant commit or PR
- test result
- code path checked
- duplicate link

### Outcome B: Relevant But Not For This Run

Leave a comment explaining:
- what you verified
- why it is not currently finishable in one PR
- which prerequisite or architectural blocker prevents progress
- that you are skipping it and continuing to the next candidate

Do not close it.

### Outcome C: Relevant And Selected

This becomes the one active workstream for the run. You may bundle directly related bug issues into the same fix if warranted.

## Worktree Setup

1. Inspect the user's existing checkout before doing anything else:
   - record whether it is clean or dirty
   - if dirty, leave it untouched for the entire run
2. Fetch the latest remote state:
   - `git fetch origin`
3. Create a fresh branch from `origin/main`.
4. Create and use a dedicated worktree.
5. Suggested branch format:
   - `fix/issue-<number>-<slug>`
6. If bundling multiple issues, use the primary issue number in the branch name.

Perform all edits, tests, git operations, PR creation, rebases, merges, and monitoring inside this dedicated worktree.

## Investigation Requirements

Before writing a fix:

1. Reproduce the issue consistently if possible.
2. Read the relevant code paths carefully.
3. Check nearby code for the same failure mode.
4. Check recent commits and merged PRs for overlap.
5. Determine root cause before editing.

Do not guess. Do not stack speculative fixes.

## Fix Attempt Policy

For the core issue fix, you may make at most two root-cause-driven fix attempts.

A fix attempt means:
- you identified a concrete root cause hypothesis
- you changed code to address it
- you reran the relevant reproduction and verification

If two attempts fail to resolve the issue, stop core-fix iteration. Leave a detailed comment on the issue including:
- what you reproduced
- what you tried
- what still fails
- what deeper blocker or missing prerequisite remains

Then terminate the run cleanly without forcing a PR.

CI-only follow-up fixes after opening a PR do not count toward this two-attempt limit.

## Implementation Rules

1. Add or update a failing test when practical.
2. If a minimal automated test is not practical, create the smallest reliable reproduction possible.
3. Fix the root cause, not only the symptom.
4. Keep the scope tight unless bundling related bug issues is clearly justified.
5. If bundling issues, verify that each bundled issue is truly addressed by the same PR.

## Verification Requirements

Before opening a PR, run the relevant local verification for the changed area. At minimum:
- targeted tests for the changed crate or module
- any new regression test you added

If the change affects shared behavior, run broader verification as needed.

When claiming a bug is fixed, rely on actual command results, not reasoning alone.

## PR Creation

Use the repository-local PR workflow:
- `bash scripts/create-pr.sh ...`

Follow the repository rules already vendored in this repo. Do not bypass them.

The PR must:
- clearly describe the bug and the root cause
- summarize the fix
- summarize the verification performed
- include `Fixes #...` lines for every bundled issue that the PR truly resolves

If you only determined an issue is irrelevant, do not create a PR for that. Leave an issue comment and close it directly.

## Auto-Merge And CI Monitoring

After creating the PR:

1. Enable auto-merge if allowed.
2. Monitor checks using the repository-local monitor script:
   - `bash scripts/monitor-pr-checks.sh <pr-url-or-number> --interval 30`
3. Continue working the PR until one of these happens:
   - the PR merges
   - the overall monitoring window reaches 30 minutes

## Handling CI Failures

If CI fails and the failure is plausibly caused by your changes:
- inspect the failing job immediately
- fix the issue
- rerun the relevant local verification
- push again
- continue monitoring

CI fixes do not count toward the two core fix attempts.

Coverage, docs, formatting, and repository policy checks are part of this CI follow-up work and must be handled if they fail because of your PR.

## Handling A Stale PR

If the PR becomes out of date with `origin/main` and auto-merge cannot proceed:

1. Fetch the latest `origin/main`.
2. Rebase or merge `origin/main` into your branch autonomously.
3. Resolve conflicts carefully.
4. Rerun the necessary local verification.
5. Push the updated branch.
6. Re-enable auto-merge if it was disabled.
7. Resume monitoring.

You are allowed to perform this rebase or merge without user confirmation.

## Timeout Behavior

If 30 minutes pass after PR creation and the PR is still not merged:

1. Leave the PR open.
2. If auto-merge is enabled and still appropriate, keep it enabled.
3. Leave a concise PR comment summarizing:
   - current status
   - any remaining blocker
   - whether auto-merge is still armed
4. Terminate the run cleanly.

## Final Cleanup

Cleanup is mandatory on every exit path:
- issue closed as irrelevant
- issue skipped after blocker comment
- fix abandoned after two core attempts
- PR merged
- PR left open after timeout

Cleanup steps:

1. Make sure any branch state you need is pushed.
2. Remove the dedicated worktree.
3. Delete run-specific large temporary artifacts created during this run.
4. If the user's existing checkout was clean at run start:
   - return to it
   - fetch `origin`
   - sync it back to `origin/main`
5. If the user's existing checkout was dirty at run start:
   - leave it untouched in its original dirty state
   - do not stash, reset, pull, rebase, or checkout over it

### Large Artifact Cleanup

Avoid leaving large temporary outputs behind. Prefer repo-external temporary paths for:
- `CARGO_TARGET_DIR`
- temporary repro projects
- coverage outputs
- generated docs output
- logs
- crash dumps

At the end of the run, delete large artifacts created specifically for this run. Do not delete shared caches or pre-existing user-managed artifacts.

## Run Completion Criteria

The run is complete only when one of these is true:

1. You closed an issue as clearly irrelevant with evidence.
2. You fixed the selected issue, created a PR, and confirmed it merged.
3. You determined the issue is relevant but not currently executable in one PR, left a blocker comment, and terminated cleanly.
4. You exhausted two core fix attempts, documented the blocker on the issue, and terminated cleanly.
5. You opened a PR, monitored it for up to 30 minutes, left a status comment if still open, and terminated cleanly.

In all cases, final cleanup is required.
