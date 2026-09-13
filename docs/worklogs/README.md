# Work Logs

Keep lightweight decision records for nontrivial multi-phase changes,
non-obvious design choices, or performance experiments, following the shared
[Work Logs And Design Records policy](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#work-logs-and-design-records).
Small fixes and AI assistance alone do not require a work log.

Use one file per change theme and link it from the PR. Start with a few lines:

```markdown
# <Change theme>

## Decisions
- Chosen approach and why; important rejected alternatives, if any.

## Verification conclusions and constraints
- What the final state establishes, and what remains unverified or limited.
- Links to detailed evidence or special reproduction conditions, when needed.
```

Do not include commands run (Cargo or otherwise), files read, agent activity,
or edit/review chronology. Mention failed attempts only when they explain a
choice or limitation. Update the same record when its conclusions change,
not after every correction; do not rewrite historical logs for this format.

Keep complete performance results in linked experiment evidence rather than
copying them into the work log. Required validation is unchanged. Durable
architecture and public contracts belong in `docs/design/`; link them instead
of repeating their contents. Work logs survive either squash or non-squash
merge; no merge-policy change is required.
