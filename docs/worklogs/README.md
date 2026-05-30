# Work logs

Work logs are reviewer-facing records for completed nontrivial work. Use them
when a PR includes a refactor, cleanup stream, AI-assisted implementation, or
explicit design tradeoff.

Unlike `docs/plans/`, work logs describe what actually happened during the
session and why the final design was chosen. They should be curated summaries,
not raw transcripts.

Include the following sections when they are relevant:

- Session summary
- Context read
- Reference code or prior art consulted
- Decisions made
- Rejected or deferred alternatives
- Verification performed
- Remaining risks or follow-up work

If a decision should guide future implementation beyond the current PR, record
that durable design intent in `docs/design/` as well and link it from the work
log.
