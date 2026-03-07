Check whether this repository's managed agent assets are current with the upstream `template-rs` bundle.

Workflow:

1. Treat this as a read-only command.
2. Run `bash scripts/check-agent-assets.sh "$@"`.
3. Report whether the repo is `up-to-date`, `update-available`, or `unable-to-check`.
4. If the script reports local managed-file drift, call that out explicitly.

Use `--quiet` when this is being run as a background freshness check.
