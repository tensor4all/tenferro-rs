Sync this repository's managed agent assets from the upstream `template-rs` bundle.

Workflow:

1. Run `bash scripts/sync-agent-assets.sh "$@"`.
2. Do not overwrite locally modified managed files unless the user explicitly requested `--force`.
3. After sync, summarize which files were refreshed and which upstream revision was installed.

This command updates vendored common rules, project-local commands, helper scripts, and `ai/agent-assets.lock`.
