#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-claude-solve-bug.sh [options] [-- <extra claude args>]

Run Claude Code headlessly from the repository root using a prompt file under ai/.

Options:
  --prompt PATH   Prompt file. Relative paths are resolved from this script's directory.
                  Default: solve_bug_issue.md
  --model MODEL   Pass --model MODEL to claude.
  --run-dir PATH  Directory for logs.
                  Default: a fresh temporary directory.
  --text          Disable stream-json output and print plain text instead.
  -h, --help      Show this help.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
repo_name="$(basename "$repo_root")"

prompt_arg="solve_bug_issue.md"
model=""
run_dir=""
json_mode=1
extra_args=()

while (($# > 0)); do
  case "$1" in
    --prompt)
      prompt_arg="${2:?missing value for --prompt}"
      shift 2
      ;;
    --model)
      model="${2:?missing value for --model}"
      shift 2
      ;;
    --run-dir)
      run_dir="${2:?missing value for --run-dir}"
      shift 2
      ;;
    --text)
      json_mode=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$prompt_arg" = /* ]]; then
  prompt_path="$prompt_arg"
else
  prompt_path="$script_dir/$prompt_arg"
fi

if [[ ! -f "$prompt_path" ]]; then
  echo "prompt file not found: $prompt_path" >&2
  exit 1
fi

if [[ -z "$run_dir" ]]; then
  run_dir="$(mktemp -d "${TMPDIR:-/tmp}/${repo_name}-claude-solve-bug.XXXXXX")"
else
  mkdir -p "$run_dir"
  run_dir="$(cd -- "$run_dir" && pwd -P)"
fi

log_path="$run_dir/output.log"
if ((json_mode)); then
  log_path="$run_dir/events.jsonl"
fi

prompt_text="$(cat "$prompt_path")"

cmd=(
  claude
  --print
  --dangerously-skip-permissions
)

if ((json_mode)); then
  cmd+=(--output-format stream-json)
else
  cmd+=(--output-format text)
fi

if [[ -n "$model" ]]; then
  cmd+=(--model "$model")
fi

cmd+=("${extra_args[@]}" "$prompt_text")

echo "repo_root=$repo_root"
echo "prompt_path=$prompt_path"
echo "run_dir=$run_dir"
echo "log_path=$log_path"

(
  cd "$repo_root"
  "${cmd[@]}"
) | tee "$log_path"
