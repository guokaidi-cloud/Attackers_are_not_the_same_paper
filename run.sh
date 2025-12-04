#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CMD="python main.py --attack cluster --use_emb --dataset mnist --num_pass 4 --lr_attack 0.01 --attack_id 0 --dispersion"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh [--log-dir DIR] [--cmd COMMAND]... [--cmd-file FILE]

Options:
  --log-dir DIR    Directory to store command list and logs.
                   Defaults to logs/<timestamp> under repo root.
  --cmd COMMAND    Command to execute (can be used multiple times).
  --cmd-file FILE  File that lists commands (one per line, '#' for comments).
  -h, --help       Show this message.

Notes:
  * If neither --cmd nor --cmd-file is provided, a default command will run:
    ${DEFAULT_CMD}
  * Commands are executed from the repository root.
  * Each command's stdout/stderr is mirrored to the console and stored as
    <log-dir>/<index>_<slug>.log
EOF
}

LOG_DIR=""
COMMANDS_FILE=""
declare -a COMMANDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --log-dir" >&2; exit 1; }
      LOG_DIR="$2"
      shift 2
      ;;
    --cmd)
      [[ $# -lt 2 ]] && { echo "Missing value for --cmd" >&2; exit 1; }
      COMMANDS+=("$2")
      shift 2
      ;;
    --cmd-file)
      [[ $# -lt 2 ]] && { echo "Missing value for --cmd-file" >&2; exit 1; }
      COMMANDS_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$COMMANDS_FILE" && "$COMMANDS_FILE" != /* ]]; then
  COMMANDS_FILE="${REPO_ROOT}/${COMMANDS_FILE}"
fi

if [[ -n "$COMMANDS_FILE" ]]; then
  [[ -f "$COMMANDS_FILE" ]] || { echo "Command file not found: $COMMANDS_FILE" >&2; exit 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    COMMANDS+=("$line")
  done < "$COMMANDS_FILE"
fi

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  COMMANDS+=("$DEFAULT_CMD")
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${REPO_ROOT}/logs/$(date +%y%m%d_%H%M%S)"
elif [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="${REPO_ROOT}/${LOG_DIR}"
fi

mkdir -p "$LOG_DIR"

COMMAND_LIST_FILE="${LOG_DIR}/commands.txt"
{
  echo "# Generated: $(date '+%F %T')"
  echo "# Repo root: ${REPO_ROOT}"
} > "$COMMAND_LIST_FILE"

slugify() {
  local text="$1"
  local slug
  slug=$(echo "$text" | tr -cs '[:alnum:]_-' '_' | sed -e 's/^_//' -e 's/_$//')
  if [[ -z "$slug" ]]; then
    slug="command"
  fi
  echo "${slug:0:80}"
}

cd "$REPO_ROOT"

for i in "${!COMMANDS[@]}"; do
  idx=$((i + 1))
  cmd="${COMMANDS[$i]}"
  printf '[%02d/%02d] %s\n' "$idx" "${#COMMANDS[@]}" "$cmd"
  echo "[$(printf '%02d' "$idx")] $cmd" >> "$COMMAND_LIST_FILE"

  slug=$(slugify "$cmd")
  log_file="${LOG_DIR}/$(printf '%02d' "$idx")_${slug}.log"

  {
    echo "===== Command ====="
    echo "$cmd"
    echo "===== Start: $(date '+%F %T') ====="
  } >> "$log_file"

  if eval "$cmd" 2>&1 | tee -a "$log_file"; then
    echo "===== End: $(date '+%F %T') | Status: OK =====" >> "$log_file"
  else
    status=$?
    echo "===== End: $(date '+%F %T') | Status: FAIL (${status}) =====" >> "$log_file"
    exit "$status"
  fi
done