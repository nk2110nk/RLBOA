#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-results}"
MODE="${MODE:-expert}"
EPISODES="${EPISODES:-100}"
N_ACTIONS="${N_ACTIONS:-10}"
USE_DOCKER="${USE_DOCKER:-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-mipn_negotiator_cpy:dev}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"
STOCHASTIC="${STOCHASTIC:-0}"
NOISE="${NOISE:-0}"
PLOT="${PLOT:-0}"
CASES="${CASES:-}"
GENERAL_ISSUES="${GENERAL_ISSUES:-Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A Coffee Camera Lunch SmartPhone Kitchen}"
GENERAL_AGENTS="${GENERAL_AGENTS:-Boulware Linear Conceder Atlas3}"

usage() {
  cat <<'EOF'
Usage:
  ./run_test_cases.sh [case...]

Examples:
  ./run_test_cases.sh 1
  ./run_test_cases.sh case2 case5
  CASES="1 3 6" ./run_test_cases.sh
  EPISODES=5 ./run_test_cases.sh 4
  DRY_RUN=1 ./run_test_cases.sh 1

Environment:
  ROOT_DIR        Results root. Default: results
  MODE            expert or general. Default: expert
  EPISODES        Test episodes per model. Default: 100
  N_ACTIONS       RLBOA action bins. Default: 10
  USE_DOCKER      Run through Docker. Default: 1
  DOCKER_IMAGE    Docker image name. Default: mipn_negotiator_cpy:dev
  DRY_RUN         Print commands without running. Default: 0
  LIMIT           Run only first N models per case. Default: 0 means no limit
  DETERMINISTIC   Add --deterministic when 1. Default: 0
  STOCHASTIC      Add --stochastic when 1. Default: 0
  NOISE           Add --noise when 1. Default: 0
  PLOT            Add --plot when 1. Default: 0
  GENERAL_ISSUES  Issues used when MODE=general.
  GENERAL_AGENTS  Agents used when MODE=general.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$DETERMINISTIC" == "1" && "$STOCHASTIC" == "1" ]]; then
  echo "DETERMINISTIC=1 and STOCHASTIC=1 cannot be used together." >&2
  exit 1
fi

if [[ "$MODE" != "expert" && "$MODE" != "general" ]]; then
  echo "MODE must be expert or general. Got: $MODE" >&2
  exit 1
fi

if [[ "$USE_DOCKER" == "1" && "${IN_TEST_CASE_DOCKER:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker command not found; running directly in the current environment."
    USE_DOCKER=0
  else
    exec docker run --rm \
      --user "$(id -u):$(id -g)" \
      --entrypoint /bin/bash \
      -v "$PWD":/work \
      -w /work \
      -e MPLCONFIGDIR=/tmp/mplconfig \
      -e IN_TEST_CASE_DOCKER=1 \
      -e USE_DOCKER=0 \
      -e ROOT_DIR="$ROOT_DIR" \
      -e MODE="$MODE" \
      -e EPISODES="$EPISODES" \
      -e N_ACTIONS="$N_ACTIONS" \
      -e DRY_RUN="$DRY_RUN" \
      -e LIMIT="$LIMIT" \
      -e DETERMINISTIC="$DETERMINISTIC" \
      -e STOCHASTIC="$STOCHASTIC" \
      -e NOISE="$NOISE" \
      -e PLOT="$PLOT" \
      -e CASES="$CASES" \
      -e GENERAL_ISSUES="$GENERAL_ISSUES" \
      -e GENERAL_AGENTS="$GENERAL_AGENTS" \
      "$DOCKER_IMAGE" \
      ./run_test_cases.sh "$@"
  fi
fi

raw_cases=()
if (( $# > 0 )); then
  raw_cases=("$@")
elif [[ -n "$CASES" ]]; then
  read -r -a raw_cases <<< "$CASES"
else
  raw_cases=(1 2 3 4 5 6)
fi

normalize_case() {
  case "$1" in
    1|case1|results_case1) echo "results_case1" ;;
    2|case2|results_case2) echo "results_case2" ;;
    3|case3|results_case3) echo "results_case3" ;;
    4|case4|results_case4) echo "results_case4" ;;
    5|case5|results_case5) echo "results_case5" ;;
    6|case6|results_case6) echo "results_case6" ;;
    *)
      echo "Unknown case: $1" >&2
      exit 1
      ;;
  esac
}

model_arg() {
  local model_dir="$1"
  if [[ "$model_dir" = /* ]]; then
    printf '%s/' "${model_dir%/}"
  else
    printf './%s/' "${model_dir#./}"
  fi
}

build_test_command() {
  local issue="$1"
  local agent0="$2"
  local agent1="$3"
  local model_dir="$4"

  cmd=(
    python3 test_negotiator.py
    -a "$agent0" "$agent1"
    -i "$issue"
    -m "$(model_arg "$model_dir")"
    -e "$EPISODES"
    --n-actions "$N_ACTIONS"
  )

  if [[ "$DETERMINISTIC" == "1" ]]; then
    cmd+=(--deterministic)
  fi
  if [[ "$STOCHASTIC" == "1" ]]; then
    cmd+=(--stochastic)
  fi
  if [[ "$NOISE" == "1" ]]; then
    cmd+=(--noise)
  fi
  if [[ "$PLOT" == "1" ]]; then
    cmd+=(--plot)
  fi
}

build_general_test_command() {
  local model_dir="$1"
  local issues=()
  local agents=()

  read -r -a issues <<< "$GENERAL_ISSUES"
  read -r -a agents <<< "$GENERAL_AGENTS"

  cmd=(
    python3 test_negotiator.py
    --model-type general
    -a "${agents[@]}"
    -i "${issues[@]}"
    -m "$(model_arg "$model_dir")"
    -e "$EPISODES"
    --n-actions "$N_ACTIONS"
  )

  if [[ "$DETERMINISTIC" == "1" ]]; then
    cmd+=(--deterministic)
  fi
  if [[ "$STOCHASTIC" == "1" ]]; then
    cmd+=(--stochastic)
  fi
  if [[ "$NOISE" == "1" ]]; then
    cmd+=(--noise)
  fi
  if [[ "$PLOT" == "1" ]]; then
    cmd+=(--plot)
  fi
}

run_case() {
  local case_name="$1"
  local case_dir="${ROOT_DIR%/}/${case_name}"

  if [[ ! -d "$case_dir" ]]; then
    echo "Missing case directory: $case_dir" >&2
    exit 1
  fi

  mapfile -t model_dirs < <(
    find "$case_dir" -mindepth 3 -maxdepth 3 -type d -name RLBOA_Negotiator | sort
  )

  local total="${#model_dirs[@]}"
  local current=0
  local ran=0

  echo "=== ${case_name}: ${total} models ==="

  for model_dir in "${model_dirs[@]}"; do
    current=$((current + 1))
    if [[ "$LIMIT" -gt 0 && "$ran" -ge "$LIMIT" ]]; then
      echo "Reached LIMIT=${LIMIT} for ${case_name}"
      break
    fi

    if [[ ! -f "${model_dir%/}/checkpoint.zip" ]]; then
      echo "[${case_name} ${current}/${total}] skip: checkpoint.zip not found: ${model_dir}"
      continue
    fi

    local timestamp_dir
    local experiment_dir
    local experiment_name
    local issue
    local pair
    local agent0
    local agent1

    timestamp_dir="$(dirname "$model_dir")"
    experiment_dir="$(dirname "$timestamp_dir")"
    experiment_name="$(basename "$experiment_dir")"
    issue="${experiment_name%_*}"
    pair="${experiment_name##*_}"
    agent0="${pair%-*}"
    agent1="${pair#*-}"

    build_test_command "$issue" "$agent0" "$agent1" "$model_dir"

    echo "[${case_name} ${current}/${total}] issue=${issue} agents=${agent0}-${agent1}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '  %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
    ran=$((ran + 1))
  done
}

run_general_case() {
  local case_name="$1"
  local case_dir="${ROOT_DIR%/}/${case_name}"

  if [[ ! -d "$case_dir" ]]; then
    echo "Missing case directory: $case_dir" >&2
    exit 1
  fi

  mapfile -t model_dirs < <(
    find "$case_dir" -mindepth 3 -maxdepth 3 -type d -name RLBOA_Negotiator | sort | while read -r model_dir; do
      timestamp_dir="$(dirname "$model_dir")"
      experiment_dir="$(dirname "$timestamp_dir")"
      experiment_name="$(basename "$experiment_dir")"
      is_general=1
      for agent_name in $GENERAL_AGENTS; do
        if [[ "$experiment_name" != *"$agent_name"* ]]; then
          is_general=0
          break
        fi
      done
      if [[ "$is_general" == "1" && -f "${model_dir%/}/checkpoint.zip" ]]; then
        printf '%s\n' "$model_dir"
      fi
    done
  )

  local total="${#model_dirs[@]}"
  local current=0
  local ran=0

  echo "=== ${case_name}: ${total} general models ==="
  if [[ "$total" -eq 0 ]]; then
    echo "No general RLBOA_Negotiator/checkpoint.zip found under ${case_dir} containing agents: ${GENERAL_AGENTS}" >&2
    exit 1
  fi

  for model_dir in "${model_dirs[@]}"; do
    current=$((current + 1))
    if [[ "$LIMIT" -gt 0 && "$ran" -ge "$LIMIT" ]]; then
      echo "Reached LIMIT=${LIMIT} for ${case_name}"
      break
    fi

    build_general_test_command "$model_dir"

    echo "[${case_name} ${current}/${total}] general model=${model_dir}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '  %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
    ran=$((ran + 1))
  done
}

for raw_case in "${raw_cases[@]}"; do
  if [[ "$MODE" == "general" ]]; then
    run_general_case "$(normalize_case "$raw_case")"
  else
    run_case "$(normalize_case "$raw_case")"
  fi
done
