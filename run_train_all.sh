#!/usr/bin/env bash
set -euo pipefail

DOMAINS=(
  Laptop
  ItexvsCypress
  IS_BT_Acquisition
  Grocery
  thompson
  Car
  EnergySmall_A
)

AGENTS=(
  Boulware
  Linear
  Conceder
  Atlas3
)

TIMESTEPS="${TIMESTEPS:-100000}"
N_ENVS="${N_ENVS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
N_ACTIONS="${N_ACTIONS:-10}"
SAVE_ROOT="${SAVE_ROOT:-}"
NO_NOISE="${NO_NOISE:-0}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA_ARGS=()
if [[ "$NO_NOISE" == "1" ]]; then
  EXTRA_ARGS+=(--no-noise)
fi

total=$(( ${#DOMAINS[@]} * (${#AGENTS[@]} * (${#AGENTS[@]} + 1) / 2) ))
current=0

for domain in "${DOMAINS[@]}"; do
  for ((i = 0; i < ${#AGENTS[@]}; i++)); do
    for ((j = i; j < ${#AGENTS[@]}; j++)); do
      current=$((current + 1))
      agent0="${AGENTS[$i]}"
      agent1="${AGENTS[$j]}"
      SAVE_ARGS=()
      if [[ -n "$SAVE_ROOT" ]]; then
        timestamp="$(date +%Y%m%d-%H%M%S)"
        SAVE_ARGS=(-sp "${SAVE_ROOT%/}/${domain}_${agent0}-${agent1}/${timestamp}-TA/")
      fi

      cmd=(
        python3 train.py
        -a "$agent0" "$agent1" \
        -i "$domain" \
        -t "$TIMESTEPS" \
        -n "$N_ENVS" \
        --eval-episodes "$EVAL_EPISODES" \
        --n-actions "$N_ACTIONS" \
        "${SAVE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
      )

      echo "[$current/$total] domain=$domain agents=$agent0-$agent1"
      if [[ "$DRY_RUN" == "1" ]]; then
        printf '  %q' "${cmd[@]}"
        printf '\n'
      else
        "${cmd[@]}"
      fi
    done
  done
done
