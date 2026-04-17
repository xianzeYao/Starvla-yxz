#!/usr/bin/env bash
set -euo pipefail

policy_host="${POLICY_HOST:-127.0.0.1}"
policy_port="${POLICY_PORT:-10093}"
control_dt="${CONTROL_DT:-0.05}"
execute_horizon="${EXECUTE_HORIZON:-4}"
max_episode_steps="${MAX_EPISODE_STEPS:-400}"
task_prompt="${TASK_PROMPT:-stack the two paper cups on top of the paper cup closest to the shelf one by one and place the stacked cups on the shelf}"
python_bin="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED=1

"${python_bin}" deployment/model_server/arx/client_policy_arx.py \
  --policy_host "${policy_host}" \
  --policy_port "${policy_port}" \
  --control_dt "${control_dt}" \
  --execute_horizon "${execute_horizon}" \
  --max_episode_steps "${max_episode_steps}" \
  --task_prompt "${task_prompt}"
