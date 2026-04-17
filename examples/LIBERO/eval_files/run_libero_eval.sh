#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

###########################################################################################
# === Please modify the following paths according to your environment ===
CKPT_PATH="${CKPT_PATH:-/data/yxz/starvla/outputs/libero_qwen3_gr00t_test/checkpoints/steps_30000_pytorch_model.pt}"
STAR_PYTHON="${STAR_PYTHON:-/data/yxz/conda/envs/starVLA/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data/yxz/conda/envs/libero/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/home/yxz/LIBERO}"

GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-5694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-5}"
HOST="${HOST:-127.0.0.1}"
# === End of environment variable configuration ===
###########################################################################################

export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"

folder_name="$(echo "${CKPT_PATH}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/results/libero_eval}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-${EVAL_ROOT}/${folder_name}/results/${TASK_SUITE_NAME}}"
LOG_DIR="${LOG_DIR:-${EVAL_ROOT}/${folder_name}/logs/${TASK_SUITE_NAME}}"
SERVER_LOG="${LOG_DIR}/server.log"
EVAL_LOG="${LOG_DIR}/eval.log"

mkdir -p "${VIDEO_OUT_PATH}" "${LOG_DIR}"

SERVER_PID=""

cleanup() {
  local exit_code=$?
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Checkpoint: ${CKPT_PATH}"
echo "[INFO] Task suite: ${TASK_SUITE_NAME}"
echo "[INFO] Trials per task: ${NUM_TRIALS_PER_TASK}"
echo "[INFO] Log dir: ${LOG_DIR}"
echo "[INFO] Video dir: ${VIDEO_OUT_PATH}"

echo "[INFO] Starting policy server on ${HOST}:${PORT} ..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STAR_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[INFO] Waiting for policy server readiness ..."
for _ in $(seq 1 90); do
  if "${STAR_PYTHON}" - <<PY
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("${HOST}", int("${PORT}")))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY
  then
    echo "[INFO] Policy server is ready."
    break
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Policy server exited early. Check ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[ERROR] Policy server is not running. Check ${SERVER_LOG}" >&2
  exit 1
fi

if ! "${STAR_PYTHON}" - <<PY
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("${HOST}", int("${PORT}")))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY
then
  echo "[ERROR] Timed out waiting for policy server. Check ${SERVER_LOG}" >&2
  exit 1
fi

echo "[INFO] Starting LIBERO evaluation ..."
"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT_PATH}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  2>&1 | tee "${EVAL_LOG}"

echo "[INFO] LIBERO evaluation finished."
