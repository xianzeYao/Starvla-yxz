# ARX Deployment

This directory contains the deployment flow for ARX robot-side execution with a
remote StarVLA policy server.

The current path is:

- the GPU machine runs `server_policy_arx.py`
- the robot machine runs `client_policy_arx.py`
- the two sides communicate through websocket
- in practice, the robot machine should usually connect through an SSH local tunnel

The deployment code now supports configurable:

- camera selection through `--camera_keys`
- single-arm or dual-arm execution through `--arm_side`
- optional state input through `--no_state`

If the client-side configuration does not match the server-side policy metadata,
the client should fail early instead of silently running with the wrong setup.

## Files

- `server_policy_arx.py`: policy server entrypoint on the GPU machine
- `client_policy_arx.py`: robot-side deployment loop
- `joint_action_utils.py`: action chunk sizing and action unnormalization helpers
- `run_server_arx.sh`: helper script for launching the server
- `run_client_arx.sh`: helper script for launching the client

## Server Responsibilities

The server loads the checkpoint, runs `predict_action`, and returns:

- `raw_actions`
- `action_chunk_size`
- `action_dim`
- `state_dim`
- `include_state`
- `normalization_mode`
- `unnorm_key`
- `num_inference_timesteps`
- optionally `camera_keys`

Use `raw_actions` directly on the robot side. The client does not need local
checkpoint files.

## Client Responsibilities

The client is responsible for:

- capturing the selected camera views in the exact order of `--camera_keys`
- optionally collecting state
- sending `image` and `lang`, and optionally `state`
- executing the returned `raw_actions`
- validating that client-side settings match server-side metadata

## Request Contract

The request sent to the server looks like:

```python
{
    "examples": [{
        "image": [img_0, img_1, img_2],
        "lang": task_prompt,
        "state": state[None, :],  # only when state is enabled
    }],
    "do_sample": False,
}
```

Notes:

- `image` is an ordered list, not a dict
- image order must match `--camera_keys`
- `state` must be omitted when `--no_state` is used

## Recommended Runtime Defaults

For current ARX deployment:

- dataset fps: `20`
- control dt: `0.05`
- execute horizon: `4`
- chunk length: `16`
- inference steps: `4`

This means:

- the server predicts a 16-step action chunk
- the client executes the first 4 actions
- the client then re-queries the server

## Dual Fold Blanket V2 Example

The following example matches the current dual-arm blanket setup:

- checkpoint:
  `/data/yxz/starvla/outputs/dual_fold_blanket_v2_qwen3gr00t_50k/checkpoints/steps_50000_pytorch_model.pt`
- dataset:
  `/data/yxz/datasets/dual_fold_blanket_v2`
- action dimension: `14`
- state dimension: `14`
- camera keys:
  - `camera_h`
  - `camera_l`
  - `camera_r`
- deployment mode:
  - dual arm: `--arm_side both`
  - no state to action head: `--no_state`

## Start The Policy Server

Run this on the GPU machine:

```bash
CUDA_VISIBLE_DEVICES=1 /data/yxz/conda/envs/starVLA/bin/python \
  /home/yxz/starVLA-yxz/deployment/model_server/arx/server_policy_arx.py \
  --ckpt_path /data/yxz/starvla/arx/dual_fold_blanket_v3_baseline/checkpoints/steps_50000_pytorch_model.pt \
  --port 10093 \
  --use_bf16 \
  --num_inference_timesteps_override 4 \
  --camera_keys camera_h,camera_l,camera_r
```

Why `--camera_keys` is recommended:

- it is not strictly required for server inference
- but it lets the server expose the expected camera order in metadata
- then the client can fail fast if camera order or camera count is wrong

The checkpoint run directory must also contain:

- `config.yaml`
- `dataset_statistics.json`

## Check The Server Port

On the GPU machine, verify that the server is listening:

```bash
ss -ltnp | grep 10093
```

You should see a `LISTEN` entry for port `10093`.

## SSH Tunnel

The robot machine should usually not connect to the public inference port
directly. Instead, create a local SSH tunnel on the robot machine.

In this setup:

- SSH entrypoint of the server machine: `112.25.93.66:5054`
- server websocket port on the GPU machine: `10093`
- local forwarded port on the robot machine: `10093`

Run this on the robot machine:

```bash
ssh -N \
  -L 10093:127.0.0.1:10093 \
  -p 5054 \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  yxz@112.25.93.66
```

What this means:

- the robot machine listens on local `127.0.0.1:10093`
- traffic sent there is forwarded through SSH
- on the remote side it lands on `127.0.0.1:10093` of the server machine

So after the tunnel is created, the robot-side client should connect to:

- host: `127.0.0.1`
- port: `10093`

If local port `10093` is already occupied on the robot machine, use another
local port such as `11093`:

```bash
ssh -N \
  -L 11093:127.0.0.1:10093 \
  -p 5054 \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  yxz@112.25.93.66
```

Then run the client with `--policy_port 11093`.

## Verify The Tunnel

On the robot machine, after starting the tunnel, verify connectivity:

```bash
nc -vz 127.0.0.1 10093
```

If successful, you should see a message similar to:

```text
Connection to 127.0.0.1 10093 port [tcp/*] succeeded!
```

You can also check whether the local forwarded port is listening:

```bash
ss -ltnp | grep 10093
```

## Start The Robot Client

Run this on the robot machine after the tunnel is up:

```bash
python3 deployment/model_server/arx/client_policy_arx.py \
  --policy_host 127.0.0.1 \
  --policy_port 10093 \
  --control_dt 0.05 \
  --execute_horizon 4 \
  --max_episode_steps 200 \
  --task_prompt "dual fold blanket" \
  --arm_side both \
  --camera_keys camera_h,camera_l,camera_r \
  --image_size 640,480 \
  --no_state
```

Important:

- do not set `--policy_host` to the public server IP when using the tunnel
- when the tunnel is active, the client should always connect to local
  `127.0.0.1:<forwarded_port>`

## Live Deployment Checklist

GPU machine:

- checkpoint `.pt` exists
- matching `config.yaml` exists
- matching `dataset_statistics.json` exists
- server process is listening on port `10093`

Robot machine:

- can import the ARX SDK used by `client_policy_arx.py`
- can capture the selected camera views
- camera order matches `--camera_keys`
- can execute dual-arm control when `--arm_side both` is used
- tunnel is active
- `nc -vz 127.0.0.1 10093` succeeds before starting the client

## Common Failure Modes

- Wrong camera order:
  the client may now fail early if server metadata says the deployment expects
  `camera_h,camera_l,camera_r` but the client is configured differently.

- Wrong arm mode:
  `--arm_side both` expects a 14D action layout. If the server exposes a
  single-arm policy, the client should reject it.

- State mismatch:
  for this blanket setup, use `--no_state`. If the client tries to send state
  while the deployment is configured as state-free, the metadata check should
  reject it.

- Tunnel confusion:
  if you use SSH port forwarding, the client does not connect to the public IP.
  It connects to its own local forwarded port.
