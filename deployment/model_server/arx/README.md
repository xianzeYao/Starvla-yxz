# ARX / Gravity Single Deployment

This deployment flow is tailored to the single-arm, 7D joint-action policy trained on:

- two RGB images: `cam_high`, `cam_right_wrist`
- one 7D state vector: `[joint_0 ... joint_5, gripper]`
- one 7D action vector: `[joint_0 ... joint_5, gripper]`

## Files

- `server_policy_arx.py`: start the policy server on the GPU machine
- `client_policy_arx.py`: robot-side deployment loop with a single adapter class to fill in
- `remote_arx_client_pseudocode.py`: robot-side pseudocode template
- `joint_action_utils.py`: server-side action unnormalization / schedule helpers
- `run_server_arx.sh`: local helper script to start the server
- `run_client_arx.sh`: robot-side helper script

## Start The Policy Server

```bash
bash deployment/model_server/arx/run_server_arx.sh
```

Or run manually:

```bash
CUDA_VISIBLE_DEVICES=5 /data/yxz/conda/envs/starVLA/bin/python deployment/model_server/arx/server_policy_arx.py \
  --ckpt_path /path/to/steps_10000_pytorch_model.pt \
  --port 10093 \
  --use_bf16 \
  --num_inference_timesteps_override 4
```

The checkpoint path must point to the `.pt` file inside the run directory. The server also expects:

- `config.yaml`
- `dataset_statistics.json`

to exist under the same run folder.

## What The Server Returns

The base model predicts `normalized_actions`.

The ARX server additionally returns:

- `raw_actions`: action chunk after unnormalization
- `action_chunk_size`
- `unnorm_key`
- `normalization_mode`
- `num_inference_timesteps`

For your robot-side integration, use `raw_actions` directly.
The robot-side client no longer needs the checkpoint path and will raise an error if the server does not return `raw_actions`.

## Robot-Side Request Contract

The request should look like:

```python
{
    "examples": [{
        "image": [cam_high, cam_right_wrist],
        "lang": task_prompt,
        "state": state[None, :]
    }]
}
```

Required conventions:

- image order:
  - `cam_high`
  - `cam_right_wrist`
- state order:
  - `joint_0`
  - `joint_1`
  - `joint_2`
  - `joint_3`
  - `joint_4`
  - `joint_5`
  - `gripper`

## Robot-Side Integration

If you want the fewest changes on the robot machine, edit only:

- `deployment/model_server/arx/client_policy_arx.py`

Inside that file, replace `PlaceholderARXRobotAdapter` with your real implementation for:

- `reset(task_prompt)`
- `get_images()`
- `get_state()`
- `step_joint(action)`
- optionally `should_stop(step_idx)`

Then start the robot-side loop with:

```bash
bash deployment/model_server/arx/run_client_arx.sh
```

Or manually:

```bash
python deployment/model_server/arx/client_policy_arx.py \
  --policy_host <server_ip> \
  --policy_port 10093 \
  --control_dt 0.05 \
  --execute_horizon 4 \
  --max_episode_steps 400 \
  --task_prompt "your task prompt"
```

## Recommended Deployment Settings

For the current trained model:

- action chunk size: `16`
- recommended client `control_dt`: `0.05s`
- recommended execute horizon: `4`
- recommended policy query frequency: about `5 Hz`
- recommended flow-matching inference steps: `4`

Meaning:

- the policy predicts a 16-step action chunk
- you execute only the first 4 steps
- then you re-query the policy

`execute_horizon` is a robot-side control-loop choice.
The server only returns the trained chunk size and the action outputs.
The client chooses its own `control_dt` and how many steps to execute before re-querying.

This is a good starting point because it balances:

- inference latency
- network latency
- closed-loop replanning frequency

If inference is too slow:

- increase `execute_horizon` to `6` or `8`
- or reduce `num_inference_timesteps_override` to `2` or `3`

If control feels too open-loop:

- reduce `execute_horizon` back toward `4`
- or try `num_inference_timesteps_override=6`

## Deployment Requirements Checklist

Server-side machine:

- trained checkpoint `.pt`
- matching `config.yaml`
- matching `dataset_statistics.json`
- `starVLA` conda environment
- GPU with enough memory for Qwen3-VL + QwenGR00T

Robot-side machine:

- can capture the two RGB images
- can read the current 7D robot state
- can execute `step_joint(action)`
- can send websocket messages to the server machine
- can provide the task string explicitly as `lang`
- does not need a local StarVLA checkpoint

You should make a ssh tunnel from the robot machine to the server machine for secure communication:

usage as an example:

```bash
ssh -N \
  -L 10093:127.0.0.1:10093 \
  -p 5054 \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  yxz@112.25.93.66
```