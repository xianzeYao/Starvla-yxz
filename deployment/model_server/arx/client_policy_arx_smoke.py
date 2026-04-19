from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.arx.client_utils import (
    build_deployment_config_from_args,
    configure_logging,
    connect_policy_client,
    dual_arm_model_order_to_robot_order,
    ensure_numpy_available,
    query_policy,
    resolve_repo_path,
    scalar_value,
    vector_value,
)


@dataclass
class SmokeTestConfig:
    dataset_root: Path
    output_dir: Path
    episode_index: int | None = None
    video_backend: str = "pyav"
    random_seed: int | None = None


def load_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "Smoke test requires LeRobot dataset dependencies. "
            "Use a Python environment with `lerobot`, `datasets`, `pyarrow`, and video backends installed."
        ) from exc
    return LeRobotDataset


def read_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset info file: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def read_total_episodes(dataset_root: Path) -> int:
    total_episodes = int(read_dataset_info(dataset_root).get("total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError(f"Invalid total_episodes for dataset {dataset_root}: {total_episodes}")
    return total_episodes


def load_action_names(dataset_root: Path) -> list[str]:
    names = read_dataset_info(dataset_root).get("features", {}).get("action", {}).get("names", [])
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return [f"action_{idx}" for idx in range(7)]


def load_repo_id(dataset_root: Path) -> str:
    source_path = dataset_root / "arx_collect_source.json"
    if source_path.is_file():
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            repo_id = str(payload.get("repo_id", "")).strip()
            if repo_id:
                return repo_id
        except Exception:
            pass
    return f"local/{dataset_root.name}"


def infer_dataset_task_prompt(dataset_root: Path) -> str | None:
    source_path = dataset_root / "arx_collect_source.json"
    if not source_path.is_file():
        return None

    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    source_episodes = payload.get("source_episodes", [])
    if not source_episodes:
        return None

    task_prompt = str(source_episodes[0].get("task", "")).strip()
    return task_prompt or None


def dataset_image_to_rgb_uint8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "convert") and callable(getattr(image, "convert")):
        image = np.asarray(image.convert("RGB"))

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected dataset image with 3 dims, but got shape {array.shape}")

    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB dataset image, but got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.max(array)) if array.size > 0 else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def sanitize_stem_component(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in value)
    return sanitized.strip("_") or "policy"


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def plot_alignment(
    figure_path: Path,
    per_dim_dir: Path,
    time_s: np.ndarray,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    action_names: list[str],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[smoke] matplotlib not installed, skip plotting", flush=True)
        return

    per_dim_dir.mkdir(parents=True, exist_ok=True)
    num_dims = gt_actions.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(14, max(3, 2.2 * num_dims)), sharex=True)
    if num_dims == 1:
        axes = [axes]

    for dim_idx, ax in enumerate(axes):
        ax.plot(time_s, gt_actions[:, dim_idx], label="gt", linewidth=1.2)
        ax.plot(time_s, pred_actions[:, dim_idx], label="pred", linewidth=1.0)
        ax.set_ylabel(action_names[dim_idx] if dim_idx < len(action_names) else f"action_{dim_idx}")
        ax.grid(True, alpha=0.25)
        if dim_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def choose_episode_index(smoke_cfg: SmokeTestConfig, total_episodes: int) -> int:
    if smoke_cfg.episode_index is not None:
        episode_index = int(smoke_cfg.episode_index)
    else:
        episode_index = random.Random(smoke_cfg.random_seed).randrange(total_episodes)

    if episode_index < 0 or episode_index >= total_episodes:
        raise IndexError(
            f"Smoke test episode_index={episode_index} is out of range for total_episodes={total_episodes}"
        )
    return episode_index


def capture_dataset_observation(
    item: dict[str, Any],
    camera_keys: tuple[str, ...],
    include_state: bool,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    images = [
        dataset_image_to_rgb_uint8(item[f"observation.images.{camera_key}"])
        for camera_key in camera_keys
    ]

    state = None
    if include_state:
        if "observation.state" not in item:
            raise KeyError("Dataset item is missing 'observation.state'")
        state = vector_value(item["observation.state"])
    return images, state


def get_dataset_task(item: dict[str, Any], fallback: str) -> str:
    task_value = item.get("task")
    if task_value is None:
        return fallback
    return str(task_value)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=10093)
    parser.add_argument("--control_dt", type=float, default=0.05)
    parser.add_argument("--execute_horizon", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--task_prompt", type=str, default="")
    parser.add_argument("--arm_side", type=str, required=True)
    parser.add_argument("--camera_keys", type=str, default="camera_h,camera_r")
    parser.add_argument("--image_size", type=str, default="640,480")
    parser.add_argument("--no_state", action="store_true")
    parser.add_argument("--dataset_root", "--smoke_test", dest="dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="deployment/dryrun_records/arx_smoke_test")
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--video_backend", type=str, default="pyav")
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def build_smoke_test_config(args: argparse.Namespace) -> SmokeTestConfig:
    return SmokeTestConfig(
        dataset_root=resolve_repo_path(args.dataset_root),
        output_dir=resolve_repo_path(args.output_dir),
        episode_index=args.episode_index,
        video_backend=args.video_backend,
        random_seed=args.random_seed,
    )


def run_smoke_test(args: argparse.Namespace) -> None:
    cfg = build_deployment_config_from_args(args)
    smoke_cfg = build_smoke_test_config(args)
    dataset_root = smoke_cfg.dataset_root
    output_root = smoke_cfg.output_dir
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Smoke test dataset root not found: {dataset_root}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    total_episodes = read_total_episodes(dataset_root)
    episode_index = choose_episode_index(smoke_cfg, total_episodes)
    prompt = cfg.task_prompt.strip() or infer_dataset_task_prompt(dataset_root)
    if not prompt:
        raise RuntimeError("Smoke test requires a task prompt, but none was provided or inferred")

    LeRobotDataset = load_lerobot_dataset()
    repo_id = load_repo_id(dataset_root)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[episode_index],
        download_videos=False,
        video_backend=smoke_cfg.video_backend,
    )

    total_steps = min(len(dataset), cfg.max_episode_steps)
    if total_steps <= 0:
        raise RuntimeError(f"Episode {episode_index} is empty in dataset {dataset_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_string()
    stem = (
        f"{sanitize_stem_component(f'{cfg.policy_host}_{cfg.policy_port}')}_"
        f"{dataset_root.name}_ep{episode_index:03d}_{stamp}"
    )
    per_dim_dir = output_root / f"{stem}_dry_run"
    records_path = per_dim_dir / f"{stem}.json"
    summary_path = per_dim_dir / f"{stem}_summary.json"
    figure_path = per_dim_dir / f"{stem}.png"

    client, metadata, action_chunk_size = connect_policy_client(cfg)
    try:
        print(
            f"[smoke] server={cfg.policy_host}:{cfg.policy_port} dataset={dataset_root} "
            f"episode={episode_index} steps={total_steps} arm_side={cfg.arm_side} "
            f"camera_keys={list(cfg.camera_keys)} include_state={cfg.include_state} "
            f"execute_horizon={cfg.execute_horizon} action_chunk_size={action_chunk_size}",
            flush=True,
        )

        episode_start_ts = scalar_value(dataset[0]["timestamp"])
        records: list[dict[str, Any]] = []
        step_idx = 0

        while step_idx < total_steps:
            item = dataset[step_idx]
            images, state = capture_dataset_observation(
                item,
                camera_keys=cfg.camera_keys,
                include_state=cfg.include_state,
            )

            query_start = time.perf_counter()
            action_chunk = query_policy(client, images, state, prompt, cfg=cfg, metadata=metadata)
            latency_ms = (time.perf_counter() - query_start) * 1000.0

            execute_count = min(cfg.execute_horizon, len(action_chunk), total_steps - step_idx)
            if execute_count <= 0:
                raise RuntimeError(f"No predicted action available at step {step_idx}")

            for local_idx in range(execute_count):
                dataset_step = step_idx + local_idx
                current_item = item if local_idx == 0 else dataset[dataset_step]
                gt_action = vector_value(current_item["action"])
                pred_action = np.asarray(action_chunk[local_idx], dtype=np.float32).reshape(-1)
                if cfg.arm_side == "both" and pred_action.shape[0] == 14 and gt_action.shape[0] == 14:
                    pred_action = dual_arm_model_order_to_robot_order(pred_action)
                if pred_action.shape != gt_action.shape:
                    raise RuntimeError(
                        f"Predicted action shape {pred_action.shape} does not match gt {gt_action.shape}"
                    )

                records.append(
                    {
                        "step": dataset_step,
                        "episode_index": int(scalar_value(current_item["episode_index"])),
                        "frame_index": int(scalar_value(current_item["frame_index"])),
                        "time_s": scalar_value(current_item["timestamp"]) - episode_start_ts,
                        "dataset_timestamp_s": scalar_value(current_item["timestamp"]),
                        "plan_origin_step": step_idx,
                        "chunk_offset": local_idx,
                        "latency_ms": latency_ms,
                        "task": get_dataset_task(current_item, prompt),
                        "gt_action": gt_action,
                        "pred_action": pred_action,
                        "abs_error": np.abs(pred_action - gt_action),
                    }
                )

            step_idx += execute_count
            print(
                f"[smoke] step={step_idx}/{total_steps} query_latency={latency_ms:.2f}ms execute_count={execute_count}",
                flush=True,
            )

        time_s = np.asarray([record["time_s"] for record in records], dtype=np.float32)
        gt_actions = np.asarray([record["gt_action"] for record in records], dtype=np.float32)
        pred_actions = np.asarray([record["pred_action"] for record in records], dtype=np.float32)
        abs_error = np.abs(pred_actions - gt_actions)
        sq_error = (pred_actions - gt_actions) ** 2
        action_names = load_action_names(dataset_root)
        if len(action_names) < gt_actions.shape[1]:
            action_names.extend([f"action_{idx}" for idx in range(len(action_names), gt_actions.shape[1])])
        action_names = action_names[: gt_actions.shape[1]]

        summary = {
            "policy_type": "remote_server",
            "rollout_mode": "smoke_test",
            "model_path": f"ws://{cfg.policy_host}:{cfg.policy_port}",
            "policy_host": cfg.policy_host,
            "policy_port": cfg.policy_port,
            "server_metadata": metadata,
            "dataset_root": str(dataset_root),
            "repo_id": repo_id,
            "episode_index": int(episode_index),
            "total_steps": int(total_steps),
            "chunk_length": int(action_chunk_size),
            "model_chunk_length": int(action_chunk_size),
            "replan_interval": int(cfg.execute_horizon),
            "chunk_method": "replace",
            "control_dt": float(cfg.control_dt),
            "execute_horizon": int(cfg.execute_horizon),
            "max_episode_steps": int(cfg.max_episode_steps),
            "arm_side": cfg.arm_side,
            "camera_keys": list(cfg.camera_keys),
            "include_state": cfg.include_state,
            "task": prompt,
            "action_names": action_names,
            "mae_per_dim": abs_error.mean(axis=0),
            "rmse_per_dim": np.sqrt(sq_error.mean(axis=0)),
            "mae_mean": float(abs_error.mean()),
            "rmse_mean": float(np.sqrt(sq_error.mean())),
            "records_path": str(records_path),
            "figure_path": str(figure_path),
            "per_dim_dir": str(per_dim_dir),
        }

        save_json(records_path, records)
        save_json(summary_path, summary)
        plot_alignment(
            figure_path=figure_path,
            per_dim_dir=per_dim_dir,
            time_s=time_s,
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            action_names=action_names,
            title=f"ARX smoke test | {cfg.policy_host}:{cfg.policy_port} | {dataset_root.name} | episode {episode_index}",
        )

        print(f"[smoke] records saved to: {records_path}", flush=True)
        print(f"[smoke] summary saved to: {summary_path}", flush=True)
        print(f"[smoke] figure saved to: {figure_path}", flush=True)
    finally:
        client.close()


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    ensure_numpy_available()
    configure_logging(args.log_level)
    run_smoke_test(args)


if __name__ == "__main__":
    main()
