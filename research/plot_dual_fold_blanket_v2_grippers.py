from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEFT_GRIPPER_INDEX = 6
RIGHT_GRIPPER_INDEX = 13


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_dataframe(dataset_root: Path) -> pd.DataFrame:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    frames = [pd.read_parquet(path) for path in parquet_paths]
    return pd.concat(frames, ignore_index=True)


def extract_grippers(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    states = np.stack(series.to_list()).astype(np.float32)
    if states.ndim != 2 or states.shape[1] <= RIGHT_GRIPPER_INDEX:
        raise ValueError(f"Unexpected observation.state shape: {states.shape}")
    return states[:, LEFT_GRIPPER_INDEX], states[:, RIGHT_GRIPPER_INDEX]


def plot_episode(
    episode_df: pd.DataFrame,
    fps: float,
    output_path: Path,
) -> dict[str, float | int]:
    ordered = episode_df.sort_values("frame_index").reset_index(drop=True)
    left_gripper, right_gripper = extract_grippers(ordered["observation.state"])
    time_s = ordered["frame_index"].to_numpy(dtype=np.float32) / float(fps)
    episode_index = int(ordered["episode_index"].iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(time_s, left_gripper, color="#1f77b4", linewidth=1.4)
    axes[0].set_ylabel("left_gripper")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_s, right_gripper, color="#d62728", linewidth=1.4)
    axes[1].set_ylabel("right_gripper")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"dual_fold_blanket_v2 | episode {episode_index} | "
        f"frames={len(ordered)} | duration={time_s[-1]:.2f}s"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "episode_index": episode_index,
        "num_frames": int(len(ordered)),
        "duration_s": float(time_s[-1]) if len(time_s) else 0.0,
        "left_min": float(left_gripper.min()),
        "left_max": float(left_gripper.max()),
        "left_mean": float(left_gripper.mean()),
        "right_min": float(right_gripper.min()),
        "right_max": float(right_gripper.max()),
        "right_mean": float(right_gripper.mean()),
        "plot_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/data/yxz/datasets/dual_fold_blanket_v2"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/dual_fold_blanket_v2_grippers"),
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    info = load_info(dataset_root)
    fps = float(info.get("fps", 20))

    df = load_dataframe(dataset_root)
    required_columns = {"episode_index", "frame_index", "observation.state"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    summary_rows: list[dict[str, float | int]] = []
    for episode_index, episode_df in df.groupby("episode_index", sort=True):
        output_path = output_dir / f"episode_{int(episode_index):03d}_grippers.png"
        row = plot_episode(episode_df, fps=fps, output_path=output_path)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("episode_index")
    summary_csv = output_dir / "episode_gripper_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved {len(summary_df)} episode plots to {output_dir}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
