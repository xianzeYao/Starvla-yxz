from __future__ import annotations

import argparse
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


@dataclass
class DeploymentConfig:
    policy_host: str = "127.0.0.1"
    policy_port: int = 10093
    control_dt: float = 0.05
    execute_horizon: int = 4
    max_episode_steps: int = 400
    task_prompt: str = (
        "stack the two paper cups on top of the paper cup closest to the shelf "
        "one by one and place the stacked cups on the shelf"
    )


class RobotAdapterBase(ABC):
    """
    Replace this adapter with your real robot interface.

    Training assumptions for the current policy:
    - two RGB images in this exact order:
      1. cam_high
      2. cam_right_wrist
    - one 7D joint state:
      [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper]
    - one 7D absolute joint action in the same order
    """

    def reset(self, task_prompt: str) -> None:
        """Optional: move the robot to the start pose for a new episode."""

    @abstractmethod
    def get_images(self) -> list[np.ndarray]:
        """Return [cam_high, cam_right_wrist], uint8 RGB, shape (H, W, 3)."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current 7D state in training order."""

    @abstractmethod
    def step_joint(self, action: np.ndarray) -> None:
        """Execute one 7D absolute joint action."""

    def should_stop(self, step_idx: int) -> bool:
        """
        Optional stop condition.

        Override this to stop on success, timeout, teleop interrupt, safety trigger, etc.
        """
        return False

    def close(self) -> None:
        """Optional cleanup hook."""


class PlaceholderARXRobotAdapter(RobotAdapterBase):
    """
    Minimal placeholder for the robot-side machine.

    Edit only this class on the real robot machine if you want to keep the rest
    of the deployment loop unchanged.
    """

    def reset(self, task_prompt: str) -> None:
        logging.info("Reset requested for task: %s", task_prompt)
        raise NotImplementedError("Implement reset() for your robot.")

    def get_images(self) -> list[np.ndarray]:
        raise NotImplementedError("Implement get_images() and return [cam_high, cam_right_wrist].")

    def get_state(self) -> np.ndarray:
        raise NotImplementedError("Implement get_state() and return a 7D joint state.")

    def step_joint(self, action: np.ndarray) -> None:
        raise NotImplementedError("Implement step_joint() for your robot controller.")


class ARXPolicyClientRunner:
    def __init__(self, cfg: DeploymentConfig, robot: RobotAdapterBase) -> None:
        self.cfg = cfg
        self.robot = robot
        self.client = WebsocketClientPolicy(host=cfg.policy_host, port=cfg.policy_port)
        self.server_metadata = self.client.get_server_metadata()

        action_chunk_size = self.server_metadata.get("action_chunk_size")

        if action_chunk_size is None:
            raise KeyError(
                "Server metadata is missing 'action_chunk_size'. "
                "Please start the policy with deployment/model_server/arx/server_policy_arx.py."
            )
        if cfg.execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive")
        if cfg.execute_horizon > int(action_chunk_size):
            raise ValueError(
                f"execute_horizon={cfg.execute_horizon} exceeds action_chunk_size={action_chunk_size}"
            )
        if cfg.control_dt <= 0:
            raise ValueError("control_dt must be positive")

        self.action_chunk_size = int(action_chunk_size)
        self.control_dt = float(cfg.control_dt)
        self.query_hz = 1.0 / (self.control_dt * float(cfg.execute_horizon))

        logging.info("Connected to policy server at %s:%s", cfg.policy_host, cfg.policy_port)
        logging.info("Server metadata: %s", self.server_metadata)
        logging.info(
            "Client execution config: action_chunk_size=%s execute_horizon=%s control_dt=%.4f query_hz=%.3f",
            self.action_chunk_size,
            cfg.execute_horizon,
            self.control_dt,
            self.query_hz,
        )

    def _validate_images(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if len(images) != 2:
            raise ValueError(f"Expected exactly 2 images, but got {len(images)}")

        validated = []
        for idx, image in enumerate(images):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"Image {idx} must have shape (H, W, 3), but got {array.shape}")
            if array.dtype != np.uint8:
                array = array.astype(np.uint8)
            validated.append(array)
        return validated

    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (7,):
            raise ValueError(f"Expected state shape (7,), but got {state.shape}")
        return state.reshape(1, -1)

    def build_request(self, task_prompt: str) -> dict:
        images = self._validate_images(self.robot.get_images())
        state = self._validate_state(self.robot.get_state())
        return {
            "examples": [
                {
                    "image": images,
                    "lang": task_prompt,
                    "state": state,
                }
            ],
            "do_sample": False,
        }

    def query_policy(self, task_prompt: str) -> np.ndarray:
        request_data = self.build_request(task_prompt)
        response = self.client.predict_action(request_data)
        data = response["data"]

        if "raw_actions" not in data:
            raise KeyError(
                "Server response is missing 'raw_actions'. "
                "Use deployment/model_server/arx/server_policy_arx.py so unnormalization stays on the server side."
            )

        raw_actions = np.asarray(data["raw_actions"][0], dtype=np.float32)
        if raw_actions.ndim != 2 or raw_actions.shape[-1] != 7:
            raise ValueError(f"Expected raw_actions shape [T, 7], but got {raw_actions.shape}")
        return raw_actions

    def run_episode(self, task_prompt: str | None = None) -> None:
        prompt = task_prompt or self.cfg.task_prompt
        step_idx = 0

        logging.info("Starting episode with task prompt: %s", prompt)
        self.robot.reset(prompt)

        try:
            while step_idx < self.cfg.max_episode_steps:
                if self.robot.should_stop(step_idx):
                    logging.info("Stop requested before querying policy at step %s", step_idx)
                    break

                query_start = time.time()
                action_chunk = self.query_policy(prompt)
                query_latency = time.time() - query_start

                execute_count = min(self.cfg.execute_horizon, len(action_chunk))
                for local_idx in range(execute_count):
                    if self.robot.should_stop(step_idx):
                        logging.info("Stop requested before executing step %s", step_idx)
                        return

                    action = action_chunk[local_idx]
                    action_start = time.time()
                    self.robot.step_joint(action)
                    step_idx += 1

                    action_latency = time.time() - action_start
                    sleep_time = max(0.0, self.control_dt - action_latency)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                logging.info(
                    "step=%s query_latency=%.3fs execute_count=%s action_chunk_size=%s",
                    step_idx,
                    query_latency,
                    execute_count,
                    self.action_chunk_size,
                )
        finally:
            self.robot.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=10093)
    parser.add_argument("--control_dt", type=float, default=0.05)
    parser.add_argument("--execute_horizon", type=int, default=4)
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--task_prompt", type=str, required=True)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        force=True,
    )

    cfg = DeploymentConfig(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        control_dt=args.control_dt,
        execute_horizon=args.execute_horizon,
        max_episode_steps=args.max_episode_steps,
        task_prompt=args.task_prompt,
    )

    robot = PlaceholderARXRobotAdapter()
    runner = ARXPolicyClientRunner(cfg=cfg, robot=robot)
    runner.run_episode()


if __name__ == "__main__":
    main()
