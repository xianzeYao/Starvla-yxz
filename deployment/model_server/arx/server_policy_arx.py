from __future__ import annotations

import argparse
import logging
import socket

import torch

from deployment.model_server.arx.joint_action_utils import (
    get_action_chunk_size,
    get_action_stats,
    unnormalize_joint_actions,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


class JointActionPolicyWrapper:
    """Wrap a StarVLA checkpoint and return both normalized and raw joint actions."""

    def __init__(
        self,
        ckpt_path: str,
        unnorm_key: str | None = None,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:
        self.ckpt_path = ckpt_path
        self.policy = baseframework.from_pretrained(ckpt_path)
        self.unnorm_key, self.action_stats = get_action_stats(
            ckpt_path,
            unnorm_key=unnorm_key,
            action_mode=action_mode,
        )
        self.action_chunk_size = get_action_chunk_size(ckpt_path)
        self.normalization_mode = normalization_mode

    def to(self, *args, **kwargs):
        self.policy = self.policy.to(*args, **kwargs)
        return self

    def eval(self):
        self.policy = self.policy.eval()
        return self

    def predict_action(self, examples, **kwargs):
        result = self.policy.predict_action(examples=examples, **kwargs)
        normalized_actions = result["normalized_actions"]
        raw_actions = unnormalize_joint_actions(
            normalized_actions,
            self.action_stats,
            normalization_mode=self.normalization_mode,
        )
        result["raw_actions"] = raw_actions
        result["unnorm_key"] = self.unnorm_key
        result["action_chunk_size"] = self.action_chunk_size
        result["normalization_mode"] = self.normalization_mode
        return result


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--normalization_mode", type=str, default="min_max")
    parser.add_argument(
        "--num_inference_timesteps_override",
        type=int,
        default=None,
        help="Override the checkpoint's flow-matching inference steps at deployment time.",
    )
    return parser


def main(args) -> None:
    policy = JointActionPolicyWrapper(
        ckpt_path=args.ckpt_path,
        unnorm_key=args.unnorm_key,
        action_mode="abs",
        normalization_mode=args.normalization_mode,
    )

    if args.num_inference_timesteps_override is not None:
        override_steps = int(args.num_inference_timesteps_override)
        if override_steps <= 0:
            raise ValueError("num_inference_timesteps_override must be positive")
        policy.policy.action_model.num_inference_timesteps = override_steps
        policy.policy.config.framework.action_model.num_inference_timesteps = override_steps

    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating joint-action policy server at host=%s ip=%s", hostname, local_ip)
    logging.info("Resolved unnorm_key=%s", policy.unnorm_key)
    logging.info(
        "Using num_inference_timesteps=%s",
        policy.policy.action_model.num_inference_timesteps,
    )

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "env": "gravity_single_joint",
            "unnorm_key": policy.unnorm_key,
            "normalization_mode": args.normalization_mode,
            "action_chunk_size": policy.action_chunk_size,
            "num_inference_timesteps": policy.policy.action_model.num_inference_timesteps,
        },
    )
    logging.info("server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
