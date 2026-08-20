"""Paired checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .config import ArmConfig, EnvConfig
from .env import observations, sample_scenarios
from .metrics import communication_bytes, paired_bootstrap_ci, paired_mean_bootstrap_ci, rollout
from .models import make_policy
from .env import radius_edges


def load_policy(path: Path, device: torch.device, eval_env_cfg: EnvConfig | None = None):
    checkpoint = torch.load(path, map_location=device)
    train_env_cfg = EnvConfig(**checkpoint["env_cfg"])
    env_cfg = eval_env_cfg or train_env_cfg
    arm_cfg = ArmConfig(**checkpoint["arm_cfg"])
    dummy = sample_scenarios(2, env_cfg, device=device)
    eval_obs_dim = observations(dummy.state0, dummy, env_cfg).shape[-1]
    if eval_obs_dim != arm_cfg.obs_dim:
        raise ValueError(
            f"eval EnvConfig gives obs_dim={eval_obs_dim}, "
            f"but checkpoint {path} expects obs_dim={arm_cfg.obs_dim}"
        )
    policy = make_policy(checkpoint["arm"], env_cfg, arm_cfg).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return checkpoint["arm"], policy, env_cfg, train_env_cfg


def env_overrides_from_args(base: EnvConfig, args: argparse.Namespace) -> EnvConfig:
    updates = {}
    for field, attr in (
        ("n_agents", "eval_n_agents"),
        ("max_force", "eval_max_force"),
        ("world_x", "eval_world_x"),
        ("world_y", "eval_world_y"),
        ("comm_radius", "eval_comm_radius"),
        ("scenario_difficulty", "eval_scenario_difficulty"),
        ("gap_y_range", "eval_gap_y_range"),
        ("gap_half_base", "eval_gap_half_base"),
        ("gap_half_difficulty_scale", "eval_gap_half_difficulty_scale"),
        ("gap_half_jitter", "eval_gap_half_jitter"),
        ("wall_x_abs", "eval_wall_x_abs"),
        ("source_x", "eval_source_x"),
        ("goal_x", "eval_goal_x"),
        ("wall_sensor_range", "eval_wall_sensor_range"),
    ):
        value = getattr(args, attr)
        if value is not None:
            updates[field] = value
    if args.eval_local_observation is not None:
        updates["local_observation"] = args.eval_local_observation
    if args.eval_strict_local_observation is not None:
        updates["strict_local_observation"] = args.eval_strict_local_observation
        if args.eval_strict_local_observation:
            updates["local_observation"] = True
    return replace(base, **updates) if updates else base


@torch.no_grad()
def paired_eval(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    first_checkpoint = torch.load(args.checkpoints[0], map_location="cpu")
    env_cfg = env_overrides_from_args(EnvConfig(**first_checkpoint["env_cfg"]), args)
    loaded = [load_policy(Path(path), device, env_cfg) for path in args.checkpoints]
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    successes = {arm: [] for arm, _, _, _ in loaded}
    collisions = {arm: [] for arm, _, _, _ in loaded}
    finish_steps = {arm: [] for arm, _, _, _ in loaded}
    done = 0
    while done < args.episodes:
        current = min(args.batch_size, args.episodes - done)
        scenario = sample_scenarios(current, env_cfg, device=device, generator=generator)
        for arm, policy, _, _ in loaded:
            result = rollout(policy, scenario, env_cfg)
            successes[arm].append(result["success"].float().cpu())
            collisions[arm].append(result["collision"].float().cpu())
            finish_steps[arm].append(result["finish_step"].float().cpu())
        done += current

    edge_index = radius_edges(env_cfg, device=device)
    summary = {
        "episodes": args.episodes,
        "seed": args.seed,
        "env_config": asdict(env_cfg),
        "arms": {},
        "paired": {},
    }
    success_tensors = {}
    collision_tensors = {}
    finish_tensors = {}
    for arm, policy, _, _ in loaded:
        success = torch.cat(successes[arm])
        collision = torch.cat(collisions[arm])
        finish = torch.cat(finish_steps[arm])
        success_tensors[arm] = success
        collision_tensors[arm] = collision
        finish_tensors[arm] = finish
        summary["arms"][arm] = {
            "success_rate": float(success.mean().item()),
            "collision_rate": float(collision.mean().item()),
            "mean_finish_step": float(finish.mean().item()),
            "wire_bytes_total": communication_bytes(
                policy,
                edge_index,
                batch_size=args.episodes,
                steps=env_cfg.max_steps,
            ),
            "wire_bytes_per_episode": communication_bytes(
                policy,
                edge_index,
                batch_size=1,
                steps=env_cfg.max_steps,
            ),
            "message_floats_per_directed_edge_round": policy.communication_floats_per_directed_edge_round(),
            "communication_rounds": policy.communication_rounds(),
            "checkpoint_env_config": asdict(
                next(train_cfg for loaded_arm, _, _, train_cfg in loaded if loaded_arm == arm)
            ),
        }

    for left in success_tensors:
        for right in success_tensors:
            if left == right:
                continue
            summary["paired"][f"{left}_minus_{right}"] = paired_bootstrap_ci(
                success_tensors[left],
                success_tensors[right],
                seed=args.seed,
            )
            summary["paired"][f"{left}_collision_minus_{right}"] = paired_bootstrap_ci(
                collision_tensors[left],
                collision_tensors[right],
                seed=args.seed + 1,
            )
            summary["paired"][f"{left}_finish_minus_{right}"] = paired_mean_bootstrap_ci(
                finish_tensors[left],
                finish_tensors[right],
                seed=args.seed + 2,
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=60601)
    parser.add_argument("--out", default="")
    parser.add_argument("--eval-n-agents", type=int, default=None)
    parser.add_argument("--eval-max-force", type=float, default=None)
    parser.add_argument("--eval-world-x", type=float, default=None)
    parser.add_argument("--eval-world-y", type=float, default=None)
    parser.add_argument("--eval-comm-radius", type=float, default=None)
    parser.add_argument("--eval-scenario-difficulty", type=float, default=None)
    parser.add_argument("--eval-gap-y-range", type=float, default=None)
    parser.add_argument("--eval-gap-half-base", type=float, default=None)
    parser.add_argument("--eval-gap-half-difficulty-scale", type=float, default=None)
    parser.add_argument("--eval-gap-half-jitter", type=float, default=None)
    parser.add_argument("--eval-wall-x-abs", type=float, default=None)
    parser.add_argument("--eval-source-x", type=float, default=None)
    parser.add_argument("--eval-goal-x", type=float, default=None)
    parser.add_argument("--eval-wall-sensor-range", type=float, default=None)
    parser.set_defaults(eval_local_observation=None)
    parser.add_argument(
        "--eval-local-observation", dest="eval_local_observation", action="store_true"
    )
    parser.add_argument(
        "--eval-global-observation", dest="eval_local_observation", action="store_false"
    )
    parser.set_defaults(eval_strict_local_observation=None)
    parser.add_argument(
        "--eval-strict-local-observation", dest="eval_strict_local_observation", action="store_true"
    )
    parser.add_argument(
        "--eval-nonstrict-local-observation",
        dest="eval_strict_local_observation",
        action="store_false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = paired_eval(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
