"""Evaluation and communication metrics."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
from torch import Tensor

from .config import EnvConfig
from .env import (
    PianoBatch,
    collision_mask,
    observations,
    radius_edges,
    sample_scenarios,
    step_dynamics,
    success_mask,
)
from .models import BasePolicy


def communication_bytes(
    policy: BasePolicy,
    edge_index: Tensor,
    *,
    batch_size: int,
    steps: int,
) -> int:
    directed_edges = 2 * int(edge_index.shape[0])
    floats = (
        batch_size
        * steps
        * directed_edges
        * policy.communication_rounds()
        * policy.communication_floats_per_directed_edge_round()
    )
    return int(floats * 4)


@torch.no_grad()
def rollout(
    policy: BasePolicy,
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    max_steps: int | None = None,
) -> dict[str, Tensor]:
    edge_index = radius_edges(cfg, device=scenario.state0.device)
    steps = max_steps or cfg.max_steps
    state = scenario.state0.clone()
    hidden = policy.init_hidden(state.shape[0], state.device)
    alive = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    succeeded = torch.zeros_like(alive)
    collided = torch.zeros_like(alive)
    finish_step = torch.full((state.shape[0],), steps, dtype=torch.long, device=state.device)

    for step in range(steps):
        obs = observations(state, scenario, cfg)
        out = policy(obs, edge_index, hidden)
        hidden = out.hidden
        next_state = step_dynamics(state, out.forces, cfg)
        hit = collision_mask(next_state, scenario, cfg) & alive
        ok = success_mask(next_state, scenario, cfg) & alive & (~hit)
        collided |= hit
        succeeded |= ok
        finished = hit | ok
        finish_step = torch.where(
            finished & alive, torch.full_like(finish_step, step + 1), finish_step
        )
        alive &= ~finished
        state = torch.where(alive[:, None], next_state, state)
        if not bool(alive.any()):
            break

    return {
        "success": succeeded,
        "collision": collided,
        "finish_step": finish_step,
        "final_state": state,
    }


@torch.no_grad()
def evaluate_policy(
    policy: BasePolicy,
    cfg: EnvConfig,
    *,
    episodes: int,
    batch_size: int,
    seed: int,
    device: torch.device | str,
) -> dict[str, float | int | dict[str, float]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    successes = []
    collisions = []
    finish_steps = []
    edge_index = radius_edges(cfg, device=device)
    done = 0
    while done < episodes:
        current = min(batch_size, episodes - done)
        scenario = sample_scenarios(current, cfg, device=device, generator=generator)
        result = rollout(policy, scenario, cfg)
        successes.append(result["success"].float().cpu())
        collisions.append(result["collision"].float().cpu())
        finish_steps.append(result["finish_step"].float().cpu())
        done += current
    success = torch.cat(successes)
    collision = torch.cat(collisions)
    finish = torch.cat(finish_steps)
    bytes_total = communication_bytes(policy, edge_index, batch_size=episodes, steps=cfg.max_steps)
    return {
        "episodes": episodes,
        "success_rate": float(success.mean().item()),
        "collision_rate": float(collision.mean().item()),
        "mean_finish_step": float(finish.mean().item()),
        "wire_bytes_total": bytes_total,
        "wire_bytes_per_episode": float(bytes_total / max(episodes, 1)),
        "edge_count": int(edge_index.shape[0]),
        "directed_edge_count": int(2 * edge_index.shape[0]),
        "env_config": asdict(cfg),
    }


def paired_bootstrap_ci(
    left_success: Tensor,
    right_success: Tensor,
    *,
    seed: int = 0,
    samples: int = 4000,
) -> dict[str, float]:
    """Bootstrap CI for paired success-rate difference left - right."""

    if left_success.shape != right_success.shape:
        raise ValueError("paired tensors must have the same shape")
    n = left_success.numel()
    generator = torch.Generator()
    generator.manual_seed(seed)
    diffs = []
    delta = left_success.float().cpu() - right_success.float().cpu()
    for _ in range(samples):
        idx = torch.randint(0, n, (n,), generator=generator)
        diffs.append(delta[idx].mean())
    values = torch.stack(diffs).sort().values
    lo = values[int(0.025 * (samples - 1))]
    hi = values[int(0.975 * (samples - 1))]
    mean = delta.mean()
    se = delta.std(unbiased=True) / math.sqrt(max(n, 1))
    return {
        "mean_diff": float(mean.item()),
        "normal_se": float(se.item()),
        "bootstrap95_low": float(lo.item()),
        "bootstrap95_high": float(hi.item()),
    }


def paired_mean_bootstrap_ci(
    left: Tensor,
    right: Tensor,
    *,
    seed: int = 0,
    samples: int = 4000,
) -> dict[str, float]:
    """Bootstrap CI for paired mean difference left - right."""

    if left.shape != right.shape:
        raise ValueError("paired tensors must have the same shape")
    n = left.numel()
    generator = torch.Generator()
    generator.manual_seed(seed)
    delta = left.float().cpu() - right.float().cpu()
    diffs = []
    for _ in range(samples):
        idx = torch.randint(0, n, (n,), generator=generator)
        diffs.append(delta[idx].mean())
    values = torch.stack(diffs).sort().values
    lo = values[int(0.025 * (samples - 1))]
    hi = values[int(0.975 * (samples - 1))]
    mean = delta.mean()
    se = delta.std(unbiased=True) / math.sqrt(max(n, 1))
    return {
        "mean_diff": float(mean.item()),
        "normal_se": float(se.item()),
        "bootstrap95_low": float(lo.item()),
        "bootstrap95_high": float(hi.item()),
    }
