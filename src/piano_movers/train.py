"""Train one PianoMovers-Force arm."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import ArmConfig, EnvConfig
from .env import (
    body_to_world,
    collision_mask,
    desired_maneuver,
    expert_policy,
    observations,
    radius_edges,
    sample_scenarios,
    sample_training_states,
    step_dynamics,
    success_mask,
    t_shape_points,
    wrap_angle,
)
from .metrics import evaluate_policy
from .models import make_policy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def model_induced_states(
    policy,
    scenario,
    cfg: EnvConfig,
    edge_index: torch.Tensor,
    *,
    rollout_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect model-induced DAgger states from the current policy."""

    state = scenario.state0.clone()
    hidden = policy.init_hidden(state.shape[0], state.device)
    snapshots = []
    hidden_snapshots = []
    for _ in range(rollout_steps):
        snapshots.append(state.clone())
        hidden_snapshots.append(hidden.detach().clone())
        obs = observations(state, scenario, cfg)
        out = policy(obs, edge_index, hidden)
        next_state = step_dynamics(state, out.forces, cfg)
        done = collision_mask(next_state, scenario, cfg) | success_mask(next_state, scenario, cfg)
        reset_hidden = policy.init_hidden(state.shape[0], state.device)
        state = torch.where(done[:, None], scenario.state0, next_state)
        hidden = torch.where(done[:, None, None], reset_hidden, out.hidden)
    stacked = torch.stack(snapshots, dim=1)
    hidden_stacked = torch.stack(hidden_snapshots, dim=1)
    pick = torch.randint(0, rollout_steps, (state.shape[0],), device=state.device)
    row = torch.arange(state.shape[0], device=state.device)
    return stacked[row, pick], hidden_stacked[row, pick]


def smooth_safety_loss(
    next_state: torch.Tensor,
    scenario,
    cfg: EnvConfig,
) -> torch.Tensor:
    """Differentiable margin loss for wall slits and world bounds."""

    return smooth_safety_per_sample(next_state, scenario, cfg).mean()


def smooth_safety_per_sample(
    next_state: torch.Tensor,
    scenario,
    cfg: EnvConfig,
) -> torch.Tensor:
    """Per-episode differentiable collision-risk proxy.

    This is kept separate from ``smooth_safety_loss`` so rollout training can
    preserve safety margins relative to the expert without using a hard
    collision gate.  Lower is safer.
    """

    points = body_to_world(t_shape_points(next_state.device), next_state)
    wall_dx = (points[:, :, None, 0] - scenario.wall_x[:, None, :]).abs()
    rel_y = (points[:, :, None, 1] - scenario.gap_y[:, None, :]).abs()
    gap_limit = scenario.gap_half[:, None, :] - 0.10
    gap_violation = (rel_y - gap_limit).clamp_min(0.0)
    near_wall = torch.exp(-((wall_dx / (5.0 * cfg.wall_slab)).square()))
    wall_loss = (near_wall * gap_violation.square()).mean(dim=(1, 2))

    y_margin = (points[:, :, 1].abs() - (cfg.world_y - 0.10)).clamp_min(0.0).square().mean(dim=1)
    x_margin = (points[:, :, 0].abs() - (cfg.world_x - 0.12)).clamp_min(0.0).square().mean(dim=1)
    return wall_loss + y_margin + x_margin


def state_tracking_loss(
    pred_next: torch.Tensor, expert_next: torch.Tensor, cfg: EnvConfig
) -> torch.Tensor:
    """Normalized one-step rigid-state tracking used inside differentiable rollout."""

    pred_xy = torch.stack((pred_next[:, 0] / cfg.world_x, pred_next[:, 1] / cfg.world_y), dim=-1)
    expert_xy = torch.stack(
        (expert_next[:, 0] / cfg.world_x, expert_next[:, 1] / cfg.world_y), dim=-1
    )
    pos_loss = F.mse_loss(pred_xy, expert_xy)
    angle_loss = (wrap_angle(pred_next[:, 2] - expert_next[:, 2]) / torch.pi).square().mean()
    vel_loss = F.mse_loss(pred_next[:, 3:5] / 2.0, expert_next[:, 3:5] / 2.0)
    omega_loss = F.mse_loss(pred_next[:, 5] / 2.0, expert_next[:, 5] / 2.0)
    return pos_loss + 0.75 * angle_loss + 0.25 * vel_loss + 0.25 * omega_loss


def section_loss_fn(pred_section: torch.Tensor, target_section: torch.Tensor) -> torch.Tensor:
    """Supervise the physical prefix of a possibly wider learned section.

    The benchmark's interpretable section target has 8 coordinates:
    load twist, net wrench, passage offset, and progress.  Wider sheaf
    variants can use extra edge-stalk coordinates as learned coordination
    channels, but those coordinates should not be forced to zero.
    """

    width = min(pred_section.shape[-1], target_section.shape[-1])
    target = target_section[:, None, :width].expand_as(pred_section[..., :width])
    return F.mse_loss(pred_section[..., :width], target.detach())


def pose_potential(state: torch.Tensor, target: torch.Tensor, cfg: EnvConfig) -> torch.Tensor:
    """Smooth distance-to-current-maneuver target; lower is better."""

    xy = torch.stack(
        (
            (state[:, 0] - target[:, 0]) / cfg.world_x,
            (state[:, 1] - target[:, 1]) / cfg.world_y,
        ),
        dim=-1,
    )
    pos = torch.linalg.vector_norm(xy, dim=-1)
    angle = wrap_angle(state[:, 2] - target[:, 2]).abs() / torch.pi
    return pos + 0.65 * angle


def source_progress(state: torch.Tensor, scenario) -> torch.Tensor:
    denom = (scenario.goal[:, 0] - scenario.source[:, 0]).clamp_min(1e-4)
    return ((state[:, 0] - scenario.source[:, 0]) / denom).clamp(0.0, 1.0)


def differentiable_rollout_loss(
    policy,
    scenario,
    cfg: EnvConfig,
    edge_index: torch.Tensor,
    *,
    horizon: int,
    initial_fraction: float,
    force_weight: float,
    state_weight: float,
    target_weight: float,
    progress_weight: float,
    safety_weight: float,
    goal_capture_weight: float,
    section_weight: float,
    source_margin_weight: float,
    source_margin: float,
    clearance_margin_weight: float,
    clearance_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Short-horizon closed-loop loss, applied identically to all arms."""

    batch = scenario.state0.shape[0]
    device = scenario.state0.device
    state = sample_training_states(scenario, cfg)
    if initial_fraction > 0.0:
        initial_mask = torch.rand(batch, device=device) < initial_fraction
        state = torch.where(initial_mask[:, None], scenario.state0, state)
    hidden = policy.init_hidden(batch, device)

    force_terms = []
    state_terms = []
    target_terms = []
    progress_terms = []
    safety_terms = []
    source_margin_terms = []
    clearance_margin_terms = []
    goal_capture_terms = []
    section_terms = []

    for _ in range(horizon):
        obs = observations(state, scenario, cfg)
        target_forces, target_section, _ = expert_policy(state, scenario, cfg)
        output = policy(obs, edge_index, hidden)
        hidden = output.hidden

        pred_next = step_dynamics(state, output.forces, cfg)
        expert_next = step_dynamics(state, target_forces, cfg).detach()
        target_pose, _ = desired_maneuver(state, scenario)
        target_pose = target_pose.detach()
        expert_potential = pose_potential(expert_next, target_pose, cfg)
        next_potential = pose_potential(pred_next, target_pose, cfg)
        pred_progress = source_progress(pred_next, scenario)
        expert_progress = source_progress(expert_next, scenario).detach()
        current_progress = source_progress(state, scenario).detach()
        pred_safety = smooth_safety_per_sample(pred_next, scenario, cfg)
        expert_safety = smooth_safety_per_sample(expert_next, scenario, cfg).detach()

        force_terms.append(
            F.mse_loss(output.forces / cfg.max_force, target_forces.detach() / cfg.max_force)
        )
        state_terms.append(state_tracking_loss(pred_next, expert_next, cfg))
        target_terms.append(next_potential.square().mean())
        progress_terms.append(
            (next_potential - expert_potential + 0.01).clamp_min(0.0).square().mean()
        )
        safety_terms.append(pred_safety.mean())
        source_regret = (expert_progress - pred_progress + source_margin).clamp_min(0.0)
        backtrack = (current_progress - pred_progress + 0.25 * source_margin).clamp_min(0.0)
        source_margin_terms.append((source_regret.square() + backtrack.square()).mean())
        clearance_regret = (pred_safety - expert_safety + clearance_margin).clamp_min(0.0)
        clearance_margin_terms.append(clearance_regret.mean())
        late_phase = (state[:, 0] > scenario.wall_x[:, 1] + 0.25).float()
        goal_potential = pose_potential(pred_next, scenario.goal.detach(), cfg)
        stop_energy = (pred_next[:, 3:5] / 2.0).square().sum(dim=-1) + (
            pred_next[:, 5] / 2.0
        ).square()
        goal_capture = late_phase * (goal_potential.square() + 0.50 * stop_energy)
        goal_capture_terms.append(goal_capture.sum() / late_phase.sum().clamp_min(1.0))
        if "section" in output.aux:
            section_terms.append(section_loss_fn(output.aux["section"], target_section))

        state = pred_next

    force_loss = torch.stack(force_terms).mean()
    state_loss = torch.stack(state_terms).mean()
    target_loss = torch.stack(target_terms).mean()
    progress_loss = torch.stack(progress_terms).mean()
    safety_loss = torch.stack(safety_terms).mean()
    source_margin_loss = torch.stack(source_margin_terms).mean()
    clearance_margin_loss = torch.stack(clearance_margin_terms).mean()
    goal_capture_loss = torch.stack(goal_capture_terms).mean()
    section_loss = (
        torch.stack(section_terms).mean() if section_terms else torch.zeros((), device=device)
    )
    total = (
        force_weight * force_loss
        + state_weight * state_loss
        + target_weight * target_loss
        + progress_weight * progress_loss
        + safety_weight * safety_loss
        + source_margin_weight * source_margin_loss
        + clearance_margin_weight * clearance_margin_loss
        + goal_capture_weight * goal_capture_loss
        + section_weight * section_loss
    )
    return total, {
        "rollout_force_loss": force_loss,
        "rollout_state_loss": state_loss,
        "rollout_target_loss": target_loss,
        "rollout_progress_loss": progress_loss,
        "rollout_safety_loss": safety_loss,
        "rollout_source_margin_loss": source_margin_loss,
        "rollout_clearance_margin_loss": clearance_margin_loss,
        "rollout_goal_capture_loss": goal_capture_loss,
        "rollout_section_loss": section_loss,
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    env_cfg = EnvConfig(
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        max_force=args.max_force,
        comm_radius=args.comm_radius,
        world_x=args.world_x,
        world_y=args.world_y,
        success_pos_tol=args.success_pos_tol,
        success_angle_tol=args.success_angle_tol,
        scenario_difficulty=args.scenario_difficulty,
        gap_y_range=args.gap_y_range,
        gap_half_base=args.gap_half_base,
        gap_half_difficulty_scale=args.gap_half_difficulty_scale,
        gap_half_jitter=args.gap_half_jitter,
        wall_x_abs=args.wall_x_abs,
        source_x=args.source_x,
        goal_x=args.goal_x,
        local_observation=args.local_observation or args.strict_local_observation,
        strict_local_observation=args.strict_local_observation,
        wall_sensor_range=args.wall_sensor_range,
    )
    dummy = sample_scenarios(2, env_cfg, device=device)
    obs_dim = observations(dummy.state0, dummy, env_cfg).shape[-1]
    arm_cfg = ArmConfig(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        comm_rounds=args.comm_rounds,
        gbp_steps=args.gbp_steps,
        edge_sigma2=args.edge_sigma2,
        restriction_residual_scale=args.restriction_residual_scale,
        edge_dim=args.edge_dim,
        temporal_window=args.temporal_window,
        temporal_precision=args.temporal_precision,
        force_residual_scale=args.force_residual_scale,
        analytic_force_scale=args.analytic_force_scale,
        mean_only_messages=args.mean_only_messages,
        decoder_extra_context=args.decoder_extra_context,
        edge_conditioned_restrictions=args.edge_conditioned_restrictions,
        decoder_section_only=args.decoder_section_only,
    )
    policy = make_policy(args.arm, env_cfg, arm_cfg).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("use at most one of --init-checkpoint and --resume-checkpoint")
    edge_index = radius_edges(env_cfg, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{args.arm}_metrics.jsonl"
    best_success = -1.0
    step_offset = args.step_offset

    def build_checkpoint(step: int, eval_row: dict | None = None) -> dict:
        checkpoint = {
            "arm": args.arm,
            "env_cfg": asdict(env_cfg),
            "arm_cfg": asdict(arm_cfg),
            "state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_success": best_success,
        }
        if eval_row is not None:
            checkpoint["eval"] = eval_row
        return checkpoint

    def load_checkpoint(
        path: str,
        *,
        resume_optimizer: bool,
        allow_env_config_mismatch: bool = False,
        allow_arm_config_mismatch: bool = False,
        compatible_weights_only: bool = False,
    ) -> dict:
        checkpoint = torch.load(path, map_location=device)
        if checkpoint.get("arm") != args.arm:
            raise ValueError(
                f"checkpoint arm {checkpoint.get('arm')} does not match --arm {args.arm}"
            )
        loaded_env_cfg = asdict(EnvConfig(**checkpoint.get("env_cfg", {})))
        env_config_matches = loaded_env_cfg == asdict(env_cfg)
        if not env_config_matches and not allow_env_config_mismatch:
            raise ValueError("checkpoint EnvConfig does not match current arguments")
        loaded_arm_cfg = asdict(ArmConfig(**checkpoint.get("arm_cfg", {})))
        arm_config_matches = loaded_arm_cfg == asdict(arm_cfg)
        if not arm_config_matches and not allow_arm_config_mismatch:
            raise ValueError("checkpoint ArmConfig does not match current arguments")
        if (not env_config_matches or not arm_config_matches) and not compatible_weights_only:
            raise ValueError(
                "config-mismatched init requires --init-compatible-weights-only "
                "to avoid accidentally loading incompatible tensors"
            )
        if compatible_weights_only:
            current_state = policy.state_dict()
            loaded_state = checkpoint["state_dict"]
            compatible = {
                key: value
                for key, value in loaded_state.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            current_state.update(compatible)
            policy.load_state_dict(current_state)
            checkpoint["_compatible_loaded_keys"] = len(compatible)
            checkpoint["_compatible_total_keys"] = len(current_state)
        else:
            policy.load_state_dict(checkpoint["state_dict"])
        if resume_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint["_env_config_matches_current"] = env_config_matches
        checkpoint["_arm_config_matches_current"] = arm_config_matches
        return checkpoint

    if args.init_checkpoint:
        loaded = load_checkpoint(
            args.init_checkpoint,
            resume_optimizer=False,
            allow_env_config_mismatch=args.allow_init_env_config_mismatch,
            allow_arm_config_mismatch=args.allow_init_arm_config_mismatch,
            compatible_weights_only=args.init_compatible_weights_only,
        )
        if (
            "eval" in loaded
            and loaded.get("_env_config_matches_current", False)
            and loaded.get("_arm_config_matches_current", False)
        ):
            best_success = float(loaded["eval"].get("success_rate", best_success))
            torch.save(
                build_checkpoint(int(loaded.get("step", step_offset)), loaded["eval"]),
                out_dir / f"{args.arm}_best.pt",
            )
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "event": "init_checkpoint_loaded",
                    "path": args.init_checkpoint,
                    "checkpoint_step": int(loaded.get("step", -1)),
                    "best_success": best_success,
                    "allow_env_config_mismatch": bool(args.allow_init_env_config_mismatch),
                    "allow_arm_config_mismatch": bool(args.allow_init_arm_config_mismatch),
                    "compatible_weights_only": bool(args.init_compatible_weights_only),
                    "compatible_loaded_keys": loaded.get("_compatible_loaded_keys"),
                    "compatible_total_keys": loaded.get("_compatible_total_keys"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif args.resume_checkpoint:
        loaded = load_checkpoint(args.resume_checkpoint, resume_optimizer=True)
        step_offset = int(loaded.get("step", step_offset))
        best_success = float(
            loaded.get("best_success", loaded.get("eval", {}).get("success_rate", best_success))
        )
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "event": "resume_checkpoint_loaded",
                    "path": args.resume_checkpoint,
                    "step_offset": step_offset,
                    "best_success": best_success,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    start = time.time()

    for local_step in range(1, args.steps + 1):
        step = step_offset + local_step
        scenario = sample_scenarios(args.batch_size, env_cfg, device=device)
        use_dagger = step >= args.dagger_start and step % args.dagger_every == 0
        hidden_context = None
        if use_dagger:
            with torch.no_grad():
                state, hidden_context = model_induced_states(
                    policy,
                    scenario,
                    env_cfg,
                    edge_index,
                    rollout_steps=args.dagger_rollout_steps,
                )
        else:
            state = sample_training_states(scenario, env_cfg)
        obs = observations(state, scenario, env_cfg)
        target_forces, target_section, _ = expert_policy(state, scenario, env_cfg)
        output = policy(obs, edge_index, hidden_context)
        force_loss = F.mse_loss(
            output.forces / env_cfg.max_force, target_forces / env_cfg.max_force
        )
        loss = force_loss
        section_loss = torch.zeros((), device=device)
        safety_loss = torch.zeros((), device=device)
        rollout_total = torch.zeros((), device=device)
        rollout_stats = {
            "rollout_force_loss": torch.zeros((), device=device),
            "rollout_state_loss": torch.zeros((), device=device),
            "rollout_target_loss": torch.zeros((), device=device),
            "rollout_progress_loss": torch.zeros((), device=device),
            "rollout_safety_loss": torch.zeros((), device=device),
            "rollout_goal_capture_loss": torch.zeros((), device=device),
            "rollout_section_loss": torch.zeros((), device=device),
        }
        if "section" in output.aux:
            section_loss = section_loss_fn(output.aux["section"], target_section)
            loss = loss + args.section_loss_weight * section_loss
        if args.safety_loss_weight > 0.0:
            predicted_next = step_dynamics(state, output.forces, env_cfg)
            safety_loss = smooth_safety_loss(predicted_next, scenario, env_cfg)
            loss = loss + args.safety_loss_weight * safety_loss
        if args.rollout_loss_weight > 0.0 and step >= args.rollout_start:
            rollout_batch = (
                args.rollout_batch_size if args.rollout_batch_size > 0 else args.batch_size
            )
            rollout_scenario = sample_scenarios(rollout_batch, env_cfg, device=device)
            rollout_total, rollout_stats = differentiable_rollout_loss(
                policy,
                rollout_scenario,
                env_cfg,
                edge_index,
                horizon=args.rollout_horizon,
                initial_fraction=args.rollout_initial_fraction,
                force_weight=args.rollout_force_weight,
                state_weight=args.rollout_state_weight,
                target_weight=args.rollout_target_weight,
                progress_weight=args.rollout_progress_weight,
                safety_weight=args.rollout_safety_weight,
                goal_capture_weight=args.rollout_goal_capture_weight,
                section_weight=args.section_loss_weight,
                source_margin_weight=args.rollout_source_margin_weight,
                source_margin=args.rollout_source_margin,
                clearance_margin_weight=args.rollout_clearance_margin_weight,
                clearance_margin=args.rollout_clearance_margin,
            )
            loss = loss + args.rollout_loss_weight * rollout_total
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "arm": args.arm,
                "loss": float(loss.detach().cpu().item()),
                "force_loss": float(force_loss.detach().cpu().item()),
                "section_loss": float(section_loss.detach().cpu().item()),
                "safety_loss": float(safety_loss.detach().cpu().item()),
                "rollout_total_loss": float(rollout_total.detach().cpu().item()),
                **{key: float(value.detach().cpu().item()) for key, value in rollout_stats.items()},
                "dagger": bool(use_dagger),
                "elapsed_sec": time.time() - start,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        if step % args.eval_every == 0 or local_step == args.steps:
            policy.eval()
            eval_row = evaluate_policy(
                policy,
                env_cfg,
                episodes=args.eval_episodes,
                batch_size=args.eval_batch_size,
                seed=args.seed + 100000 + step,
                device=device,
            )
            policy.train()
            eval_row.update({"step": step, "arm": args.arm, "elapsed_sec": time.time() - start})
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_row, sort_keys=True) + "\n")
            print(json.dumps(eval_row, sort_keys=True), flush=True)
            torch.save(build_checkpoint(step, eval_row), out_dir / f"{args.arm}_last.pt")
            if float(eval_row["success_rate"]) >= best_success:
                best_success = float(eval_row["success_rate"])
                torch.save(build_checkpoint(step, eval_row), out_dir / f"{args.arm}_best.pt")

    return {"arm": args.arm, "best_success": best_success, "out_dir": str(out_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=["sheaf_gbp", "raw_full", "comm_matched", "no_comm"], required=True
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="runs/dev")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--max-force", type=float, default=4.0)
    parser.add_argument("--world-x", type=float, default=3.5)
    parser.add_argument("--world-y", type=float, default=1.75)
    parser.add_argument("--success-pos-tol", type=float, default=0.22)
    parser.add_argument("--success-angle-tol", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--comm-rounds", type=int, default=3)
    parser.add_argument("--gbp-steps", type=int, default=3)
    parser.add_argument("--comm-radius", type=float, default=1.18)
    parser.add_argument("--scenario-difficulty", type=float, default=1.0)
    parser.add_argument("--gap-y-range", type=float, default=0.58)
    parser.add_argument("--gap-half-base", type=float, default=0.83)
    parser.add_argument("--gap-half-difficulty-scale", type=float, default=0.05)
    parser.add_argument("--gap-half-jitter", type=float, default=0.08)
    parser.add_argument("--wall-x-abs", type=float, default=0.82)
    parser.add_argument("--source-x", type=float, default=-2.35)
    parser.add_argument("--goal-x", type=float, default=2.35)
    parser.add_argument("--local-observation", action="store_true")
    parser.add_argument("--strict-local-observation", action="store_true")
    parser.add_argument("--wall-sensor-range", type=float, default=1.05)
    parser.add_argument("--edge-sigma2", type=float, default=0.18)
    parser.add_argument("--edge-dim", type=int, default=8)
    parser.add_argument("--mean-only-messages", action="store_true")
    parser.add_argument("--restriction-residual-scale", type=float, default=0.05)
    parser.add_argument("--edge-conditioned-restrictions", action="store_true")
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--temporal-precision", type=float, default=0.35)
    parser.add_argument("--force-residual-scale", type=float, default=0.60)
    parser.add_argument("--analytic-force-scale", type=float, default=1.0)
    parser.add_argument("--decoder-extra-context", action="store_true")
    parser.add_argument("--decoder-section-only", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--section-loss-weight", type=float, default=0.15)
    parser.add_argument("--safety-loss-weight", type=float, default=0.25)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--rollout-start", type=int, default=1)
    parser.add_argument("--rollout-horizon", type=int, default=12)
    parser.add_argument("--rollout-batch-size", type=int, default=384)
    parser.add_argument("--rollout-initial-fraction", type=float, default=0.50)
    parser.add_argument("--rollout-force-weight", type=float, default=1.0)
    parser.add_argument("--rollout-state-weight", type=float, default=5.0)
    parser.add_argument("--rollout-target-weight", type=float, default=0.05)
    parser.add_argument("--rollout-progress-weight", type=float, default=2.0)
    parser.add_argument("--rollout-safety-weight", type=float, default=5.0)
    parser.add_argument("--rollout-source-margin-weight", type=float, default=0.0)
    parser.add_argument("--rollout-source-margin", type=float, default=0.012)
    parser.add_argument("--rollout-clearance-margin-weight", type=float, default=0.0)
    parser.add_argument("--rollout-clearance-margin", type=float, default=0.002)
    parser.add_argument("--rollout-goal-capture-weight", type=float, default=1.0)
    parser.add_argument("--dagger-start", type=int, default=350)
    parser.add_argument("--dagger-every", type=int, default=2)
    parser.add_argument("--dagger-rollout-steps", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--allow-init-env-config-mismatch", action="store_true")
    parser.add_argument("--allow-init-arm-config-mismatch", action="store_true")
    parser.add_argument("--init-compatible-weights-only", action="store_true")
    parser.add_argument("--step-offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
