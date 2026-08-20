"""Render a real PianoMovers-Force rollout as an MP4 video."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ArmConfig, EnvConfig
from .env import (
    PianoBatch,
    body_to_world,
    contact_points,
    desired_maneuver,
    observations,
    radius_edges,
    sample_scenarios,
    step_dynamics,
    success_mask,
    collision_mask,
)
from .metrics import communication_bytes, rollout
from .models import make_policy

BODY_RENDER_SCALE = 0.55
DRAW_SHARED_WRENCH_ARROW = False


def load_policy(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    env_cfg = EnvConfig(**checkpoint["env_cfg"])
    arm_cfg = ArmConfig(**checkpoint["arm_cfg"])
    policy = make_policy(checkpoint["arm"], env_cfg, arm_cfg).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return checkpoint["arm"], policy, env_cfg, checkpoint


def slice_scenario(scenario: PianoBatch, index: int) -> PianoBatch:
    return PianoBatch(
        state0=scenario.state0[index : index + 1].clone(),
        goal=scenario.goal[index : index + 1].clone(),
        wall_x=scenario.wall_x[index : index + 1].clone(),
        gap_y=scenario.gap_y[index : index + 1].clone(),
        gap_half=scenario.gap_half[index : index + 1].clone(),
        source=scenario.source[index : index + 1].clone(),
    )


def scenario_to_device(scenario: PianoBatch, device: torch.device | str) -> PianoBatch:
    return PianoBatch(
        state0=scenario.state0.to(device),
        goal=scenario.goal.to(device),
        wall_x=scenario.wall_x.to(device),
        gap_y=scenario.gap_y.to(device),
        gap_half=scenario.gap_half.to(device),
        source=scenario.source.to(device),
    )


@torch.no_grad()
def find_case(
    policy,
    cfg: EnvConfig,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_episodes: int,
    raw_policy=None,
    require_raw_fail: bool = False,
) -> tuple[PianoBatch, dict]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    done = 0
    edge_index = radius_edges(cfg, device=device)
    while done < max_episodes:
        current = min(batch_size, max_episodes - done)
        scenario = sample_scenarios(current, cfg, device=device, generator=generator)
        sheaf_result = rollout(policy, scenario, cfg)
        mask = sheaf_result["success"].clone()
        raw_result = None
        if raw_policy is not None and require_raw_fail:
            raw_result = rollout(raw_policy, scenario, cfg)
            mask &= ~raw_result["success"]
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() > 0:
            idx = int(indices[0].item())
            meta = {
                "search_seed": seed,
                "search_episode_index": done + idx,
                "sheaf_success": bool(sheaf_result["success"][idx].item()),
                "sheaf_collision": bool(sheaf_result["collision"][idx].item()),
                "sheaf_finish_step": int(sheaf_result["finish_step"][idx].item()),
                "wire_bytes_per_episode": communication_bytes(
                    policy, edge_index, batch_size=1, steps=cfg.max_steps
                ),
            }
            if raw_result is not None:
                meta.update(
                    {
                        "raw_success": bool(raw_result["success"][idx].item()),
                        "raw_collision": bool(raw_result["collision"][idx].item()),
                        "raw_finish_step": int(raw_result["finish_step"][idx].item()),
                        "raw_wire_bytes_per_episode": communication_bytes(
                            raw_policy, edge_index, batch_size=1, steps=cfg.max_steps
                        ),
                    }
                )
            return slice_scenario(scenario, idx), meta
        done += current
    raise RuntimeError(f"no matching successful case found in {max_episodes} episodes")


@torch.no_grad()
def record_rollout(policy, scenario: PianoBatch, cfg: EnvConfig) -> list[dict]:
    edge_index = radius_edges(cfg, device=scenario.state0.device)
    state = scenario.state0.clone()
    hidden = policy.init_hidden(1, scenario.state0.device)
    frames: list[dict] = []
    for step in range(cfg.max_steps):
        obs = observations(state, scenario, cfg)
        output = policy(obs, edge_index, hidden)
        next_state = step_dynamics(state, output.forces, cfg)
        hit = bool(collision_mask(next_state, scenario, cfg)[0].item())
        ok = bool(
            (success_mask(next_state, scenario, cfg) & ~collision_mask(next_state, scenario, cfg))[
                0
            ].item()
        )
        section = output.aux.get("section")
        frames.append(
            {
                "step": step,
                "state": state[0].detach().cpu(),
                "next_state": next_state[0].detach().cpu(),
                "forces": output.forces[0].detach().cpu(),
                "section": section[0].detach().cpu() if section is not None else None,
                "success": ok,
                "collision": hit,
            }
        )
        hidden = output.hidden
        state = next_state
        if hit or ok:
            break
    return frames


def visual_contact_points(n_agents: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Aesthetic ant/handle locations on the rendered T silhouette.

    The simulator's ``contact_points`` stay unchanged: they are the actual
    force-application points used for dynamics, observation, communication, and
    metrics.  These visual points are only for the movie.  They lie on or just
    outside the rendered T boundary so the agents read as ants holding the load
    instead of detached blue markers.
    """

    if n_agents == 4:
        points = [(-0.48, 0.29), (0.48, 0.29), (-0.13, -0.53), (0.13, -0.53)]
    elif n_agents == 6:
        points = [
            (-0.48, 0.29),
            (0.00, 0.42),
            (0.48, 0.29),
            (-0.14, -0.12),
            (0.14, -0.12),
            (0.00, -0.55),
        ]
    elif n_agents == 8:
        points = [
            (-0.48, 0.29),
            (-0.18, 0.42),
            (0.18, 0.42),
            (0.48, 0.29),
            (-0.14, 0.05),
            (0.14, 0.05),
            (-0.08, -0.55),
            (0.08, -0.55),
        ]
    elif n_agents == 10:
        points = [
            (-0.48, 0.29),
            (-0.30, 0.41),
            (0.00, 0.43),
            (0.30, 0.41),
            (0.48, 0.29),
            (-0.14, 0.09),
            (0.14, 0.09),
            (-0.14, -0.25),
            (0.14, -0.25),
            (0.00, -0.55),
        ]
    elif n_agents == 12:
        points = [
            (-0.48, 0.29),
            (-0.32, 0.41),
            (-0.11, 0.43),
            (0.11, 0.43),
            (0.32, 0.41),
            (0.48, 0.29),
            (-0.14, 0.09),
            (0.14, 0.09),
            (-0.14, -0.25),
            (0.14, -0.25),
            (-0.08, -0.55),
            (0.08, -0.55),
        ]
    else:
        # Fallback for future layouts: put handles in the rendered body's scale.
        points = (BODY_RENDER_SCALE * contact_points(n_agents, "cpu")).tolist()
    return torch.tensor(points, dtype=torch.float32, device=device)


def body_rectangles_world(state: torch.Tensor) -> list[torch.Tensor]:
    scale = BODY_RENDER_SCALE
    stem = scale * torch.tensor(
        [(-0.20, -0.95), (0.20, -0.95), (0.20, 0.55), (-0.20, 0.55)],
        dtype=torch.float32,
    )
    top = scale * torch.tensor(
        [(-0.85, 0.35), (0.85, 0.35), (0.85, 0.72), (-0.85, 0.72)],
        dtype=torch.float32,
    )
    theta = state[2]
    c = torch.cos(theta)
    s = torch.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    xy = state[0:2].view(1, 2)
    return [rect @ rot.T + xy for rect in (stem, top)]


def draw_ant(
    ax, xy: torch.Tensor, heading: float, *, scale: float = 1.0, alpha: float = 0.96
) -> None:
    from matplotlib.patches import Ellipse

    x = float(xy[0])
    y = float(xy[1])
    c = math.cos(heading)
    s = math.sin(heading)
    ux, uy = c, s
    vx, vy = -s, c
    angle = math.degrees(heading)
    body_color = "#3b2416"
    edge_color = "#120b07"
    blue_handle = "#4da3c7"
    length = 0.042 * scale
    width = 0.026 * scale
    centers = [
        (x - 0.95 * length * ux, y - 0.95 * length * uy, 1.25, 1.00),
        (x, y, 1.02, 0.86),
        (x + 0.90 * length * ux, y + 0.90 * length * uy, 0.82, 0.70),
    ]
    ax.add_patch(
        Ellipse(
            (x, y),
            width=0.105 * scale,
            height=0.070 * scale,
            angle=angle,
            facecolor=blue_handle,
            edgecolor="none",
            alpha=0.18,
            zorder=5.8,
        )
    )
    for cx, cy, lw, ww in centers:
        ax.add_patch(
            Ellipse(
                (cx, cy),
                width=length * lw,
                height=width * ww,
                angle=angle,
                facecolor=body_color,
                edgecolor=edge_color,
                linewidth=0.35 * scale,
                alpha=alpha,
                zorder=7,
            )
        )
    for offset in (-0.72, 0.0, 0.72):
        base_x = x + offset * length * ux
        base_y = y + offset * length * uy
        for side in (-1.0, 1.0):
            knee_x = base_x + side * 0.030 * scale * vx
            knee_y = base_y + side * 0.030 * scale * vy
            foot_x = knee_x - 0.018 * scale * ux
            foot_y = knee_y - 0.018 * scale * uy
            ax.plot(
                [base_x, knee_x, foot_x],
                [base_y, knee_y, foot_y],
                color=edge_color,
                linewidth=0.55 * scale,
                alpha=0.86 * alpha,
                solid_capstyle="round",
                zorder=6.6,
            )


def ant_headings(points: torch.Tensor, forces: torch.Tensor, state: torch.Tensor) -> list[float]:
    headings: list[float] = []
    theta = float(state[2])
    for point, force in zip(points, forces, strict=True):
        if float(torch.linalg.vector_norm(force).item()) > 1e-4:
            headings.append(float(math.atan2(float(force[1]), float(force[0]))))
        else:
            rel = point - state[0:2]
            headings.append(float(math.atan2(float(rel[1]), float(rel[0])) + theta))
    return headings


def draw_force_arrow(ax, point: torch.Tensor, force: torch.Tensor, *, scale: float) -> None:
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe

    if float(torch.linalg.vector_norm(force).item()) < 1e-5:
        return
    x = float(point[0])
    y = float(point[1])
    dx = float(scale * force[0])
    dy = float(scale * force[1])
    arrow = FancyArrowPatch(
        (x, y),
        (x + dx, y + dy),
        arrowstyle="-|>",
        mutation_scale=13.5,
        linewidth=2.0,
        color="#d23b2a",
        alpha=0.96,
        zorder=9.0,
        shrinkA=0.0,
        shrinkB=0.0,
        capstyle="round",
        joinstyle="round",
    )
    arrow.set_path_effects(
        [
            pe.Stroke(linewidth=4.0, foreground="#fffaf0", alpha=0.92),
            pe.Normal(),
        ]
    )
    ax.add_patch(arrow)


def render_video(
    frames: list[dict],
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    out_path: Path,
    title: str,
    fps: int,
    dpi: int,
    meta: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.patches import Polygon, Rectangle

    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_index = radius_edges(cfg, device="cpu")
    contacts = visual_contact_points(cfg.n_agents, "cpu")
    render_scenario = scenario_to_device(scenario, "cpu")
    source = render_scenario.source[0]
    goal = render_scenario.goal[0]
    wall_x = render_scenario.wall_x[0]
    gap_y = render_scenario.gap_y[0]
    gap_half = render_scenario.gap_half[0]
    centers = torch.stack([frame["state"][0:2] for frame in frames])
    view_points_x = torch.cat((centers[:, 0], source[0:1], goal[0:1], wall_x))
    view_x_min = max(-cfg.world_x - 0.08, float(view_points_x.min()) - 0.46)
    view_x_max = min(cfg.world_x + 0.08, float(view_points_x.max()) + 0.46)
    view_y_min = -cfg.world_y - 0.06
    view_y_max = cfg.world_y + 0.06

    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    fig.patch.set_facecolor("#fbfaf7")

    def draw_static() -> None:
        ax.set_xlim(view_x_min, view_x_max)
        ax.set_ylim(view_y_min, view_y_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#fbfaf7")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(
            Rectangle(
                (-cfg.world_x, -cfg.world_y),
                2 * cfg.world_x,
                2 * cfg.world_y,
                fill=False,
                edgecolor="#8a8177",
                linewidth=1.2,
                alpha=0.75,
            )
        )
        for wi in range(2):
            x = float(wall_x[wi])
            gy = float(gap_y[wi])
            gh = float(gap_half[wi])
            slab = cfg.wall_slab
            low_h = max(0.0, gy - gh + cfg.world_y)
            high_y = gy + gh
            high_h = max(0.0, cfg.world_y - high_y)
            ax.add_patch(
                Rectangle(
                    (x - slab, -cfg.world_y),
                    2 * slab,
                    low_h,
                    color="#4e4a45",
                    alpha=0.88,
                    linewidth=0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x - slab, high_y),
                    2 * slab,
                    high_h,
                    color="#4e4a45",
                    alpha=0.88,
                    linewidth=0,
                )
            )
            ax.plot([x - 0.12, x + 0.12], [gy - gh, gy - gh], color="#b3aaa0", lw=1.0)
            ax.plot([x - 0.12, x + 0.12], [gy + gh, gy + gh], color="#b3aaa0", lw=1.0)
        ax.scatter(
            [source[0]],
            [source[1]],
            marker="o",
            s=70,
            color="#7aa974",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.scatter(
            [goal[0]],
            [goal[1]],
            marker="*",
            s=165,
            color="#d55e00",
            edgecolor="white",
            linewidth=0.7,
        )

    def update(frame_index: int):
        ax.clear()
        draw_static()
        frame = frames[frame_index]
        state = frame["state"]
        state_batch = state.view(1, 6)
        contact_world = body_to_world(contacts, state_batch)[0].detach().cpu()
        target_pose, _ = desired_maneuver(state_batch, render_scenario)
        target = target_pose[0].detach().cpu()

        ax.plot(
            centers[: frame_index + 1, 0],
            centers[: frame_index + 1, 1],
            color="#2b6f9e",
            linewidth=2.4,
            alpha=0.76,
        )
        ax.scatter([target[0]], [target[1]], marker="x", s=72, color="#7b5ea7", linewidth=1.8)
        for rect in body_rectangles_world(state):
            ax.add_patch(
                Polygon(
                    rect.numpy(),
                    closed=True,
                    facecolor="#f1bf58",
                    edgecolor="#2f2a25",
                    linewidth=1.35,
                    alpha=0.95,
                )
            )

        for edge in edge_index.tolist():
            a, b = edge
            ax.plot(
                [contact_world[a, 0], contact_world[b, 0]],
                [contact_world[a, 1], contact_world[b, 1]],
                color="#4da3c7",
                linewidth=1.25,
                alpha=0.30,
            )

        forces = frame["forces"]
        headings = ant_headings(contact_world, forces, state)
        for point, heading in zip(contact_world, headings, strict=True):
            draw_ant(ax, point, heading, scale=1.15)

        section = frame["section"]
        section_text = "section unavailable"
        if section is not None:
            mean_section = section.mean(dim=0)
            sx, sy, tau = float(mean_section[3]), float(mean_section[4]), float(mean_section[5])
            if DRAW_SHARED_WRENCH_ARROW:
                ax.arrow(
                    float(state[0]),
                    float(state[1]),
                    0.45 * sx,
                    0.45 * sy,
                    head_width=0.060,
                    head_length=0.080,
                    color="#12805c",
                    alpha=0.82,
                    length_includes_head=True,
                )
            section_text = f"sheaf section: shared wrench=({sx:+.2f}, {sy:+.2f}, τ={tau:+.2f})"

        status = (
            "SUCCESS" if frame["success"] else ("COLLISION" if frame["collision"] else "running")
        )
        bytes_per_episode = meta.get("wire_bytes_per_episode", 0)
        ax.set_title(
            f"{title}\n"
            f"step {frame['step']:02d}/{cfg.max_steps} · {status} · "
            f"{cfg.n_agents} agents · {bytes_per_episode / 1024:.0f} KiB/episode",
            fontsize=13.5,
            color="#2b2b2b",
            pad=10,
        )
        ax.text(
            0.015,
            0.985,
            "ants = fixed force handles on the T payload · pale blue links = local communication graph\n"
            "text reports the inferred shared wrench\n"
            f"{section_text}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            color="#2d2a26",
            bbox=dict(
                boxstyle="round,pad=0.35", facecolor="#fffdf8", edgecolor="#d6cfc3", alpha=0.90
            ),
        )
        return []

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400, metadata={"title": title})
    animation.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


def render_comparison_video(
    sheaf_frames: list[dict],
    baseline_frames: list[dict],
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    out_path: Path,
    title: str,
    baseline_label: str,
    fps: int,
    dpi: int,
    meta: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.patches import Polygon, Rectangle

    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_index = radius_edges(cfg, device="cpu")
    contacts = visual_contact_points(cfg.n_agents, "cpu")
    render_scenario = scenario_to_device(scenario, "cpu")
    source = render_scenario.source[0]
    goal = render_scenario.goal[0]
    wall_x = render_scenario.wall_x[0]
    gap_y = render_scenario.gap_y[0]
    gap_half = render_scenario.gap_half[0]
    sheaf_centers = torch.stack([frame["state"][0:2] for frame in sheaf_frames])
    baseline_centers = torch.stack([frame["state"][0:2] for frame in baseline_frames])
    all_centers = torch.cat((sheaf_centers, baseline_centers), dim=0)
    view_points_x = torch.cat((all_centers[:, 0], source[0:1], goal[0:1], wall_x))
    view_x_min = max(-cfg.world_x - 0.08, float(view_points_x.min()) - 0.46)
    view_x_max = min(cfg.world_x + 0.08, float(view_points_x.max()) + 0.46)
    view_y_min = -cfg.world_y - 0.06
    view_y_max = cfg.world_y + 0.06

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.9), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf7")

    def draw_static(ax) -> None:
        ax.set_xlim(view_x_min, view_x_max)
        ax.set_ylim(view_y_min, view_y_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#fbfaf7")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(
            Rectangle(
                (-cfg.world_x, -cfg.world_y),
                2 * cfg.world_x,
                2 * cfg.world_y,
                fill=False,
                edgecolor="#8a8177",
                linewidth=1.1,
                alpha=0.75,
            )
        )
        for wi in range(2):
            x = float(wall_x[wi])
            gy = float(gap_y[wi])
            gh = float(gap_half[wi])
            slab = cfg.wall_slab
            low_h = max(0.0, gy - gh + cfg.world_y)
            high_y = gy + gh
            high_h = max(0.0, cfg.world_y - high_y)
            ax.add_patch(
                Rectangle(
                    (x - slab, -cfg.world_y),
                    2 * slab,
                    low_h,
                    color="#4e4a45",
                    alpha=0.88,
                    linewidth=0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x - slab, high_y),
                    2 * slab,
                    high_h,
                    color="#4e4a45",
                    alpha=0.88,
                    linewidth=0,
                )
            )
            ax.plot([x - 0.12, x + 0.12], [gy - gh, gy - gh], color="#b3aaa0", lw=0.95)
            ax.plot([x - 0.12, x + 0.12], [gy + gh, gy + gh], color="#b3aaa0", lw=0.95)
        ax.scatter(
            [source[0]],
            [source[1]],
            marker="o",
            s=58,
            color="#7aa974",
            edgecolor="white",
            linewidth=0.7,
        )
        ax.scatter(
            [goal[0]],
            [goal[1]],
            marker="*",
            s=138,
            color="#d55e00",
            edgecolor="white",
            linewidth=0.6,
        )

    def draw_panel(
        ax,
        frames: list[dict],
        centers: torch.Tensor,
        frame_index: int,
        panel_title: str,
        bytes_per_episode: float,
        path_color: str,
    ):
        idx = min(frame_index, len(frames) - 1)
        frame = frames[idx]
        state = frame["state"]
        state_batch = state.view(1, 6)
        contact_world = body_to_world(contacts, state_batch)[0].detach().cpu()
        target_pose, _ = desired_maneuver(state_batch, render_scenario)
        target = target_pose[0].detach().cpu()

        draw_static(ax)
        ax.plot(
            centers[: idx + 1, 0],
            centers[: idx + 1, 1],
            color=path_color,
            linewidth=2.35,
            alpha=0.78,
        )
        ax.scatter([target[0]], [target[1]], marker="x", s=62, color="#7b5ea7", linewidth=1.6)
        for rect in body_rectangles_world(state):
            ax.add_patch(
                Polygon(
                    rect.numpy(),
                    closed=True,
                    facecolor="#f1bf58",
                    edgecolor="#2f2a25",
                    linewidth=1.20,
                    alpha=0.95,
                )
            )
        for edge in edge_index.tolist():
            a, b = edge
            ax.plot(
                [contact_world[a, 0], contact_world[b, 0]],
                [contact_world[a, 1], contact_world[b, 1]],
                color="#4da3c7",
                linewidth=1.05,
                alpha=0.26,
            )

        headings = ant_headings(contact_world, frame["forces"], state)
        for point, heading in zip(contact_world, headings, strict=True):
            draw_ant(ax, point, heading, scale=1.02)

        section = frame["section"]
        section_text = "no learned sheaf section"
        if section is not None:
            mean_section = section.mean(dim=0)
            sx, sy, tau = float(mean_section[3]), float(mean_section[4]), float(mean_section[5])
            if DRAW_SHARED_WRENCH_ARROW:
                ax.arrow(
                    float(state[0]),
                    float(state[1]),
                    0.43 * sx,
                    0.43 * sy,
                    head_width=0.055,
                    head_length=0.075,
                    color="#12805c",
                    alpha=0.82,
                    length_includes_head=True,
                )
            section_text = f"shared wrench=({sx:+.2f}, {sy:+.2f}, τ={tau:+.2f})"

        status = (
            "SUCCESS" if frame["success"] else ("COLLISION" if frame["collision"] else "running")
        )
        ax.set_title(
            f"{panel_title}\n"
            f"step {frame['step']:02d}/{cfg.max_steps} · {status} · {bytes_per_episode / 1024:.0f} KiB/episode",
            fontsize=12.2,
            color="#2b2b2b",
            pad=8,
        )
        ax.text(
            0.015,
            0.985,
            section_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            color="#2d2a26",
            bbox=dict(
                boxstyle="round,pad=0.28", facecolor="#fffdf8", edgecolor="#d6cfc3", alpha=0.88
            ),
        )

    def update(frame_index: int):
        for ax in axes:
            ax.clear()
        draw_panel(
            axes[0],
            sheaf_frames,
            sheaf_centers,
            frame_index,
            "Sheaf-GBP",
            float(meta.get("wire_bytes_per_episode", 0.0)),
            "#3b6fb6",
        )
        draw_panel(
            axes[1],
            baseline_frames,
            baseline_centers,
            frame_index,
            baseline_label,
            float(meta.get("raw_wire_bytes_per_episode", 0.0)),
            "#cc6677",
        )
        fig.suptitle(
            f"{title}\n"
            "same initial state and map · ants are fixed force handles · pale blue links are radius-limited communication",
            fontsize=13.2,
            color="#2b2b2b",
        )
        return []

    frames_total = max(len(sheaf_frames), len(baseline_frames))
    animation = FuncAnimation(fig, update, frames=frames_total, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=3200, metadata={"title": title})
    animation.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-checkpoint", default="")
    parser.add_argument("--require-raw-fail", action="store_true")
    parser.add_argument("--comparison", action="store_true")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-search-episodes", type=int, default=2048)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--title", default="PianoMovers-Force: real Sheaf-GBP success")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    arm, policy, cfg, checkpoint = load_policy(Path(args.checkpoint), device)
    if arm != "sheaf_gbp":
        raise ValueError(f"--checkpoint must be sheaf_gbp, got {arm}")
    raw_policy = None
    raw_checkpoint_meta = None
    if args.raw_checkpoint:
        raw_arm, raw_policy, raw_cfg, raw_checkpoint = load_policy(
            Path(args.raw_checkpoint), device
        )
        if raw_cfg != cfg:
            raise ValueError("raw checkpoint EnvConfig must match sheaf checkpoint EnvConfig")
        raw_checkpoint_meta = {
            "arm": raw_arm,
            "path": args.raw_checkpoint,
            "checkpoint_step": raw_checkpoint.get("step"),
            "eval": raw_checkpoint.get("eval", {}),
        }
    scenario, meta = find_case(
        policy,
        cfg,
        device=device,
        seed=args.seed,
        batch_size=args.batch_size,
        max_episodes=args.max_search_episodes,
        raw_policy=raw_policy,
        require_raw_fail=args.require_raw_fail,
    )
    frames = record_rollout(policy, scenario, cfg)
    baseline_frames = (
        record_rollout(raw_policy, scenario, cfg)
        if args.comparison and raw_policy is not None
        else []
    )
    out_path = Path(args.out)
    meta.update(
        {
            "arm": arm,
            "checkpoint": args.checkpoint,
            "checkpoint_step": checkpoint.get("step"),
            "checkpoint_eval": checkpoint.get("eval", {}),
            "raw_checkpoint": raw_checkpoint_meta,
            "env_config": asdict(cfg),
            "rendered_steps": len(frames),
            "render_success": bool(frames[-1]["success"]),
            "render_collision": bool(frames[-1]["collision"]),
            "comparison": bool(args.comparison and raw_policy is not None),
            "baseline_label": args.baseline_label if raw_policy is not None else "",
            "baseline_rendered_steps": len(baseline_frames),
            "baseline_render_success": bool(baseline_frames[-1]["success"])
            if baseline_frames
            else None,
            "baseline_render_collision": bool(baseline_frames[-1]["collision"])
            if baseline_frames
            else None,
            "out": str(out_path),
        }
    )
    if args.comparison and raw_policy is not None:
        render_comparison_video(
            frames,
            baseline_frames,
            scenario,
            cfg,
            out_path=out_path,
            title=args.title,
            baseline_label=args.baseline_label,
            fps=args.fps,
            dpi=args.dpi,
            meta=meta,
        )
    else:
        render_video(
            frames,
            scenario,
            cfg,
            out_path=out_path,
            title=args.title,
            fps=args.fps,
            dpi=args.dpi,
            meta=meta,
        )
    out_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
