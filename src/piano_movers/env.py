"""Vectorized PianoMovers-Force rigid-body benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import EnvConfig


@dataclass
class PianoBatch:
    state0: Tensor
    goal: Tensor
    wall_x: Tensor
    gap_y: Tensor
    gap_half: Tensor
    source: Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def contact_points(n_agents: int, device: torch.device | str | None = None) -> Tensor:
    """Return fixed handle locations in payload body coordinates."""

    if n_agents == 4:
        points = [(-0.70, 0.48), (0.70, 0.48), (-0.28, -0.55), (0.28, -0.55)]
    elif n_agents == 6:
        points = [
            (-0.78, 0.48),
            (0.00, 0.58),
            (0.78, 0.48),
            (-0.32, -0.35),
            (0.32, -0.35),
            (0.00, -0.92),
        ]
    elif n_agents == 8:
        points = [
            (-0.78, 0.48),
            (-0.28, 0.60),
            (0.28, 0.60),
            (0.78, 0.48),
            (-0.34, 0.02),
            (0.34, 0.02),
            (-0.18, -0.86),
            (0.18, -0.86),
        ]
    elif n_agents == 10:
        points = [
            (-0.78, 0.48),
            (-0.46, 0.58),
            (0.00, 0.62),
            (0.46, 0.58),
            (0.78, 0.48),
            (-0.34, 0.08),
            (0.34, 0.08),
            (-0.24, -0.42),
            (0.24, -0.42),
            (0.00, -0.88),
        ]
    elif n_agents == 12:
        points = [
            (-0.80, 0.47),
            (-0.52, 0.58),
            (-0.18, 0.63),
            (0.18, 0.63),
            (0.52, 0.58),
            (0.80, 0.47),
            (-0.34, 0.12),
            (0.34, 0.12),
            (-0.28, -0.34),
            (0.28, -0.34),
            (-0.16, -0.86),
            (0.16, -0.86),
        ]
    else:
        raise ValueError("supported n_agents are 4, 6, 8, 10, and 12")
    return torch.tensor(points, dtype=torch.float32, device=device)


def t_shape_points(device: torch.device | str | None = None) -> Tensor:
    """Support points of a compound T body, used for slit collision checks."""

    hull_scale = 0.55
    # Stem box: x in [-0.20, 0.20], y in [-0.95, 0.55].
    stem = [(-0.20, -0.95), (0.20, -0.95), (-0.20, 0.55), (0.20, 0.55)]
    # Top box: x in [-0.85, 0.85], y in [0.35, 0.72].
    top = [(-0.85, 0.35), (0.85, 0.35), (-0.85, 0.72), (0.85, 0.72)]
    # A few internal edge samples make discrete wall crossing less brittle.
    samples = [(-0.55, 0.54), (0.55, 0.54), (0.0, -0.55), (0.0, 0.0)]
    return hull_scale * torch.tensor(stem + top + samples, dtype=torch.float32, device=device)


def t_shape_rectangles(device: torch.device | str | None = None) -> Tensor:
    """Two local rectangles whose union is the rendered/colliding T payload."""

    hull_scale = 0.55
    stem = [(-0.20, -0.95), (0.20, -0.95), (0.20, 0.55), (-0.20, 0.55)]
    top = [(-0.85, 0.35), (0.85, 0.35), (0.85, 0.72), (-0.85, 0.72)]
    return hull_scale * torch.tensor([stem, top], dtype=torch.float32, device=device)


def rotate(points: Tensor, theta: Tensor) -> Tensor:
    """Rotate `[P,2]` body points by batch angles `[B]`."""

    c = torch.cos(theta).unsqueeze(-1)
    s = torch.sin(theta).unsqueeze(-1)
    x = points[:, 0].view(1, -1)
    y = points[:, 1].view(1, -1)
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def body_to_world(points: Tensor, state: Tensor) -> Tensor:
    rel = rotate(points, state[:, 2])
    return rel + state[:, None, 0:2]


def body_rectangles_to_world(rectangles: Tensor, state: Tensor) -> Tensor:
    """Rotate local rectangles `[R,4,2]` by batch poses `[B,6]`."""

    batch = state.shape[0]
    flat = rectangles.reshape(-1, 2)
    world = body_to_world(flat, state)
    return world.reshape(batch, rectangles.shape[0], rectangles.shape[1], 2)


def _convex_rect_intersects(poly: Tensor, rect: Tensor) -> Tensor:
    """Vectorized SAT intersection for convex quads `[B,4,2]`."""

    axes = []
    for edge_id in (0, 1):
        edge = poly[:, edge_id + 1] - poly[:, edge_id]
        normal = torch.stack((-edge[:, 1], edge[:, 0]), dim=-1)
        axes.append(normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-8))
    x_axis = torch.zeros_like(axes[0])
    x_axis[:, 0] = 1.0
    y_axis = torch.zeros_like(axes[0])
    y_axis[:, 1] = 1.0
    axes.extend((x_axis, y_axis))
    stacked = torch.stack(axes, dim=1)
    poly_proj = torch.einsum("bpd,bad->bpa", poly, stacked)
    rect_proj = torch.einsum("bpd,bad->bpa", rect, stacked)
    separated = (poly_proj.amax(dim=1) < rect_proj.amin(dim=1)) | (
        rect_proj.amax(dim=1) < poly_proj.amin(dim=1)
    )
    return ~separated.any(dim=1)


def sample_scenarios(
    batch_size: int,
    cfg: EnvConfig,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    difficulty: float = 1.0,
) -> PianoBatch:
    """Sample paired procedural three-room tasks."""

    dev = torch.device(device)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, device=dev, generator=generator)

    effective_difficulty = float(difficulty) * float(cfg.scenario_difficulty)
    gap1 = (rand(batch_size) - 0.5) * cfg.gap_y_range
    gap2 = (rand(batch_size) - 0.5) * cfg.gap_y_range
    half = (
        cfg.gap_half_base
        - cfg.gap_half_difficulty_scale * effective_difficulty
        + cfg.gap_half_jitter * rand(batch_size, 2)
    )
    wall_x = (
        torch.tensor([-cfg.wall_x_abs, cfg.wall_x_abs], device=dev)
        .view(1, 2)
        .expand(batch_size, -1)
    )
    gap_y = torch.stack((gap1, gap2), dim=-1)
    source = torch.zeros(batch_size, 3, device=dev)
    source[:, 0] = cfg.source_x
    source[:, 1] = gap1 + 0.10 * (rand(batch_size) - 0.5)
    source[:, 2] = 0.0
    goal = torch.zeros(batch_size, 3, device=dev)
    goal[:, 0] = cfg.goal_x
    goal[:, 1] = gap2 + 0.05 * (rand(batch_size) - 0.5)
    goal[:, 2] = 0.0
    state0 = torch.zeros(batch_size, 6, device=dev)
    state0[:, :3] = source
    return PianoBatch(
        state0=state0,
        goal=goal,
        wall_x=wall_x,
        gap_y=gap_y,
        gap_half=half,
        source=source,
    )


def sample_training_states(
    scenario: PianoBatch,
    cfg: EnvConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample states along the task corridor for expert imitation."""

    batch = scenario.state0.shape[0]
    device = scenario.state0.device

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, device=device, generator=generator)

    state = torch.zeros(batch, 6, device=device)
    progress = rand(batch)
    state[:, 0] = scenario.source[:, 0] + progress * (scenario.goal[:, 0] - scenario.source[:, 0])
    interp_gap = torch.where(progress < 0.48, scenario.gap_y[:, 0], scenario.gap_y[:, 1])
    state[:, 1] = interp_gap + 0.52 * (rand(batch) - 0.5)
    theta1 = torch.full((batch,), 0.80, device=device)
    theta2 = torch.full((batch,), -0.80, device=device)
    nominal_theta = torch.where(
        state[:, 0] < scenario.wall_x[:, 0] + 0.20,
        theta1,
        torch.where(state[:, 0] < scenario.wall_x[:, 1] + 0.20, theta2, 0.0),
    )
    state[:, 2] = wrap_angle(nominal_theta + 0.65 * (rand(batch) - 0.5))
    state[:, 3:5] = 0.25 * torch.randn(batch, 2, device=device, generator=generator)
    state[:, 5] = 0.25 * torch.randn(batch, device=device, generator=generator)
    # Keep a slice of initial states in every batch.
    initial_mask = rand(batch) < 0.20
    state = torch.where(initial_mask[:, None], scenario.state0, state)
    return state


def desired_maneuver(state: Tensor, scenario: PianoBatch) -> tuple[Tensor, Tensor]:
    """Return target pose and normalized progress features for the expert."""

    x = state[:, 0]
    batch = x.shape[0]
    theta1 = torch.full((batch,), 0.95, device=state.device)
    theta2 = torch.full((batch,), -0.95, device=state.device)
    target = scenario.goal.clone()

    before_1 = x < scenario.wall_x[:, 0] - 0.22
    in_1 = (x >= scenario.wall_x[:, 0] - 0.22) & (x < scenario.wall_x[:, 0] + 0.36)
    before_2 = (x >= scenario.wall_x[:, 0] + 0.36) & (x < scenario.wall_x[:, 1] - 0.22)
    in_2 = (x >= scenario.wall_x[:, 1] - 0.22) & (x < scenario.wall_x[:, 1] + 0.36)

    target[:, 0] = torch.where(before_1, scenario.wall_x[:, 0] - 0.28, target[:, 0])
    target[:, 1] = torch.where(before_1, scenario.gap_y[:, 0], target[:, 1])
    target[:, 2] = torch.where(before_1, theta1, target[:, 2])

    target[:, 0] = torch.where(in_1, scenario.wall_x[:, 0] + 0.38, target[:, 0])
    target[:, 1] = torch.where(in_1, scenario.gap_y[:, 0], target[:, 1])
    target[:, 2] = torch.where(in_1, theta1, target[:, 2])

    target[:, 0] = torch.where(before_2, scenario.wall_x[:, 1] - 0.28, target[:, 0])
    target[:, 1] = torch.where(before_2, scenario.gap_y[:, 1], target[:, 1])
    target[:, 2] = torch.where(before_2, theta2, target[:, 2])

    target[:, 0] = torch.where(in_2, scenario.wall_x[:, 1] + 0.38, target[:, 0])
    target[:, 1] = torch.where(in_2, scenario.gap_y[:, 1], target[:, 1])
    target[:, 2] = torch.where(in_2, theta2, target[:, 2])

    denom = (scenario.goal[:, 0] - scenario.source[:, 0]).clamp_min(1e-4)
    progress = ((state[:, 0] - scenario.source[:, 0]) / denom).clamp(0.0, 1.0)
    active_gap = torch.where(state[:, 0] < 0.0, scenario.gap_y[:, 0], scenario.gap_y[:, 1])
    passage_offset = (state[:, 1] - active_gap).clamp(-1.0, 1.0)
    phase = torch.stack((passage_offset, progress), dim=-1)
    return target, phase


def expert_policy(
    state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute local expert forces, normalized shared section, and net wrench."""

    contacts = contact_points(cfg.n_agents, state.device)
    r_world = rotate(contacts, state[:, 2])
    target, phase = desired_maneuver(state, scenario)
    pos_error = target[:, 0:2] - state[:, 0:2]
    angle_error = wrap_angle(target[:, 2] - state[:, 2])
    desired_twist = torch.cat(
        (
            (1.55 * pos_error - 0.45 * state[:, 3:5]).clamp(-1.7, 1.7),
            (1.80 * angle_error - 0.42 * state[:, 5]).clamp(-1.8, 1.8).unsqueeze(-1),
        ),
        dim=-1,
    )
    force_xy = (4.4 * pos_error - 1.35 * state[:, 3:5]).clamp(
        -cfg.n_agents * cfg.max_force * 0.75,
        cfg.n_agents * cfg.max_force * 0.75,
    )
    tau = (3.2 * angle_error - 0.90 * state[:, 5]).clamp(
        -cfg.n_agents * cfg.max_force * 0.62,
        cfg.n_agents * cfg.max_force * 0.62,
    )
    wrench = torch.cat((force_xy, tau.unsqueeze(-1)), dim=-1)

    batch = state.shape[0]
    grasp = torch.zeros(batch, 3, 2 * cfg.n_agents, device=state.device)
    grasp[:, 0, 0::2] = 1.0
    grasp[:, 1, 1::2] = 1.0
    grasp[:, 2, 0::2] = -r_world[:, :, 1]
    grasp[:, 2, 1::2] = r_world[:, :, 0]
    gram = grasp @ grasp.transpose(-1, -2)
    eye = torch.eye(3, device=state.device).view(1, 3, 3)
    dual = torch.linalg.solve(gram + 0.08 * eye, wrench.unsqueeze(-1)).squeeze(-1)
    forces = (grasp.transpose(-1, -2) @ dual.unsqueeze(-1)).squeeze(-1)
    forces = forces.view(batch, cfg.n_agents, 2)
    norms = torch.linalg.vector_norm(forces, dim=-1, keepdim=True).clamp_min(1e-6)
    forces = forces * (cfg.max_force / norms).clamp_max(1.0)

    wrench_scale = cfg.n_agents * cfg.max_force
    section = torch.cat(
        (
            desired_twist[:, 0:2] / 1.7,
            desired_twist[:, 2:3] / 1.8,
            wrench[:, 0:2] / wrench_scale,
            wrench[:, 2:3] / (0.75 * wrench_scale),
            phase,
        ),
        dim=-1,
    )
    return forces, section, wrench


def observations(state: Tensor, scenario: PianoBatch, cfg: EnvConfig) -> Tensor:
    """Local per-handle observations under the Force abstraction."""

    contacts = contact_points(cfg.n_agents, state.device)
    contact_world = body_to_world(contacts, state)
    r_world = contact_world - state[:, None, 0:2]
    contact_velocity = torch.stack(
        (
            state[:, None, 3] - state[:, None, 5] * r_world[:, :, 1],
            state[:, None, 4] + state[:, None, 5] * r_world[:, :, 0],
        ),
        dim=-1,
    )
    batch = state.shape[0]
    n = cfg.n_agents
    target, phase = desired_maneuver(state, scenario)
    goal_delta = scenario.goal[:, 0:2] - state[:, 0:2]
    source_delta = state[:, 0:2] - scenario.source[:, 0:2]
    angle_goal = wrap_angle(scenario.goal[:, 2] - state[:, 2])
    angle_target = wrap_angle(target[:, 2] - state[:, 2])

    if cfg.strict_local_observation:
        # Strict local sensing intentionally removes the repeated global
        # waypoint/progress/gap features used by earlier experiments.  Each
        # agent sees only its handle, its handle velocity, the final goal from
        # its own contact point, and wall/slit geometry within sensor range.
        repeated = torch.zeros(batch, n, 0, device=state.device)
        visibility = (
            (contact_world[:, :, 0:1] - scenario.wall_x[:, None, :]).abs() <= cfg.wall_sensor_range
        ).float()
        wall_rel_x = (
            scenario.wall_x[:, None, :] - contact_world[:, :, 0:1]
        ) / cfg.wall_sensor_range
        gap_rel_y = (scenario.gap_y[:, None, :] - contact_world[:, :, 1:2]) / cfg.world_y
        gap_half = scenario.gap_half[:, None, :] / cfg.world_y
        local_wall = torch.cat(
            (
                visibility * wall_rel_x,
                visibility * gap_rel_y,
                visibility * gap_half,
                visibility,
            ),
            dim=-1,
        )
        goal_delta_contact = (scenario.goal[:, None, 0:2] - contact_world) / 2.6
        strict_local = torch.cat(
            (
                torch.sin(state[:, 2:3]).unsqueeze(1).expand(-1, n, -1),
                torch.cos(state[:, 2:3]).unsqueeze(1).expand(-1, n, -1),
                contacts.view(1, n, 2).expand(batch, -1, -1),
                contact_world[:, :, 0:1] / 2.6,
                contact_world[:, :, 1:2] / cfg.world_y,
                contact_velocity / 2.0,
                goal_delta_contact,
            ),
            dim=-1,
        )
        return torch.cat((strict_local, local_wall), dim=-1)

    if cfg.local_observation:
        progress = (
            (state[:, 0:1] - scenario.source[:, 0:1])
            / (scenario.goal[:, 0:1] - scenario.source[:, 0:1]).clamp_min(1e-4)
        ).clamp(0.0, 1.0)
        global_features = torch.cat(
            (
                state[:, 0:2] / 2.6,
                torch.sin(state[:, 2:3]),
                torch.cos(state[:, 2:3]),
                state[:, 3:5] / 2.0,
                state[:, 5:6] / 2.0,
                goal_delta / 2.6,
                torch.sin(angle_goal).unsqueeze(-1),
                torch.cos(angle_goal).unsqueeze(-1),
                source_delta / 2.6,
                progress,
            ),
            dim=-1,
        )
        repeated = global_features.unsqueeze(1).expand(-1, n, -1)
        visibility = (
            (contact_world[:, :, 0:1] - scenario.wall_x[:, None, :]).abs() <= cfg.wall_sensor_range
        ).float()
        wall_rel_x = (
            scenario.wall_x[:, None, :] - contact_world[:, :, 0:1]
        ) / cfg.wall_sensor_range
        gap_rel_y = (scenario.gap_y[:, None, :] - contact_world[:, :, 1:2]) / cfg.world_y
        gap_half = scenario.gap_half[:, None, :] / cfg.world_y
        local_wall = torch.cat(
            (
                visibility * wall_rel_x,
                visibility * gap_rel_y,
                visibility * gap_half,
                visibility,
            ),
            dim=-1,
        )
    else:
        global_features = torch.cat(
            (
                state[:, 0:2] / 2.6,
                torch.sin(state[:, 2:3]),
                torch.cos(state[:, 2:3]),
                state[:, 3:5] / 2.0,
                state[:, 5:6] / 2.0,
                goal_delta / 2.6,
                torch.sin(angle_goal).unsqueeze(-1),
                torch.cos(angle_goal).unsqueeze(-1),
                source_delta / 2.6,
                (target[:, 0:2] - state[:, 0:2]) / 2.6,
                torch.sin(angle_target).unsqueeze(-1),
                torch.cos(angle_target).unsqueeze(-1),
                phase,
            ),
            dim=-1,
        )
        wall_features = torch.cat(
            (
                (scenario.wall_x - state[:, 0:1]) / 2.6,
                (scenario.gap_y - state[:, 1:2]) / cfg.world_y,
                scenario.gap_half,
            ),
            dim=-1,
        )
        repeated = (
            torch.cat((global_features, wall_features), dim=-1).unsqueeze(1).expand(-1, n, -1)
        )
        local_wall = torch.zeros(batch, n, 0, device=state.device)
    local = torch.cat(
        (
            contacts.view(1, n, 2).expand(batch, -1, -1),
            r_world / 1.2,
            contact_world[:, :, 0:1] / 2.6,
            contact_world[:, :, 1:2] / cfg.world_y,
        ),
        dim=-1,
    )
    return torch.cat((repeated, local_wall, local), dim=-1)


def radius_edges(cfg: EnvConfig, *, device: torch.device | str) -> Tensor:
    contacts = contact_points(cfg.n_agents, device)
    distances = torch.cdist(contacts, contacts)
    rows, cols = torch.where((distances <= cfg.comm_radius) & (distances > 0))
    undirected = rows < cols
    edges = torch.stack((rows[undirected], cols[undirected]), dim=-1)
    if edges.numel() == 0:
        raise ValueError("communication radius produced an empty graph")
    return edges.long()


def step_dynamics(state: Tensor, forces: Tensor, cfg: EnvConfig) -> Tensor:
    contacts = contact_points(cfg.n_agents, state.device)
    r_world = rotate(contacts, state[:, 2])
    net_force = forces.sum(dim=1)
    torque = (r_world[:, :, 0] * forces[:, :, 1] - r_world[:, :, 1] * forces[:, :, 0]).sum(dim=1)
    next_state = state.clone()
    next_state[:, 3:5] = (next_state[:, 3:5] + cfg.dt * net_force / cfg.mass) * (
        1.0 - cfg.linear_damping * cfg.dt
    )
    next_state[:, 5] = (next_state[:, 5] + cfg.dt * torque / cfg.inertia) * (
        1.0 - cfg.angular_damping * cfg.dt
    )
    next_state[:, 0:2] = next_state[:, 0:2] + cfg.dt * next_state[:, 3:5]
    next_state[:, 2] = wrap_angle(next_state[:, 2] + cfg.dt * next_state[:, 5])
    return next_state


def collision_mask(state: Tensor, scenario: PianoBatch, cfg: EnvConfig) -> Tensor:
    body_rects = body_rectangles_to_world(t_shape_rectangles(state.device), state)
    batch = state.shape[0]
    wall_hit = torch.zeros(batch, dtype=torch.bool, device=state.device)
    for body_id in range(body_rects.shape[1]):
        body = body_rects[:, body_id]
        for wall_id in range(2):
            wall_x = scenario.wall_x[:, wall_id]
            gap_y = scenario.gap_y[:, wall_id]
            gap_half = (scenario.gap_half[:, wall_id] - 0.03).clamp_min(0.0)
            x0 = wall_x - cfg.wall_slab
            x1 = wall_x + cfg.wall_slab
            lower_y1 = gap_y - gap_half
            upper_y0 = gap_y + gap_half
            lower = torch.stack(
                (
                    torch.stack((x0, torch.full_like(x0, -cfg.world_y)), dim=-1),
                    torch.stack((x1, torch.full_like(x0, -cfg.world_y)), dim=-1),
                    torch.stack((x1, lower_y1), dim=-1),
                    torch.stack((x0, lower_y1), dim=-1),
                ),
                dim=1,
            )
            upper = torch.stack(
                (
                    torch.stack((x0, upper_y0), dim=-1),
                    torch.stack((x1, upper_y0), dim=-1),
                    torch.stack((x1, torch.full_like(x0, cfg.world_y)), dim=-1),
                    torch.stack((x0, torch.full_like(x0, cfg.world_y)), dim=-1),
                ),
                dim=1,
            )
            has_lower = lower_y1 > -cfg.world_y
            has_upper = upper_y0 < cfg.world_y
            wall_hit |= has_lower & _convex_rect_intersects(body, lower)
            wall_hit |= has_upper & _convex_rect_intersects(body, upper)
    vertices = body_rects.reshape(batch, -1, 2)
    y_hit = vertices[:, :, 1].abs().amax(dim=1) > cfg.world_y
    x_hit = vertices[:, :, 0].abs().amax(dim=1) > cfg.world_x
    return wall_hit | y_hit | x_hit


def success_mask(state: Tensor, scenario: PianoBatch, cfg: EnvConfig) -> Tensor:
    pos_ok = torch.linalg.vector_norm(state[:, 0:2] - scenario.goal[:, 0:2], dim=-1) < (
        cfg.success_pos_tol
    )
    angle_ok = wrap_angle(state[:, 2] - scenario.goal[:, 2]).abs() < cfg.success_angle_tol
    return pos_ok & angle_ok
