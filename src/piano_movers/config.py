"""Configuration objects for PianoMovers-Force."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    n_agents: int = 6
    dt: float = 0.08
    mass: float = 4.0
    inertia: float = 1.35
    linear_damping: float = 0.18
    angular_damping: float = 0.16
    max_force: float = 4.0
    world_x: float = 3.5
    world_y: float = 1.75
    wall_slab: float = 0.055
    success_pos_tol: float = 0.22
    success_angle_tol: float = 0.35
    max_steps: int = 90
    comm_radius: float = 1.18
    scenario_difficulty: float = 1.0
    gap_y_range: float = 0.58
    gap_half_base: float = 0.83
    gap_half_difficulty_scale: float = 0.05
    gap_half_jitter: float = 0.08
    wall_x_abs: float = 0.82
    source_x: float = -2.35
    goal_x: float = 2.35
    local_observation: bool = False
    strict_local_observation: bool = False
    wall_sensor_range: float = 1.05
    device: str = "cpu"


@dataclass(frozen=True)
class ArmConfig:
    obs_dim: int
    hidden_dim: int = 96
    stalk_dim: int = 32
    edge_dim: int = 8
    message_dim: int = 32
    comm_rounds: int = 3
    gbp_steps: int = 3
    temporal_window: int = 4
    temporal_precision: float = 0.35
    damping: float = 0.55
    restriction_rank: int = 4
    restriction_residual_scale: float = 0.05
    edge_sigma2: float = 0.18
    force_residual_scale: float = 0.60
    analytic_force_scale: float = 1.0
    mean_only_messages: bool = False
    decoder_extra_context: bool = False
    edge_conditioned_restrictions: bool = False
    decoder_section_only: bool = False
