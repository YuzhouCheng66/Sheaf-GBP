from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from piano_movers.config import ArmConfig, EnvConfig
from piano_movers.compare import paired_eval
from piano_movers.env import (
    contact_points,
    expert_policy,
    observations,
    radius_edges,
    sample_scenarios,
    sample_training_states,
)
from piano_movers.metrics import evaluate_policy
from piano_movers.models import make_policy


def test_env_and_arms_smoke_cpu():
    cfg = EnvConfig(n_agents=6, max_steps=8)
    scenario = sample_scenarios(4, cfg, device="cpu")
    state = sample_training_states(scenario, cfg)
    obs = observations(state, scenario, cfg)
    forces, section, wrench = expert_policy(state, scenario, cfg)
    assert obs.shape[:2] == (4, 6)
    assert forces.shape == (4, 6, 2)
    assert section.shape == (4, 8)
    assert wrench.shape == (4, 3)
    edge_index = radius_edges(cfg, device="cpu")
    arm_cfg = ArmConfig(obs_dim=obs.shape[-1], hidden_dim=32, comm_rounds=1, gbp_steps=1)
    for arm in ("no_comm", "raw_full", "comm_matched", "sheaf_gbp"):
        policy = make_policy(arm, cfg, arm_cfg)
        out = policy(obs, edge_index)
        assert out.forces.shape == (4, 6, 2)
        assert torch.isfinite(out.forces).all()


def test_strict_local_observation_smoke_cpu():
    loose_cfg = EnvConfig(n_agents=6, max_steps=3, local_observation=True)
    strict_cfg = EnvConfig(
        n_agents=6, max_steps=3, local_observation=True, strict_local_observation=True
    )
    scenario = sample_scenarios(2, strict_cfg, device="cpu")
    loose_obs = observations(scenario.state0, scenario, loose_cfg)
    strict_obs = observations(scenario.state0, scenario, strict_cfg)
    assert strict_obs.shape[:2] == (2, 6)
    assert strict_obs.shape[-1] < loose_obs.shape[-1]
    edge_index = radius_edges(strict_cfg, device="cpu")
    arm_cfg = ArmConfig(obs_dim=strict_obs.shape[-1], hidden_dim=24, comm_rounds=1, gbp_steps=1)
    policy = make_policy("sheaf_gbp", strict_cfg, arm_cfg)
    out = policy(strict_obs, edge_index)
    assert out.forces.shape == (2, 6, 2)
    assert torch.isfinite(out.forces).all()


def test_more_agent_layouts_smoke_cpu():
    for n_agents in (10, 12):
        cfg = EnvConfig(
            n_agents=n_agents,
            max_steps=3,
            max_force=1.6,
            comm_radius=0.72,
            local_observation=True,
            strict_local_observation=True,
        )
        contacts = contact_points(n_agents, "cpu")
        assert contacts.shape == (n_agents, 2)
        scenario = sample_scenarios(2, cfg, device="cpu")
        obs = observations(scenario.state0, scenario, cfg)
        edge_index = radius_edges(cfg, device="cpu")
        forces, section, wrench = expert_policy(scenario.state0, scenario, cfg)
        arm_cfg = ArmConfig(
            obs_dim=obs.shape[-1],
            hidden_dim=24,
            edge_dim=8,
            gbp_steps=1,
            temporal_window=2,
            decoder_extra_context=True,
            edge_conditioned_restrictions=True,
        )
        policy = make_policy("sheaf_gbp", cfg, arm_cfg)
        out = policy(obs, edge_index)
        assert forces.shape == (2, n_agents, 2)
        assert section.shape == (2, 8)
        assert wrench.shape == (2, 3)
        assert out.forces.shape == (2, n_agents, 2)
        assert torch.isfinite(out.forces).all()


def test_short_eval_cpu():
    cfg = EnvConfig(n_agents=4, max_steps=3)
    scenario = sample_scenarios(2, cfg, device="cpu")
    obs = observations(scenario.state0, scenario, cfg)
    arm_cfg = ArmConfig(obs_dim=obs.shape[-1], hidden_dim=24, comm_rounds=1, gbp_steps=1)
    policy = make_policy("sheaf_gbp", cfg, arm_cfg)
    result = evaluate_policy(policy, cfg, episodes=4, batch_size=2, seed=11, device="cpu")
    assert result["episodes"] == 4
    assert result["wire_bytes_per_episode"] > 0


def test_mean_only_sheaf_uses_comm_matched_float_budget_cpu():
    cfg = EnvConfig(n_agents=6, max_steps=3, local_observation=True)
    scenario = sample_scenarios(2, cfg, device="cpu")
    obs = observations(scenario.state0, scenario, cfg)
    edge_index = radius_edges(cfg, device="cpu")
    arm_cfg = ArmConfig(
        obs_dim=obs.shape[-1],
        hidden_dim=24,
        edge_dim=16,
        gbp_steps=1,
        mean_only_messages=True,
    )
    policy = make_policy("sheaf_gbp", cfg, arm_cfg)
    out = policy(obs, edge_index)
    assert out.forces.shape == (2, 6, 2)
    assert out.aux["section"].shape[-1] == 16
    assert policy.communication_floats_per_directed_edge_round() == 16


def test_sheaf_decoder_extra_context_does_not_change_comm_budget_cpu():
    cfg = EnvConfig(n_agents=6, max_steps=3, local_observation=True)
    scenario = sample_scenarios(2, cfg, device="cpu")
    obs = observations(scenario.state0, scenario, cfg)
    edge_index = radius_edges(cfg, device="cpu")
    arm_cfg = ArmConfig(
        obs_dim=obs.shape[-1],
        hidden_dim=24,
        edge_dim=8,
        gbp_steps=1,
        decoder_extra_context=True,
    )
    policy = make_policy("sheaf_gbp", cfg, arm_cfg)
    out = policy(obs, edge_index)
    assert out.forces.shape == (2, 6, 2)
    assert out.aux["analytic_force"].shape == (2, 6, 2)
    assert policy.communication_floats_per_directed_edge_round() == 16


def test_edge_conditioned_restrictions_do_not_change_comm_budget_cpu():
    cfg = EnvConfig(n_agents=6, max_steps=3, local_observation=True)
    scenario = sample_scenarios(2, cfg, device="cpu")
    obs = observations(scenario.state0, scenario, cfg)
    edge_index = radius_edges(cfg, device="cpu")
    arm_cfg = ArmConfig(
        obs_dim=obs.shape[-1],
        hidden_dim=24,
        edge_dim=8,
        gbp_steps=1,
        temporal_window=0,
        edge_conditioned_restrictions=True,
    )
    policy = make_policy("sheaf_gbp", cfg, arm_cfg)
    out = policy(obs, edge_index)
    assert out.forces.shape == (2, 6, 2)
    assert out.aux["section"].shape[-1] == 8
    assert policy.communication_floats_per_directed_edge_round() == 16


def test_section_only_decoder_keeps_sheaf_comm_budget_cpu():
    cfg = EnvConfig(n_agents=6, max_steps=3, local_observation=True)
    scenario = sample_scenarios(2, cfg, device="cpu")
    obs = observations(scenario.state0, scenario, cfg)
    edge_index = radius_edges(cfg, device="cpu")
    arm_cfg = ArmConfig(
        obs_dim=obs.shape[-1],
        hidden_dim=24,
        edge_dim=8,
        gbp_steps=1,
        temporal_window=0,
        decoder_section_only=True,
        decoder_extra_context=True,
    )
    policy = make_policy("sheaf_gbp", cfg, arm_cfg)
    out = policy(obs, edge_index)
    assert out.forces.shape == (2, 6, 2)
    assert out.aux["section"].shape[-1] == 8
    assert policy.communication_floats_per_directed_edge_round() == 16


def test_paired_eval_allows_agent_force_ood_override_cpu(tmp_path):
    cfg = EnvConfig(n_agents=6, max_steps=2, local_observation=True)
    scenario = sample_scenarios(2, cfg, device="cpu")
    obs = observations(scenario.state0, scenario, cfg)
    arm_cfg = ArmConfig(obs_dim=obs.shape[-1], hidden_dim=24, comm_rounds=1, gbp_steps=1)
    checkpoints = []
    for arm in ("sheaf_gbp", "raw_full"):
        policy = make_policy(arm, cfg, arm_cfg)
        path = tmp_path / f"{arm}.pt"
        torch.save(
            {
                "arm": arm,
                "env_cfg": asdict(cfg),
                "arm_cfg": asdict(arm_cfg),
                "state_dict": policy.state_dict(),
                "step": 0,
            },
            path,
        )
        checkpoints.append(str(path))

    args = argparse.Namespace(
        checkpoints=checkpoints,
        device="cpu",
        episodes=2,
        batch_size=2,
        seed=23,
        out="",
        eval_n_agents=8,
        eval_max_force=2.5,
        eval_world_x=None,
        eval_world_y=1.55,
        eval_comm_radius=None,
        eval_scenario_difficulty=None,
        eval_gap_y_range=None,
        eval_gap_half_base=None,
        eval_gap_half_difficulty_scale=None,
        eval_gap_half_jitter=None,
        eval_wall_x_abs=None,
        eval_source_x=None,
        eval_goal_x=None,
        eval_wall_sensor_range=None,
        eval_local_observation=None,
        eval_strict_local_observation=None,
    )
    result = paired_eval(args)
    assert result["env_config"]["n_agents"] == 8
    assert result["env_config"]["max_force"] == 2.5
    assert result["env_config"]["world_y"] == 1.55
    assert set(result["arms"]) == {"sheaf_gbp", "raw_full"}
