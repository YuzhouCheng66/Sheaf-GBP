"""Policy arms for PianoMovers-Force."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ArmConfig, EnvConfig
from .env import contact_points
from .gbp import DiagSheafGBP, homogeneous_topology


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        modules.extend((nn.Linear(last, hidden_dim), nn.GELU()))
        last = hidden_dim
    modules.append(nn.Linear(last, out_dim))
    return nn.Sequential(*modules)


@dataclass
class PolicyOutput:
    forces: Tensor
    hidden: Tensor
    aux: dict[str, Tensor]


class BasePolicy(nn.Module):
    arm_name = "base"

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.arm_cfg = arm_cfg
        self.obs_encoder = mlp(arm_cfg.obs_dim, arm_cfg.hidden_dim, arm_cfg.hidden_dim)
        self.gru = nn.GRUCell(arm_cfg.hidden_dim, arm_cfg.hidden_dim)

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(
            batch_size,
            self.env_cfg.n_agents,
            self.arm_cfg.hidden_dim,
            device=device,
        )

    def recurrent_encode(self, obs: Tensor, hidden: Tensor | None) -> Tensor:
        batch, n_agents, _ = obs.shape
        encoded = self.obs_encoder(obs)
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)
        flat = self.gru(
            encoded.reshape(batch * n_agents, -1),
            hidden.reshape(batch * n_agents, -1),
        )
        return flat.view(batch, n_agents, -1)

    def communication_floats_per_directed_edge_round(self) -> int:
        return 0

    def communication_rounds(self) -> int:
        return 0


class NoCommPolicy(BasePolicy):
    arm_name = "no_comm"

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig):
        super().__init__(env_cfg, arm_cfg)
        self.decoder = mlp(arm_cfg.hidden_dim + arm_cfg.obs_dim, arm_cfg.hidden_dim, 2)

    def forward(
        self, obs: Tensor, edge_index: Tensor, hidden: Tensor | None = None
    ) -> PolicyOutput:
        del edge_index
        next_hidden = self.recurrent_encode(obs, hidden)
        raw_force = self.decoder(torch.cat((obs, next_hidden), dim=-1))
        forces = self.env_cfg.max_force * torch.tanh(raw_force)
        return PolicyOutput(forces=forces, hidden=next_hidden, aux={})


class MPNNPolicy(BasePolicy):
    """Recurrent graph baseline with explicit float-message accounting."""

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig, *, message_dim: int, name: str):
        super().__init__(env_cfg, arm_cfg)
        self.message_dim = message_dim
        self.arm_name = name
        self.send = nn.Linear(arm_cfg.hidden_dim, message_dim)
        self.recv = mlp(message_dim, arm_cfg.hidden_dim, arm_cfg.hidden_dim, layers=1)
        self.update = mlp(2 * arm_cfg.hidden_dim, arm_cfg.hidden_dim, arm_cfg.hidden_dim, layers=1)
        self.decoder = mlp(arm_cfg.hidden_dim + arm_cfg.obs_dim, arm_cfg.hidden_dim, 2)

    def communication_floats_per_directed_edge_round(self) -> int:
        return self.message_dim

    def communication_rounds(self) -> int:
        return self.arm_cfg.comm_rounds

    def forward(
        self, obs: Tensor, edge_index: Tensor, hidden: Tensor | None = None
    ) -> PolicyOutput:
        h = self.recurrent_encode(obs, hidden)
        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        directed_src = torch.cat((src, dst), dim=0)
        directed_dst = torch.cat((dst, src), dim=0)
        for _ in range(self.arm_cfg.comm_rounds):
            message = self.send(h.index_select(1, directed_src))
            agg_msg = h.new_zeros(h.shape[0], h.shape[1], self.message_dim)
            agg_msg.index_add_(1, directed_dst, message)
            degree = h.new_zeros(h.shape[1])
            degree.index_add_(0, directed_dst, torch.ones_like(directed_dst, dtype=h.dtype))
            agg_msg = agg_msg / degree.clamp_min(1.0).view(1, -1, 1)
            agg = self.recv(agg_msg)
            h = h + self.update(torch.cat((h, agg), dim=-1))
        raw_force = self.decoder(torch.cat((obs, h), dim=-1))
        forces = self.env_cfg.max_force * torch.tanh(raw_force)
        return PolicyOutput(forces=forces, hidden=h, aux={})


class SheafGBPPolicy(BasePolicy):
    """Learned coordination sheaf with geometry-grounded stalk restrictions."""

    arm_name = "sheaf_gbp"

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig):
        super().__init__(env_cfg, arm_cfg)
        if arm_cfg.temporal_window < 0:
            raise ValueError("temporal_window must be non-negative")
        if arm_cfg.edge_dim < 8:
            raise ValueError("sheaf edge_dim must be at least 8 for the physical section")
        self.mean_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.stalk_dim)
        self.precision_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.stalk_dim)
        self.u_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.edge_dim * arm_cfg.restriction_rank)
        self.v_head = nn.Linear(arm_cfg.hidden_dim, arm_cfg.stalk_dim * arm_cfg.restriction_rank)
        if arm_cfg.edge_conditioned_restrictions:
            # Directed restriction residuals can depend on the pair of handles
            # connected by the communication edge.  This keeps the message
            # budget unchanged but lets the sheaf express that different
            # contact pairs should agree on different projected coordinates.
            self.edge_u_head = mlp(
                arm_cfg.hidden_dim + 7,
                arm_cfg.hidden_dim,
                arm_cfg.edge_dim * arm_cfg.restriction_rank,
                layers=1,
            )
        self.gbp = DiagSheafGBP(
            arm_cfg.gbp_steps,
            damping=arm_cfg.damping,
            mean_only_messages=arm_cfg.mean_only_messages,
        )
        decoder_in_dim = arm_cfg.obs_dim + arm_cfg.edge_dim
        if not arm_cfg.decoder_section_only:
            decoder_in_dim += arm_cfg.stalk_dim
        if arm_cfg.decoder_extra_context:
            # analytic force (2) plus temporal memory count (1).  These are
            # locally available after GBP and do not add wireless messages.
            decoder_in_dim += 3
        self.decoder = mlp(decoder_in_dim, arm_cfg.hidden_dim, 2)

    def init_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        memory_dim = self.arm_cfg.temporal_window * (self.arm_cfg.stalk_dim + 1)
        return torch.zeros(
            batch_size,
            self.env_cfg.n_agents,
            self.arm_cfg.hidden_dim + memory_dim,
            device=device,
        )

    def split_hidden(self, hidden: Tensor | None, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if hidden is None:
            hidden = self.init_hidden(obs.shape[0], obs.device)
        hidden_dim = self.arm_cfg.hidden_dim
        rec_hidden = hidden[..., :hidden_dim]
        window = self.arm_cfg.temporal_window
        if window == 0:
            history = obs.new_zeros(obs.shape[0], obs.shape[1], 0, self.arm_cfg.stalk_dim)
            mask = obs.new_zeros(obs.shape[0], obs.shape[1], 0)
            return rec_hidden, history, mask
        offset = hidden_dim
        history_values = window * self.arm_cfg.stalk_dim
        history = hidden[..., offset : offset + history_values].view(
            obs.shape[0],
            obs.shape[1],
            window,
            self.arm_cfg.stalk_dim,
        )
        mask = hidden[..., offset + history_values : offset + history_values + window]
        return rec_hidden, history, mask

    def pack_hidden(
        self, rec_hidden: Tensor, history: Tensor, mask: Tensor, mean: Tensor
    ) -> Tensor:
        window = self.arm_cfg.temporal_window
        if window == 0:
            return rec_hidden
        next_history = torch.cat((mean.detach().unsqueeze(2), history[:, :, :-1]), dim=2)
        next_mask = torch.cat((torch.ones_like(mask[:, :, :1]), mask[:, :, :-1]), dim=2)
        return torch.cat(
            (rec_hidden, next_history.reshape(*rec_hidden.shape[:2], -1), next_mask),
            dim=-1,
        )

    def temporal_unary(self, history: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.arm_cfg.temporal_window == 0:
            shape = (*history.shape[:2], self.arm_cfg.stalk_dim)
            return history.new_zeros(shape), history.new_zeros(shape)
        weights = mask.unsqueeze(-1)
        precision = self.arm_cfg.temporal_precision * weights.sum(dim=2)
        natural = self.arm_cfg.temporal_precision * (weights * history).sum(dim=2)
        return precision, natural

    def communication_floats_per_directed_edge_round(self) -> int:
        # Default: edge natural plus diagonal precision; no full covariance.
        # Mean-only mode uses fixed precision and only transmits the natural
        # vector, so a 16-d section is comm-matched to a 16-float MPNN message.
        return (
            self.arm_cfg.edge_dim if self.arm_cfg.mean_only_messages else 2 * self.arm_cfg.edge_dim
        )

    def communication_rounds(self) -> int:
        return self.arm_cfg.gbp_steps

    def analytic_restrictions(self, batch: int, device: torch.device | str) -> Tensor:
        cfg = self.arm_cfg
        contacts = contact_points(self.env_cfg.n_agents, device)
        r = torch.zeros(batch, self.env_cfg.n_agents, cfg.edge_dim, cfg.stalk_dim, device=device)
        eye = torch.eye(cfg.edge_dim, cfg.stalk_dim, device=device)
        r[:] = eye.view(1, 1, cfg.edge_dim, cfg.stalk_dim)
        # Geometry-grounded wrench row: local force slots 3/4 also imply torque
        # through the handle's body-frame lever arm.
        r[:, :, 5, 3] = -contacts[:, 1].view(1, -1)
        r[:, :, 5, 4] = contacts[:, 0].view(1, -1)
        r[:, :, 5, 5] = 0.35
        return r

    def restrictions(self, h: Tensor) -> Tensor:
        batch = h.shape[0]
        base = self.analytic_restrictions(batch, h.device)
        rank = self.arm_cfg.restriction_rank
        u = self.u_head(h).view(batch, self.env_cfg.n_agents, self.arm_cfg.edge_dim, rank)
        v = self.v_head(h).view(batch, self.env_cfg.n_agents, self.arm_cfg.stalk_dim, rank)
        residual = torch.einsum("bner,bnvr->bnev", u, v)
        return base + self.arm_cfg.restriction_residual_scale * residual / float(rank)

    def directed_restrictions(self, node_r: Tensor, edge_index: Tensor) -> Tensor:
        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        left = node_r.index_select(1, src)
        right = node_r.index_select(1, dst)
        return torch.stack((left, right), dim=2).reshape(
            node_r.shape[0],
            2 * edge_index.shape[0],
            self.arm_cfg.edge_dim,
            self.arm_cfg.stalk_dim,
        )

    def edge_conditioned_directed_restrictions(
        self,
        node_r: Tensor,
        local_h: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        directed_src = torch.cat((src, dst), dim=0)
        directed_dst = torch.cat((dst, src), dim=0)
        base = node_r.index_select(1, directed_src)

        contacts = contact_points(self.env_cfg.n_agents, local_h.device)
        src_contact = contacts.index_select(0, directed_src)
        dst_contact = contacts.index_select(0, directed_dst)
        rel = dst_contact - src_contact
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        geom = torch.cat((src_contact, dst_contact, rel, dist), dim=-1)
        geom = geom.view(1, geom.shape[0], geom.shape[1]).expand(local_h.shape[0], -1, -1)

        h_src = local_h.index_select(1, directed_src)
        rank = self.arm_cfg.restriction_rank
        u = self.edge_u_head(torch.cat((h_src, geom), dim=-1)).view(
            local_h.shape[0],
            directed_src.shape[0],
            self.arm_cfg.edge_dim,
            rank,
        )
        v = self.v_head(h_src).view(
            local_h.shape[0],
            directed_src.shape[0],
            self.arm_cfg.stalk_dim,
            rank,
        )
        residual = torch.einsum("bder,bdvr->bdev", u, v)
        return base + self.arm_cfg.restriction_residual_scale * residual / float(rank)

    def clamp_force_norm(self, forces: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(forces, dim=-1, keepdim=True).clamp_min(1e-6)
        return forces * (self.env_cfg.max_force / norms).clamp_max(1.0)

    def analytic_force_from_section(self, section: Tensor, obs: Tensor) -> Tensor:
        """Map the consensus wrench slice of the section to local handle forces.

        Section coordinates 3:6 are trained to represent desired net wrench:
        Fx, Fy, tau. The allocation is the damped least-norm grasp-map solution
        used by the expert, with a learned residual added downstream.
        """

        cfg = self.env_cfg
        safe_section = section.clamp(-1.2, 1.2)
        wrench_scale = cfg.n_agents * cfg.max_force
        fx = safe_section[..., 3] * wrench_scale
        fy = safe_section[..., 4] * wrench_scale
        tau = safe_section[..., 5] * (0.75 * wrench_scale)

        contacts = contact_points(cfg.n_agents, obs.device)
        if cfg.strict_local_observation:
            sin_theta = obs[:, :, 0]
            cos_theta = obs[:, :, 1]
        else:
            sin_theta = obs[:, :, 2]
            cos_theta = obs[:, :, 3]
        cx = contacts[:, 0].view(1, -1)
        cy = contacts[:, 1].view(1, -1)
        rx = cos_theta * cx - sin_theta * cy
        ry = sin_theta * cx + cos_theta * cy
        sum_r2 = (rx.square() + ry.square()).sum(dim=1, keepdim=True)

        fx_per = fx / float(cfg.n_agents + 0.08)
        fy_per = fy / float(cfg.n_agents + 0.08)
        tau_coeff = tau / (sum_r2 + 0.08)
        forces = torch.stack((fx_per - tau_coeff * ry, fy_per + tau_coeff * rx), dim=-1)
        return self.clamp_force_norm(forces)

    def forward(
        self, obs: Tensor, edge_index: Tensor, hidden: Tensor | None = None
    ) -> PolicyOutput:
        rec_hidden, history, history_mask = self.split_hidden(hidden, obs)
        encoded = self.obs_encoder(obs)
        batch, n_agents, _ = obs.shape
        local_h = self.gru(
            encoded.reshape(batch * n_agents, -1),
            rec_hidden.reshape(batch * n_agents, -1),
        ).view(batch, n_agents, -1)
        mu = self.mean_head(local_h)
        precision = F.softplus(self.precision_head(local_h)) + 1e-3
        natural = precision * mu
        temporal_precision, temporal_natural = self.temporal_unary(history, history_mask)
        precision = precision + temporal_precision
        natural = natural + temporal_natural
        node_r = self.restrictions(local_h)
        if self.arm_cfg.edge_conditioned_restrictions:
            directed_r = self.edge_conditioned_directed_restrictions(node_r, local_h, edge_index)
        else:
            directed_r = self.directed_restrictions(node_r, edge_index)
        topology = homogeneous_topology(
            edge_index,
            obs.shape[0],
            self.env_cfg.n_agents,
            device=obs.device,
        )
        sigma2 = obs.new_full(
            (obs.shape[0], 2 * edge_index.shape[0], self.arm_cfg.edge_dim),
            self.arm_cfg.edge_sigma2,
        )
        result = self.gbp(precision, natural, directed_r, sigma2, topology)
        section = torch.einsum("bnev,bnv->bne", node_r, result.mean)
        analytic_force = self.analytic_force_from_section(section, obs)
        decoder_inputs = [obs, section]
        if not self.arm_cfg.decoder_section_only:
            decoder_inputs.insert(1, result.mean)
        memory_count = history_mask.sum(dim=-1)
        if self.arm_cfg.decoder_extra_context:
            memory_scale = float(max(1, self.arm_cfg.temporal_window))
            decoder_inputs.append(analytic_force / self.env_cfg.max_force)
            decoder_inputs.append((memory_count / memory_scale).unsqueeze(-1))
        raw_residual = self.decoder(torch.cat(decoder_inputs, dim=-1))
        residual_force = (
            self.arm_cfg.force_residual_scale * self.env_cfg.max_force * torch.tanh(raw_residual)
        )
        forces = self.clamp_force_norm(
            self.arm_cfg.analytic_force_scale * analytic_force + residual_force
        )
        next_hidden = self.pack_hidden(local_h, history, history_mask, result.mean)
        return PolicyOutput(
            forces=forces,
            hidden=next_hidden,
            aux={
                "section": section,
                "analytic_force": analytic_force,
                "stalk_mean": result.mean,
                "stalk_precision": result.precision,
                "temporal_memory_count": memory_count,
            },
        )


def make_policy(arm: str, env_cfg: EnvConfig, arm_cfg: ArmConfig) -> BasePolicy:
    if arm == "no_comm":
        return NoCommPolicy(env_cfg, arm_cfg)
    if arm == "raw_full":
        return MPNNPolicy(env_cfg, arm_cfg, message_dim=32, name="raw_full")
    if arm == "comm_matched":
        return MPNNPolicy(env_cfg, arm_cfg, message_dim=2 * arm_cfg.edge_dim, name="comm_matched")
    if arm == "sheaf_gbp":
        return SheafGBPPolicy(env_cfg, arm_cfg)
    raise ValueError(f"unknown arm: {arm}")
