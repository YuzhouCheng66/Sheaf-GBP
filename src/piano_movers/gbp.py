"""Diagonal-message Sheaf Gaussian belief propagation.

This is a lean vendored subset of the local `sheaf-admm-repro` PyTorch GBP
frontend.  It keeps only the irregular-free homogeneous topology used by
PianoMovers-Force and communicates diagonal edge precision plus natural vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GBPTopology:
    incidence_vertex: Tensor
    reverse_incidence: Tensor
    flat_incidence_index: Tensor
    valid_incidence: Tensor


@dataclass
class GBPResult:
    mean: Tensor
    precision: Tensor
    omega: Tensor
    xi: Tensor


def homogeneous_topology(
    edge_index: Tensor,
    batch_size: int,
    num_nodes: int,
    *,
    device: torch.device | str | None = None,
) -> GBPTopology:
    """Build directed incidences from undirected edges.

    Edges are ordered `[u, v]`, and directed incidences are `[u, v, ...]`, so
    `incidence ^ 1` is the reverse message slot.
    """

    target_device = torch.device(device) if device is not None else edge_index.device
    edges = edge_index.to(device=target_device, dtype=torch.long)
    if edges.ndim != 2:
        raise ValueError("edge_index must have shape [E,2] or [2,E]")
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.transpose(0, 1)
    if edges.shape[1] != 2:
        raise ValueError("edge_index must have shape [E,2] or [2,E]")
    incidence = edges.reshape(-1)
    directed = incidence.numel()
    offsets = torch.arange(batch_size, device=target_device).unsqueeze(1) * num_nodes
    flat = (incidence.unsqueeze(0) + offsets).reshape(-1)
    reverse = torch.arange(directed, device=target_device, dtype=torch.long).bitwise_xor(1)
    return GBPTopology(
        incidence_vertex=incidence,
        reverse_incidence=reverse,
        flat_incidence_index=flat,
        valid_incidence=torch.ones(directed, dtype=torch.bool, device=target_device),
    )


class DiagSheafGBP(nn.Module):
    """Diagonal-message Gaussian BP over partial-consensus sheaf factors."""

    def __init__(
        self,
        num_steps: int,
        damping: float = 0.55,
        eps: float = 1e-6,
        max_precision: float = 1e5,
        mean_only_messages: bool = False,
    ):
        super().__init__()
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be in [0,1]")
        self.num_steps = num_steps
        self.damping = damping
        self.eps = eps
        self.max_precision = max_precision
        self.mean_only_messages = mean_only_messages

    def _incoming(
        self,
        q: Tensor,
        h: Tensor,
        r: Tensor,
        r2: Tensor,
        topology: GBPTopology,
        omega: Tensor,
        xi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, num_nodes, vertex_dim = q.shape
        omega_reverse = omega.index_select(1, topology.reverse_incidence)
        xi_reverse = xi.index_select(1, topology.reverse_incidence)
        incidence_precision = (r2 * omega_reverse.unsqueeze(-1)).sum(dim=-2)
        incidence_natural = (r * xi_reverse.unsqueeze(-1)).sum(dim=-2)

        flat_size = batch * num_nodes
        node_precision = q.reshape(flat_size, vertex_dim).clone()
        node_natural = h.reshape(flat_size, vertex_dim).clone()
        node_precision.index_add_(
            0,
            topology.flat_incidence_index,
            incidence_precision.reshape(-1, vertex_dim),
        )
        node_natural.index_add_(
            0,
            topology.flat_incidence_index,
            incidence_natural.reshape(-1, vertex_dim),
        )
        return (
            node_precision.view(batch, num_nodes, vertex_dim),
            node_natural.view(batch, num_nodes, vertex_dim),
        )

    def sweep(
        self,
        q: Tensor,
        h: Tensor,
        r: Tensor,
        r2: Tensor,
        sigma2: Tensor,
        topology: GBPTopology,
        omega: Tensor,
        xi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        node_precision, node_natural = self._incoming(q, h, r, r2, topology, omega, xi)
        batch, _, vertex_dim = q.shape

        omega_reverse = omega.index_select(1, topology.reverse_incidence)
        xi_reverse = xi.index_select(1, topology.reverse_incidence)
        incidence_precision = (r2 * omega_reverse.unsqueeze(-1)).sum(dim=-2)
        incidence_natural = (r * xi_reverse.unsqueeze(-1)).sum(dim=-2)

        gathered_precision = node_precision.index_select(1, topology.incidence_vertex)
        gathered_natural = node_natural.index_select(1, topology.incidence_vertex)
        cavity_precision = (gathered_precision - incidence_precision).clamp_min(self.eps)
        cavity_natural = gathered_natural - incidence_natural
        cavity_variance = cavity_precision.reciprocal()
        cavity_mean = cavity_variance * cavity_natural

        edge_mean = (r * cavity_mean.unsqueeze(-2)).sum(dim=-1)
        edge_variance = (r2 * cavity_variance.unsqueeze(-2)).sum(dim=-1)
        if self.mean_only_messages:
            # Communication-efficient variant: edge precision is a fixed,
            # deterministic hyperparameter, so only the natural/mean vector has
            # to be transmitted.  This spends all communicated floats on the
            # shared section rather than half on dynamic diagonal precision.
            omega_target = sigma2.clamp_min(self.eps).reciprocal().clamp_max(self.max_precision)
        else:
            omega_target = (
                (sigma2 + edge_variance)
                .clamp_min(self.eps)
                .reciprocal()
                .clamp_max(self.max_precision)
            )
        xi_target = omega_target * edge_mean
        omega_new = (
            omega_target
            if self.mean_only_messages
            else torch.lerp(omega, omega_target, self.damping)
        )
        xi_new = torch.lerp(xi, xi_target, self.damping)
        valid = topology.valid_incidence.view(1, -1, 1)
        return omega_new * valid, xi_new * valid

    def beliefs(
        self,
        q: Tensor,
        h: Tensor,
        r: Tensor,
        r2: Tensor,
        topology: GBPTopology,
        omega: Tensor,
        xi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        node_precision, node_natural = self._incoming(q, h, r, r2, topology, omega, xi)
        return node_natural / node_precision.clamp_min(self.eps), node_precision

    def forward(
        self,
        q: Tensor,
        h: Tensor,
        r: Tensor,
        sigma2: Tensor,
        topology: GBPTopology,
        initial_omega: Tensor | None = None,
        initial_xi: Tensor | None = None,
    ) -> GBPResult:
        r2 = r.square()
        batch, directed, edge_dim, _ = r.shape
        if initial_omega is not None:
            omega = initial_omega
        elif self.mean_only_messages:
            omega = sigma2.clamp_min(self.eps).reciprocal().clamp_max(self.max_precision)
        else:
            omega = q.new_zeros((batch, directed, edge_dim))
        xi = q.new_zeros((batch, directed, edge_dim)) if initial_xi is None else initial_xi
        for _ in range(self.num_steps):
            omega, xi = self.sweep(q, h, r, r2, sigma2, topology, omega, xi)
        mean, precision = self.beliefs(q, h, r, r2, topology, omega, xi)
        return GBPResult(mean=mean, precision=precision, omega=omega, xi=xi)
