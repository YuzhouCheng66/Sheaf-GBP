"""Browser demo for PianoMovers-Force Sheaf-GBP rollouts.

This intentionally avoids Streamlit/Gradio or ffmpeg.  The server runs the
trained policies on CPU and returns a compact JSON rollout; the browser renders
the map, T payload, ant handles, communication graph, and trajectories on a
Canvas.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from .config import EnvConfig
from .env import PianoBatch, radius_edges, sample_scenarios
from .metrics import communication_bytes
from .video import find_case, load_policy, record_rollout, visual_contact_points


REPO_ROOT = Path(__file__).resolve().parents[2]

PRESETS: dict[int, dict[str, Any]] = {
    8: {
        "label": "8 ants · transferred map-medium",
        "default_seed": 20261941,
        "sheaf": "runs/map_medium_8/sheaf_gbp/sheaf_gbp_best.pt",
        "raw": "runs/map_medium_8/raw_full/raw_full_best.pt",
        "paired": {
            "sheaf_success": 0.9470,
            "raw_success": 0.4058,
            "sheaf_bytes": 274560,
            "raw_bytes": 366080,
        },
    },
    12: {
        "label": "12 ants · map-medium",
        "default_seed": 20261921,
        "sheaf": "runs/map_medium_12/sheaf_gbp/sheaf_gbp_best.pt",
        "raw": "runs/map_medium_12/raw_full/raw_full_best.pt",
        "paired": {
            "sheaf_success": 0.9683,
            "raw_success": 0.2715,
            "sheaf_bytes": 295680,
            "raw_bytes": 394240,
        },
    },
}

BODY_RECTS = [
    [
        [-0.20 * 0.55, -0.95 * 0.55],
        [0.20 * 0.55, -0.95 * 0.55],
        [0.20 * 0.55, 0.55 * 0.55],
        [-0.20 * 0.55, 0.55 * 0.55],
    ],
    [
        [-0.85 * 0.55, 0.35 * 0.55],
        [0.85 * 0.55, 0.35 * 0.55],
        [0.85 * 0.55, 0.72 * 0.55],
        [-0.85 * 0.55, 0.72 * 0.55],
    ],
]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def scenario_to_json(scenario: PianoBatch) -> dict[str, Any]:
    return {
        "state0": tensor_list(scenario.state0[0]),
        "goal": tensor_list(scenario.goal[0]),
        "wall_x": tensor_list(scenario.wall_x[0]),
        "gap_y": tensor_list(scenario.gap_y[0]),
        "gap_half": tensor_list(scenario.gap_half[0]),
        "source": tensor_list(scenario.source[0]),
    }


def frame_to_json(frame: dict[str, Any]) -> dict[str, Any]:
    section = frame.get("section")
    section_mean = None
    if section is not None:
        section_mean = tensor_list(section.mean(dim=0))
    return {
        "step": int(frame["step"]),
        "state": tensor_list(frame["state"]),
        "next_state": tensor_list(frame["next_state"]),
        "forces": tensor_list(frame["forces"]),
        "section_mean": section_mean,
        "success": bool(frame["success"]),
        "collision": bool(frame["collision"]),
    }


def arm_summary(frames: list[dict[str, Any]], bytes_per_episode: int | float) -> dict[str, Any]:
    terminal = frames[-1]
    return {
        "success": bool(terminal["success"]),
        "collision": bool(terminal["collision"]),
        "finish_step": int(terminal["step"] + 1),
        "wire_bytes_per_episode": float(bytes_per_episode),
    }


@lru_cache(maxsize=8)
def cached_policy(path_string: str):
    path = repo_path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return load_policy(path, torch.device("cpu"))


def sample_direct_map(
    cfg: EnvConfig, seed: int, map_index: int
) -> tuple[PianoBatch, dict[str, Any]]:
    if map_index < 0:
        raise ValueError("map_index must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scenario = sample_scenarios(map_index + 1, cfg, device="cpu", generator=generator)
    selected = PianoBatch(
        state0=scenario.state0[map_index : map_index + 1].clone(),
        goal=scenario.goal[map_index : map_index + 1].clone(),
        wall_x=scenario.wall_x[map_index : map_index + 1].clone(),
        gap_y=scenario.gap_y[map_index : map_index + 1].clone(),
        gap_half=scenario.gap_half[map_index : map_index + 1].clone(),
        source=scenario.source[map_index : map_index + 1].clone(),
    )
    return selected, {
        "search_seed": seed,
        "search_episode_index": map_index,
        "contrast_search": False,
    }


@torch.no_grad()
def build_rollout_payload(
    *,
    agents: int,
    seed: int,
    contrast_search: bool,
    max_search_episodes: int,
) -> dict[str, Any]:
    if agents not in PRESETS:
        raise ValueError(f"unsupported agent count {agents}; available: {sorted(PRESETS)}")
    preset = PRESETS[agents]
    sheaf_arm, sheaf_policy, cfg, sheaf_checkpoint = cached_policy(preset["sheaf"])
    raw_arm, raw_policy, raw_cfg, raw_checkpoint = cached_policy(preset["raw"])
    if sheaf_arm != "sheaf_gbp":
        raise ValueError(f"expected sheaf_gbp checkpoint, got {sheaf_arm}")
    if raw_cfg != cfg:
        raise ValueError("raw checkpoint EnvConfig does not match sheaf checkpoint EnvConfig")

    if contrast_search:
        try:
            scenario, meta = find_case(
                sheaf_policy,
                cfg,
                device=torch.device("cpu"),
                seed=seed,
                batch_size=64,
                max_episodes=max_search_episodes,
                raw_policy=raw_policy,
                require_raw_fail=True,
            )
            meta["contrast_search"] = True
        except RuntimeError:
            scenario, meta = sample_direct_map(cfg, seed, 0)
            meta["contrast_search"] = False
            meta["contrast_search_fallback"] = True
    else:
        scenario, meta = sample_direct_map(cfg, seed, 0)

    sheaf_frames = record_rollout(sheaf_policy, scenario, cfg)
    raw_frames = record_rollout(raw_policy, scenario, cfg)
    edges = radius_edges(cfg, device="cpu")
    sheaf_bytes = communication_bytes(sheaf_policy, edges, batch_size=1, steps=cfg.max_steps)
    raw_bytes = communication_bytes(raw_policy, edges, batch_size=1, steps=cfg.max_steps)

    return {
        "preset": {
            "agents": agents,
            "label": preset["label"],
            "paired": preset["paired"],
            "sheaf_checkpoint_step": sheaf_checkpoint.get("step"),
            "raw_checkpoint_step": raw_checkpoint.get("step"),
        },
        "meta": meta,
        "cfg": asdict(cfg),
        "scenario": scenario_to_json(scenario),
        "edges": tensor_list(edges),
        "contacts": tensor_list(visual_contact_points(cfg.n_agents, "cpu")),
        "body_rects": BODY_RECTS,
        "arms": {
            "sheaf": {
                "name": "Sheaf-GBP",
                "path_color": "#386cb0",
                "frames": [frame_to_json(frame) for frame in sheaf_frames],
                "summary": arm_summary(sheaf_frames, sheaf_bytes),
            },
            "raw": {
                "name": "raw-full",
                "path_color": "#cc6677",
                "frames": [frame_to_json(frame) for frame in raw_frames],
                "summary": arm_summary(raw_frames, raw_bytes),
            },
        },
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PianoMovers-Force · Sheaf-GBP demo</title>
  <style>
    :root {
      --paper: #fbfaf7;
      --ink: #292724;
      --muted: #756f66;
      --hair: #ded7cb;
      --card: #fffdf8;
      --blue: #386cb0;
      --rose: #cc6677;
      --gold: #f1bf58;
      --green: #7aa974;
      --orange: #d55e00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(220, 210, 190, 0.35), transparent 36rem),
        linear-gradient(180deg, #fffefb 0%, var(--paper) 100%);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1420px, calc(100vw - 32px));
      margin: 20px auto 28px;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 14px;
    }
    h1 {
      font-size: clamp(26px, 3vw, 42px);
      line-height: 1.02;
      letter-spacing: -0.045em;
      margin: 0 0 6px;
      font-weight: 650;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.45;
      max-width: 860px;
    }
    .controls {
      display: flex;
      gap: 10px;
      align-items: end;
      justify-content: flex-end;
      flex-wrap: wrap;
      padding: 12px;
      border: 1px solid var(--hair);
      border-radius: 18px;
      background: rgba(255, 253, 248, 0.78);
      box-shadow: 0 10px 28px rgba(78, 74, 69, 0.08);
      backdrop-filter: blur(10px);
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.02em;
    }
    select, input {
      height: 38px;
      min-width: 122px;
      border: 1px solid #d7d0c4;
      border-radius: 12px;
      background: #fffefb;
      color: var(--ink);
      padding: 0 11px;
      font: inherit;
      outline: none;
    }
    button {
      height: 38px;
      border: 0;
      border-radius: 999px;
      padding: 0 16px;
      color: #fffefb;
      background: #2f2a25;
      font-weight: 640;
      cursor: pointer;
      box-shadow: 0 8px 16px rgba(47, 42, 37, 0.16);
    }
    button.secondary {
      background: #fffdf8;
      color: var(--ink);
      border: 1px solid var(--hair);
      box-shadow: none;
    }
    button:disabled {
      opacity: 0.48;
      cursor: wait;
    }
    .stage {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--hair);
      border-radius: 24px;
      background: var(--paper);
      box-shadow: 0 18px 45px rgba(78, 74, 69, 0.10);
    }
    canvas {
      display: block;
      width: 100%;
      height: min(66vh, 760px);
      min-height: 520px;
      background: var(--paper);
    }
    .hud {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      overflow: hidden;
      border: 1px solid var(--hair);
      border-radius: 20px;
      margin-top: 12px;
      background: var(--hair);
    }
    .tile {
      background: rgba(255, 253, 248, 0.86);
      padding: 13px 15px;
      min-height: 74px;
    }
    .tile small {
      display: block;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .tile strong {
      display: block;
      font-size: 19px;
      letter-spacing: -0.02em;
    }
    .tile span {
      color: var(--muted);
      font-size: 12px;
    }
    .note {
      margin: 11px 3px 0;
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.45;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      min-width: 210px;
      align-self: center;
    }
    @media (max-width: 900px) {
      header { grid-template-columns: 1fr; }
      .controls { justify-content: flex-start; }
      .hud { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      canvas { min-height: 460px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>PianoMovers-Force</h1>
        <p class="subtitle">
          A T-shaped load moves through two narrow slits. Ants are fixed force handles;
          pale links are the radius-limited communication graph. The left panel shares
          a 12-d sheaf section, the right panel sends raw 32-d latents.
        </p>
      </div>
      <div class="controls">
        <label>
          agents
          <select id="agents">
            <option value="8">8 ants</option>
            <option value="12" selected>12 ants</option>
          </select>
        </label>
        <label>
          map seed
          <input id="seed" type="number" value="20261921" step="1" />
        </label>
        <button id="render">Render case</button>
        <button id="play" class="secondary">Pause</button>
        <button id="restart" class="secondary">Restart</button>
        <div id="status" class="status">Ready.</div>
      </div>
    </header>

    <section class="stage">
      <canvas id="canvas"></canvas>
    </section>

    <section class="hud">
      <div class="tile"><small>Sheaf-GBP</small><strong id="sheaf-status">—</strong><span id="sheaf-detail">—</span></div>
      <div class="tile"><small>raw-full</small><strong id="raw-status">—</strong><span id="raw-detail">—</span></div>
      <div class="tile"><small>communication</small><strong id="comm-detail">—</strong><span>bytes per episode, same map</span></div>
      <div class="tile"><small>map</small><strong id="map-detail">—</strong><span id="map-extra">procedural two-slit room</span></div>
    </section>
    <p class="note">
      The seed directly samples one procedural map. A single movie is only a
      demo; the scientific claim still comes from the paired 4096-episode reports.
    </p>
  </main>

<script>
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
let payload = null;
let frame = 0;
let playing = true;
let lastTime = performance.now();
let fps = 10;

const palette = {
  paper: "#fbfaf7",
  ink: "#292724",
  muted: "#756f66",
  wall: "#4e4a45",
  wallTick: "#b3aaa0",
  blue: "#386cb0",
  rose: "#cc6677",
  link: "#4da3c7",
  gold: "#f1bf58",
  green: "#7aa974",
  orange: "#d55e00",
  purple: "#7b5ea7",
  ant: "#3b2416",
  antEdge: "#120b07",
};

function resize() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}
window.addEventListener("resize", resize);

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function statusText(summary) {
  if (!summary) return "—";
  if (summary.success) return "SUCCESS";
  if (summary.collision) return "COLLISION";
  return "timeout";
}
function kib(bytes) { return `${Math.round(bytes / 1024)} KiB`; }
function fmtPct(x) { return `${(100 * x).toFixed(1)}%`; }

function panelGeometry(panel) {
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  const gap = 18;
  const margin = 42;
  const panelW = (w - 2 * margin - gap) / 2;
  const panelH = h - 92;
  const x0 = margin + panel * (panelW + gap);
  const y0 = 62;
  const cfg = payload.cfg;
  const sx = panelW / (2 * cfg.world_x + 0.34);
  const sy = panelH / (2 * cfg.world_y + 0.22);
  const scale = Math.min(sx, sy);
  return {
    x: x0, y: y0, w: panelW, h: panelH, scale,
    cx: x0 + panelW / 2,
    cy: y0 + panelH / 2,
  };
}

function worldToScreen(g, p) {
  return [g.cx + p[0] * g.scale, g.cy - p[1] * g.scale];
}

function rotatePoint(p, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [c * p[0] - s * p[1], s * p[0] + c * p[1]];
}

function bodyToWorld(local, state) {
  const r = rotatePoint(local, state[2]);
  return [state[0] + r[0], state[1] + r[1]];
}

function drawRoundRect(x, y, w, h, r, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.stroke(); }
}

function drawStar(x, y, r, fill) {
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5;
    const rr = i % 2 === 0 ? r : r * 0.42;
    const px = rr * Math.cos(a), py = rr * Math.sin(a);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = "#fffefb";
  ctx.lineWidth = 1.1;
  ctx.stroke();
  ctx.restore();
}

function drawEllipse(x, y, rx, ry, angle, fill, stroke, alpha = 1, lineWidth = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
  ctx.restore();
}

function drawAnt(g, xyWorld, heading, scaleWorld) {
  const [x, y] = worldToScreen(g, xyWorld);
  const scale = g.scale * scaleWorld;
  const c = Math.cos(heading), s = Math.sin(heading);
  const ux = c, uy = -s;       // screen y is inverted
  const vx = -s, vy = -c;
  drawEllipse(x, y, 0.052 * scale, 0.035 * scale, -heading, "#4da3c7", null, 0.18);
  const length = 0.042 * scale;
  const width = 0.026 * scale;
  const centers = [
    [-0.95 * length, 1.25, 1.00],
    [0.00, 1.02, 0.86],
    [0.90 * length, 0.82, 0.70],
  ];
  ctx.save();
  ctx.lineCap = "round";
  for (const offset of [-0.72, 0.0, 0.72]) {
    const bx = x + offset * length * ux;
    const by = y + offset * length * uy;
    for (const side of [-1, 1]) {
      const kx = bx + side * 0.030 * scale * vx;
      const ky = by + side * 0.030 * scale * vy;
      const fx = kx - 0.018 * scale * ux;
      const fy = ky - 0.018 * scale * uy;
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(kx, ky);
      ctx.lineTo(fx, fy);
      ctx.strokeStyle = palette.antEdge;
      ctx.globalAlpha = 0.82;
      ctx.lineWidth = 0.72 * scaleWorld;
      ctx.stroke();
    }
  }
  ctx.restore();
  for (const [off, lw, ww] of centers) {
    drawEllipse(
      x + off * ux,
      y + off * uy,
      0.5 * length * lw,
      0.5 * width * ww,
      -heading,
      palette.ant,
      palette.antEdge,
      0.96,
      0.45 * scaleWorld
    );
  }
}

function forceHeading(point, force, state) {
  const norm = Math.hypot(force[0], force[1]);
  if (norm > 1e-4) return Math.atan2(force[1], force[0]);
  return Math.atan2(point[1] - state[1], point[0] - state[0]) + state[2];
}

function drawStatic(g, panelTitle) {
  const cfg = payload.cfg;
  const sc = payload.scenario;
  ctx.save();
  ctx.beginPath();
  ctx.rect(g.x, g.y, g.w, g.h);
  ctx.clip();
  ctx.fillStyle = palette.paper;
  ctx.fillRect(g.x, g.y, g.w, g.h);

  const lowerLeft = worldToScreen(g, [-cfg.world_x, -cfg.world_y]);
  const upperRight = worldToScreen(g, [cfg.world_x, cfg.world_y]);
  ctx.strokeStyle = "rgba(138,129,119,0.72)";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(lowerLeft[0], upperRight[1], upperRight[0] - lowerLeft[0], lowerLeft[1] - upperRight[1]);

  for (let wi = 0; wi < 2; wi++) {
    const x = sc.wall_x[wi];
    const gy = sc.gap_y[wi];
    const gh = Math.max(0, sc.gap_half[wi] - 0.03); // match exact collision aperture
    const slab = cfg.wall_slab;
    const lowTop = gy - gh;
    const highBottom = gy + gh;
    const r1a = worldToScreen(g, [x - slab, -cfg.world_y]);
    const r1b = worldToScreen(g, [x + slab, lowTop]);
    const r2a = worldToScreen(g, [x - slab, highBottom]);
    const r2b = worldToScreen(g, [x + slab, cfg.world_y]);
    ctx.fillStyle = "rgba(78,74,69,0.90)";
    ctx.fillRect(r1a[0], r1b[1], r1b[0] - r1a[0], r1a[1] - r1b[1]);
    ctx.fillRect(r2a[0], r2b[1], r2b[0] - r2a[0], r2a[1] - r2b[1]);
    ctx.strokeStyle = palette.wallTick;
    ctx.lineWidth = 1.0;
    for (const y of [gy - gh, gy + gh]) {
      const a = worldToScreen(g, [x - 0.12, y]);
      const b = worldToScreen(g, [x + 0.12, y]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
  }
  const src = worldToScreen(g, sc.source);
  ctx.beginPath();
  ctx.arc(src[0], src[1], 5.8, 0, 2 * Math.PI);
  ctx.fillStyle = palette.green;
  ctx.fill();
  ctx.strokeStyle = "#fffefb";
  ctx.lineWidth = 1.1;
  ctx.stroke();
  const goal = worldToScreen(g, sc.goal);
  drawStar(goal[0], goal[1], 9.5, palette.orange);
  ctx.restore();

  ctx.fillStyle = palette.ink;
  ctx.font = "600 18px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(panelTitle, g.x + g.w / 2, 31);
}

function drawBody(g, state) {
  for (const rect of payload.body_rects) {
    ctx.beginPath();
    rect.forEach((p, i) => {
      const w = bodyToWorld(p, state);
      const s = worldToScreen(g, w);
      if (i === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
    });
    ctx.closePath();
    ctx.fillStyle = palette.gold;
    ctx.globalAlpha = 0.96;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#2f2a25";
    ctx.lineWidth = 1.35;
    ctx.stroke();
  }
}

function drawPath(g, frames, idx, color) {
  ctx.beginPath();
  for (let i = 0; i <= idx; i++) {
    const s = worldToScreen(g, frames[Math.min(i, frames.length - 1)].state);
    if (i === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
  }
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.76;
  ctx.lineWidth = 2.7;
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawPanel(panel, armKey, globalFrame) {
  const arm = payload.arms[armKey];
  const frames = arm.frames;
  const idx = Math.min(globalFrame, frames.length - 1);
  const f = frames[idx];
  const g = panelGeometry(panel);
  drawStatic(g, arm.name);
  drawPath(g, frames, idx, arm.path_color);

  // maneuver target marker: section-independent visual target from payload is not sent;
  // use goal marker and path as the clean explanatory visual.
  const contactsWorld = payload.contacts.map((p) => bodyToWorld(p, f.state));
  for (const [a, b] of payload.edges) {
    const pa = worldToScreen(g, contactsWorld[a]);
    const pb = worldToScreen(g, contactsWorld[b]);
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.strokeStyle = palette.link;
    ctx.globalAlpha = 0.26;
    ctx.lineWidth = 1.15;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  drawBody(g, f.state);
  contactsWorld.forEach((p, i) => drawAnt(g, p, forceHeading(p, f.forces[i], f.state), 1.12));

  const summary = arm.summary;
  const st = f.success ? "SUCCESS" : (f.collision ? "COLLISION" : "running");
  const line = `step ${String(f.step).padStart(2, "0")}/${payload.cfg.max_steps} · ${st} · ${kib(summary.wire_bytes_per_episode)}/episode`;
  ctx.fillStyle = palette.muted;
  ctx.font = "500 14px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(line, g.x + g.w / 2, 53);

  const text = f.section_mean
    ? `shared wrench=(${f.section_mean[3].toFixed(2)}, ${f.section_mean[4].toFixed(2)}, τ=${f.section_mean[5].toFixed(2)})`
    : "no shared 8-d sheaf section";
  const box = worldToScreen(g, [-payload.cfg.world_x + 0.10, payload.cfg.world_y - 0.10]);
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  const tw = ctx.measureText(text).width + 18;
  drawRoundRect(box[0], box[1] - 20, tw, 25, 7, "rgba(255,253,248,0.90)", "rgba(214,207,195,0.95)");
  ctx.fillStyle = palette.ink;
  ctx.textAlign = "left";
  ctx.fillText(text, box[0] + 9, box[1] - 4);
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = palette.paper;
  ctx.fillRect(0, 0, rect.width, rect.height);
  if (!payload) {
    ctx.fillStyle = palette.muted;
    ctx.font = "500 18px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Choose agents and seed, then render a case.", rect.width / 2, rect.height / 2);
    return;
  }
  ctx.fillStyle = palette.ink;
  ctx.font = "650 20px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  const meta = payload.meta;
  const title = `same initial state and map · seed ${meta.search_seed} · case ${meta.search_episode_index}`;
  ctx.fillText(title, rect.width / 2, 23);
  drawPanel(0, "sheaf", frame);
  drawPanel(1, "raw", frame);
}

function updateHud() {
  if (!payload) return;
  const sheaf = payload.arms.sheaf.summary;
  const raw = payload.arms.raw.summary;
  $("sheaf-status").textContent = statusText(sheaf);
  $("sheaf-detail").textContent = `finish step ${sheaf.finish_step} · paired SR ${fmtPct(payload.preset.paired.sheaf_success)}`;
  $("raw-status").textContent = statusText(raw);
  $("raw-detail").textContent = `finish step ${raw.finish_step} · paired SR ${fmtPct(payload.preset.paired.raw_success)}`;
  $("comm-detail").textContent = `${kib(sheaf.wire_bytes_per_episode)} < ${kib(raw.wire_bytes_per_episode)}`;
  $("map-detail").textContent = `${payload.preset.agents} agents · seed ${payload.meta.search_seed}`;
  const gh = payload.scenario.gap_half.map((x) => (x - 0.03).toFixed(2)).join(", ");
  $("map-extra").textContent = `effective slit half-widths ${gh}`;
}

async function renderCase() {
  const agents = $("agents").value;
  const seed = $("seed").value;
  $("render").disabled = true;
  $("status").textContent = "Loading policies and rolling out this seed map…";
  try {
    const res = await fetch(`/api/rollout?agents=${encodeURIComponent(agents)}&seed=${encodeURIComponent(seed)}&contrast=0`);
    if (!res.ok) throw new Error(await res.text());
    payload = await res.json();
    frame = 0;
    playing = true;
    $("play").textContent = "Pause";
    updateHud();
    draw();
    $("status").textContent = `Rendered seed ${payload.meta.search_seed}.`;
  } catch (err) {
    console.error(err);
    $("status").textContent = `Error: ${err.message || err}`;
  } finally {
    $("render").disabled = false;
  }
}

function tick(now) {
  if (payload && playing && now - lastTime > 1000 / fps) {
    lastTime = now;
    const total = Math.max(payload.arms.sheaf.frames.length, payload.arms.raw.frames.length);
    frame = Math.min(frame + 1, total - 1);
    if (frame >= total - 1) playing = false;
    $("play").textContent = playing ? "Pause" : "Play";
    draw();
  }
  requestAnimationFrame(tick);
}

$("render").addEventListener("click", renderCase);
$("play").addEventListener("click", () => {
  playing = !playing;
  $("play").textContent = playing ? "Pause" : "Play";
});
$("restart").addEventListener("click", () => {
  frame = 0;
  playing = true;
  $("play").textContent = "Pause";
  draw();
});
$("agents").addEventListener("change", () => {
  if ($("agents").value === "8") $("seed").value = "20261941";
  if ($("agents").value === "12") $("seed").value = "20261921";
});

resize();
requestAnimationFrame(tick);
renderCase();
</script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "PianoMoversDemo/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args)
        )

    def send_bytes(
        self, data: bytes, *, status: int = 200, content_type: str = "application/octet-stream"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: Any, *, status: int = 200) -> None:
        self.send_bytes(
            json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def send_error_json(self, message: str, *, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"error": message}, status=int(status))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                self.send_bytes(INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/presets":
                self.send_json(
                    {
                        str(k): {
                            "label": v["label"],
                            "default_seed": v["default_seed"],
                            "paired": v["paired"],
                            "available": repo_path(v["sheaf"]).exists()
                            and repo_path(v["raw"]).exists(),
                        }
                        for k, v in PRESETS.items()
                    }
                )
                return
            if parsed.path == "/api/rollout":
                query = parse_qs(parsed.query)
                agents = int(query.get("agents", ["12"])[0])
                fallback_preset = PRESETS[12]
                seed = int(
                    query.get("seed", [str(PRESETS.get(agents, fallback_preset)["default_seed"])])[
                        0
                    ]
                )
                contrast = query.get("contrast", ["0"])[0] not in ("0", "false", "False", "no")
                max_search = int(query.get("max_search_episodes", ["2048"])[0])
                payload = build_rollout_payload(
                    agents=agents,
                    seed=seed,
                    contrast_search=contrast,
                    max_search_episodes=max_search,
                )
                self.send_json(payload)
                return
            if parsed.path == "/favicon.ico":
                self.send_bytes(b"", content_type="image/x-icon")
                return
            path = (REPO_ROOT / parsed.path.lstrip("/")).resolve()
            if not path.is_file() or REPO_ROOT not in path.parents:
                self.send_error_json("not found", status=HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_bytes(path.read_bytes(), content_type=content_type)
        except Exception as exc:  # Keep demo errors visible in the browser.
            self.send_error_json(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the PianoMovers-Force browser demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--smoke", action="store_true", help="build one payload and exit")
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261921)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        payload = build_rollout_payload(
            agents=args.agents,
            seed=args.seed,
            contrast_search=False,
            max_search_episodes=256,
        )
        compact = {
            "agents": payload["preset"]["agents"],
            "seed": payload["meta"]["search_seed"],
            "case": payload["meta"]["search_episode_index"],
            "sheaf": payload["arms"]["sheaf"]["summary"],
            "raw": payload["arms"]["raw"]["summary"],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
        return
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"PianoMovers demo serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
