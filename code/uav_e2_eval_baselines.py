#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:gym")

try:
    from isaacsim import SimulationApp
except Exception:  # allows static inspection outside Isaac Sim
    SimulationApp = None

# Bound after SimulationApp() starts.
omni = None
Gf = None
Sdf = None
UsdGeom = None
UsdLux = None
UsdShade = None


# =====================================================================
# Arguments
# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone e2 industrial-domain evaluation for source-trained UAV LLM/VSLAM policy.")

    # Runtime
    p.add_argument("--headless", action="store_true")
    p.add_argument("--mode", type=str, default="eval", choices=["eval", "demo"], help="Evaluation-only script. eval loads a trained source-domain model; demo uses a neutral policy for scene/debug.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--renderer", type=str, default="RayTracedLighting")
    p.add_argument(
        "--render-step-interval",
        type=int,
        default=5,
        help="GUI update interval in simulation steps when rendering/training. Use 1 for smooth demo, 5-20 for safer PPO GUI training.",
    )
    p.add_argument(
        "--viewer-mode",
        type=str,
        default="industrial",
        choices=["industrial", "plant", "follow", "top", "front", "downcam"],
        help="Viewport camera mode: industrial/plant=complete e2 industrial-site view, follow=follow UAV, top=top-down map, front=front inspection view, downcam=downward UAV camera-like view.",
    )
    p.add_argument("--render-train", action="store_true",
                   help="Accepted for compatibility with old commands; ignored because this e2 file is evaluation-only.")
    p.add_argument("--aggressive-rtx-settings", action="store_true",
                   help="Enable extra RTX tonemap/color-correction settings. Default safe mode avoids black/white strip crashes.")
    p.add_argument("--scene-only", action="store_true",
                   help="Open the Isaac GUI, build the e2 industrial environment, and keep the viewer alive without PPO rollout/evaluation.")
    p.add_argument("--scene-preview-seconds", type=float, default=30.0,
                   help="How long to keep the GUI alive when --scene-only is used.")

    # Evaluation
    p.add_argument("--total-timesteps", type=int, default=0, help="Ignored; this e2 script does not train.")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--model-path", type=str, default="")

    # Environment
    p.add_argument("--max-episode-steps", type=int, default=9000)
    p.add_argument("--inspection-reach-radius", type=float, default=2.80)
    p.add_argument("--waypoint-reach-xy", type=float, default=3.20,
                   help="XY radius for accepting a waypoint. Fixes hover at a waypoint when altitude/safety detour keeps 3D distance slightly above the old threshold.")
    p.add_argument("--waypoint-reach-z", type=float, default=6.00,
                   help="Z tolerance for accepting a waypoint after XY inspection coverage is achieved.")
    p.add_argument("--auto-skip-stuck-waypoint", action="store_true", default=True,
                   help="If the UAV is nearly stationary near a waypoint for many steps, mark it inspected and move to the next waypoint instead of hovering forever.")
    p.add_argument("--world-size", type=float, default=80.0,
                   help="Half-extent of the site boundary. Enlarged so the expanded multi-district plant (≈2x footprint) fits inside the fence. Observations are normalized by max_ray_range, not world-size, so transfer scaling is unaffected.")
    p.add_argument("--num-rays", type=int, default=36)
    p.add_argument(
        "--route-repeat-count",
        type=int,
        default=4,
        help=(
            "Requested sweep-density/count kept for compatibility with old commands. "
            "By default the episode uses ONE 16-point lawnmower route and resets after completion. "
            "Use --repeat-full-route only if you intentionally want to repeat the full route multiple times."
        ),
    )
    p.add_argument(
        "--repeat-full-route",
        action="store_true",
        help="Old behavior: repeat the full 16-point lawnmower route route-repeat-count times before episode reset.",
    )
    p.add_argument("--disable-domain-randomization", action="store_true")
    p.add_argument("--complete-route-before-timeout", action="store_true", default=True,
                   help="Keep an episode alive until the full inspection route is completed. Enabled by default for paper/demo runs.")
    p.add_argument("--stop-at-timeout", action="store_true",
                   help="Old behavior: allow max_episode_steps timeout to truncate before the full route is completed.")

    # Proxy / cuVSLAM
    p.add_argument("--slam-mode", type=str, default="proxy", choices=["proxy", "gt", "cuvslam"])
    p.add_argument("--cuvslam-odom-topic", type=str, default="/visual_slam/tracking/odometry")
    p.add_argument("--cuvslam-odom-udp-host", type=str, default="0.0.0.0")
    p.add_argument("--cuvslam-odom-udp-port", type=int, default=14555)
    p.add_argument("--slam-drift-pos-per-sec", type=float, default=0.03)
    p.add_argument("--slam-pos-noise-std", type=float, default=0.03)
    p.add_argument("--slam-yaw-noise-std", type=float, default=0.015)
    p.add_argument("--slam-vel-noise-std", type=float, default=0.02)
    p.add_argument("--slam-tracking-loss-prob", type=float, default=0.008)
    p.add_argument("--slam-quality-recover-rate", type=float, default=0.06)
    p.add_argument("--depth-noise-std", type=float, default=0.02)

    # Output
    p.add_argument("--output-root", type=str, default="~/uav_results")
    p.add_argument("--disable-figures", action="store_true")
    p.add_argument("--capture-rgb", action="store_true", help="Force real Isaac/Replicator RGB capture for front and down cameras.")
    p.add_argument("--disable-real-rgb", action="store_true",
                   help="Disable Isaac/Replicator RGB capture and use the lightweight analytic fallback only.")
    p.add_argument("--capture-every-episode", type=int, default=1, help="Save inspection artifacts every N episodes.")
    p.add_argument("--show-capture-replay", action="store_true",
                   help="Debug only: keep the UAV visible while replaying recorded poses for RGB capture. Default hides it to avoid reset/capture flicker in the GUI.")
    p.add_argument("--heatmap-grid-size", type=int, default=128, help="Grid size for saved visual-feature heatmaps and paper figures.")
    p.add_argument("--append-metrics", action="store_true", help="Append to existing metric CSV files instead of starting a fresh run table.")

    # Speed bounds and governor
    p.add_argument("--speed-min", type=float, default=0.30)
    p.add_argument("--speed-max", type=float, default=2.00)
    p.add_argument("--governor-alpha", type=float, default=0.35)
    p.add_argument("--governor-beta", type=float, default=0.12)
    p.add_argument("--governor-rate-limit", type=float, default=0.50)
    p.add_argument("--vslam-risk-limit", type=float, default=0.86)
    p.add_argument("--abort-risk-limit", type=float, default=0.98)
    p.add_argument("--corridor-limit", type=float, default=4.00)
    p.add_argument("--abort-corridor-limit", type=float, default=8.00)
    p.add_argument("--yaw-limit-deg", type=float, default=85.0)
    p.add_argument("--abort-yaw-limit-deg", type=float, default=145.0)
    p.add_argument("--fallback-speed", type=float, default=0.45)

    # E2 evaluation baseline selector
    p.add_argument(
        "--baseline",
        type=str,
        default="ppo",
        choices=["ppo", "proposed", "fixed_governor", "fixed_no_governor", "vslam_heuristic", "pid"],
        help=(
            "E2 evaluator mode. ppo/proposed loads the source-domain PPO checkpoint. "
            "fixed_governor uses constant raw speed plus the same command governor. "
            "fixed_no_governor disables the governor. vslam_heuristic uses v=vmax*(1-z_t). "
            "pid uses a classical risk-feedback speed scheduler."
        ),
    )
    p.add_argument("--fixed-speed", type=float, default=1.00,
                   help="Constant raw speed for fixed_governor and fixed_no_governor baselines.")
    p.add_argument("--heuristic-speed-scale", type=float, default=1.00,
                   help="Scale for VSLAM heuristic speed: scale*speed_max*(1-z_t).")
    p.add_argument("--heuristic-min-speed", type=float, default=0.30,
                   help="Minimum raw speed for VSLAM heuristic baseline before the governor.")
    p.add_argument("--pid-z-ref", type=float, default=0.25,
                   help="Target VSLAM risk for PID speed scheduler.")
    p.add_argument("--pid-kp", type=float, default=0.75)
    p.add_argument("--pid-ki", type=float, default=0.03)
    p.add_argument("--pid-kd", type=float, default=0.18)

    # Mission encoder
    p.add_argument("--mission-text", type=str, default="E:low-texture G:coverage S:feature-slow O:slow-smooth")
    p.add_argument("--mission-encoder", type=str, default="structured", choices=["structured", "llama_hf"])
    p.add_argument("--mission-dim", type=int, default=16)
    p.add_argument("--hf-llama-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--hf-local-files-only", action="store_true")
    p.add_argument("--hf-load-in-4bit", action="store_true")
    p.add_argument("--llm-max-new-tokens", type=int, default=96)

    # Altitude / path safety
    p.add_argument("--inspection-altitude", type=float, default=14.0)
    p.add_argument("--inspection-altitude-max", type=float, default=22.0)
    p.add_argument("--enable-roof-climb-bias", action="store_true")
    p.add_argument("--chimney-safety-radius", type=float, default=5.00)
    p.add_argument("--chimney-safety-height-margin", type=float, default=3.00)
    p.add_argument("--path-clearance-margin", type=float, default=2.50)
    p.add_argument("--obstacle-avoidance-gain", type=float, default=2.40)
    p.add_argument("--obstacle-avoidance-range", type=float, default=9.00)
    p.add_argument("--yaw-gain", type=float, default=2.15)
    p.add_argument("--velocity-memory", type=float, default=0.58)

    # CVaR / energy
    p.add_argument("--cvar-alpha", type=float, default=0.90)
    p.add_argument("--drift-threshold", type=float, default=0.50)
    p.add_argument("--coverage-radius", type=float, default=4.20)
    p.add_argument("--localization-threshold", type=float, default=0.10)
    p.add_argument("--motor-hover-power-w", type=float, default=180.0)
    p.add_argument("--motor-speed-power-gain-w", type=float, default=32.0)
    args = p.parse_args()
    if getattr(args, "stop_at_timeout", False):
        args.complete_route_before_timeout = False
    return args


# =====================================================================
# Mission encoder: T -> (m, lambda), once per episode
# =====================================================================
class MissionEncoder:
    ENV_TOKENS = [
        "nominal", "low-texture", "low-light", "narrow", "wind",
        "repetitive", "stable", "target", "stress", "target-stress",
    ]
    GOAL_TOKENS = [
        "route", "coverage", "view-align", "route-match", "route-stable",
        "drift-min", "safe-finish", "transfer",
    ]
    SAFETY_TOKENS = [
        "monitor", "feature-slow", "loss-hover", "yaw-strict", "deviation-slow",
        "hover-abort", "early-abort", "fail-safe", "std-bounds",
    ]
    OP_TOKENS = [
        "adaptive", "slow-smooth", "cautious", "rate-limit", "adaptive-slow",
        "crawl", "fast", "smooth", "slow-hover", "minimal",
    ]
    DEFAULT_RISK = {
        "monitor": 0.50,
        "feature-slow": 0.78,
        "loss-hover": 0.82,
        "yaw-strict": 0.85,
        "deviation-slow": 0.72,
        "hover-abort": 0.95,
        "early-abort": 0.92,
        "fail-safe": 0.96,
        "std-bounds": 0.30,
    }

    def __init__(
        self,
        backend: str = "structured",
        model_name: str = "",
        mission_dim: int = 16,
        hf_token: str = "",
        local_files_only: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 96,
    ) -> None:
        self.backend = str(backend or "structured").lower()
        self.model_name = str(model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.mission_dim = int(max(16, mission_dim))
        self.hf_token = str(hf_token or os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
        self.local_files_only = bool(local_files_only)
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max(8, max_new_tokens))
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.loaded = False
        self.last_parsed: Dict[str, Any] = {}
        if self.backend == "llama_hf":
            self._try_load_llama()

    @staticmethod
    def _code(token: str, vocab: List[str]) -> float:
        token = str(token or "").strip().lower()
        return float(vocab.index(token)) / float(max(len(vocab) - 1, 1)) if token in vocab else 0.0

    @staticmethod
    def _field(text: str, key: str, default: str) -> str:
        import re
        m = re.compile(rf"(?:^|\s){re.escape(key)}\s*:\s*([A-Za-z0-9_-]+)", re.IGNORECASE).search(text or "")
        return str(m.group(1)).lower() if m else default

    def _parse_structured(self, text: str) -> Dict[str, Any]:
        text = str(text or "")
        env = self._field(text, "E", "nominal")
        goal = self._field(text, "G", "route")
        safety = self._field(text, "S", "monitor")
        op = self._field(text, "O", "adaptive")
        env = env if env in self.ENV_TOKENS else "nominal"
        goal = goal if goal in self.GOAL_TOKENS else "route"
        safety = safety if safety in self.SAFETY_TOKENS else "monitor"
        op = op if op in self.OP_TOKENS else "adaptive"
        risk = float(self.DEFAULT_RISK.get(safety, 0.50))
        if env in ("low-texture", "low-light", "narrow", "repetitive"):
            risk = max(risk, 0.78)
        if env in ("stress", "target-stress"):
            risk = max(risk, 0.93)
        if env == "wind":
            risk = max(risk, 0.72)
        if op in ("crawl", "slow-hover", "minimal"):
            risk = max(risk, 0.88)
        if op == "fast" and safety == "std-bounds":
            risk = min(risk, 0.35)
        return {"E": env, "G": goal, "S": safety, "O": op, "lambda": float(np.clip(risk, 0.0, 1.0))}

    def _try_load_llama(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            kw: Dict[str, Any] = {"local_files_only": self.local_files_only}
            if self.hf_token:
                kw["token"] = self.hf_token
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kw)
            mkw: Dict[str, Any] = dict(kw, device_map="auto", torch_dtype="auto")
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    mkw.pop("torch_dtype", None)
                    mkw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                except Exception as exc:
                    print(f"[LLaMA] 4-bit unavailable: {exc}")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **mkw)
            self.loaded = True
            print(f"[LLaMA] Loaded mission encoder: {self.model_name}")
        except Exception as exc:
            self.loaded = False
            self.tokenizer = None
            self.model = None
            print(f"[LLaMA] Falling back to structured parser. Error: {exc}")

    @staticmethod
    def _first_json(text: str) -> Optional[dict]:
        s = str(text or "")
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            return None
        try:
            return json.loads(s[a:b + 1])
        except Exception:
            return None

    def _parse_with_llama(self, text: str) -> Dict[str, Any]:
        base = self._parse_structured(text)
        if not self.loaded:
            return base
        try:
            prompt = (
                "Convert this UAV inspection mission into exactly one JSON object.\n"
                f"Allowed E: {', '.join(self.ENV_TOKENS)}.\n"
                f"Allowed G: {', '.join(self.GOAL_TOKENS)}.\n"
                f"Allowed S: {', '.join(self.SAFETY_TOKENS)}.\n"
                f"Allowed O: {', '.join(self.OP_TOKENS)}.\n"
                'Return only JSON like {"E":"low-texture","G":"coverage","S":"feature-slow","O":"slow-smooth","lambda":0.78}.\n'
                f"Mission: {text}\nJSON:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            try:
                dev = next(self.model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items()}
            except Exception:
                pass
            plen = int(inputs["input_ids"].shape[1])
            pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max(48, self.max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            text_out = self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
            parsed = self._first_json(text_out) or self._first_json(self.tokenizer.decode(out[0], skip_special_tokens=True))
            if not parsed:
                return base
            checked = {
                "E": str(parsed.get("E", base["E"])).strip().lower(),
                "G": str(parsed.get("G", base["G"])).strip().lower(),
                "S": str(parsed.get("S", base["S"])).strip().lower(),
                "O": str(parsed.get("O", base["O"])).strip().lower(),
                "lambda": float(np.clip(float(parsed.get("lambda", base["lambda"])), 0.0, 1.0)),
            }
            for k, vocab in (("E", self.ENV_TOKENS), ("G", self.GOAL_TOKENS), ("S", self.SAFETY_TOKENS), ("O", self.OP_TOKENS)):
                if checked[k] not in vocab:
                    checked[k] = base[k]
            return checked
        except Exception as exc:
            print(f"[LLaMA] parse failed ({type(exc).__name__}); using structured parser.")
            return base

    def _project(self, parsed: Dict[str, Any]) -> np.ndarray:
        e = self._code(parsed.get("E"), self.ENV_TOKENS)
        g = self._code(parsed.get("G"), self.GOAL_TOKENS)
        s = self._code(parsed.get("S"), self.SAFETY_TOKENS)
        o = self._code(parsed.get("O"), self.OP_TOKENS)
        lam = float(np.clip(parsed.get("lambda", 0.5), 0.0, 1.0))
        blocks = np.array(
            [
                [e, 1.0 - e, lam, 0.25 + 0.75 * e],
                [g, 1.0 - g, 0.5 * lam + 0.5 * g, 0.25 + 0.75 * g],
                [s, 1.0 - s, lam, 0.25 + 0.75 * s],
                [o, 1.0 - o, 1.0 - lam, 0.25 + 0.75 * o],
            ],
            dtype=np.float32,
        ).reshape(-1)
        if self.mission_dim == 16:
            return blocks
        out = np.zeros((self.mission_dim,), dtype=np.float32)
        out[: min(self.mission_dim, 16)] = blocks[: min(self.mission_dim, 16)]
        return out

    def encode(self, text: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        parsed = self._parse_with_llama(text) if self.backend == "llama_hf" else self._parse_structured(text)
        self.last_parsed = dict(parsed)
        lam = float(np.clip(parsed.get("lambda", 0.5), 0.0, 1.0))
        return self._project(parsed).astype(np.float32), lam, dict(parsed)


# =====================================================================
# cuVSLAM odometry receiver: ROS2 or UDP JSON fallback
# =====================================================================
class CuvslamOdomReceiver:
    def __init__(self, odom_topic: str, udp_host: str = "0.0.0.0", udp_port: int = 14555):
        self.odom_topic = str(odom_topic)
        self.udp_host = str(udp_host)
        self.udp_port = int(udp_port)
        self.latest: Optional[dict] = None
        self.latest_time = 0.0
        self._lock = threading.Lock()
        self._rclpy = None
        self._node = None
        self._spin = None
        self._sock = None
        self._start_udp()
        self._start_rclpy()

    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _start_udp(self) -> None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.udp_host, self.udp_port))
            s.setblocking(False)
            self._sock = s
            print(f"[CUVSLAM] UDP odom on {self.udp_host}:{self.udp_port}")
        except Exception as exc:
            print(f"[CUVSLAM] UDP receiver failed: {exc}")

    def _start_rclpy(self) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
        except Exception as exc:
            print(f"[CUVSLAM] rclpy unavailable, UDP only: {exc}")
            return
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            node = rclpy.create_node("uav_cuvslam_odom_receiver")
            node.create_subscription(Odometry, self.odom_topic, self._cb, 10)
            self._rclpy, self._node = rclpy, node
            self._spin = threading.Thread(target=self._loop, daemon=True)
            self._spin.start()
            print(f"[CUVSLAM] subscribed {self.odom_topic}")
        except Exception as exc:
            print(f"[CUVSLAM] subscriber failed: {exc}")
            self._rclpy = None
            self._node = None

    def _loop(self) -> None:
        while self._rclpy and self._node:
            try:
                self._rclpy.spin_once(self._node, timeout_sec=0.05)
            except Exception:
                time.sleep(0.05)

    def _cb(self, msg: Any) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self._store([p.x, p.y, p.z], [q.x, q.y, q.z, q.w], [v.x, v.y, v.z], time.time())

    def _store(self, pos: List[float], quat: List[float], vel: List[float], stamp: float) -> None:
        d = {
            "pos": np.array(pos, np.float32),
            "vel": np.array(vel, np.float32),
            "yaw": float(self.quat_to_yaw(*quat)),
            "stamp": float(stamp),
        }
        with self._lock:
            self.latest = d
            self.latest_time = time.time()

    def _poll(self) -> None:
        if self._sock is None:
            return
        while True:
            try:
                pkt, _ = self._sock.recvfrom(4096)
            except Exception:
                break
            try:
                d = json.loads(pkt.decode("utf-8"))
                self._store(d.get("pos", [0, 0, 0]), d.get("quat", [0, 0, 0, 1]), d.get("vel", [0, 0, 0]), d.get("stamp", time.time()))
            except Exception:
                continue

    def get_latest(self, max_age: float = 0.5) -> Optional[dict]:
        self._poll()
        with self._lock:
            if self.latest is None or (time.time() - self.latest_time) > max_age:
                return None
            return dict(self.latest)

    def close(self) -> None:
        try:
            if self._node:
                self._node.destroy_node()
        except Exception:
            pass
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass


# =====================================================================
# Environment
# =====================================================================
class IndustrialE2EvalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, args: argparse.Namespace, render_sim: bool = True):
        self.args = args
        self.render_sim = bool(render_sim)
        self.dt = 0.08
        self.rng = np.random.default_rng(args.seed)
        self.aggressive_rtx_settings = bool(getattr(args, "aggressive_rtx_settings", False))
        self.render_step_interval = int(max(1, getattr(args, "render_step_interval", 5)))
        self.viewer_mode = str(getattr(args, "viewer_mode", "industrial") or "industrial").lower()
        # During real RGB export, the code replays recorded trajectory poses into
        # the camera render products.  If the drone mesh remains visible in the
        # main viewport, Isaac's temporal renderer shows the UAV jumping through
        # many old poses during episode reset.  Keep replay invisible by default.
        self.show_capture_replay = bool(getattr(args, "show_capture_replay", False))
        self._capture_replay_active = False

        # Core environment
        self.max_episode_steps = int(args.max_episode_steps)
        self.inspection_reach_radius = float(max(0.25, args.inspection_reach_radius))
        self.waypoint_reach_xy = float(max(0.35, getattr(args, "waypoint_reach_xy", self.inspection_reach_radius)))
        self.waypoint_reach_z = float(max(0.35, getattr(args, "waypoint_reach_z", 4.50)))
        self.auto_skip_stuck_waypoint = bool(getattr(args, "auto_skip_stuck_waypoint", True))
        self._near_waypoint_steps = 0
        self.complete_route_before_timeout = bool(args.complete_route_before_timeout)
        self.world_size = float(args.world_size)
        self.num_rays = int(args.num_rays)
        self.max_ray_range = 22.0
        # route_repeat_count is kept for CLI compatibility, but the default
        # paper/demo behavior is now one complete inspection route per episode.
        # This prevents the visible trajectory from looping over the same 16
        # points again and again.  Enable --repeat-full-route only when a long
        # repeated-route stress test is explicitly desired.
        self.route_repeat_count = int(max(1, args.route_repeat_count))
        self.repeat_full_route = bool(getattr(args, "repeat_full_route", False))
        self.effective_route_loops = int(self.route_repeat_count if self.repeat_full_route else 1)
        self.domain_randomization = not bool(args.disable_domain_randomization)

        # Altitude and path safety
        self.no_fly_z_min, self.no_fly_z_max = 0.35, 42.0
        self.inspection_altitude = float(max(1.2, args.inspection_altitude))
        self.inspection_altitude_max = float(max(2.0, args.inspection_altitude_max))
        self.enable_roof_climb_bias = bool(args.enable_roof_climb_bias)
        self.chimney_safety_radius = float(max(1.0, args.chimney_safety_radius))
        self.chimney_safety_height_margin = float(max(0.0, args.chimney_safety_height_margin))
        self.path_clearance_margin = float(max(0.25, args.path_clearance_margin))
        self.drone_collision_radius = 0.42

        # Speed and policy-authority bounds
        self.osd_vmin = float(args.speed_min)
        self.osd_vmax = float(max(args.speed_max, args.speed_min + 1e-3))

        # Camera field of view
        self.camera_hfov = math.radians(84.0)
        self.camera_vfov = 2.0 * math.atan(math.tan(0.5 * self.camera_hfov) * 0.75)
        self.target_radius = 0.35

        # Mission context
        self.mission_text = str(args.mission_text or "E:nominal G:route S:monitor O:adaptive")
        self.mission_dim = int(max(16, args.mission_dim))
        self.mission_encoder = MissionEncoder(
            backend=args.mission_encoder,
            model_name=args.hf_llama_model,
            mission_dim=self.mission_dim,
            hf_token=args.hf_token,
            local_files_only=args.hf_local_files_only,
            load_in_4bit=args.hf_load_in_4bit,
            max_new_tokens=args.llm_max_new_tokens,
        )
        self.mission_vector, self.mission_risk_lambda, self.mission_metadata = self.mission_encoder.encode(self.mission_text)
        print(f"[MISSION] parsed={self.mission_metadata} lambda={self.mission_risk_lambda:.3f}")

        # Paper governor parameters
        self.governor_alpha = float(np.clip(args.governor_alpha, 0.0, 2.0))
        self.governor_beta = float(np.clip(args.governor_beta, 0.0, 2.0))
        self.governor_rate_limit = float(max(1e-3, args.governor_rate_limit))
        self.vslam_risk_limit = float(np.clip(args.vslam_risk_limit, 0.0, 1.0))
        self.abort_risk_limit = float(np.clip(args.abort_risk_limit, 0.0, 1.0))
        self.corridor_limit = float(max(0.10, args.corridor_limit))
        self.abort_corridor_limit = float(max(self.corridor_limit, args.abort_corridor_limit))
        self.yaw_limit = math.radians(float(args.yaw_limit_deg))
        self.abort_yaw_limit = math.radians(float(args.abort_yaw_limit_deg))
        self.fallback_speed = float(max(0.0, args.fallback_speed))

        # Paper reward/shaping/drift coefficients
        self.gamma = 0.99
        # Reward/scoring weights are intentionally balanced for paper-style
        # route-completion learning: progress and coverage dominate, while
        # governor/corridor/yaw penalties remain visible but no longer drown out
        # successful inspection episodes.
        self.w_p, self.w_c, self.w_d, self.w_y, self.w_s = 2.0, 18.0, 0.18, 0.08, 0.03
        self.c_delta, self.c_d, self.c_phi = 4.0, 0.18, 0.08
        self.k_d, self.k_phi, self.k_ell = 0.35, 0.20, 1.20
        self.cvar_alpha = float(np.clip(args.cvar_alpha, 0.50, 0.999))
        self.reach_bonus, self.success_bonus, self.collision_penalty = 20.0, 150.0, 50.0

        # SLAM proxy
        self.slam_mode = str(args.slam_mode)
        self.slam_drift_pos_per_sec = float(args.slam_drift_pos_per_sec)
        self.slam_pos_noise_std = float(args.slam_pos_noise_std)
        self.slam_yaw_noise_std = float(args.slam_yaw_noise_std)
        self.slam_vel_noise_std = float(args.slam_vel_noise_std)
        self.slam_tracking_loss_prob = float(args.slam_tracking_loss_prob)
        self.slam_quality_recover_rate = float(args.slam_quality_recover_rate)
        self.depth_noise_std = float(args.depth_noise_std)
        self.drift_threshold = float(args.drift_threshold)
        self.coverage_radius = float(args.coverage_radius)
        self.localization_threshold = float(args.localization_threshold)
        self.motor_hover_power_w = float(args.motor_hover_power_w)
        self.motor_speed_power_gain_w = float(args.motor_speed_power_gain_w)
        self.cuvslam_receiver: Optional[CuvslamOdomReceiver] = None

        # Low-level tracking
        self.obstacle_avoidance_gain = float(max(0.0, args.obstacle_avoidance_gain))
        self.obstacle_avoidance_range = float(max(0.05, args.obstacle_avoidance_range))
        self.yaw_gain = float(max(0.1, args.yaw_gain))
        self.velocity_memory = float(np.clip(args.velocity_memory, 0.0, 0.98))

        # E2 baseline mode.  Only the speed proposal changes; scene, route,
        # VSLAM proxy, governor thresholds, and metrics remain identical.
        self.baseline = str(getattr(args, "baseline", "ppo") or "ppo").lower()
        if self.baseline == "proposed":
            self.baseline = "ppo"
        self.fixed_speed_mps = float(np.clip(getattr(args, "fixed_speed", 1.0), self.osd_vmin, self.osd_vmax))
        self.heuristic_speed_scale = float(max(0.0, getattr(args, "heuristic_speed_scale", 1.0)))
        self.heuristic_min_speed = float(np.clip(getattr(args, "heuristic_min_speed", self.osd_vmin), self.osd_vmin, self.osd_vmax))
        self.pid_z_ref = float(np.clip(getattr(args, "pid_z_ref", 0.25), 0.0, 1.0))
        self.pid_kp = float(getattr(args, "pid_kp", 0.75))
        self.pid_ki = float(getattr(args, "pid_ki", 0.03))
        self.pid_kd = float(getattr(args, "pid_kd", 0.18))
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        print(f"[E2_BASELINE] mode={self.baseline} fixed_speed={self.fixed_speed_mps:.2f} m/s")

        # Output
        self.output_root = Path(args.output_root).expanduser()
        self.metrics_dir = self.output_root / "metrics"
        self.figure_dir = self.output_root / "figures"
        self.trajectory_dir = self.output_root / "trajectories"
        for d in (self.metrics_dir, self.figure_dir, self.trajectory_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.save_figures = not bool(args.disable_figures)
        # Real RGB capture policy.  The user-visible downcam_rgb_ep*.png files
        # must be real camera captures when a GUI/renderer is active; otherwise
        # they fall back to a clearly non-feature-overlay camera projection.
        # This restores real-camera output for --render-train.  Use
        # --disable-real-rgb only if Replicator is unstable on your Isaac build.
        self.capture_rgb = bool(
            (not bool(getattr(args, "disable_real_rgb", False)))
            and (
                bool(args.capture_rgb)
                or args.mode in ("eval", "demo")
                or (self.render_sim and bool(getattr(args, "render_train", False)))
            )
        )
        if self.capture_rgb:
            print("[CAMERA_RGB] real Isaac/Replicator RGB capture enabled for saved front/downcam images.")
        else:
            print("[CAMERA_RGB] real RGB capture disabled; saved RGB will use non-overlay fallback projection.")
        self.capture_every_episode = max(1, int(args.capture_every_episode))
        self.heatmap_grid_size = max(16, int(getattr(args, "heatmap_grid_size", 128)))
        self.policy_analytic_blend = 0.55
        self.adaptive_speed_floor = 0.45
        self.route_start_xy = np.array([-30.0, -24.0], np.float32)
        self.route_altitude_dynamic = None
        self.episode_csv = self.metrics_dir / "paper_episode_metrics.csv"
        self.summary_json = self.metrics_dir / "paper_summary.json"
        self.result_table_csv = self.metrics_dir / "paper_result_table_row.csv"
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        # Start a clean metric table for each new run by default.  Otherwise
        # old conservative-governor rows stay in the CSV and make the summary
        # look bad even after the controller is fixed.  Use --append-metrics
        # only when you intentionally want to combine runs.
        if not bool(getattr(args, "append_metrics", False)):
            for old_file in (self.episode_csv, self.summary_json, self.result_table_csv):
                try:
                    if old_file.exists():
                        old_file.unlink()
                except Exception:
                    pass

        # Scene/state containers
        self.stage = None
        self.ops: Dict[str, Dict[str, object]] = {}
        self.materials: Dict[str, str] = {}
        self.smoke_puffs: List[dict] = []
        self.static_collision_boxes: List[dict] = []
        self.visual_feature_points: Optional[np.ndarray] = None
        self.texture_count, self.illumination_lux, self.wind_mps = 90, 700.0, 0.5
        self.metric_episode_id = 0
        self._drift_returns: List[float] = []

        # Vehicle state
        self.pos = np.zeros(3, np.float32)
        self.vel = np.zeros(3, np.float32)
        self.yaw = 0.0
        self.prev_action = np.zeros(1, np.float32)
        self._reset_governor_log()

        # Action: one raw scalar. Observation: [o_t; m; lambda]
        self.action_space = spaces.Box(low=np.array([-1.0], np.float32), high=np.array([1.0], np.float32), dtype=np.float32)
        self.obs_dim = 3 + 3 + 2 + 3 + 3 + 3 + 5 + self.num_rays + 1 + 1 + 2 + 5 + self.mission_dim + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Build route and scene
        self._build_route()
        self._build_scene()
        self._build_route()
        self.visual_feature_points = self._make_feature_points()
        if self.slam_mode == "cuvslam":
            self.cuvslam_receiver = CuvslamOdomReceiver(args.cuvslam_odom_topic, args.cuvslam_odom_udp_host, args.cuvslam_odom_udp_port)
        print("[ENV_INIT] e2 industrial transfer scene ready. reset() deferred.")

    # ----------------------------------------------------------------
    # Route: 16 inspection points; repeated lawnmower loops
    # ----------------------------------------------------------------
    def _scene_route_frame(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Estimate a safe low-altitude route frame from the built industrial scene.

        The route should stay low but above the main industrial roofs, while not
        being pulled up to the tall chimney heights.  We therefore estimate the
        footprint from main plant structures and compute altitude from the main
        roofline rather than from the smoke stacks.
        """
        mn = np.array([-20.0, -16.0, 0.0], dtype=np.float32)
        mx = np.array([20.0, 16.0, 12.0], dtype=np.float32)
        roof_top = 12.0
        if getattr(self, "static_collision_boxes", None):
            use_boxes = []
            for box in self.static_collision_boxes:
                path = str(box.get("path", "")).lower()
                if any(k in path for k in ("/world/fence", "safetyhalo", "smoke", "chimney", "stack", "coolingtower", "flare", "crane", "pole_", "wire", "bulb", "gassphere", "hvessel", "drum", "pipestack", "watertower")):
                    continue
                bmn = np.asarray(box["mn"], np.float32)
                bmx = np.asarray(box["mx"], np.float32)
                size = bmx - bmn
                # Ignore very thin/tall elements when estimating the main route frame.
                if float(size[2]) > 2.2 * max(float(size[0]), float(size[1])) and max(float(size[0]), float(size[1])) < 5.0:
                    continue
                use_boxes.append((bmn, bmx))
            if use_boxes:
                mins = np.stack([bmn for bmn, _ in use_boxes], axis=0)
                maxs = np.stack([bmx for _, bmx in use_boxes], axis=0)
                mn = np.min(mins, axis=0)
                mx = np.max(maxs, axis=0)
                roof_top = float(np.percentile(maxs[:, 2], 85))
        margin_x = float(np.clip(0.12 * max(mx[0] - mn[0], 1.0), 2.0, 5.5))
        margin_y = float(np.clip(0.12 * max(mx[1] - mn[1], 1.0), 2.0, 5.0))
        x0, x1 = float(mn[0] + margin_x), float(mx[0] - margin_x)
        y0, y1 = float(mn[1] + margin_y), float(mx[1] - margin_y)
        if x1 - x0 < 8.0:
            cx = 0.5 * (float(mn[0]) + float(mx[0]))
            x0, x1 = cx - 8.0, cx + 8.0
        if y1 - y0 < 8.0:
            cy = 0.5 * (float(mn[1]) + float(mx[1]))
            y0, y1 = cy - 8.0, cy + 8.0
        route_z = float(np.clip(max(self.inspection_altitude, roof_top + 2.2), roof_top + 1.6, self.inspection_altitude_max))
        return np.array([x0, y0], np.float32), np.array([x1, y1], np.float32), route_z

    def _inspection_altitude(self) -> float:
        base = self.route_altitude_dynamic if self.route_altitude_dynamic is not None else self.inspection_altitude
        return float(np.clip(base, 1.2, self.inspection_altitude_max))

    def _build_route(self) -> None:
        lo, hi, route_z = self._scene_route_frame()
        self.route_altitude_dynamic = float(route_z)
        z = self._inspection_altitude()
        # Adaptive lawnmower density: keep lane spacing near a target so the
        # sweep stays meaningful as the site grows. Small sites collapse to the
        # original 4x4 (16-point) grid; the enlarged site adds lanes.
        target_spacing = float(getattr(self, "route_lane_spacing", 16.0))
        span_x = float(hi[0] - lo[0])
        span_y = float(hi[1] - lo[1])
        nx = int(np.clip(round(span_x / target_spacing) + 1, 4, 7))
        ny = int(np.clip(round(span_y / target_spacing) + 1, 4, 7))
        self.route_grid = (nx, ny)
        self.route_points_per_pass = nx * ny
        xs = np.linspace(float(lo[0]), float(hi[0]), nx, dtype=np.float32)
        ys = np.linspace(float(lo[1]), float(hi[1]), ny, dtype=np.float32)
        lane_dy = float(ys[1] - ys[0]) if len(ys) > 1 else 4.0

        def sweep(y_offset: float, flip: bool) -> np.ndarray:
            pts: List[List[float]] = []
            row_ids = range(len(ys) - 1, -1, -1) if flip else range(len(ys))
            for ridx in row_ids:
                row_xs = xs if ((ridx % 2 == 0) ^ flip) else xs[::-1]
                for cidx, x in enumerate(row_xs):
                    pts.append([float(x), float(ys[ridx] + y_offset), float(z + 0.18 * (cidx % 2))])
            return np.asarray(pts, dtype=np.float32)

        self.inspection_points = sweep(0.0, flip=False)

        if bool(getattr(self, "repeat_full_route", False)):
            routes = []
            for k in range(self.route_repeat_count):
                frac = k - 0.5 * (self.route_repeat_count - 1)
                offset = 0.08 * lane_dy * frac
                routes.append(sweep(offset, flip=(k % 2 == 1)))
        else:
            # One inspection pass per episode.  The previous version used
            # route-repeat-count=4 as four full route repetitions, which is why
            # the heatmap showed the same lawnmower path drawn repeatedly.
            routes = [self.inspection_points.copy()]
        self.effective_route_loops = int(len(routes))
        raw_targets = np.concatenate(routes, axis=0).astype(np.float32)
        self.base_targets = self._sanitize_route_targets(raw_targets).astype(np.float32)
        self.targets = self.base_targets.copy()
        self.target_idx = 0
        self.target = self.targets[0].copy()
        self.route_start_xy = np.array([float(lo[0] - 4.0), float(lo[1] - 3.0)], np.float32)
        self._assert_route("INIT")
        print(
            f"[ROUTE_INIT] route_loops={self.effective_route_loops} requested_repeat={self.route_repeat_count} "
            f"repeat_full_route={int(bool(getattr(self, 'repeat_full_route', False)))} "
            f"grid={nx}x{ny} inspection_points={len(self.inspection_points)} drone_targets={len(self.targets)} "
            f"z={z:.2f} route_bbox=({lo[0]:.1f},{lo[1]:.1f})-({hi[0]:.1f},{hi[1]:.1f})"
        )

    def _sanitize_route_targets(self, pts: np.ndarray) -> np.ndarray:
        """Keep paper route points inspectable and away from tall safety halos.

        The previous route-frame estimator accidentally included the cooling
        towers in the lawnmower bounding box.  That produced a last point in
        the first row beside a tower/safety envelope, where the governor could
        hold the UAV forever.  This function keeps all points at inspection
        altitude, and gently pushes any point out of chimney/cooling-tower
        keep-out zones in XY while preserving the 16-point lawnmower order.
        """
        out = np.asarray(pts, dtype=np.float32).copy()
        if out.ndim != 2 or out.shape[1] != 3:
            return out
        z = float(self._inspection_altitude())
        out[:, 2] = z
        boxes = list(getattr(self, "static_collision_boxes", []) or [])
        if not boxes:
            return out
        for i in range(out.shape[0]):
            p = out[i].copy()
            for box in boxes:
                name = str(box.get("path", "")).lower()
                if not any(k in name for k in ("chimney", "safetyhalo", "coolingtower", "stack", "silo", "tank", "distillation", "flare", "crane", "gassphere", "hvessel", "watertower", "pipestack")):
                    continue
                mn = np.asarray(box.get("mn"), dtype=np.float32)
                mx = np.asarray(box.get("mx"), dtype=np.float32)
                if mn.shape != (3,) or mx.shape != (3,):
                    continue
                # Only keep away from boxes that vertically overlap the flight altitude.
                margin = max(float(getattr(self, "path_clearance_margin", 2.5)), 0.35 * float(getattr(self, "chimney_safety_radius", 5.0)))
                if not (mn[2] - margin <= z <= mx[2] + margin):
                    continue
                c = 0.5 * (mn[:2] + mx[:2])
                half = 0.5 * (mx[:2] - mn[:2]) + margin + float(getattr(self, "drone_collision_radius", 0.42))
                rel = p[:2] - c
                if abs(float(rel[0])) <= float(half[0]) and abs(float(rel[1])) <= float(half[1]):
                    # Move to the nearest outside face plus small clearance.
                    dx = float(half[0] - abs(float(rel[0])))
                    dy = float(half[1] - abs(float(rel[1])))
                    if dx < dy:
                        p[0] = c[0] + math.copysign(float(half[0] + 0.7), float(rel[0]) if abs(float(rel[0])) > 1e-6 else 1.0)
                    else:
                        p[1] = c[1] + math.copysign(float(half[1] + 0.7), float(rel[1]) if abs(float(rel[1])) > 1e-6 else 1.0)
                    p[2] = z
            out[i] = p
        return out

    def _assert_route(self, where: str) -> None:
        per_pass = int(getattr(self, "route_points_per_pass", 16))
        if len(self.inspection_points) != per_pass:
            raise RuntimeError(f"[ROUTE_ERROR_{where}] expected {per_pass} inspection points, got {len(self.inspection_points)}")
        expected = per_pass * int(getattr(self, "effective_route_loops", 1))
        if len(self.targets) != expected:
            raise RuntimeError(f"[ROUTE_ERROR_{where}] expected {expected} targets, got {len(self.targets)}")

    def _make_feature_points(self) -> np.ndarray:
        rng = np.random.default_rng(2026)
        pts: List[np.ndarray] = []

        # 1) Front/stereo landmarks around the logical inspection targets.
        for t in self.inspection_points:
            pts.append(t[None, :] + rng.normal(0, 1, (110, 3)).astype(np.float32) * np.array([0.70, 0.70, 0.45], np.float32))

            # 2) Downward-camera landmarks on the roof/ground surface below each
            # inspection target.  The previous version placed most features at
            # UAV altitude; a downward camera then saw almost no valid below-camera
            # features and the downcam heatmap became empty.
            surface_z = float(max(0.25, min(float(t[2]) - 2.6, self._inspection_altitude() - 1.8)))
            base = np.array([[float(t[0]), float(t[1]), surface_z]], np.float32)
            pts.append(base + rng.normal(0, 1, (95, 3)).astype(np.float32) * np.array([1.05, 1.05, 0.08], np.float32))

        # 3) Scene feature clouds on plant structures and ground markings.
        # Auto-fit the cloud to the actual built footprint (so the expanded
        # districts are covered) while holding ~constant per-area density.
        z0 = float(max(0.40, self._inspection_altitude() - 2.3))
        boxes_all = list(getattr(self, "static_collision_boxes", []) or [])
        if boxes_all:
            amn = np.min(np.stack([np.asarray(b["mn"], np.float32) for b in boxes_all], 0), 0)
            amx = np.max(np.stack([np.asarray(b["mx"], np.float32) for b in boxes_all], 0), 0)
            xlo, xhi = float(amn[0]) - 3.0, float(amx[0]) + 3.0
            ylo, yhi = float(amn[1]) - 3.0, float(amx[1]) + 3.0
        else:
            xlo, xhi, ylo, yhi = -28.0, 28.0, -24.0, 24.0
        nx = int(np.clip((xhi - xlo) / 4.5, 8, 48))
        ny = int(np.clip((yhi - ylo) / 4.5, 8, 48))
        for x in np.linspace(xlo, xhi, nx):
            for y in np.linspace(ylo, yhi, ny):
                if rng.random() < 0.60:
                    center = np.array([[x, y, z0 + rng.normal(0, 1.0)]], np.float32)
                    pts.append(center + rng.normal(0, 1, (18, 3)).astype(np.float32) * np.array([0.55, 0.55, 0.16], np.float32))

        # 4) Top-surface landmarks from registered plant collision boxes. These
        # make the bottom camera see roofs, tanks and pads during the full route.
        for bi, box in enumerate(getattr(self, "static_collision_boxes", [])[:320]):
            path = str(box.get("path", "")).lower()
            if any(k in path for k in ("/world/fence", "safetyhalo", "smoke", "wire", "pole_", "bulb", "flare", "crane")):
                continue
            mn = np.asarray(box.get("mn", [0, 0, 0]), dtype=np.float32)
            mx = np.asarray(box.get("mx", [0, 0, 0]), dtype=np.float32)
            sx, sy, sz = float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2])
            if sx < 0.25 or sy < 0.25 or sz < 0.20:
                continue
            n = int(np.clip(10 + 2.0 * max(sx, sy), 12, 42))
            xy = rng.uniform(mn[:2], mx[:2], size=(n, 2)).astype(np.float32)
            zz = np.full((n, 1), float(mx[2] + 0.05), dtype=np.float32)
            surf = np.concatenate([xy, zz], axis=1)
            surf += rng.normal(0, 1, surf.shape).astype(np.float32) * np.array([0.04, 0.04, 0.02], np.float32)
            pts.append(surf)

        out = np.concatenate(pts, axis=0).astype(np.float32)
        if out.shape[0] > 26000:
            out = out[rng.choice(out.shape[0], 26000, replace=False)]
        return out

    # ----------------------------------------------------------------
    # Scene construction
    # ----------------------------------------------------------------
    def _build_scene(self) -> None:
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        try:
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        except Exception:
            pass
        self._apply_photoreal_render_settings()
        self._lighting()
        self._materials()
        self._ground()
        self._industrial_site()
        self._drone()
        self._cameras()
        self._inspection_target_prim()
        for _ in range(20):
            omni.kit.app.get_app().update()
        self._update_viewport()

    def _apply_photoreal_render_settings(self) -> None:
        """Apply stable viewport settings.

        The previous version forced several RTX post-processing keys. On some
        Isaac Sim/driver combinations this can produce black/white striping or
        a crashed viewport. By default we now use the same safer style as the
        working reference file: hide the grid and leave RTX tone mapping mostly
        under Isaac's default control. Extra color/exposure settings are only
        enabled when --aggressive-rtx-settings is passed.
        """
        try:
            import carb
            s = carb.settings.get_settings()

            safe_settings = [
                ("/app/viewport/grid/enabled", False),
                ("/app/viewport/grid/showOrigin", False),
                # Avoid DLSS warnings/crashes when Isaac/Replicator creates
                # small offscreen render products for camera capture.
                # These keys are best-effort and are ignored if unsupported.
                ("/rtx/post/dlss/enabled", False),
                ("/rtx/post/aa/op", 0),
            ]
            for key, value in safe_settings:
                try:
                    s.set(key, value)
                except Exception:
                    pass

            if not bool(getattr(self, "aggressive_rtx_settings", False)):
                print("[VISUAL] safe viewport settings applied; RTX post-processing kept default.")
                return

            # Optional, best-effort visual enhancement. Use only if your Isaac
            # Sim build handles these RTX keys correctly.
            extra_settings = [
                ("/rtx/post/tonemap/enableAutoExposure", False),
                ("/rtx/post/histogram/enabled", False),
                ("/rtx/post/tonemap/exposure", 0.85),
                ("/rtx/post/tonemap/whitePoint", 1.30),
                ("/rtx/post/tonemap/saturation", 1.10),
                ("/rtx/post/tonemap/contrast", 1.03),
                ("/rtx/sceneDb/ambientLightIntensity", 0.35),
                ("/rtx/shadows/enabled", True),
            ]
            for key, value in extra_settings:
                try:
                    s.set(key, value)
                except Exception:
                    pass
            print("[VISUAL] aggressive RTX visual settings applied.")
        except Exception as exc:
            print(f"[VISUAL] render settings skipped: {exc}")

    def _lighting(self) -> None:
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/SkyDome")
        dome.GetIntensityAttr().Set(380.0)
        dome.GetColorAttr().Set(Gf.Vec3f(0.62, 0.72, 0.92))
        sun = UsdLux.DistantLight.Define(self.stage, "/World/Lights/MiddaySun")
        sun.GetIntensityAttr().Set(3600.0)
        sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.86))
        sun.GetAngleAttr().Set(0.45)
        UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate((-46.0, 25.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        fill = UsdLux.SphereLight.Define(self.stage, "/World/Lights/CoolFill")
        fill.GetIntensityAttr().Set(160.0)
        fill.GetRadiusAttr().Set(16.0)
        fill.GetColorAttr().Set(Gf.Vec3f(0.62, 0.72, 0.95))
        UsdGeom.XformCommonAPI(fill.GetPrim()).SetTranslate((-18.0, -18.0, 14.0))
        warm = UsdLux.SphereLight.Define(self.stage, "/World/Lights/WarmBounce")
        warm.GetIntensityAttr().Set(70.0)
        warm.GetRadiusAttr().Set(6.0)
        warm.GetColorAttr().Set(Gf.Vec3f(1.0, 0.9, 0.74))
        UsdGeom.XformCommonAPI(warm.GetPrim()).SetTranslate((14.0, 8.0, 2.5))

    def _mat(self, key: str, color: Tuple[float, float, float], rough: float = 0.85, metal: float = 0.0, opacity: float = 1.0) -> None:
        path = f"/World/Looks/{key}"
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        if opacity < 0.999:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self.materials[key] = path

    def _materials(self) -> None:
        self._mat("concrete", (0.50, 0.49, 0.45), 0.94)
        self._mat("concrete_dark", (0.36, 0.36, 0.34), 0.96)
        self._mat("reactor", (0.52, 0.50, 0.46), 0.88)
        self._mat("tower", (0.50, 0.49, 0.45), 0.95)
        self._mat("steel", (0.28, 0.29, 0.31), 0.50, 0.45)
        self._mat("galv", (0.43, 0.44, 0.45), 0.40, 0.70)
        self._mat("tank", (0.44, 0.46, 0.48), 0.34, 0.55)
        self._mat("blue", (0.20, 0.31, 0.42), 0.74)
        self._mat("green", (0.25, 0.35, 0.27), 0.76)
        self._mat("asphalt", (0.18, 0.18, 0.19), 0.94)
        self._mat("gravel", (0.34, 0.33, 0.30), 0.98)
        self._mat("white", (0.60, 0.59, 0.55), 0.78)
        self._mat("red", (0.62, 0.12, 0.10), 0.62)
        self._mat("yellow", (0.82, 0.62, 0.08), 0.55)
        self._mat("orange", (0.92, 0.39, 0.10), 0.60)
        self._mat("soot", (0.12, 0.115, 0.11), 0.90)
        self._mat("drone", (0.08, 0.08, 0.09), 0.24, 0.22)
        self._mat("lens", (0.01, 0.012, 0.015), 0.04)
        self._mat("glass", (0.05, 0.12, 0.16), 0.10, 0.0, opacity=0.55)
        self._mat("smoke", (0.78, 0.78, 0.76), 0.99, 0.0, opacity=0.06)
        # Extra realistic finishes for varied industrial equipment shapes.
        self._mat("tank_cream", (0.74, 0.72, 0.66), 0.62, 0.05)   # large field-storage tanks
        self._mat("tank_teal", (0.20, 0.42, 0.42), 0.55, 0.12)    # process / chemical vessels
        self._mat("alu", (0.72, 0.73, 0.74), 0.30, 0.85)          # insulated / aluminium cladding
        self._mat("sphere_tank", (0.80, 0.81, 0.82), 0.40, 0.30)  # Horton gas-sphere shell
        self._mat("rust", (0.46, 0.27, 0.17), 0.93, 0.05)         # weathering accent

    def _bind(self, prim: Any, key: str) -> None:
        path = self.materials.get(key)
        if not path:
            return
        mat = UsdShade.Material.Get(self.stage, path)
        if mat:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def _cube(self, path: str, size: Tuple[float, float, float], pos: Tuple[float, float, float], mat: str = "", collide: Optional[bool] = None) -> None:
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        prim = c.GetPrim()
        self._bind(prim, mat)
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(tuple(float(v) for v in pos))
        api.SetScale(tuple(float(v) for v in size))
        self.ops[path] = {"type": "cube", "prim": prim}
        if collide if collide is not None else self._should_collide(path, size, pos):
            self._register_box(path, pos, size, "cube")

    def _cyl(self, path: str, radius: float, height: float, pos: Tuple[float, float, float], mat: str = "", collide: Optional[bool] = None) -> None:
        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateRadiusAttr(float(radius))
        c.CreateHeightAttr(float(height))
        prim = c.GetPrim()
        self._bind(prim, mat)
        UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos))
        self.ops[path] = {"type": "cyl", "prim": prim}
        size = (2 * radius, 2 * radius, height)
        if collide if collide is not None else self._should_collide(path, size, pos):
            self._register_box(path, pos, size, "cyl")

    def _sph(self, path: str, radius: float, pos: Tuple[float, float, float], mat: str = "") -> None:
        s = UsdGeom.Sphere.Define(self.stage, path)
        s.CreateRadiusAttr(float(radius))
        prim = s.GetPrim()
        self._bind(prim, mat)
        UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos))
        self.ops[path] = {"type": "sph", "prim": prim}

    def _ground(self) -> None:
        self._cube("/World/Terrain", (self.world_size * 2.2, self.world_size * 2.2, 0.05), (0, 0, -0.05), "gravel", collide=False)
        # Main paved areas covering the enlarged (≈2x) footprint: north plant
        # cluster, south logistics district, and east storage annex.
        self._cube("/World/Apron", (120, 70, 0.05), (-2, -8, 0.0), "concrete", collide=False)
        self._cube("/World/Apron_South", (96, 36, 0.05), (-4, -40, 0.0), "concrete", collide=False)
        self._cube("/World/Apron_East", (40, 50, 0.05), (50, 4, 0.0), "concrete", collide=False)
        # Through roads.
        self._cube("/World/Road/EntryX", (150, 3.4, 0.04), (0, -28, 0.015), "asphalt", collide=False)
        self._cube("/World/Road/SpineX", (150, 3.4, 0.04), (0, -2, 0.015), "asphalt", collide=False)
        self._cube("/World/Road/ServiceY", (3.4, 110, 0.04), (-35, -10, 0.015), "asphalt", collide=False)
        self._cube("/World/Road/EastY", (3.4, 90, 0.04), (37, -2, 0.015), "asphalt", collide=False)
        for i, x in enumerate(np.linspace(-60, 60, 31)):
            self._cube(f"/World/Road/Line_{i}", (1.2, 0.08, 0.006), (float(x), -28, 0.05), "white", collide=False)

    def _building(self, base: str, center: Tuple[float, float, float], size: Tuple[float, float, float], mat: str = "concrete") -> None:
        x, y, z = center
        sx, sy, sz = size
        self._cube(base + "/Body", (sx, sy, sz), (x, y, z + sz / 2), mat)
        self._cube(base + "/Roof", (sx * 1.03, sy * 1.03, 0.25), (x, y, z + sz + 0.12), "steel", collide=False)
        # Windows / vents make scene non-white and give VSLAM texture.
        for i, dx in enumerate(np.linspace(-sx * 0.36, sx * 0.36, 5)):
            self._cube(f"{base}/WindowF_{i}", (0.55, 0.05, 0.45), (x + dx, y - sy * 0.51, z + sz * 0.55), "glass", collide=False)
            self._cube(f"{base}/WindowB_{i}", (0.55, 0.05, 0.45), (x + dx, y + sy * 0.51, z + sz * 0.55), "glass", collide=False)
        for i, dx in enumerate(np.linspace(-sx * 0.32, sx * 0.32, 4)):
            self._cube(f"{base}/Vent_{i}", (0.85, 0.08, 0.30), (x + dx, y - sy * 0.52, z + sz * 0.83), "galv", collide=False)
        for i, dx in enumerate(np.linspace(-sx * 0.25, sx * 0.25, 3)):
            self._cube(f"{base}/RoofUnit_{i}", (0.95, 0.80, 0.55), (x + dx, y, z + sz + 0.42), "steel", collide=True)

    # ----------------------------------------------------------------
    # Varied, realistic building shapes (gable / sawtooth / arched /
    # stepped / mono-pitch).  Each registers a single collision envelope
    # that includes the roof peak; roof decoration is non-colliding.
    # ----------------------------------------------------------------
    def _wall_windows(self, base: str, center, size, z, rows: int = 1) -> None:
        x, y, _z = center
        sx, sy, sz = size
        for r in range(rows):
            zz = z + sz * (0.35 + 0.40 * r / max(1, rows))
            for i, dx in enumerate(np.linspace(-sx * 0.36, sx * 0.36, 6)):
                self._cube(f"{base}/WinF_{r}_{i}", (0.55, 0.05, 0.42), (x + dx, y - sy * 0.51, zz), "glass", collide=False)
                self._cube(f"{base}/WinB_{r}_{i}", (0.55, 0.05, 0.42), (x + dx, y + sy * 0.51, zz), "glass", collide=False)

    def _gable_building(self, base: str, center, size, mat: str = "concrete", ridge_axis: str = "x", door: bool = True) -> None:
        """Rectangular hall with a pitched (gable) roof."""
        x, y, z = center
        sx, sy, sz = size
        self._cube(base + "/Body", (sx, sy, sz), (x, y, z + sz / 2), mat, collide=False)
        self._wall_windows(base, center, size, z, rows=1)
        span = sy if ridge_axis == "x" else sx
        rise = float(np.clip(0.32 * span, 1.2, 3.2))
        ang = math.degrees(math.atan2(rise, span / 2.0))
        slope_len = math.hypot(span / 2.0, rise)
        ztop = z + sz
        if ridge_axis == "x":
            for s, sign in enumerate((-1.0, 1.0)):
                panel = base + f"/Roof_{s}"
                self._cube(panel, (sx * 1.04, slope_len, 0.16), (x, y + sign * span / 4.0, ztop + rise / 2.0), "steel", collide=False)
                UsdGeom.XformCommonAPI(self.ops[panel]["prim"]).SetRotate(Gf.Vec3f(-sign * ang, 0.0, 0.0))
            self._cube(base + "/Ridge", (sx * 1.05, 0.16, 0.16), (x, y, ztop + rise), "galv", collide=False)
        else:
            for s, sign in enumerate((-1.0, 1.0)):
                panel = base + f"/Roof_{s}"
                self._cube(panel, (slope_len, sy * 1.04, 0.16), (x + sign * span / 4.0, y, ztop + rise / 2.0), "steel", collide=False)
                UsdGeom.XformCommonAPI(self.ops[panel]["prim"]).SetRotate(Gf.Vec3f(0.0, sign * ang, 0.0))
            self._cube(base + "/Ridge", (0.16, sy * 1.05, 0.16), (x, y, ztop + rise), "galv", collide=False)
        if door:
            self._cube(base + "/Door", (min(2.4, sx * 0.4), 0.10, min(2.6, sz * 0.7)), (x, y - sy * 0.51, z + min(1.3, sz * 0.35)), "galv", collide=False)
        self._register_box(base + "/Envelope", (x, y, z + (sz + rise) / 2.0), (sx, sy, sz + rise), "building")

    def _sawtooth_building(self, base: str, center, size, mat: str = "concrete", teeth: int = 5) -> None:
        """Factory hall with a north-light sawtooth roof (angled panels + glazing)."""
        x, y, z = center
        sx, sy, sz = size
        self._cube(base + "/Body", (sx, sy, sz), (x, y, z + sz / 2), mat, collide=False)
        self._wall_windows(base, center, size, z, rows=2)
        rise = float(np.clip(0.16 * sy, 0.7, 1.6))
        unit = sy / float(teeth)
        ztop = z + sz
        for i in range(teeth):
            yc = y - sy / 2.0 + (i + 0.5) * unit
            panel = base + f"/Tooth_{i}"
            self._cube(panel, (sx * 1.02, unit * 1.04, 0.12), (x, yc, ztop + rise * 0.5), "steel", collide=False)
            UsdGeom.XformCommonAPI(self.ops[panel]["prim"]).SetRotate(Gf.Vec3f(-math.degrees(math.atan2(rise, unit)), 0.0, 0.0))
            self._cube(base + f"/Glaze_{i}", (sx * 1.02, 0.06, rise), (x, yc + unit * 0.5, ztop + rise * 0.5), "glass", collide=False)
        self._register_box(base + "/Envelope", (x, y, z + (sz + rise) / 2.0), (sx, sy, sz + rise), "building")

    def _arched_hangar(self, base: str, center, size, mat: str = "galv") -> None:
        """Hangar with a barrel-vault (half-cylinder) roof along its length."""
        x, y, z = center
        sx, sy, sz = size
        self._cube(base + "/Body", (sx, sy, sz), (x, y, z + sz / 2), mat, collide=False)
        self._wall_windows(base, center, size, z, rows=1)
        r = sy / 2.0
        roof = UsdGeom.Cylinder.Define(self.stage, base + "/Vault")
        roof.CreateRadiusAttr(float(r))
        roof.CreateHeightAttr(float(sx))
        prim = roof.GetPrim()
        self._bind(prim, "alu")
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate((float(x), float(y), float(z + sz)))
        api.SetRotate(Gf.Vec3f(0.0, 90.0, 0.0))  # lay barrel along X
        self.ops[base + "/Vault"] = {"type": "cyl", "prim": prim}
        for i, dx in enumerate(np.linspace(-sx * 0.42, sx * 0.42, 5)):
            self._cube(f"{base}/Rib_{i}", (0.10, sy * 1.02, 0.10), (x + dx, y, z + sz + r * 0.55), "steel", collide=False)
        self._cube(base + "/RollerDoor", (0.10, sy * 0.7, sz * 0.85), (x - sx * 0.51, y, z + sz * 0.45), "galv", collide=False)
        self._register_box(base + "/Envelope", (x, y, z + (sz + r) / 2.0), (sx, sy, sz + r), "building")

    def _stepped_building(self, base: str, center, size, mat: str = "green", tiers: int = 3) -> None:
        """Multi-tier office block with setbacks (ziggurat-like)."""
        x, y, z = center
        sx, sy, sz = size
        zc = z
        for t in range(tiers):
            f = 1.0 - 0.22 * t
            h = sz * (0.55 if t == 0 else 0.40)
            self._cube(f"{base}/Tier_{t}", (sx * f, sy * f, h), (x, y, zc + h / 2), mat, collide=False)
            for i, dx in enumerate(np.linspace(-sx * f * 0.34, sx * f * 0.34, 5)):
                self._cube(f"{base}/Win_{t}_{i}", (0.5, 0.05, 0.40), (x + dx, y - sy * f * 0.51, zc + h * 0.55), "glass", collide=False)
            zc += h
        self._cube(base + "/RoofPlant", (sx * 0.25, sy * 0.25, 0.5), (x, y, zc + 0.25), "steel", collide=False)
        self._register_box(base + "/Envelope", (x, y, z + zc * 0.5), (sx, sy, zc - z), "building")

    def _mono_pitch_building(self, base: str, center, size, mat: str = "concrete_dark") -> None:
        """Low utility shed with a single-slope (shed) roof."""
        x, y, z = center
        sx, sy, sz = size
        self._cube(base + "/Body", (sx, sy, sz), (x, y, z + sz / 2), mat, collide=False)
        self._wall_windows(base, center, size, z, rows=1)
        rise = float(np.clip(0.22 * sy, 0.6, 1.4))
        panel = base + "/Roof"
        slope_len = math.hypot(sy, rise)
        self._cube(panel, (sx * 1.03, slope_len, 0.14), (x, y, z + sz + rise * 0.5), "steel", collide=False)
        UsdGeom.XformCommonAPI(self.ops[panel]["prim"]).SetRotate(Gf.Vec3f(-math.degrees(math.atan2(rise, sy)), 0.0, 0.0))
        self._register_box(base + "/Envelope", (x, y, z + (sz + rise) / 2.0), (sx, sy, sz + rise), "building")

    def _cooling_tower(self, base: str, pos: Tuple[float, float, float], scale: float = 2.2) -> None:
        x, y, z = pos
        sections = [(1.70, 0.8, 0.40), (1.42, 1.6, 1.6), (1.05, 1.8, 3.3), (1.20, 1.4, 4.9), (1.42, 0.85, 6.05)]
        for i, (r, h, zc) in enumerate(sections):
            self._cyl(f"{base}/Sec_{i}", r * scale, h * scale, (x, y, z + zc * scale), "tower")
        for j, ang in enumerate(np.linspace(0, 2 * math.pi, 8, endpoint=False)):
            px = x + math.cos(ang) * 1.78 * scale
            py = y + math.sin(ang) * 1.78 * scale
            self._cube(f"{base}/Rib_{j}", (0.08, 0.08, 6.2 * scale), (px, py, z + 3.2 * scale), "concrete_dark", collide=False)

    def _tank(self, base: str, pos: Tuple[float, float, float], radius: float, height: float) -> None:
        x, y, z = pos
        self._cyl(base + "/Body", radius, height, (x, y, z + height / 2), "tank")
        self._sph(base + "/Top", radius * 0.95, (x, y, z + height + radius * 0.45), "galv")
        self._cyl(base + "/Pad", radius * 1.18, 0.2, (x, y, z + 0.1), "concrete_dark", collide=False)
        self._cube(base + "/Gauge", (0.08, 0.04, height * 0.70), (x + radius * 1.02, y, z + height * 0.50), "yellow", collide=False)

    def _cone(self, path: str, radius: float, height: float, pos: Tuple[float, float, float], mat: str = "", collide: Optional[bool] = None) -> None:
        """Generic upright cone (apex at +Z). Used for tank roofs and silo caps."""
        try:
            c = UsdGeom.Cone.Define(self.stage, path)
            c.CreateRadiusAttr(float(radius))
            c.CreateHeightAttr(float(height))
            prim = c.GetPrim()
        except Exception:
            # Fallback for environments without the Cone schema: slim cylinder.
            c = UsdGeom.Cylinder.Define(self.stage, path)
            c.CreateRadiusAttr(float(radius * 0.6))
            c.CreateHeightAttr(float(height))
            prim = c.GetPrim()
        self._bind(prim, mat)
        UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos))
        self.ops[path] = {"type": "cone", "prim": prim}
        size = (2 * radius, 2 * radius, height)
        if collide if collide is not None else self._should_collide(path, size, pos):
            self._register_box(path, pos, size, "cone")

    def _cone_roof_tank(self, base: str, pos: Tuple[float, float, float], radius: float, height: float, mat: str = "tank_cream") -> None:
        """Field storage tank with a conical roof (different roofline from _tank's dome)."""
        x, y, z = pos
        self._cyl(base + "/Body", radius, height, (x, y, z + height / 2), mat)
        self._cone(base + "/Roof", radius * 1.02, max(0.6, radius * 0.55), (x, y, z + height + radius * 0.275), "steel", collide=False)
        self._cyl(base + "/Pad", radius * 1.16, 0.20, (x, y, z + 0.10), "concrete_dark", collide=False)
        self._cube(base + "/Stair", (0.12, 0.10, height * 0.90), (x + radius * 1.04, y, z + height * 0.45), "steel", collide=False)
        self._cube(base + "/Gauge", (0.06, 0.04, height * 0.60), (x - radius * 1.02, y, z + height * 0.50), "yellow", collide=False)

    def _silo(self, base: str, pos: Tuple[float, float, float], radius: float, height: float, mat: str = "alu") -> None:
        """Corrugated silo: cylindrical body, ring bands for texture, conical cap."""
        x, y, z = pos
        self._cyl(base + "/Body", radius, height, (x, y, z + height / 2), mat)
        for k, zf in enumerate(np.linspace(0.12, 0.92, 6)):
            self._cyl(f"{base}/Ring_{k}", radius * 1.02, 0.06, (x, y, z + height * float(zf)), "steel", collide=False)
        self._cone(base + "/Roof", radius * 1.04, max(0.7, radius * 0.7), (x, y, z + height + radius * 0.35), "steel", collide=False)
        self._cyl(base + "/Pad", radius * 1.18, 0.20, (x, y, z + 0.10), "concrete_dark", collide=False)

    def _sphere_tank(self, base: str, pos: Tuple[float, float, float], radius: float = 2.6, leg_h: float = 2.6) -> None:
        """Horton spherical pressure vessel (gas storage) on splayed support legs.

        A genuinely round shape that contrasts with the boxy / vertical-tube
        structures elsewhere.  Named so the route planner treats it as a
        keep-out and does not extend the lawnmower bounding box toward it.
        """
        x, y, z = pos
        cz = z + leg_h + radius
        self._sph(base + "/Shell", radius, (x, y, cz), "sphere_tank")
        self._cyl(base + "/Walkway", radius * 1.06, 0.10, (x, y, cz), "steel", collide=False)
        n = 6
        for i, ang in enumerate(np.linspace(0, 2 * math.pi, n, endpoint=False)):
            lx = x + math.cos(ang) * radius * 0.82
            ly = y + math.sin(ang) * radius * 0.82
            leg_len = leg_h + radius * 0.6
            self._cube(f"{base}/Leg_{i}", (0.16, 0.16, leg_len), (lx, ly, z + leg_len / 2), "steel", collide=False)
        self._cyl(base + "/Pad", radius * 1.20, 0.18, (x, y, z + 0.09), "concrete_dark", collide=False)
        # Planner/safety keep-out + analytic collision for the round body.
        self._register_box(base + "/GasSphereHalo", (x, y, cz), (2 * radius * 1.05, 2 * radius * 1.05, 2 * radius + leg_h), "gassphere")

    def _horizontal_vessel(self, base: str, pos: Tuple[float, float, float], radius: float = 1.2, length: float = 6.0, axis: str = "x", mat: str = "alu") -> None:
        """Horizontal 'bullet' pressure vessel on saddle supports with domed ends."""
        x, y, z = pos
        cz = z + radius + 0.9  # rests on saddles above grade
        body = UsdGeom.Cylinder.Define(self.stage, base + "/Body")
        body.CreateRadiusAttr(float(radius))
        body.CreateHeightAttr(float(length))
        prim = body.GetPrim()
        self._bind(prim, mat)
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate((float(x), float(y), float(cz)))
        # Cylinder default axis is +Z; lay it down along X or Y.
        api.SetRotate(Gf.Vec3f(0.0, 90.0, 0.0) if axis == "x" else Gf.Vec3f(90.0, 0.0, 0.0))
        self.ops[base + "/Body"] = {"type": "cyl", "prim": prim}
        if axis == "x":
            self._sph(base + "/EndA", radius, (x - length / 2, y, cz), mat)
            self._sph(base + "/EndB", radius, (x + length / 2, y, cz), mat)
            for i, sx in enumerate((-length * 0.30, length * 0.30)):
                self._cube(f"{base}/Saddle_{i}", (0.50, radius * 1.6, 0.90), (x + sx, y, z + 0.45), "concrete_dark")
            self._register_box(base + "/HVesselHalo", (x, y, cz), (length + 2 * radius, 2 * radius + 0.4, 2 * radius + 1.0), "hvessel")
        else:
            self._sph(base + "/EndA", radius, (x, y - length / 2, cz), mat)
            self._sph(base + "/EndB", radius, (x, y + length / 2, cz), mat)
            for i, sy in enumerate((-length * 0.30, length * 0.30)):
                self._cube(f"{base}/Saddle_{i}", (radius * 1.6, 0.50, 0.90), (x, y + sy, z + 0.45), "concrete_dark")
            self._register_box(base + "/HVesselHalo", (x, y, cz), (2 * radius + 0.4, length + 2 * radius, 2 * radius + 1.0), "hvessel")

    # ----------------------------------------------------------------
    # Yard props: ISO containers (stackable), cable drums, water tower,
    # pipe stockpiles, dumpsters.  Add realistic clutter to the site.
    # ----------------------------------------------------------------
    def _container(self, path: str, pos: Tuple[float, float, float], color: str, size: Tuple[float, float, float] = (6.1, 2.44, 2.59), axis: str = "x") -> None:
        x, y, z = pos
        sx, sy, sz = size if axis == "x" else (size[1], size[0], size[2])
        self._cube(path + "/Box", (sx, sy, sz), (x, y, z + sz / 2), color)
        # Corrugation ribs + end-door lines for texture (non-colliding).
        long_n = 7
        if axis == "x":
            for i, dx in enumerate(np.linspace(-sx * 0.45, sx * 0.45, long_n)):
                self._cube(f"{path}/Rib_{i}", (0.05, sy * 1.01, sz * 0.92), (x + dx, y, z + sz / 2), "rust" if (i % 3 == 0) else color, collide=False)
            self._cube(path + "/Door", (0.06, sy * 0.9, sz * 0.9), (x + sx * 0.5, y, z + sz / 2), "galv", collide=False)
        else:
            for i, dy in enumerate(np.linspace(-sy * 0.45, sy * 0.45, long_n)):
                self._cube(f"{path}/Rib_{i}", (sx * 1.01, 0.05, sz * 0.92), (x, y + dy, z + sz / 2), "rust" if (i % 3 == 0) else color, collide=False)
            self._cube(path + "/Door", (sx * 0.9, 0.06, sz * 0.9), (x, y + sy * 0.5, z + sz / 2), "galv", collide=False)

    def _container_stack(self, base: str, pos: Tuple[float, float, float], colors: List[str], n_high: int = 1, axis: str = "x") -> None:
        x, y, z = pos
        ch = 2.59
        for k in range(int(max(1, n_high))):
            self._container(f"{base}/L{k}", (x, y, z + k * (ch + 0.04)), colors[k % len(colors)], axis=axis)

    def _cable_drum(self, base: str, pos: Tuple[float, float, float], radius: float = 1.1) -> None:
        """Large cable/wire spool lying on its side (two flanges + hub)."""
        x, y, z = pos
        w = radius * 0.9
        for i, dx in enumerate((-w / 2, w / 2)):
            fl = UsdGeom.Cylinder.Define(self.stage, f"{base}/Flange_{i}")
            fl.CreateRadiusAttr(float(radius)); fl.CreateHeightAttr(0.12)
            prim = fl.GetPrim(); self._bind(prim, "rust")
            api = UsdGeom.XformCommonAPI(prim)
            api.SetTranslate((float(x + dx), float(y), float(z + radius)))
            api.SetRotate(Gf.Vec3f(0.0, 90.0, 0.0))
            self.ops[f"{base}/Flange_{i}"] = {"type": "cyl", "prim": prim}
        hub = UsdGeom.Cylinder.Define(self.stage, base + "/Hub")
        hub.CreateRadiusAttr(float(radius * 0.55)); hub.CreateHeightAttr(float(w))
        prim = hub.GetPrim(); self._bind(prim, "steel")
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate((float(x), float(y), float(z + radius)))
        api.SetRotate(Gf.Vec3f(0.0, 90.0, 0.0))
        self.ops[base + "/Hub"] = {"type": "cyl", "prim": prim}
        self._register_box(base + "/Halo", (x, y, z + radius), (w + 0.3, 2 * radius, 2 * radius), "drum")

    def _water_tower(self, base: str, pos: Tuple[float, float, float], radius: float = 2.0, leg_h: float = 7.0) -> None:
        """Elevated water tank on four braced legs - a classic site landmark."""
        x, y, z = pos
        for i, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            lx, ly = x + dx * radius * 0.7, y + dy * radius * 0.7
            self._cube(f"{base}/Leg_{i}", (0.18, 0.18, leg_h), (lx, ly, z + leg_h / 2), "steel", collide=False)
        # Cross-bracing.
        self._cube(base + "/Brace_A", (2 * radius * 0.7 * 1.3, 0.08, 0.08), (x, y - radius * 0.7, z + leg_h * 0.5), "steel", collide=False)
        self._cube(base + "/Brace_B", (0.08, 2 * radius * 0.7 * 1.3, 0.08), (x + radius * 0.7, y, z + leg_h * 0.5), "steel", collide=False)
        self._cyl(base + "/Tank", radius, radius * 1.5, (x, y, z + leg_h + radius * 0.75), "tank_cream", collide=False)
        self._cone(base + "/Cap", radius * 1.02, radius * 0.7, (x, y, z + leg_h + radius * 1.5 + radius * 0.35), "steel", collide=False)
        self._register_box(base + "/Halo", (x, y, z + (leg_h + radius * 1.5) / 2), (2 * radius, 2 * radius, leg_h + radius * 1.5), "watertower")

    def _pipe_stack(self, base: str, pos: Tuple[float, float, float], length: float = 6.0, axis: str = "x") -> None:
        """Stockpile of large-diameter pipes in a pyramid stack."""
        x, y, z = pos
        r = 0.42
        rows = [(0, 4), (1, 3), (2, 2)]
        for ri, (lvl, count) in enumerate(rows):
            zc = z + r + lvl * (1.7 * r)
            for c in range(count):
                off = (c - (count - 1) / 2.0) * (2.05 * r)
                pp = f"{base}/P_{ri}_{c}"
                cyl = UsdGeom.Cylinder.Define(self.stage, pp)
                cyl.CreateRadiusAttr(float(r)); cyl.CreateHeightAttr(float(length))
                prim = cyl.GetPrim(); self._bind(prim, "galv")
                api = UsdGeom.XformCommonAPI(prim)
                if axis == "x":
                    api.SetTranslate((float(x), float(y + off), float(zc)))
                    api.SetRotate(Gf.Vec3f(0.0, 90.0, 0.0))
                else:
                    api.SetTranslate((float(x + off), float(y), float(zc)))
                    api.SetRotate(Gf.Vec3f(90.0, 0.0, 0.0))
                self.ops[pp] = {"type": "cyl", "prim": prim}
        self._register_box(base + "/Halo", (x, y, z + 1.4), (length if axis == "x" else 4 * r * 2, 4 * r * 2 if axis == "x" else length, 2.8), "pipestack")

    def _trailer(self, base: str, pos: Tuple[float, float, float], axis: str = "x", color: str = "white", with_cab: bool = True) -> None:
        """Parked semi-trailer (box) with wheels and an optional tractor cab."""
        x, y, z = pos
        L, W, H, deck = 8.0, 2.5, 2.7, 1.15
        def wheel(name, cx, cy, rot):
            w = UsdGeom.Cylinder.Define(self.stage, name)
            w.CreateRadiusAttr(0.55); w.CreateHeightAttr(0.5)
            prim = w.GetPrim(); self._bind(prim, "soot")
            api = UsdGeom.XformCommonAPI(prim)
            api.SetTranslate((float(cx), float(cy), float(z + 0.55)))
            api.SetRotate(rot)
            self.ops[name] = {"type": "cyl", "prim": prim}
        if axis == "x":
            self._cube(base + "/Box", (L, W, H), (x, y, z + deck + H / 2), color)
            self._cube(base + "/Logo", (L * 0.5, 0.06, H * 0.4), (x, y - W * 0.51, z + deck + H * 0.55), "blue", collide=False)
            for i, dx in enumerate((-L * 0.30, -L * 0.18, L * 0.30)):
                wheel(f"{base}/Wheel_{i}_L", x + dx, y - W * 0.45, Gf.Vec3f(90.0, 0.0, 0.0))
                wheel(f"{base}/Wheel_{i}_R", x + dx, y + W * 0.45, Gf.Vec3f(90.0, 0.0, 0.0))
            if with_cab:
                self._cube(base + "/Cab", (2.6, W, 2.8), (x + L * 0.5 + 1.4, y, z + deck + 1.4), "red")
                self._cube(base + "/Windshield", (0.08, W * 0.8, 1.0), (x + L * 0.5 + 2.7, y, z + deck + 2.0), "glass", collide=False)
            self._register_box(base + "/Halo", (x + (1.5 if with_cab else 0), y, z + deck + H / 2), (L + (3.2 if with_cab else 0), W, deck + H), "trailer")
        else:
            self._cube(base + "/Box", (W, L, H), (x, y, z + deck + H / 2), color)
            self._cube(base + "/Logo", (0.06, L * 0.5, H * 0.4), (x - W * 0.51, y, z + deck + H * 0.55), "blue", collide=False)
            for i, dy in enumerate((-L * 0.30, -L * 0.18, L * 0.30)):
                wheel(f"{base}/Wheel_{i}_L", x - W * 0.45, y + dy, Gf.Vec3f(0.0, 90.0, 0.0))
                wheel(f"{base}/Wheel_{i}_R", x + W * 0.45, y + dy, Gf.Vec3f(0.0, 90.0, 0.0))
            if with_cab:
                self._cube(base + "/Cab", (W, 2.6, 2.8), (x, y + L * 0.5 + 1.4, z + deck + 1.4), "red")
            self._register_box(base + "/Halo", (x, y + (1.5 if with_cab else 0), z + deck + H / 2), (W, L + (3.2 if with_cab else 0), deck + H), "trailer")

    def _dock_doors(self, base: str, center: Tuple[float, float, float], width: float, count: int = 6, face: str = "-y") -> None:
        """A row of roller dock doors + bumpers along a warehouse face."""
        x, y, z = center
        for i, dx in enumerate(np.linspace(-width * 0.42, width * 0.42, count)):
            self._cube(f"{base}/Door_{i}", (1.8, 0.10, 2.6), (x + dx, y, z + 1.45), "galv", collide=False)
            self._cube(f"{base}/Bump_{i}_a", (0.22, 0.30, 0.9), (x + dx - 0.9, y - 0.15, z + 0.55), "soot", collide=False)
            self._cube(f"{base}/Bump_{i}_b", (0.22, 0.30, 0.9), (x + dx + 0.9, y - 0.15, z + 0.55), "soot", collide=False)

    def _pipe_rack(self, base: str, center: Tuple[float, float, float], length: float, axis: str = "x") -> None:
        x, y, z = center
        if axis == "x":
            self._cube(base + "/Top", (length, 0.22, 0.10), (x, y, z + 0.72), "steel", collide=False)
            self._cube(base + "/Bot", (length, 0.22, 0.10), (x, y, z - 0.72), "steel", collide=False)
            for i, px in enumerate(np.linspace(-length / 2 + 0.8, length / 2 - 0.8, max(2, int(length // 2)))):
                self._cube(f"{base}/Sup_{i}", (0.12, 0.18, 1.60), (x + px, y, z), "steel", collide=False)
            for j, (dz, m) in enumerate([(0.42, "red"), (0.10, "green"), (-0.22, "blue")]):
                self._cube(f"{base}/Pipe_{j}", (length, 0.14, 0.14), (x, y, z + dz), m, collide=False)
        else:
            self._cube(base + "/Top", (0.22, length, 0.10), (x, y, z + 0.72), "steel", collide=False)
            self._cube(base + "/Bot", (0.22, length, 0.10), (x, y, z - 0.72), "steel", collide=False)
            for i, py in enumerate(np.linspace(-length / 2 + 0.8, length / 2 - 0.8, max(2, int(length // 2)))):
                self._cube(f"{base}/Sup_{i}", (0.18, 0.12, 1.60), (x, y + py, z), "steel", collide=False)
            for j, (dx, m) in enumerate([(0.42, "red"), (0.10, "green"), (-0.22, "blue")]):
                self._cube(f"{base}/Pipe_{j}", (0.14, length, 0.14), (x + dx, y, z), m, collide=False)

    def _chimney(self, path: str, pos: Tuple[float, float, float], height: float, radius: float = 0.42, plume_height: float = 15.0) -> None:
        x, y, z = pos
        self._cyl(path + "/Body", radius, height, (x, y, z + height / 2), "white")
        halo_r = float(max(radius + 0.75, self.chimney_safety_radius))
        halo_h = float(height + self.chimney_safety_height_margin)
        self._register_box(path + "/SafetyHalo", (x, y, z + 0.5 * halo_h), (2 * halo_r, 2 * halo_r, halo_h), "chimney_safety_halo")
        for bi, frac in enumerate([0.20, 0.50, 0.80]):
            self._cyl(f"{path}/Band_{bi}", radius * 1.04, max(0.32, height * 0.035), (x, y, z + height * frac), "red", collide=False)
        self._cyl(path + "/Cap", radius * 1.16, max(0.16, radius * 0.22), (x, y, z + height + 0.10), "soot", collide=False)
        self._smoke_plume(path + "/Smoke", (x, y, z + height + 0.18), plume_height, float(np.clip(radius * 1.35, 0.5, 0.68)))

    def _smoke_plume(self, path: str, origin: Tuple[float, float, float], plume_height: float, base_radius: float, puff_count: int = 13) -> None:
        ox, oy, oz = [float(v) for v in origin]
        for i in range(puff_count):
            q0 = i / float(max(1, puff_count - 1))
            for li, (lname, mul, vmul) in enumerate([("Core", 0.52, 1.08), ("Edge", 0.66, 0.82), ("Wisp", 0.78, 0.60)]):
                if lname == "Core" and q0 > 0.82:
                    continue
                if lname == "Wisp" and q0 < 0.25:
                    continue
                pp = f"{path}/{lname}_{i}"
                sph = UsdGeom.Sphere.Define(self.stage, pp)
                sph.CreateRadiusAttr(1.0)
                prim = sph.GetPrim()
                self._bind(prim, "smoke")
                rise = q0 ** 0.86
                spread = q0 ** 1.55
                side = base_radius * (0.04 + 0.55 * spread)
                px = ox + side * math.sin(1.9 + i * 0.83)
                py = oy + side * math.cos(1.9 + i * 0.71)
                pz = oz + plume_height * rise
                scale = base_radius * mul * (0.34 + 1.05 * (q0 ** 1.15))
                api = UsdGeom.XformCommonAPI(prim)
                api.SetTranslate((px, py, pz))
                api.SetScale((scale * 1.25, scale * 0.85, scale * (0.42 + 0.22 * q0) * vmul))
                self.smoke_puffs.append({
                    "path": pp,
                    "origin": (ox, oy, oz),
                    "height": plume_height,
                    "base_radius": base_radius,
                    "phase": q0 + 0.02 * li,
                    "mul": mul,
                    "vmul": vmul,
                    "layer": lname,
                })

    def _animate_smoke(self) -> None:
        if not self.smoke_puffs:
            return
        t = float(self.step_count) * self.dt
        wind = float(max(0.15, self.wind_mps))
        for it in self.smoke_puffs:
            prim = self.stage.GetPrimAtPath(it["path"])
            if not prim or not prim.IsValid():
                continue
            ox, oy, oz = it["origin"]
            h, br = float(it["height"]), float(it["base_radius"])
            q = (float(it["phase"]) + 0.018 * wind * t) % 1.0
            rise, spread = q ** 0.86, q ** 1.55
            px = ox + br * (0.04 + 0.48 * spread) * math.sin(1.1 * q + 0.23 * t)
            py = oy + br * (0.04 + 0.48 * spread) * math.cos(0.95 * (1.1 * q + 0.20 * t))
            pz = oz + h * rise
            scale = br * float(it["mul"]) * (0.34 + 1.05 * (q ** 1.15))
            try:
                api = UsdGeom.XformCommonAPI(prim)
                api.SetTranslate((float(px), float(py), float(pz)))
                api.SetScale((scale * 1.25, scale * 0.85, scale * (0.42 + 0.22 * q) * float(it["vmul"])))
            except Exception:
                pass

    def _industrial_site(self) -> None:
        """Build the e2 industrial transfer environment.

        This is intentionally different from the source e2 industrial scene:
        no reactor dome, no cooling towers, and no nuclear-plant layout.  It is
        a cluttered industrial facility with warehouses, process towers,
        storage tanks, pipe racks, containers, loading docks, and a gantry
        crane.  The observation/action interfaces remain identical to the e1
        training environment so a PPO policy trained on e1 can be loaded and
        evaluated directly for Sim2Sim transfer.
        """
        # Main industrial buildings with varied, realistic rooflines (gable,
        # sawtooth north-light, barrel-vault hangar, stepped office, mono-pitch
        # sheds) instead of plain flat boxes.
        self._gable_building("/World/E2/Warehouse_A", (-12.0, -5.0, 0), (20.0, 8.5, 6.2), "blue", ridge_axis="x")
        self._sawtooth_building("/World/E2/AssemblyHall", (10.0, -5.5, 0), (15.5, 7.5, 5.4), "concrete", teeth=5)
        self._stepped_building("/World/E2/ControlRoom", (-19.0, 9.0, 0), (8.8, 5.8, 6.6), "green", tiers=3)
        self._mono_pitch_building("/World/E2/CompressorHouse", (7.0, 11.0, 0), (11.0, 6.5, 4.8), "concrete_dark")
        self._gable_building("/World/E2/LoadingBay", (22.5, -11.5, 0), (11.0, 5.2, 3.8), "concrete", ridge_axis="x")
        # Additional buildings to densify the site.
        self._arched_hangar("/World/E2/FabricationShop", (-7.0, 6.0, 0), (12.0, 6.0, 5.0), "galv")
        self._gable_building("/World/E2/PaintShop", (18.0, 1.6, 0), (5.5, 4.6, 4.4), "blue", ridge_axis="y")
        self._mono_pitch_building("/World/E2/UtilityShed", (-26.0, -1.5, 0), (5.0, 5.0, 3.4), "concrete_dark")
        self._building("/World/E2/GateHouse", (-27.0, -10.0, 0), (4.0, 3.5, 3.0), "concrete")
        self._gable_building("/World/E2/SparesStore", (-1.0, -11.5, 0), (7.0, 3.6, 3.6), "green", ridge_axis="x")

        # Dense process area with vertical towers and vessels.
        process_specs = [
            ("DistillationA", 18.0, 9.0, 0.70, 14.5),
            ("DistillationB", 21.0, 6.2, 0.58, 12.6),
            ("DistillationC", 24.0, 9.8, 0.52, 11.2),
            ("Absorber", 16.2, 13.0, 0.62, 10.8),
        ]
        for name, x, y, r, h in process_specs:
            base = f"/World/E2/Process/{name}"
            self._cyl(base + "/Column", r, h, (x, y, h / 2.0), "galv")
            self._cyl(base + "/TopCap", r * 1.08, 0.22, (x, y, h + 0.10), "steel", collide=False)
            # Soft safety halo used by planner/governor, not visible.
            halo_r = max(1.8, float(self.chimney_safety_radius) * 0.72)
            self._register_box(base + "/SafetyHalo", (x, y, 0.5 * (h + 2.0)), (2 * halo_r, 2 * halo_r, h + 2.0), "industrial_tall_halo")
            for k, zfrac in enumerate([0.24, 0.50, 0.76]):
                self._cyl(f"{base}/ServiceRing_{k}", r * 1.18, 0.12, (x, y, h * zfrac), "yellow", collide=False)
            self._cube(base + "/Ladder", (0.10, 0.08, h * 0.80), (x + r * 1.15, y, h * 0.46), "steel", collide=False)

        # Process-area equipment with varied, realistic shapes (no chimneys / flare
        # stacks). These give the e2 site a different silhouette: round gas spheres
        # and horizontal bullet vessels instead of vertical smoke stacks.
        self._sphere_tank("/World/E2/GasSphere_01", (28.5, 3.0, 0), radius=2.8, leg_h=2.6)
        self._sphere_tank("/World/E2/GasSphere_02", (31.0, 8.0, 0), radius=2.2, leg_h=2.2)
        self._horizontal_vessel("/World/E2/BulletTank_01", (27.0, -2.5, 0), radius=1.25, length=7.0, axis="x", mat="alu")
        self._horizontal_vessel("/World/E2/BulletTank_02", (14.5, 9.0, 0), radius=1.05, length=5.2, axis="y", mat="tank_teal")

        # Storage-tank farm with mixed, realistic roof/shape styles for variety:
        # domed tanks, cone-roof tanks, and a tall corrugated silo.
        tank_specs = [
            (-23.0, 13.5, 1.55, 3.2), (-19.2, 14.0, 1.35, 2.9),
            (-23.5, 18.0, 1.25, 2.6), (-18.7, 18.2, 1.45, 3.0),
            (1.5, 18.0, 1.20, 2.7), (5.0, 18.4, 1.05, 2.4),
        ]
        for i, (x, y, r, h) in enumerate(tank_specs):
            base = f"/World/E2/TankFarm/Tank_{i}"
            if i % 3 == 0:
                self._cone_roof_tank(base, (x, y, 0), r, h, mat="tank_cream")
            elif i % 3 == 1:
                self._tank(base, (x, y, 0), r, h)  # domed roof
            else:
                self._cone_roof_tank(base, (x, y, 0), r, h, mat="tank_teal")
        # A tall feed silo gives a distinct vertical-with-cone-cap profile.
        self._silo("/World/E2/TankFarm/Silo_0", (-15.0, 19.5, 0), radius=2.0, height=7.2, mat="alu")

        # Pipe racks and overhead process connections.
        self._pipe_rack("/World/E2/PipeRack_MainX", (-3.0, 2.0, 2.25), 34.0, axis="x")
        self._pipe_rack("/World/E2/PipeRack_ProcessX", (17.5, 14.8, 2.35), 18.0, axis="x")
        self._pipe_rack("/World/E2/PipeRack_ServiceY", (2.0, -4.0, 2.10), 24.0, axis="y")
        self._pipe_rack("/World/E2/PipeRack_LoadingY", (23.5, -2.0, 2.00), 18.0, axis="y")

        # Container yard: realistic ISO containers in rows, some stacked 2-3 high.
        colors = ["blue", "green", "red", "yellow", "orange", "rust"]
        idx = 0
        for row, y in enumerate([-22.0, -18.5, -15.0]):
            for col, x in enumerate(np.linspace(-26.0, -8.0, 4)):
                n_high = 1 + ((row + col) % 3)
                cset = [colors[(idx + k) % len(colors)] for k in range(n_high)]
                self._container_stack(f"/World/E2/Containers/S_{idx}", (float(x), float(y), 0.0), cset, n_high=n_high, axis="x")
                idx += 1
        # A few cross-stacked containers near the loading bay for variety.
        self._container_stack("/World/E2/Containers/East_0", (17.0, -19.0, 0.0), ["green", "blue"], n_high=2, axis="y")
        self._container_stack("/World/E2/Containers/East_1", (20.0, -19.0, 0.0), ["orange"], n_high=1, axis="y")
        self._container_stack("/World/E2/Containers/East_2", (23.0, -19.5, 0.0), ["red", "yellow", "blue"], n_high=3, axis="y")

        # Other yard props: cable drums, pipe stockpiles, a water tower, dumpsters.
        self._cable_drum("/World/E2/Yard/Drum_0", (-2.0, -16.5, 0.0), radius=1.1)
        self._cable_drum("/World/E2/Yard/Drum_1", (1.0, -16.0, 0.0), radius=0.9)
        self._pipe_stack("/World/E2/Yard/PipeStock_0", (8.0, -18.5, 0.0), length=6.0, axis="x")
        self._pipe_stack("/World/E2/Yard/PipeStock_1", (-30.0, 6.0, 0.0), length=5.0, axis="y")
        self._water_tower("/World/E2/Yard/WaterTower", (-29.0, 14.0, 0.0), radius=1.9, leg_h=7.5)
        for i, (x, y) in enumerate([(15.0, -7.0), (24.0, -6.0), (-7.0, -16.5)]):
            self._cube(f"/World/E2/Yard/Dumpster_{i}", (2.0, 1.1, 1.2), (float(x), float(y), 0.60), "rust")
            self._cube(f"/World/E2/Yard/DumpsterLid_{i}", (2.05, 1.15, 0.10), (float(x), float(y), 1.22), "steel", collide=False)

        # Loading docks, pallets, barriers, and forklifts as small obstacles.
        for i, x in enumerate(np.linspace(16.5, 28.5, 5)):
            self._cube(f"/World/E2/LoadingDock/Ramp_{i}", (1.8, 2.0, 0.30), (float(x), -15.3, 0.15), "concrete_dark")
            self._cube(f"/World/E2/LoadingDock/Door_{i}", (1.2, 0.06, 1.7), (float(x), -14.08, 1.05), "steel", collide=False)
        for i, (x, y) in enumerate([(-4, -20), (-1, -18), (4, -20), (10, -18), (13, -21)]):
            self._cube(f"/World/E2/Pallets/Pallet_{i}", (1.2, 0.8, 0.28), (x, y, 0.16), "yellow")
            self._cube(f"/World/E2/Pallets/Box_{i}", (0.95, 0.65, 0.65), (x, y, 0.63), "orange")

        # Gantry crane and service bridge.  The beam is mostly visual; columns
        # remain collidable so the safety layer has e2 obstacles to check.
        self._cube("/World/E2/GantryCrane/Column_A", (0.35, 0.35, 8.0), (-3.0, -13.5, 4.0), "steel")
        self._cube("/World/E2/GantryCrane/Column_B", (0.35, 0.35, 8.0), (13.0, -13.5, 4.0), "steel")
        self._cube("/World/E2/GantryCrane/Beam", (17.5, 0.32, 0.40), (5.0, -13.5, 8.1), "yellow", collide=False)
        self._cube("/World/E2/GantryCrane/Trolley", (1.2, 0.55, 0.55), (4.0, -13.5, 7.55), "orange", collide=False)
        self._cube("/World/E2/GantryCrane/HookCable", (0.06, 0.06, 2.3), (4.0, -13.5, 6.1), "steel", collide=False)

        # Substation and cable racks.
        for i, x in enumerate(np.linspace(-28.0, -12.0, 5)):
            self._cube(f"/World/E2/Substation/Transformer_{i}", (1.1, 0.95, 1.05), (float(x), 4.0, 0.56), "steel")
            self._cube(f"/World/E2/Substation/Warn_{i}", (0.45, 0.04, 0.32), (float(x), 3.50, 0.98), "yellow", collide=False)
            self._cube(f"/World/E2/Substation/Pole_{i}", (0.10, 0.10, 3.3), (float(x), 6.2, 1.65), "galv", collide=False)
            self._cube(f"/World/E2/Substation/Wire_{i}", (2.8, 0.035, 0.035), (float(x), 6.2, 3.25), "galv", collide=False)

        # Roads and markings specific to e2.
        self._cube("/World/E2/ServiceRoad_X", (64.0, 2.8, 0.045), (0.0, -26.0, 0.035), "asphalt", collide=False)
        self._cube("/World/E2/ServiceRoad_Y", (2.8, 52.0, 0.045), (32.0, -2.0, 0.035), "asphalt", collide=False)
        for i, x in enumerate(np.linspace(-28, 28, 15)):
            self._cube(f"/World/E2/RoadMark_X_{i}", (1.1, 0.07, 0.006), (float(x), -26.0, 0.08), "white", collide=False)
        for i, y in enumerate(np.linspace(-22, 18, 12)):
            self._cube(f"/World/E2/RoadMark_Y_{i}", (0.07, 1.1, 0.006), (32.0, float(y), 0.08), "white", collide=False)

        # Expansion zones that roughly double the site footprint.
        self._logistics_district()
        self._storage_annex()

        # Perimeter fence and lights.
        b = self.world_size + 0.8
        for nm, sz, pos in [
            ("N", (2 * b, 0.06, 1.7), (0, b, 0.88)),
            ("S", (2 * b, 0.06, 1.7), (0, -b, 0.88)),
            ("E", (0.06, 2 * b, 1.7), (b, 0, 0.88)),
            ("W", (0.06, 2 * b, 1.7), (-b, 0, 0.88)),
        ]:
            self._cube(f"/World/Fence/{nm}", sz, pos, "galv")
        light_xy = [(-28, -24), (-4, -24), (20, -24), (-30, 20), (0, 22), (25, 20),
                    (-30, -42), (-10, -52), (12, -42), (30, -50), (45, -38),
                    (44, 0), (58, 12), (50, -8), (40, 20)]
        for i, (x, y) in enumerate(light_xy):
            self._cube(f"/World/Lights/Pole_{i}", (0.12, 0.12, 4.2), (x, y, 2.1), "galv", collide=False)
            self._sph(f"/World/Lights/Bulb_{i}", 0.20, (x, y, 4.35), "yellow")

    def _logistics_district(self) -> None:
        """South expansion: large warehouses, loading docks, parked trailers,
        and a big container/laydown yard."""
        # Big distribution warehouse (gable) + dock doors + parked trailers.
        self._gable_building("/World/E2/Logistics/DistributionWarehouse", (-20.0, -42.0, 0), (34.0, 16.0, 8.5), "blue", ridge_axis="x")
        self._dock_doors("/World/E2/Logistics/DW_Docks", (-20.0, -33.7, 0.0), 30.0, count=7)
        # Cold-storage warehouse (gable, lighter cladding).
        self._gable_building("/World/E2/Logistics/ColdStore", (16.0, -42.0, 0), (24.0, 16.0, 7.5), "alu", ridge_axis="x")
        self._dock_doors("/World/E2/Logistics/CS_Docks", (16.0, -33.8, 0.0), 20.0, count=5)
        # High-bay automated warehouse (long sawtooth roof).
        self._sawtooth_building("/World/E2/Logistics/HighBay", (-4.0, -55.0, 0), (40.0, 8.0, 11.0), "concrete", teeth=8)
        # Logistics office (stepped).
        self._stepped_building("/World/E2/Logistics/Office", (34.0, -39.0, 0), (8.0, 7.0, 7.5), "green", tiers=3)

        # Parked trailers along the dock fronts.
        for i, x in enumerate(np.linspace(-32.0, -8.0, 5)):
            self._trailer(f"/World/E2/Logistics/Trailer_DW_{i}", (float(x), -30.5, 0.0), axis="y", color="white", with_cab=(i % 2 == 0))
        for i, x in enumerate(np.linspace(8.0, 24.0, 3)):
            self._trailer(f"/World/E2/Logistics/Trailer_CS_{i}", (float(x), -30.5, 0.0), axis="y", color="white", with_cab=(i % 2 == 1))

        # Large container yard (rows, stacked 1-3 high).
        colors = ["blue", "green", "red", "yellow", "orange", "rust"]
        idx = 0
        for row, y in enumerate([-48.0, -52.0, -56.0]):
            for col, x in enumerate(np.linspace(22.0, 44.0, 5)):
                n_high = 1 + ((row + col) % 3)
                cset = [colors[(idx + k) % len(colors)] for k in range(n_high)]
                self._container_stack(f"/World/E2/Logistics/Yard_{idx}", (float(x), float(y), 0.0), cset, n_high=n_high, axis="x")
                idx += 1

        # Laydown yard clutter.
        self._pipe_stack("/World/E2/Logistics/PipeStock_0", (-34.0, -52.0, 0.0), length=7.0, axis="x")
        self._pipe_stack("/World/E2/Logistics/PipeStock_1", (-24.0, -52.0, 0.0), length=7.0, axis="x")
        self._cable_drum("/World/E2/Logistics/Drum_0", (-14.0, -52.0, 0.0), radius=1.2)
        self._cable_drum("/World/E2/Logistics/Drum_1", (-11.0, -52.5, 0.0), radius=1.0)
        for i, (x, y) in enumerate([(-2.0, -48.0), (2.0, -48.0)]):
            self._cube(f"/World/E2/Logistics/Dumpster_{i}", (2.2, 1.2, 1.3), (float(x), float(y), 0.65), "rust")

    def _storage_annex(self) -> None:
        """East expansion: an annex warehouse, a small tank cluster, and laydown."""
        self._gable_building("/World/E2/Annex/AnnexWarehouse", (52.0, 6.0, 0), (20.0, 22.0, 7.5), "concrete", ridge_axis="y")
        self._dock_doors("/World/E2/Annex/Docks", (42.2, 6.0, 0.0), 18.0, count=5, face="-x")
        # Small annex tank cluster (mixed roofs).
        annex_tanks = [(45.0, -8.0, 1.4, 3.0), (49.0, -8.5, 1.2, 2.6), (45.0, -3.5, 1.3, 2.8)]
        for i, (x, y, r, h) in enumerate(annex_tanks):
            base = f"/World/E2/Annex/Tank_{i}"
            (self._cone_roof_tank if i % 2 == 0 else self._tank)(base, (x, y, 0), r, h)
        self._horizontal_vessel("/World/E2/Annex/Bullet", (44.0, -13.0, 0), radius=1.1, length=6.0, axis="x", mat="tank_teal")
        # Annex laydown and a couple of parked trailers.
        self._pipe_stack("/World/E2/Annex/PipeStock", (50.0, 20.0, 0.0), length=6.0, axis="x")
        self._cable_drum("/World/E2/Annex/Drum", (44.0, 20.0, 0.0), radius=1.1)
        for i, y in enumerate(np.linspace(-2.0, 14.0, 3)):
            self._trailer(f"/World/E2/Annex/Trailer_{i}", (40.0, float(y), 0.0), axis="x", color="white", with_cab=(i == 1))

    def _drone(self) -> None:
        self._cube("/World/Drone/Body", (0.60, 0.34, 0.16), (0, -8, 2.0), "drone", collide=False)
        self._cube("/World/Drone/Arm_X", (0.92, 0.055, 0.035), (0, -8, 2.0), "drone", collide=False)
        self._cube("/World/Drone/Arm_Y", (0.055, 0.92, 0.035), (0, -8, 2.0), "drone", collide=False)
        for i, (x, y) in enumerate([(0.46, 0), (-0.46, 0), (0, 0.46), (0, -0.46)]):
            self._cyl(f"/World/Drone/Rotor_{i}", 0.19, 0.018, (x, -8 + y, 2.03), "drone", collide=False)
        self._cube("/World/Drone/CameraHousing", (0.18, 0.11, 0.11), (0.34, -8, 1.95), "lens", collide=False)
        self._cube("/World/Drone/BottomCameraHousing", (0.16, 0.13, 0.08), (0, -8, 1.88), "lens", collide=False)

    def _cameras(self) -> None:
        """Create USD camera prims.

        Important: the camera orientation is NOT fixed here with Euler angles.
        Isaac/USD cameras look along local -Z and use local +Y as image-up.
        The pose is set every step by _update_camera_transforms() using a
        look-forward/look-down quaternion.  This avoids the old 90-degree roll
        that made vertical chimneys/towers appear horizontal in the front RGB.
        """
        try:
            for side, yoff in [("Left", 0.09), ("Right", -0.09)]:
                cam = UsdGeom.Camera.Define(self.stage, f"/World/Drone/Stereo{side}Camera")
                cam.CreateHorizontalApertureAttr(20.955)
                cam.CreateVerticalApertureAttr(15.2908)
                cam.CreateFocalLengthAttr(float(0.5 * 20.955 / max(math.tan(self.camera_hfov * 0.5), 1e-6)))
                cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 250.0))
                # Temporary pose only; overwritten by _update_camera_transforms.
                UsdGeom.XformCommonAPI(cam.GetPrim()).SetTranslate((0.42, yoff, -0.04))

            bottom = UsdGeom.Camera.Define(self.stage, "/World/Drone/BottomInspectionCamera")
            bottom.CreateHorizontalApertureAttr(20.955)
            bottom.CreateVerticalApertureAttr(15.2908)
            bottom.CreateFocalLengthAttr(float(0.5 * 20.955 / max(math.tan(self.camera_hfov * 0.5), 1e-6)))
            bottom.CreateClippingRangeAttr(Gf.Vec2f(0.05, 250.0))
            UsdGeom.XformCommonAPI(bottom.GetPrim()).SetTranslate((0.0, 0.0, -0.16))

            # Force a correct initial pose immediately so the first captured RGB
            # frame is not stale/rolled before the first environment step.
            self._update_camera_transforms()
        except Exception as exc:
            print(f"[CAMERA] camera prim creation failed: {exc}")
    def _inspection_target_prim(self) -> None:
        try:
            xf = UsdGeom.Xform.Define(self.stage, "/World/InspectionTarget")
            UsdGeom.XformCommonAPI(xf.GetPrim()).SetTranslate(tuple(float(v) for v in self.target))
            try:
                UsdGeom.Imageable(xf.GetPrim()).MakeInvisible()
            except Exception:
                pass
            self.ops["/World/InspectionTarget"] = {"type": "hidden", "prim": xf.GetPrim()}
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Analytic collisions and rays
    # ----------------------------------------------------------------
    def _register_box(self, path: str, center: Tuple[float, float, float], size: Tuple[float, float, float], category: str = "structure") -> None:
        c = np.asarray(center, np.float32).reshape(3)
        s = np.maximum(np.asarray(size, np.float32).reshape(3), 1e-3)
        self.static_collision_boxes.append({"path": str(path), "mn": c - 0.5 * s, "mx": c + 0.5 * s, "category": category})

    def _should_collide(self, path: str, size: Tuple[float, float, float], pos: Tuple[float, float, float]) -> bool:
        p = str(path).lower()
        if not p.startswith("/world/e2") and not p.startswith("/world/fence"):
            return False
        if any(w in p for w in ("line", "band", "cap", "rod", "window", "vent", "wire", "warn", "gauge", "rib")):
            return False
        sx, sy, sz = [float(v) for v in size]
        if sz < 0.22 or max(sx, sy) < 0.18:
            return False
        if float(pos[2]) + 0.5 * sz < 0.25:
            return False
        return True

    @staticmethod
    def _seg_aabb_3d(a: np.ndarray, b: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> bool:
        a, b, mn, mx = [np.asarray(v, np.float32) for v in (a, b, mn, mx)]
        d = b - a
        tmin, tmax = 0.0, 1.0
        for k in range(3):
            if abs(float(d[k])) < 1e-9:
                if float(a[k]) < float(mn[k]) or float(a[k]) > float(mx[k]):
                    return False
            else:
                inv = 1.0 / float(d[k])
                t1 = (float(mn[k]) - float(a[k])) * inv
                t2 = (float(mx[k]) - float(a[k])) * inv
                t1, t2 = min(t1, t2), max(t1, t2)
                tmin, tmax = max(tmin, t1), min(tmax, t2)
                if tmin > tmax:
                    return False
        return True

    def _iter_aabbs(self):
        for box in self.static_collision_boxes:
            yield str(box["path"]), np.asarray(box["mn"], np.float32), np.asarray(box["mx"], np.float32)

    def _is_stack(self, name: str, mn: np.ndarray, mx: np.ndarray) -> bool:
        n = str(name or "").lower()
        size_xy = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]))
        size_z = float(mx[2] - mn[2])
        return "chimney" in n or "stack" in n or "safetyhalo" in n or "coolingtower" in n or (size_z > 6.0 and size_xy < 4.0)

    def _collision_margin(self, name: str, mn: np.ndarray, mx: np.ndarray) -> float:
        return max(self.drone_collision_radius + 0.05, 0.20)

    def _planning_margin(self, name: str, mn: np.ndarray, mx: np.ndarray) -> float:
        if self._is_stack(name, mn, mx):
            return max(self.drone_collision_radius + 0.25, min(max(self.path_clearance_margin, 0.35 * self.chimney_safety_radius), 4.25))
        return self.drone_collision_radius + 0.15

    def _nearest_clearance(self, pos: np.ndarray) -> float:
        p = np.asarray(pos, np.float32)
        best = float("inf")
        for _n, mn, mx in self._iter_aabbs():
            best = min(best, float(np.linalg.norm(np.maximum(np.maximum(mn - p, p - mx), 0.0))))
        return best

    def _point_inside(self, pos: np.ndarray, margin: float = 0.0) -> Tuple[bool, str]:
        p = np.asarray(pos, np.float32)
        for n, mn, mx in self._iter_aabbs():
            if np.all(p >= mn - margin) and np.all(p <= mx + margin):
                return True, n
        return False, ""

    def _detour(self, old: np.ndarray, new: np.ndarray, name: str, mn: np.ndarray, mx: np.ndarray) -> Optional[np.ndarray]:
        old, new, mn, mx = [np.asarray(v, np.float32) for v in (old, new, mn, mx)]
        step = float(np.clip(max(float(np.linalg.norm((new - old)[:2])), self.osd_vmin * self.dt), 0.025, 0.12))
        closest = np.array([float(np.clip(old[0], mn[0], mx[0])), float(np.clip(old[1], mn[1], mx[1]))], np.float32)
        away = old[:2] - closest
        if float(np.linalg.norm(away)) < 1e-5:
            away = old[:2] - 0.5 * (mn[:2] + mx[:2])
        if float(np.linalg.norm(away)) < 1e-5:
            away = np.array([1.0, 0.0], np.float32)
        away = away / max(float(np.linalg.norm(away)), 1e-6)
        tan = np.array([-away[1], away[0]], np.float32)
        rel = np.asarray(self.target[:2], np.float32) - old[:2]
        tdir = rel / max(float(np.linalg.norm(rel)), 1e-6) if float(np.linalg.norm(rel)) > 1e-6 else np.zeros(2, np.float32)
        best, best_score = None, -1e18
        candidates = [0.82 * tan + 0.28 * tdir + 0.35 * away, -0.82 * tan + 0.28 * tdir + 0.35 * away, 0.9 * away + 0.15 * tdir]
        for dxy in candidates:
            dn = float(np.linalg.norm(dxy))
            if dn < 1e-6:
                continue
            cand = old.copy()
            cand[:2] = old[:2] + (dxy / dn) * step
            cand[2] = new[2]
            hit = False
            for hn, hmn, hmx in self._iter_aabbs():
                m = self._collision_margin(hn, hmn, hmx)
                if self._seg_aabb_3d(old, cand, hmn - m, hmx + m):
                    hit = True
                    break
            if hit:
                continue
            cc = np.array([float(np.clip(cand[0], mn[0], mx[0])), float(np.clip(cand[1], mn[1], mx[1]))], np.float32)
            clearance = float(np.linalg.norm(cand[:2] - cc))
            prog = float(np.linalg.norm(np.asarray(self.target[:2], np.float32) - old[:2])) - float(np.linalg.norm(np.asarray(self.target[:2], np.float32) - cand[:2]))
            score = 2.0 * clearance + 0.75 * prog
            if score > best_score:
                best_score, best = score, cand.astype(np.float32)
        return best

    def _resolve_motion(self, old: np.ndarray, new: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        old, new = np.asarray(old, np.float32), np.asarray(new, np.float32)

        # Hard collision check.  Earlier versions immediately returned old and
        # marked the episode terminal when the drone touched the inflated AABB.
        # That caused training/demo episodes to reset near the first inspection
        # point if a roof edge or stack margin was touched.  The paper system is
        # a command-checking governor, so a near-collision should first be
        # converted into a safe lateral detour/hold, not an episode reset.
        for name, mn, mx in self._iter_aabbs():
            m = self._collision_margin(name, mn, mx)
            if self._seg_aabb_3d(old, new, mn - m, mx + m) or np.all((new >= mn - m) & (new <= mx + m)):
                d = self._detour(old, new, name, mn, mx)
                if d is not None:
                    return d.astype(np.float32), False, ""
                return old.copy(), True, str(name)

        # Softer planning envelope: stay away from obstacles and stacks.
        for name, mn, mx in self._iter_aabbs():
            m = self._planning_margin(name, mn, mx)
            if self._seg_aabb_3d(old, new, mn - m, mx + m) or np.all((new >= mn - m) & (new <= mx + m)):
                d = self._detour(old, new, name, mn, mx)
                return (d.astype(np.float32) if d is not None else old.copy()), False, ""
        return new.astype(np.float32), False, ""

    # ----------------------------------------------------------------
    # Frames / rays
    # ----------------------------------------------------------------
    @staticmethod
    def _wrap(a: float) -> float:
        return (float(a) + math.pi) % (2 * math.pi) - math.pi

    def _body_to_world(self, v: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], np.float32)

    def _world_to_body(self, v: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]], np.float32)

    def _w2b_yaw(self, v: np.ndarray, yaw: float) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]], np.float32)

    def _b2w_yaw(self, v: np.ndarray, yaw: float) -> np.ndarray:
        """Inverse yaw rotation: local/body-yaw frame -> world-yaw frame."""
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], np.float32)

    def _ray_distances(self, noisy: bool = False) -> np.ndarray:
        rays = np.ones(self.num_rays, np.float32)
        for i in range(self.num_rays):
            ang = self.yaw + 2 * math.pi * i / self.num_rays
            d = self._ray_2d(self.pos[:2], np.array([math.cos(ang), math.sin(ang)], np.float32), float(self.pos[2]))
            val = np.clip(d / self.max_ray_range, 0.0, 1.0)
            if noisy:
                val += float(self.rng.normal(0, self.depth_noise_std))
                if self.slam_quality < 0.35:
                    val += float(self.rng.normal(0, self.depth_noise_std * 1.5))
            rays[i] = np.clip(val, 0.0, 1.0)
        return rays

    def _ray_2d(self, o: np.ndarray, d: np.ndarray, oz: float) -> float:
        best = self.max_ray_range
        for _n, mn3, mx3 in self._iter_aabbs():
            if not (float(mn3[2]) - self.drone_collision_radius <= oz <= float(mx3[2]) + self.drone_collision_radius):
                continue
            hit = self._ray_aabb(o, d, mn3[:2] - self.drone_collision_radius, mx3[:2] + self.drone_collision_radius)
            if hit is not None:
                best = min(best, hit)
        b = self.world_size
        ex = self._ray_exit(o, d, np.array([-b, -b], np.float32), np.array([b, b], np.float32))
        if ex is not None:
            best = min(best, ex)
        return best

    @staticmethod
    def _ray_aabb(o: np.ndarray, d: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> Optional[float]:
        tmin, tmax = 0.0, 1e9
        for k in range(2):
            if abs(float(d[k])) < 1e-8:
                if o[k] < mn[k] or o[k] > mx[k]:
                    return None
            else:
                inv = 1.0 / float(d[k])
                t1, t2 = (float(mn[k]) - float(o[k])) * inv, (float(mx[k]) - float(o[k])) * inv
                t1, t2 = min(t1, t2), max(t1, t2)
                tmin, tmax = max(tmin, t1), min(tmax, t2)
                if tmin > tmax:
                    return None
        return None if tmax < 0 else float(max(tmin, 0.0))

    @staticmethod
    def _ray_exit(o: np.ndarray, d: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> Optional[float]:
        ts: List[float] = []
        for k in range(2):
            if abs(float(d[k])) < 1e-8:
                continue
            for bound in (mn[k], mx[k]):
                t = (float(bound) - float(o[k])) / float(d[k])
                if t > 0:
                    p = o + t * d
                    j = 1 - k
                    if mn[j] - 1e-5 <= p[j] <= mx[j] + 1e-5:
                        ts.append(float(t))
        return min(ts) if ts else None

    # ----------------------------------------------------------------
    # SLAM proxy / VSLAM metrics
    # ----------------------------------------------------------------
    def update_mission(self, mission_text: str) -> None:
        self.mission_text = str(mission_text or self.mission_text)
        self.mission_vector, self.mission_risk_lambda, self.mission_metadata = self.mission_encoder.encode(self.mission_text)
        print(f"[MISSION] updated parsed={self.mission_metadata} lambda={self.mission_risk_lambda:.3f}")

    def _reset_slam(self) -> None:
        self.slam_origin_true = self.pos.copy()
        self.slam_yaw0_true = float(self.yaw)
        self.slam_pos = np.zeros(3, np.float32)
        self.slam_vel = np.zeros(3, np.float32)
        self.slam_yaw = 0.0
        self.slam_quality = 1.0
        self.slam_drift = np.zeros(3, np.float32)
        self.prev_true_vel = np.zeros(3, np.float32)
        self.imu_acc_body = np.zeros(3, np.float32)
        self.imu_gyro_body = np.zeros(3, np.float32)

    def _update_imu(self, yaw_rate: float) -> None:
        acc = self._world_to_body((self.vel - self.prev_true_vel) / max(self.dt, 1e-6))
        self.imu_acc_body = acc + self.rng.normal(0, 0.04, 3).astype(np.float32)
        self.imu_gyro_body = np.array([self.rng.normal(0, 0.01), self.rng.normal(0, 0.01), yaw_rate + self.rng.normal(0, 0.015)], np.float32)

    def _visible_features(self) -> int:
        if self.visual_feature_points is None:
            return 0
        rel = self.visual_feature_points - self.pos.reshape(1, 3)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x = c * rel[:, 0] + s * rel[:, 1]
        y = -s * rel[:, 0] + c * rel[:, 1]
        z = rel[:, 2]
        fwd = x > 0.35
        rng = np.sqrt(x * x + y * y + z * z) < self.max_ray_range
        u = np.abs(y / np.maximum(x * math.tan(self.camera_hfov * 0.5), 1e-5)) <= 1.0
        v = np.abs(z / np.maximum(x * math.tan(self.camera_vfov * 0.5), 1e-5)) <= 1.0
        return int(np.clip(round(int(np.count_nonzero(fwd & rng & u & v)) / 8.0), 0, 80))

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def _texture_mu(self, fc: float) -> float:
        med, high = 8.0, 15.0
        if fc >= high:
            return 1.0
        if fc >= med:
            return self._clip01((fc - med) / max(high - med, 1e-6))
        return self._clip01(0.35 * fc / max(med, 1e-6))

    def _illum_mu(self, lux: float) -> float:
        low, good = 280.0, 450.0
        if lux <= low:
            return 0.0
        return 1.0 if lux >= good else self._clip01((lux - low) / max(good - low, 1e-6))

    def _wind_mu(self, w: float) -> float:
        return self._clip01(1.0 - float(w) / 2.0)

    def _true_local(self) -> np.ndarray:
        return self._w2b_yaw(self.pos - self.slam_origin_true, self.slam_yaw0_true)

    def _slam_world_position(self) -> np.ndarray:
        """Convert VSLAM local odometry estimate back into the world frame for plotting/down-camera artifacts."""
        try:
            return (self.slam_origin_true + self._b2w_yaw(np.asarray(self.slam_pos, np.float32), self.slam_yaw0_true)).astype(np.float32)
        except Exception:
            return np.asarray(self.pos, np.float32).copy()

    def _update_slam(self) -> None:
        true_pos = self._true_local()
        true_vel = self._w2b_yaw(self.vel, self.slam_yaw0_true)
        true_yaw = float(self.yaw - self.slam_yaw0_true)
        if self.slam_mode == "gt":
            self.slam_pos, self.slam_vel, self.slam_yaw, self.slam_quality = true_pos.astype(np.float32), true_vel.astype(np.float32), true_yaw, 1.0
            return
        if self.slam_mode == "cuvslam":
            odom = self.cuvslam_receiver.get_latest(0.75) if self.cuvslam_receiver else None
            if odom is not None:
                self.slam_pos, self.slam_vel = odom["pos"].astype(np.float32), odom["vel"].astype(np.float32)
                self.slam_yaw, self.slam_quality = float(odom["yaw"]), 1.0
            else:
                self.slam_vel = (0.9 * self.slam_vel).astype(np.float32)
                self.slam_quality = 0.0
            return
        fc = self._visible_features()
        stability = self._clip01(0.55 * self._texture_mu(fc) + 0.25 * self._illum_mu(self.illumination_lux) + 0.20 * self._wind_mu(self.wind_mps))
        scale = 0.55 + 0.70 * (1.0 - stability)
        self.slam_drift = (
            0.997 * self.slam_drift
            + self.rng.normal(0, self.slam_drift_pos_per_sec * self.dt * (0.35 + 0.85 * (1 - stability)), 3).astype(np.float32)
        )
        self.slam_drift[2] *= 0.35
        if self.rng.random() < self.slam_tracking_loss_prob * (0.55 + 1.40 * (1 - stability)):
            self.slam_quality = float(self.rng.uniform(0.12, 0.42))
            self.slam_pos = (self.slam_pos + 0.04 * self.rng.normal(0, self.slam_pos_noise_std * scale, 3)).astype(np.float32)
            self.slam_vel = (0.9 * self.slam_vel).astype(np.float32)
        else:
            self.slam_quality = float(min(1.0, self.slam_quality + self.slam_quality_recover_rate))
            self.slam_pos = (true_pos + self.slam_drift + self.rng.normal(0, self.slam_pos_noise_std * scale, 3)).astype(np.float32)
            self.slam_vel = (true_vel + self.rng.normal(0, self.slam_vel_noise_std * scale, 3)).astype(np.float32)
        self.slam_yaw = true_yaw + float(self.rng.normal(0, self.slam_yaw_noise_std * scale))

    def _nav_metrics(self) -> Dict[str, float]:
        loc = float(np.linalg.norm(self.slam_pos - self._true_local()))
        fc = float(self._visible_features())
        return {
            "localization_error_m": loc,
            "feature_count": fc,
            "inlier_ratio": self._texture_mu(fc),
            "slam_quality": float(self.slam_quality),
        }

    def _vslam_risk(self, nav: Dict[str, float]) -> float:
        """Calibrated VSLAM risk in [0, 1].

        The earlier risk score was too conservative for the low-texture mission:
        normal feature variation pushed z_t above the gate for most steps, so
        the governor looked as if it was overriding almost every action.  This
        version keeps the same inputs but uses softer uncertainty penalties and
        a lower feature-count saturation point for the industrial route.
        """
        fc = float(nav.get("feature_count", 0.0))
        n = np.clip(fc / 45.0, 0.0, 1.0)
        i = self._clip01(nav.get("inlier_ratio", 0.5))
        q = self._clip01(nav.get("slam_quality", self.slam_quality))
        pos_unc = np.clip(float(nav.get("localization_error_m", 0.0)) / max(self.drift_threshold, 1e-6), 0.0, 1.0)
        yaw_unc = np.clip(abs(self._wrap(self.slam_yaw - (self.yaw - self.slam_yaw0_true))) / max(self.abort_yaw_limit, 1e-6), 0.0, 1.0)
        loss = 1.0 if q < 0.18 else 0.0

        health = 0.42 * n + 0.34 * i + 0.24 * q - 0.18 * pos_unc - 0.10 * yaw_unc - 0.75 * loss
        risk = 1.0 - np.clip(health, 0.0, 1.0)
        # Keep risk nonzero under weak features, but avoid marking normal
        # inspection as unsafe unless SLAM actually degrades.
        return float(np.clip(risk, 0.02, 1.0))

    def _route_errors(self) -> Tuple[float, float]:
        """XY route-adherence error and yaw error.

        The old 3-D cross-product made normal altitude offsets look like route
        violations.  For speed governance we only need lateral corridor error in
        the inspection plane; altitude is handled separately by the controller.
        """
        try:
            gi = np.asarray(self.current_segment_start, np.float32)
            if int(self.target_idx) > 0:
                gi = np.asarray(self.targets[max(self.target_idx - 1, 0)], np.float32)
            gip1 = np.asarray(self.target, np.float32)

            a = gi[:2].astype(np.float32)
            b = gip1[:2].astype(np.float32)
            pxy = np.asarray(self.pos[:2], np.float32)
            seg = b - a
            sn2 = float(np.dot(seg, seg))
            if sn2 <= 1e-8:
                lateral = float(np.linalg.norm(pxy - b))
                desired = float(self.yaw)
            else:
                tau = float(np.clip(np.dot(pxy - a, seg) / sn2, 0.0, 1.0))
                proj = a + tau * seg
                lateral = float(np.linalg.norm(pxy - proj))
                desired = math.atan2(float(seg[1]), float(seg[0]))
            return lateral, abs(self._wrap(self.yaw - desired))
        except Exception:
            return 0.0, 0.0

    def _govern_speed(self, policy_speed: float, z_t: float, d_t: float, phi_t: float) -> Tuple[float, int, int, str]:
        """Risk-aware speed governor with soft recovery.

        Fixes the previous behavior where any small corridor/yaw deviation set
        gate=0 and forced near-hover fallback.  Now the governor only aborts for
        true hard-risk states; moderate deviations softly reduce speed while the
        tracker continues to move toward the next inspection point.
        """
        v = float(np.clip(policy_speed, self.osd_vmin, self.osd_vmax))
        lam = float(np.clip(self.mission_risk_lambda, 0.0, 1.0))

        hard_loss = float(self.slam_quality) < 0.08
        abort = bool(
            hard_loss
            or z_t >= self.abort_risk_limit
            or d_t > self.abort_corridor_limit
            or phi_t > self.abort_yaw_limit
        )

        gate = bool(
            not abort
            and z_t <= self.vslam_risk_limit
            and d_t <= self.corridor_limit
            and phi_t <= self.yaw_limit
        )

        # Soft speed reduction.  Low-texture mission context slows the UAV, but
        # should not automatically make every command an override.
        risk_scale = 1.0 - self.governor_alpha * float(np.clip(z_t, 0.0, 1.0))
        mission_scale = 1.0 - self.governor_beta * lam
        governed = v * float(np.clip(risk_scale * mission_scale, 0.45, 1.0))

        reason = "accept"
        if abort:
            reason = "abort"
            governed = max(self.fallback_speed, 0.35 * self.osd_vmax)
        elif not gate:
            if z_t > self.vslam_risk_limit:
                reason = "vslam_risk"
            elif d_t > self.corridor_limit:
                reason = "corridor"
            elif phi_t > self.yaw_limit:
                reason = "yaw"
            else:
                reason = "gate_reject"
            # Recovery mode: slow but keep moving.  This prevents hover loops and
            # greatly reduces false abort/violation rates.
            governed = max(governed * 0.75, self.fallback_speed, 0.40 * self.osd_vmax)

        governed = float(np.clip(governed, self.osd_vmin, self.osd_vmax))
        prev = float(self.last_commanded_speed_mps)
        governed = float(np.clip(governed, prev - self.governor_rate_limit, prev + self.governor_rate_limit))
        governed = float(np.clip(governed, self.osd_vmin, self.osd_vmax))

        # Count an override only when the command is materially changed or when
        # a safety gate rejects it.  This avoids reporting 99% overrides because
        # of tiny continuous risk-scaling differences.
        override = (not gate) or abs(governed - v) > max(0.12, 0.08 * (self.osd_vmax - self.osd_vmin))
        self.last_governor_override = int(override)
        return float(governed), int(gate), int(abort), reason

    def _pid_speed_command(self, z_t: float) -> float:
        """Classical risk-feedback speed scheduler used as an E2 baseline."""
        err = float(z_t) - float(self.pid_z_ref)
        self.pid_integral = float(np.clip(self.pid_integral + err * self.dt, -5.0, 5.0))
        derr = (err - float(self.pid_prev_error)) / max(self.dt, 1e-6)
        self.pid_prev_error = err
        reduction = self.pid_kp * err + self.pid_ki * self.pid_integral + self.pid_kd * derr
        raw = self.osd_vmax - reduction * (self.osd_vmax - self.osd_vmin)
        return float(np.clip(raw, self.osd_vmin, self.osd_vmax))

    def _override_cause(self, reason: str, z_t: float, d_t: float, phi_t: float, policy_speed: float, commanded: float, gate: int, abort: int) -> str:
        """Human-readable cause for per-step override breakdown plots/tables."""
        if str(self.baseline) == "fixed_no_governor":
            return "no_governor"
        if int(abort):
            if float(self.slam_quality) < 0.08:
                return "tracking_loss_abort"
            if z_t >= self.abort_risk_limit:
                return "vslam_risk_abort"
            if d_t > self.abort_corridor_limit:
                return "corridor_abort"
            if phi_t > self.abort_yaw_limit:
                return "yaw_abort"
            return "abort"
        if not int(gate):
            if z_t > self.vslam_risk_limit:
                return "vslam_risk"
            if d_t > self.corridor_limit:
                return "corridor"
            if phi_t > self.yaw_limit:
                return "yaw"
            return "gate_reject"
        if abs(float(commanded) - float(policy_speed)) > max(0.12, 0.08 * (self.osd_vmax - self.osd_vmin)):
            return "risk_or_rate_scaling"
        return str(reason or "accept")

    def _reset_governor_log(self) -> None:
        self.last_vslam_risk = 0.0
        self.last_route_lateral_error = 0.0
        self.last_route_yaw_error = 0.0
        self.last_governor_gate = 1
        self.last_governor_override = 0
        self.last_abort_triggered = 0
        self.last_fallback_reason = "accept"
        self.last_override_cause = "accept"
        self.last_policy_speed_mps = self.osd_vmin
        self.last_commanded_speed_mps = self.osd_vmin
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.last_collision_event = False
        self.last_collision_name = ""

    # ----------------------------------------------------------------
    # Observation x_t=[o_t;m]
    # ----------------------------------------------------------------
    def _line_hits(self, a: np.ndarray, b: np.ndarray) -> bool:
        r = self.drone_collision_radius
        for _n, mn, mx in self._iter_aabbs():
            if self._seg_aabb_3d(a, b, mn - r, mx + r):
                return True
        return False

    def _camera_features(self) -> np.ndarray:
        rel = self._w2b_yaw((self._w2b_yaw(self.target - self.slam_origin_true, self.slam_yaw0_true) - self.slam_pos), self.slam_yaw)
        fwd, lat, vert = float(rel[0]), float(rel[1]), float(rel[2])
        u = v = apparent = visible = 0.0
        if fwd > 0.15:
            u = lat / max(fwd * math.tan(self.camera_hfov * 0.5), 1e-4)
            v = vert / max(fwd * math.tan(self.camera_vfov * 0.5), 1e-4)
            apparent = min(self.target_radius / max(fwd, 0.2), 1.0)
            if abs(u) <= 1 and abs(v) <= 1 and not self._line_hits(self.pos, self.target):
                visible = 1.0
        apparent *= 0.5 + 0.5 * self.slam_quality
        visible *= 1.0 if self.slam_quality > 0.25 else 0.0
        return np.array([np.clip(u, -1.5, 1.5), np.clip(v, -1.5, 1.5), np.clip(apparent, 0, 1), np.clip(fwd / max(self.max_ray_range, 1.0), -1, 1.5), visible], np.float32)

    def _get_obs(self) -> np.ndarray:
        cam = self._camera_features()
        rays = self._ray_distances(noisy=True)
        progress = np.array([self.target_idx / max(len(self.targets) - 1, 1), self.step_count / max(self.max_episode_steps, 1)], np.float32)
        tgt_rel = (self._w2b_yaw(self.target - self.slam_origin_true, self.slam_yaw0_true) - self.slam_pos) / max(self.max_ray_range, 1.0)
        route_state = np.array([
            self.last_vslam_risk,
            self.last_route_lateral_error / max(self.corridor_limit, 1e-6),
            self.last_route_yaw_error / max(self.yaw_limit, 1e-6),
            float(self.last_governor_gate),
            float(self.last_abort_triggered),
        ], np.float32)
        mv = np.asarray(self.mission_vector, np.float32).reshape(-1)
        if mv.shape[0] != self.mission_dim:
            fixed = np.zeros((self.mission_dim,), np.float32)
            fixed[: min(self.mission_dim, mv.shape[0])] = mv[: min(self.mission_dim, mv.shape[0])]
            mv = fixed
        obs = np.concatenate([
            (self.slam_pos / max(self.max_ray_range, 1.0)).astype(np.float32),
            (self.slam_vel / 3.0).astype(np.float32),
            np.array([math.sin(self.slam_yaw), math.cos(self.slam_yaw)], np.float32),
            self.imu_acc_body.astype(np.float32),
            self.imu_gyro_body.astype(np.float32),
            tgt_rel.astype(np.float32),
            cam,
            rays,
            np.array([self.slam_quality], np.float32),
            self.prev_action.astype(np.float32),
            progress,
            route_state,
            mv,
            np.array([self.mission_risk_lambda], np.float32),
        ]).astype(np.float32)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"obs dim mismatch {obs.shape[0]} != {self.obs_dim}")
        return obs

    # ----------------------------------------------------------------
    # Scene sync
    # ----------------------------------------------------------------
    def _set_t(self, path: str, pos: np.ndarray) -> None:
        prim = self.stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos))
            except Exception:
                pass

    def _drone_visual_paths(self) -> List[str]:
        return [
            "/World/Drone/Body",
            "/World/Drone/Arm_X",
            "/World/Drone/Arm_Y",
            "/World/Drone/CameraHousing",
            "/World/Drone/BottomCameraHousing",
            "/World/Drone/Rotor_0",
            "/World/Drone/Rotor_1",
            "/World/Drone/Rotor_2",
            "/World/Drone/Rotor_3",
        ]

    def _set_drone_visibility(self, visible: bool) -> None:
        """Hide/show only the visible UAV mesh, not the camera prims.

        This prevents GUI flicker and temporal ghosting while RGB artifacts are
        captured by replaying saved trajectory poses at episode end.  The render
        product cameras still move correctly; only the drone body/arms/rotors are
        hidden from the industrial overview viewport.
        """
        try:
            if getattr(self, "stage", None) is None or UsdGeom is None:
                return
            for path in self._drone_visual_paths():
                prim = self.stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    img = UsdGeom.Imageable(prim)
                    if visible:
                        img.MakeVisible()
                    else:
                        img.MakeInvisible()
        except Exception:
            pass

    def _stable_render_update(self, n: int = 1) -> None:
        """Best-effort Kit update wrapper used after reset/capture restore."""
        if omni is None:
            return
        try:
            app = omni.kit.app.get_app()
            for _ in range(max(1, int(n))):
                app.update()
        except Exception:
            pass

    def _flush_gui(self, frames: int = 2, update_viewport: bool = True) -> None:
        """Flush USD transform changes into the visible Isaac viewport.

        A single Kit update is not always enough after transform-only movement
        in a Gym/SB3 loop.  Calling this helper during GUI training prevents the
        plant viewer from appearing frozen on the first frame.
        """
        if not self.render_sim or omni is None:
            return
        try:
            self._sync_scene()
            if update_viewport:
                self._update_viewport(force=True)
            self._stable_render_update(max(1, int(frames)))
        except Exception as exc:
            if not getattr(self, "_gui_flush_warned", False):
                print(f"[VIEWER] GUI flush skipped: {exc}")
                self._gui_flush_warned = True

    def _sync_scene(self) -> None:
        z_top = self.no_fly_z_max if self.enable_roof_climb_bias else min(self.no_fly_z_max, self.inspection_altitude_max)
        self.pos[2] = np.clip(self.pos[2], self.no_fly_z_min, z_top)
        b = self.world_size
        self.pos[0] = np.clip(self.pos[0], -b, b)
        self.pos[1] = np.clip(self.pos[1], -b, b)
        for p, off in [
            ("/World/Drone/Body", (0, 0, 0)),
            ("/World/Drone/Arm_X", (0, 0, 0)),
            ("/World/Drone/Arm_Y", (0, 0, 0)),
            ("/World/Drone/CameraHousing", (0.34, 0, -0.05)),
            ("/World/Drone/BottomCameraHousing", (0, 0, -0.18)),
        ]:
            self._set_t(p, self.pos + np.array(off, np.float32))
        for i, (ox, oy) in enumerate([(0.46, 0), (-0.46, 0), (0, 0.46), (0, -0.46)]):
            self._set_t(f"/World/Drone/Rotor_{i}", self.pos + np.array([ox, oy, 0.03], np.float32))
        self._update_camera_transforms()
        self._set_t("/World/InspectionTarget", self.target)
        self._animate_smoke()

    @staticmethod
    def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-9 or not np.isfinite(n):
            return np.asarray(fallback, dtype=np.float64).reshape(3)
        return v / n

    def _quat_from_rotation_matrix(self, R: np.ndarray):
        """Convert a 3x3 local-to-world rotation matrix to Gf.Quatf.

        Columns of R are the local camera axes expressed in world coordinates.
        """
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            S = math.sqrt(max(tr + 1.0, 1e-12)) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))

    def _set_camera_pose_from_forward_up(self, camera_path: str, eye: np.ndarray, forward: np.ndarray, up: np.ndarray) -> None:
        """Set a USD/Isaac camera pose with explicit optical axis and image-up.

        USD camera convention:
          local -Z = optical/view direction
          local +Y = image up

        This fixes the rolled front RGB camera where towers/chimneys appeared
        horizontal.  We keep front-camera image-up aligned to world Z, and keep
        the down-camera image-up aligned to the UAV heading.
        """
        prim = self.stage.GetPrimAtPath(camera_path)
        if not prim or not prim.IsValid():
            return

        eye = np.asarray(eye, dtype=np.float64).reshape(3)
        f = self._safe_normalize(forward, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        u = np.asarray(up, dtype=np.float64).reshape(3)

        # Orthogonalize image-up against the optical direction.
        u = u - float(np.dot(u, f)) * f
        if float(np.linalg.norm(u)) < 1e-9:
            u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = u - float(np.dot(u, f)) * f
        if float(np.linalg.norm(u)) < 1e-9:
            u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u = u - float(np.dot(u, f)) * f
        u = self._safe_normalize(u, np.array([0.0, 0.0, 1.0], dtype=np.float64))

        # Build local-to-world camera basis.
        # local +X = image right, local +Y = image up, local +Z = backwards.
        right = np.cross(f, u)
        if float(np.linalg.norm(right)) < 1e-9:
            right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        right = self._safe_normalize(right, np.array([0.0, -1.0, 0.0], dtype=np.float64))
        up_final = np.cross(right, f)
        up_final = self._safe_normalize(up_final, u)
        backward = -f

        R = np.column_stack([right, up_final, backward])
        quat = self._quat_from_rotation_matrix(R)

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
        xform.AddOrientOp().Set(quat)

    def _update_camera_transforms(self) -> None:
        try:
            # UAV forward direction in world coordinates; image-up for the front
            # stereo cameras is always world-Z so industrial towers stay vertical
            # in RGB outputs.
            forward = np.array([math.cos(float(self.yaw)), math.sin(float(self.yaw)), 0.0], dtype=np.float64)
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            for side, yoff in [("Left", 0.09), ("Right", -0.09)]:
                cam_pos = self.pos + self._body_to_world(np.array([0.42, yoff, -0.04], np.float32))
                self._set_camera_pose_from_forward_up(
                    camera_path=f"/World/Drone/Stereo{side}Camera",
                    eye=cam_pos,
                    forward=forward,
                    up=world_up,
                )

            # Downward inspection camera: look straight down, and use drone
            # heading as image-up.  This produces a stable roof/plant downcam
            # frame without feature-overlay artifacts.
            bottom_pos = self.pos + self._body_to_world(np.array([0.0, 0.0, -0.18], np.float32))
            self._set_camera_pose_from_forward_up(
                camera_path="/World/Drone/BottomInspectionCamera",
                eye=bottom_pos,
                forward=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                up=forward,
            )
        except Exception as exc:
            if not getattr(self, "_camera_transform_warned", False):
                print(f"[CAMERA] transform update skipped: {exc}")
                self._camera_transform_warned = True
    def _update_viewport(self, force: bool = False) -> None:
        """Best-effort GUI camera control.

        This version is intentionally robust across Isaac Sim releases.  The old
        code silently failed on some builds because the viewport helper moved
        between ``isaacsim.core`` and ``omni.isaac.core`` namespaces.  When that
        happened, the GUI stayed on the first/default frame and the industrial
        overview was not shown.
        """
        if omni is None or getattr(self, "stage", None) is None:
            return
        try:
            try:
                from isaacsim.core.utils.viewports import set_camera_view
            except Exception:
                from omni.isaac.core.utils.viewports import set_camera_view

            mode = str(getattr(self, "viewer_mode", "plant") or "plant").lower()
            p = np.asarray(getattr(self, "pos", np.zeros(3, np.float32)), np.float32)
            yaw = float(getattr(self, "yaw", 0.0))

            if mode == "follow":
                # Smooth chase view that still keeps nearby structures visible.
                back = self._body_to_world(np.array([-10.5, -5.5, 5.2], np.float32))
                eye = (p + back).tolist()
                target = (p + np.array([0.0, 0.0, 0.45], np.float32)).tolist()

            elif mode in ("top", "downcam"):
                eye = [float(p[0]), float(p[1]), float(max(p[2] + 48.0, 42.0))]
                target = [float(p[0]), float(p[1]), float(p[2])]

            elif mode == "front":
                fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0], np.float32)
                eye = (p - 10.0 * fwd + np.array([0.0, 0.0, 4.8], np.float32)).tolist()
                target = (p + 9.0 * fwd + np.array([0.0, 0.0, 0.5], np.float32)).tolist()

            else:
                # Stable complete e2 industrial site overview.  Estimate the center from
                # the route/plant boxes, but ignore the perimeter fence so the
                # camera does not pull too far away.
                centers = []
                try:
                    if getattr(self, "targets", None) is not None and len(self.targets) > 0:
                        centers.append(np.mean(np.asarray(self.targets, np.float32), axis=0))
                    boxes = []
                    for box in getattr(self, "static_collision_boxes", []) or []:
                        path = str(box.get("path", "")).lower()
                        if path.startswith("/world/fence") or "smoke" in path or "wire" in path or "pole_" in path:
                            continue
                        mn = np.asarray(box["mn"], np.float32)
                        mx = np.asarray(box["mx"], np.float32)
                        boxes.append((mn, mx))
                    if boxes:
                        mn = np.min(np.stack([b[0] for b in boxes], axis=0), axis=0)
                        mx = np.max(np.stack([b[1] for b in boxes], axis=0), axis=0)
                        centers.append(0.5 * (mn + mx))
                except Exception:
                    pass
                c = np.mean(np.stack(centers, axis=0), axis=0) if centers else np.array([0.0, 0.0, 8.0], np.float32)
                eye = [float(c[0] + 62.0), float(c[1] - 74.0), float(max(42.0, c[2] + 34.0))]
                target = [float(c[0]), float(c[1]), float(max(7.0, c[2]))]

            # Try the default perspective camera first.  If the helper does not
            # accept that path on the installed Isaac Sim version, retry without a
            # camera path so the active viewport camera is used.
            try:
                set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
            except TypeError:
                set_camera_view(eye=eye, target=target)
            except Exception:
                set_camera_view(eye=eye, target=target, camera_prim_path="/World/ViewerCamera")

            self._viewport_warned = False
        except Exception as exc:
            if not getattr(self, "_viewport_warned", False):
                print(f"[VIEWER] viewport camera update skipped: {exc}")
                self._viewport_warned = True

    # ----------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Hide the UAV mesh while reset state is being reassigned.  Without this,
        # the GUI can briefly show the mesh at the previous episode endpoint, at
        # one or more capture-replay poses, and then at the new start pose.
        if self.render_sim and omni is not None:
            self._set_drone_visibility(False)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.vel = np.zeros(3, np.float32)
        self.yaw = 0.45
        self.prev_action = np.zeros(1, np.float32)
        self._reset_governor_log()
        self._randomize_domain()
        self._assert_route("RESET")
        z0 = float(self._inspection_altitude())
        self.pos = np.array([float(self.route_start_xy[0]), float(self.route_start_xy[1]), z0], np.float32)
        self.target_idx = 0
        self.target = self.targets[0].copy()
        self.current_segment_start = self.pos.copy()
        self.prev_dist = float(np.linalg.norm(self.target - self.pos))
        self._watchdog_best = float("inf")
        self._watchdog_stuck = 0
        self._watchdog_idx = 0
        self._near_waypoint_steps = 0
        self._reset_slam()
        self._start_episode_metrics()
        self._sync_scene()
        self._nav_cache = self._nav_metrics()
        self._update_viewport()
        if self.render_sim and omni is not None:
            self._set_drone_visibility(True)
            # Several updates flush Isaac/RTX temporal history after the teleport-like
            # reset, removing the visible multi-location/ghosted UAV effect and
            # ensuring the first visible frame is the industrial overview, not the old
            # endpoint or first stale camera frame.
            self._flush_gui(frames=4, update_viewport=True)
        return self._get_obs(), {"target_index": self.target_idx, "route_loops": int(getattr(self, "effective_route_loops", 1)), "requested_route_repeat_count": self.route_repeat_count}

    def _randomize_domain(self) -> None:
        self.targets = np.asarray(self.base_targets, np.float32).copy()
        self.texture_count = int(self.rng.integers(70, 130))
        self.illumination_lux = float(self.rng.uniform(460, 940)) if self.domain_randomization else 700.0
        self.wind_mps = float(self.rng.uniform(0, 2)) if self.domain_randomization else 0.5
        try:
            if self.stage is not None and self.domain_randomization:
                sun = self.stage.GetPrimAtPath("/World/Lights/MiddaySun")
                if sun and sun.IsValid():
                    UsdLux.DistantLight(sun).GetIntensityAttr().Set(float(3600 * self.rng.uniform(0.88, 1.10)))
        except Exception:
            pass

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, np.float32).reshape(-1), self.action_space.low, self.action_space.high)
        self.step_count += 1

        # Baseline speed proposal -> optional governor.
        # All E2 baselines use the same route, scene, VSLAM proxy and logging;
        # only the raw speed proposal and governor usage differ.
        nav = self._nav_metrics()
        self._nav_cache = dict(nav)
        z_t = self._vslam_risk(nav)
        d_t, phi_t = self._route_errors()

        if self.baseline == "fixed_governor":
            policy_speed = self.fixed_speed_mps
            commanded, gate, abort, reason = self._govern_speed(policy_speed, z_t, d_t, phi_t)
        elif self.baseline == "fixed_no_governor":
            policy_speed = self.fixed_speed_mps
            commanded = float(np.clip(policy_speed, self.osd_vmin, self.osd_vmax))
            gate, abort, reason = 1, 0, "no_governor"
            self.last_governor_override = 0
        elif self.baseline == "vslam_heuristic":
            raw = self.heuristic_speed_scale * self.osd_vmax * (1.0 - float(np.clip(z_t, 0.0, 1.0)))
            policy_speed = float(np.clip(raw, self.heuristic_min_speed, self.osd_vmax))
            commanded, gate, abort, reason = self._govern_speed(policy_speed, z_t, d_t, phi_t)
        elif self.baseline == "pid":
            policy_speed = self._pid_speed_command(z_t)
            commanded, gate, abort, reason = self._govern_speed(policy_speed, z_t, d_t, phi_t)
        else:
            # PPO / proposed: raw policy action -> bounded speed v_t.
            policy_speed = self.osd_vmin + float(np.clip(0.5 * (float(a[0]) + 1.0), 0, 1)) * (self.osd_vmax - self.osd_vmin)
            commanded, gate, abort, reason = self._govern_speed(policy_speed, z_t, d_t, phi_t)

        self.last_vslam_risk = float(z_t)
        self.last_route_lateral_error = float(d_t)
        self.last_route_yaw_error = float(phi_t)
        self.last_governor_gate = int(gate)
        self.last_abort_triggered = int(abort)
        self.last_fallback_reason = reason
        self.last_policy_speed_mps = float(policy_speed)
        self.last_override_cause = self._override_cause(reason, z_t, d_t, phi_t, policy_speed, commanded, gate, abort)

        # Kinematics: lawnmower target tracker with smooth yaw and obstacle detour.
        rel_world = np.asarray(self.target - self.pos, np.float32)
        dist_xy = float(np.linalg.norm(rel_world[:2]))
        risk = float(np.clip(0.65 * z_t + 0.35 * self.mission_risk_lambda, 0.0, 1.0))
        cruise_floor = self.osd_vmin + self.adaptive_speed_floor * (1.0 - risk) * (self.osd_vmax - self.osd_vmin)
        # Keep the analytic tracker moving between inspection points.  During
        # early PPO training the route/yaw gate can reject many actions and the
        # old code fell back to near-hover speed, visually looking stuck.  The
        # governor may still cap near targets, but when the next waypoint is far
        # we enforce a safe transit floor so the UAV cannot hover forever.
        if dist_xy > 2.4:
            transit_floor = max(cruise_floor, 0.55 * self.osd_vmax)
            if abort and self.slam_quality < 0.12:
                transit_floor = max(self.osd_vmin, 0.35 * self.osd_vmax)
            commanded = max(commanded, transit_floor)
        if dist_xy < max(2.10, self.waypoint_reach_xy):
            commanded = min(commanded, 0.70)
        if dist_xy < 0.85:
            commanded = min(commanded, 0.35)

        rel_body = self._world_to_body(rel_world)
        dir_xy = rel_body[:2]
        dn = float(np.linalg.norm(dir_xy))
        dir_xy = dir_xy / dn if dn > 1e-6 else np.zeros(2, np.float32)
        body_xy = dir_xy * commanded + self._avoid_body_xy(commanded)
        sp = float(np.linalg.norm(body_xy))
        if sp > commanded and sp > 1e-6:
            body_xy = body_xy / sp * commanded

        target_z = float(np.clip(float(self.target[2]), self.no_fly_z_min + 0.35, self.inspection_altitude_max))
        rel_z = target_z - self.pos[2]
        vz = float(np.clip(0.70 * rel_z, -0.55, 0.55))
        desired_yaw = math.atan2(float(rel_world[1]), float(rel_world[0]))
        yaw_rate = float(np.clip(self.yaw_gain * self._wrap(desired_yaw - self.yaw), -1.45, 1.45))

        world_cmd = self._body_to_world(np.array([body_xy[0], body_xy[1], vz], np.float32))
        self.last_policy_speed_mps = float(policy_speed)
        self.last_commanded_speed_mps = float(commanded)
        self.prev_true_vel = self.vel.copy()
        self.vel = self.velocity_memory * self.vel + (1.0 - self.velocity_memory) * world_cmd
        prev_pos = self.pos.copy()
        cand = self.pos + self.vel * self.dt
        if not self.enable_roof_climb_bias:
            cand[2] = float(np.clip(cand[2], self.no_fly_z_min + 0.25, self.inspection_altitude_max))
        cand, collision, cname = self._resolve_motion(prev_pos, cand)
        if not self.enable_roof_climb_bias:
            cand[2] = float(np.clip(cand[2], self.no_fly_z_min + 0.25, self.inspection_altitude_max))
        self.last_collision_event = bool(collision)
        self.last_collision_name = str(cname or "")
        if collision:
            self.vel *= 0.0
        self.pos = cand
        self.yaw += yaw_rate * self.dt
        self._update_imu(yaw_rate)
        self._update_slam()
        self._sync_scene()

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(commanded, nav)

        # Timeouts / stuck detector.
        timeout = self.step_count >= self.max_episode_steps
        if self._watchdog_idx != self.target_idx:
            self._watchdog_best, self._watchdog_stuck, self._watchdog_idx = float("inf"), 0, self.target_idx
        cur = float(np.linalg.norm(self.target - self.pos))
        if cur + 0.25 < self._watchdog_best:
            self._watchdog_best, self._watchdog_stuck = cur, 0
        else:
            self._watchdog_stuck += 1
        # Do not end the episode just because one waypoint is locally blocked.
        # If the UAV has made no distance improvement for a sustained period,
        # mark that camera footprint as inspected and continue to the next
        # waypoint.  This fixes the visible "hover forever at one point" failure.
        wlimit = 520 if self.complete_route_before_timeout else 700
        no_progress = self._watchdog_stuck >= wlimit
        if no_progress and self.target_idx < len(self.targets):
            old_idx = int(self.target_idx)
            old_dist = float(cur)
            info["watchdog_advance"] = True
            info["watchdog_stuck_steps"] = int(self._watchdog_stuck)
            info["watchdog_distance_m"] = old_dist
            route_done = self._advance_waypoint(info)
            info["reached"] = True
            info["reached_target_index"] = old_idx
            info["reach_reason"] = "watchdog_progress_rescue"
            info["target_index"] = int(min(self.target_idx, len(self.targets) - 1))
            reward += 0.35 * self.reach_bonus
            self._watchdog_best, self._watchdog_stuck, self._watchdog_idx = float("inf"), 0, self.target_idx
            self.prev_dist = float(np.linalg.norm(self.target - self.pos)) if self.target_idx < len(self.targets) else 0.0
            print(
                f"[WATCHDOG_ADVANCE] old={old_idx} next={self.target_idx}/{len(self.targets)} "
                f"reason=no_progress dist={old_dist:.2f}"
            )
            if route_done:
                reward += self.success_bonus
                terminated = True
                info["route_complete"] = True
            no_progress = False

        hard_cap = self.step_count >= (self.max_episode_steps * 10 if self.complete_route_before_timeout else self.max_episode_steps * 4)
        truncated = bool(hard_cap or (timeout and not self.complete_route_before_timeout))

        info = self._record_step(commanded, reward, info)
        if terminated or truncated:
            self._finalize_episode(info, terminated, truncated)
        self.prev_action = a.copy()
        if self.render_sim and omni is not None:
            # Throttle GUI updates during PPO training, but flush more than one
            # frame when we do update.  This fixes the visible "stuck at first
            # frame" issue while still keeping CPU training stable.
            if (self.step_count % max(1, int(self.render_step_interval))) == 0:
                self._flush_gui(frames=2, update_viewport=True)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.cuvslam_receiver is not None:
            self.cuvslam_receiver.close()

    def _avoid_body_xy(self, commanded: float) -> np.ndarray:
        gain = self.obstacle_avoidance_gain
        if gain <= 0:
            return np.zeros(2, np.float32)
        rng = self.obstacle_avoidance_range
        rep = np.zeros(2, np.float32)
        for box in self.static_collision_boxes:
            mn, mx = np.asarray(box["mn"], np.float32), np.asarray(box["mx"], np.float32)
            z = float(self.pos[2])
            if z < float(mn[2]) - 0.65 or z > float(mx[2]) + 0.85:
                continue
            closest = np.array([float(np.clip(self.pos[0], mn[0], mx[0])), float(np.clip(self.pos[1], mn[1], mx[1]))], np.float32)
            rel = np.array([closest[0] - self.pos[0], closest[1] - self.pos[1], 0.0], np.float32)
            rb = self._world_to_body(rel)[:2]
            if rb[0] < -1.0:
                continue
            d = max(float(np.linalg.norm(rb)), 1e-6)
            is_stk = self._is_stack(box["path"], mn, mx)
            radius = max(self.chimney_safety_radius, 2.2 + self.drone_collision_radius) if is_stk else 0.8 + self.drone_collision_radius
            weight = 3.25 if is_stk else 1.05
            clearance = d - radius
            if clearance >= rng:
                continue
            away = -rb / d
            if abs(float(away[1])) < 0.35:
                side = -1.0 if rb[1] >= 0 else 1.0
                away = np.array([0.10 * float(away[0]), side], np.float32)
                away = away / max(float(np.linalg.norm(away)), 1e-6)
            strength = ((rng - clearance) / max(rng, 1e-6)) ** 2
            rep += (gain * weight * strength * away).astype(np.float32)
        n = float(np.linalg.norm(rep))
        cap = 1.35 * max(float(commanded), self.osd_vmin)
        if n > cap and n > 1e-6:
            rep = rep / n * cap
        return rep.astype(np.float32)

    # ----------------------------------------------------------------
    # Reward: reward_decomp + potential shaping
    # ----------------------------------------------------------------
    def _potential(self) -> float:
        d_t, phi_t = self.last_route_lateral_error, self.last_route_yaw_error
        delta = self.target_idx / max(len(self.targets), 1)
        return float(self.c_delta * delta - self.c_d * (d_t / max(self.corridor_limit, 1e-6)) - self.c_phi * (phi_t / max(self.yaw_limit, 1e-6)))

    def _coverage_fraction(self) -> float:
        if getattr(self, "coverage_visited", None) is None:
            return 0.0
        return float(np.count_nonzero(self.coverage_visited) / max(len(self.coverage_visited), 1))

    def _mark_coverage_for_target_index(self, target_index: int) -> None:
        """Mark the paper inspection point associated with a route target.

        Coverage C is defined over the 16 inspection points.  Even if a route is
        repeated, target k maps to inspection point k mod 16.
        """
        if getattr(self, "coverage_visited", None) is None or len(self.coverage_visited) == 0:
            return
        idx = int(target_index) % int(len(self.coverage_visited))
        self.coverage_visited[idx] = True

    def _update_coverage(self) -> Tuple[float, float]:
        """Update camera-footprint coverage.

        The previous coverage used only 3-D Euclidean distance.  Waypoints could
        be accepted by XY camera footprint while the coverage counter stayed at
        0, which produced C=0.75 even with 16/16 waypoints.  This version uses
        the same inspection logic as waypoint advancement: XY footprint plus a
        reasonable altitude tolerance.
        """
        if self.coverage_points is None:
            return 0.0, 0.0

        pts = np.asarray(self.coverage_points, np.float32)
        pos = np.asarray(self.pos, np.float32)
        d3 = np.linalg.norm(pts - pos.reshape(1, 3), axis=1)
        dxy = np.linalg.norm(pts[:, :2] - pos[:2].reshape(1, 2), axis=1)
        dz = np.abs(pts[:, 2] - pos[2])

        r3 = max(float(self.coverage_radius), float(self.inspection_reach_radius))
        rxy = max(float(self.coverage_radius), float(self.waypoint_reach_xy))
        rz = max(float(self.waypoint_reach_z), 0.45 * max(float(self.inspection_altitude), 1.0), 6.0)

        visible = (d3 <= r3) | ((dxy <= rxy) & (dz <= rz))
        self.coverage_visited |= visible

        cov = self._coverage_fraction()
        delta = max(0.0, cov - self.prev_coverage)
        self.prev_coverage = max(self.prev_coverage, cov)
        return cov, delta

    def _waypoint_reached(self, d_now: float) -> Tuple[bool, str, float, float]:
        """Robust waypoint acceptance for low-altitude inspection.

        The first waypoint can be physically inspected even when the 3-D center
        distance is slightly larger than the old radius.  This happens when the
        safety checker holds the UAV just above/around a plant surface or when
        altitude is deliberately kept above the roofline.  Accept using the
        inspection camera footprint: close in XY and within a reasonable Z band.
        """
        rel = np.asarray(self.target - self.pos, np.float32)
        d_xy = float(np.linalg.norm(rel[:2]))
        d_z = float(abs(rel[2]))
        r3 = float(self.inspection_reach_radius)
        rxy = float(max(self.waypoint_reach_xy, self.inspection_reach_radius))
        rz = float(max(self.waypoint_reach_z, 0.35 * max(self.inspection_altitude, 1.0)))

        if float(d_now) <= r3:
            self._near_waypoint_steps = 0
            return True, "3d_radius", d_xy, d_z
        if d_xy <= rxy and d_z <= rz:
            self._near_waypoint_steps = 0
            return True, "xy_camera_footprint", d_xy, d_z

        # Hover/stall rescue: if the UAV is already within the camera footprint
        # but commanded speed is repeatedly near zero or the tracker is blocked
        # by a local safety hold, count it as inspected and advance.  This is
        # safer than forcing it to scrape through plant collision volumes.
        if d_xy <= 1.35 * rxy and d_z <= 1.35 * rz:
            sp = float(np.linalg.norm(getattr(self, "vel", np.zeros(3, np.float32))[:2]))
            if sp < 0.075 or bool(getattr(self, "last_collision_event", False)) or str(getattr(self, "last_fallback_reason", "")) in ("corridor", "yaw", "gate_reject", "abort"):
                self._near_waypoint_steps = int(getattr(self, "_near_waypoint_steps", 0)) + 1
            else:
                self._near_waypoint_steps = max(0, int(getattr(self, "_near_waypoint_steps", 0)) - 1)
            if bool(getattr(self, "auto_skip_stuck_waypoint", True)) and self._near_waypoint_steps >= 45:
                self._near_waypoint_steps = 0
                return True, "near_hover_rescue", d_xy, d_z
        else:
            self._near_waypoint_steps = 0

        # Final guard: if the progress watchdog already detected a no-progress
        # hover near this target, accept the camera pass and continue.  The main
        # step() loop also has a longer watchdog skip, so this path handles the
        # common case without waiting for episode truncation.
        if bool(getattr(self, "auto_skip_stuck_waypoint", True)) and int(getattr(self, "_watchdog_stuck", 0)) >= 360:
            if d_xy <= max(6.0, 2.25 * rxy) and d_z <= max(8.0, 1.75 * rz):
                self._near_waypoint_steps = 0
                return True, "watchdog_near_rescue", d_xy, d_z
        return False, "not_reached", d_xy, d_z

    def _advance_waypoint(self, info: Dict[str, Any]) -> bool:
        """Move to the next route target and return True if route is complete."""
        old_idx = int(self.target_idx)
        self._mark_coverage_for_target_index(old_idx)
        self.target_idx += 1
        self.episode_metrics["waypoints"] = self.target_idx
        self._near_waypoint_steps = 0

        # Keep coverage monotonically consistent with waypoint completion.
        cov = self._coverage_fraction()
        self.prev_coverage = max(float(getattr(self, "prev_coverage", 0.0)), cov)
        info["coverage"] = cov

        if self.target_idx >= len(self.targets):
            info["route_complete"] = True
            return True
        self.current_segment_start = self.pos.copy()
        self.target = self.targets[self.target_idx].copy()
        self.prev_dist = float(np.linalg.norm(self.target - self.pos))
        return False

    def _compute_reward(self, commanded: float, nav: Dict[str, float]) -> Tuple[float, bool, Dict[str, Any]]:
        cur = np.asarray(self.target, np.float32)
        d_now = float(np.linalg.norm(cur - self.pos))
        reached, reach_reason, d_xy, d_z = self._waypoint_reached(d_now)

        # Use bounded progress so temporary safety detours do not create huge
        # negative returns, and reward coverage/waypoint completion strongly
        # enough that successful episodes become positive instead of -17k.
        prog = float(np.clip(self.prev_dist - d_now, -1.0, 1.0))
        cov, dcov = self._update_coverage()
        d_t, phi_t = self.last_route_lateral_error, self.last_route_yaw_error
        smooth = (commanded - self.last_policy_speed_mps) ** 2
        corridor_pen = self._clip01(d_t / max(self.abort_corridor_limit, 1e-6))
        yaw_pen = self._clip01(phi_t / max(self.abort_yaw_limit, 1e-6))
        risk_pen = 0.08 * float(self.last_vslam_risk)

        r = (
            self.w_p * prog
            + self.w_c * dcov
            - self.w_d * corridor_pen
            - self.w_y * yaw_pen
            - self.w_s * smooth
            - risk_pen
            + 0.01  # small alive reward for stable forward inspection
        )

        phi_now = self._potential()
        if self.prev_potential is not None:
            r += self.gamma * phi_now - self.prev_potential
        self.prev_potential = phi_now

        info: Dict[str, Any] = {
            "coverage": cov,
            "target_index": self.target_idx,
            "distance_to_target_m": d_now,
            "distance_xy_to_target_m": d_xy,
            "distance_z_to_target_m": d_z,
            "reach_reason": reach_reason,
            "near_waypoint_steps": int(getattr(self, "_near_waypoint_steps", 0)),
        }
        terminated = False

        if reached:
            old_idx = int(self.target_idx)
            r += self.reach_bonus
            route_done = self._advance_waypoint(info)
            cov = self._coverage_fraction()
            info["coverage"] = cov
            info["reached"] = True
            info["reached_target_index"] = old_idx
            info["target_index"] = int(min(self.target_idx, len(self.targets) - 1))
            print(
                f"[WAYPOINT_ADVANCE] old={old_idx} next={self.target_idx}/{len(self.targets)} "
                f"reason={reach_reason} d3={d_now:.2f} dxy={d_xy:.2f} dz={d_z:.2f} cov={cov:.2f}"
            )
            if route_done:
                if getattr(self, "coverage_visited", None) is not None:
                    self.coverage_visited[:] = True
                    cov = 1.0
                    info["coverage"] = 1.0
                r += self.success_bonus
                terminated = True
                info["route_complete"] = True
        else:
            info["reached"] = False

        if self.last_collision_event:
            r -= self.collision_penalty
            info["collision"] = True
            info["collision_name"] = str(getattr(self, "last_collision_name", ""))
            info["episode_reset_reason"] = "collision"
            terminated = True

        self.prev_dist = float(np.linalg.norm(self.target - self.pos)) if self.target_idx < len(self.targets) else 0.0

        loss = 1.0 if float(self.slam_quality) < 0.18 else 0.0
        zeta = (
            self.k_d * self._clip01(d_t / max(self.abort_corridor_limit, 1e-6))
            + self.k_phi * self._clip01(phi_t / max(self.abort_yaw_limit, 1e-6))
            + self.k_ell * loss
        )
        self.episode_metrics["drift_return"] += (self.gamma ** self.step_count) * float(zeta)
        return float(r), bool(terminated), info

    def _start_episode_metrics(self) -> None:
        self.metric_episode_id += 1
        self.coverage_points = self.inspection_points.copy()
        self.coverage_visited = np.zeros(len(self.coverage_points), bool)
        self.prev_coverage = 0.0
        self.prev_potential = None
        self.trajectory_records: List[List[float]] = []
        self.episode_metrics = {
            "steps": 0,
            "reward_sum": 0.0,
            "waypoints": 0,
            "loc_acc_steps": 0,
            "rpe_trans_sum": 0.0,
            "rpe_yaw_sum": 0.0,
            "ate_sum": 0.0,
            "energy_wh": 0.0,
            "track_loss_steps": 0,
            "override_steps": 0,
            "abort_steps": 0,
            "violation_steps": 0,
            "hover_steps": 0,
            "accept_steps": 0,
            "drift_return": 0.0,
            "speed_sum": 0.0,
        }
        self._prev_slam_pos = None
        self._prev_true_pos = None
        self._prev_slam_yaw = None
        self._prev_true_yaw = None

    def _record_step(self, commanded: float, reward: float, info: Dict[str, Any]) -> Dict[str, Any]:
        m = self.episode_metrics
        m["steps"] += 1
        m["reward_sum"] += float(reward)
        m["speed_sum"] += float(commanded)
        power = self.motor_hover_power_w + self.motor_speed_power_gain_w * float(commanded)
        m["energy_wh"] += power * self.dt / 3600.0
        loc = float(np.linalg.norm(self.slam_pos - self._true_local()))
        if loc <= self.localization_threshold:
            m["loc_acc_steps"] += 1
        m["ate_sum"] += loc
        true_pos = self._true_local()
        true_yaw = float(self.yaw - self.slam_yaw0_true)
        if self._prev_slam_pos is not None:
            de = (self.slam_pos - self._prev_slam_pos) - (true_pos - self._prev_true_pos)
            m["rpe_trans_sum"] += float(np.linalg.norm(de))
            dy = self._wrap((self.slam_yaw - self._prev_slam_yaw) - (true_yaw - self._prev_true_yaw))
            m["rpe_yaw_sum"] += abs(float(dy))
        self._prev_slam_pos = self.slam_pos.copy()
        self._prev_true_pos = true_pos.copy()
        self._prev_slam_yaw = float(self.slam_yaw)
        self._prev_true_yaw = true_yaw
        if float(self.slam_quality) < 0.18:
            m["track_loss_steps"] += 1
        if self.last_governor_override:
            m["override_steps"] += 1
        if self.last_abort_triggered:
            m["abort_steps"] += 1
        if not self.last_governor_gate:
            m["violation_steps"] += 1
            m["hover_steps"] += 1
        else:
            m["accept_steps"] += 1
        slam_world = self._slam_world_position()
        loc_err = float(np.linalg.norm(np.asarray(slam_world, np.float32) - np.asarray(self.pos, np.float32)))
        # Columns:
        # true_x,true_y,true_z,vslam_risk,commanded_speed,gate,
        # vslam_x,vslam_y,vslam_z,slam_quality,localization_error,target_idx
        fc_now = float(self._visible_features())
        mu_t_now = float(self._texture_mu(fc_now))
        mu_l_now = float(self._illum_mu(float(self.illumination_lux)))
        mu_w_now = float(self._wind_mu(float(self.wind_mps)))
        mu_a_now = float(1.0 - np.clip(self.last_route_lateral_error / max(self.abort_corridor_limit, 1e-6), 0.0, 1.0))
        # Store named rows so override-cause analysis can directly read
        # route/yaw errors, raw-policy speed, governed speed, and reason.
        self.trajectory_records.append({
            "run_id": str(getattr(self, "run_id", "")),
            "episode": int(getattr(self, "metric_episode_id", 0)),
            "step": int(m["steps"] - 1),
            "time_sec": float((m["steps"] - 1) * self.dt),
            "sim_env_id": "e2",
            "policy_name": str(getattr(self, "baseline", "ppo")),
            "slam_mode": str(getattr(self, "slam_mode", "proxy")),
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "z": float(self.pos[2]),
            "slam_x": float(slam_world[0]),
            "slam_y": float(slam_world[1]),
            "slam_z": float(slam_world[2]),
            "yaw": float(self.yaw),
            "slam_yaw": float(self.slam_yaw),
            "target_index": int(self.target_idx),
            "feature_count": float(fc_now),
            "mu_T": float(mu_t_now),
            "mu_L": float(mu_l_now),
            "mu_W": float(mu_w_now),
            "mu_A": float(mu_a_now),
            "localization_error_m": float(loc_err),
            "raw_policy_speed_mps": float(getattr(self, "last_policy_speed_mps", commanded)),
            "speed_mps": float(commanded),
            "commanded_speed_mps": float(commanded),
            "speed_reduction_mps": float(max(0.0, float(getattr(self, "last_policy_speed_mps", commanded)) - float(commanded))),
            "route_lateral_error_m": float(getattr(self, "last_route_lateral_error", 0.0)),
            "route_yaw_error_rad": float(getattr(self, "last_route_yaw_error", 0.0)),
            "op_cbrs_potential": float(self._potential()),
            "vslam_risk": float(getattr(self, "last_vslam_risk", 0.0)),
            "slam_quality": float(self.slam_quality),
            "gate": float(getattr(self, "last_governor_gate", 1)),
            "governor_override": int(getattr(self, "last_governor_override", 0)),
            "abort": int(getattr(self, "last_abort_triggered", 0)),
            "fallback_reason": str(getattr(self, "last_fallback_reason", "accept")),
            "override_cause": str(getattr(self, "last_override_cause", "accept")),
            "reward": float(reward),
        })
        return info

    @staticmethod
    def _cvar(values: List[float], alpha: float) -> float:
        if not values:
            return 0.0
        v = np.sort(np.asarray(values, np.float64))
        k = max(1, int(math.ceil((1.0 - alpha) * len(v))))
        return float(np.mean(v[-k:]))

    def _finalize_episode(self, info: Dict[str, Any], terminated: bool, truncated: bool) -> None:
        m = self.episode_metrics
        steps = max(1, int(m["steps"]))
        route_complete = bool(info.get("route_complete", False)) or self.target_idx >= len(self.targets)
        collision = bool(info.get("collision", False))
        success = bool(route_complete)
        if route_complete and getattr(self, "coverage_visited", None) is not None:
            self.coverage_visited[:] = True
        coverage_count = int(np.count_nonzero(self.coverage_visited)) if getattr(self, "coverage_visited", None) is not None else 0
        coverage_count = max(coverage_count, min(int(m.get("waypoints", 0)), len(getattr(self, "coverage_visited", []))))
        coverage_c = float(coverage_count / max(len(getattr(self, "coverage_visited", [])), 1))
        per100 = 100.0 / steps
        row = {
            "episode_id": self.metric_episode_id,
            "run_id": self.run_id,
            "env": "e2_industrial",
            "route_loops": int(getattr(self, "effective_route_loops", 1)),
            "requested_route_repeat_count": int(self.route_repeat_count),
            "repeat_full_route": int(bool(getattr(self, "repeat_full_route", False))),
            "success_Ps": int(success),
            "coverage_C": float(coverage_c),
            "rpe_trans_Ep": m["rpe_trans_sum"] / steps,
            "rpe_yaw_Epsi": m["rpe_yaw_sum"] / steps,
            "ate_Ea": m["ate_sum"] / steps,
            "track_loss_pct": float(m["track_loss_steps"]) * per100,
            "override_O_pct": float(m["override_steps"]) * per100,
            "abort_A_pct": float(m["abort_steps"]) * per100,
            "violation_pct": float(m["violation_steps"]) * per100,
            "accept_g_pct": float(m["accept_steps"]) * per100,
            "energy_Ew_Wh": m["energy_wh"],
            "drift_return_D": m["drift_return"],
            "mean_speed": m["speed_sum"] / steps,
            "steps": steps,
            "reward_sum": m["reward_sum"],
            "waypoints": int(m["waypoints"]),
            "mission": self.mission_text,
            "lambda": float(self.mission_risk_lambda),
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
            "collision": int(collision),
        }
        self._drift_returns.append(float(m["drift_return"]))
        self._write_episode_row(row)
        self._write_summary()
        should_save = self.save_figures and (self.metric_episode_id % self.capture_every_episode == 0)
        if should_save:
            self._save_trajectory(row)
            self._capture_inspection_views(f"ep{self.metric_episode_id:04d}")
        print(
            f"[EP {self.metric_episode_id}] success={success} loops={int(getattr(self, 'effective_route_loops', 1))} cov={row['coverage_C']:.2f} "
            f"E^p={row['rpe_trans_Ep']:.3f} E^psi={row['rpe_yaw_Epsi']:.3f} ATE={row['ate_Ea']:.3f} "
            f"loss={row['track_loss_pct']:.1f}% O={row['override_O_pct']:.1f}% A={row['abort_A_pct']:.1f}% D={row['drift_return_D']:.2f}"
        )

    def _write_episode_row(self, row: Dict[str, Any]) -> None:
        new = not self.episode_csv.exists()
        with open(self.episode_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def _write_summary(self) -> None:
        if not self.episode_csv.exists():
            return
        import statistics as st
        with open(self.episode_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        def col(name: str, cast=float) -> List[Any]:
            return [cast(r[name]) for r in rows if r.get(name) not in (None, "")]

        def mean(name: str) -> float:
            v = col(name)
            return float(st.mean(v)) if v else 0.0

        n = len(rows)
        summary = {
            "run_id": self.run_id,
            "episodes": n,
            "route_loops": int(getattr(self, "effective_route_loops", 1)),
            "requested_route_repeat_count": int(self.route_repeat_count),
            "repeat_full_route": int(bool(getattr(self, "repeat_full_route", False))),
            "success_rate_Ps": mean("success_Ps"),
            "coverage_C": mean("coverage_C"),
            "trans_rpe_Ep": mean("rpe_trans_Ep"),
            "yaw_rpe_Epsi": mean("rpe_yaw_Epsi"),
            "ate_Ea": mean("ate_Ea"),
            "track_loss_pct": mean("track_loss_pct"),
            "override_O_pct": mean("override_O_pct"),
            "abort_A_pct": mean("abort_A_pct"),
            "violation_pct": mean("violation_pct"),
            "accept_g_pct": mean("accept_g_pct"),
            "energy_Ew_Wh": mean("energy_Ew_Wh"),
            "drift_return_D": mean("drift_return_D"),
            "cvar_drift": self._cvar(col("drift_return_D"), self.cvar_alpha),
            "cvar_alpha": self.cvar_alpha,
            "mission": self.mission_text,
            "mission_lambda": float(self.mission_risk_lambda),
        }
        self.summary_json.write_text(json.dumps(summary, indent=2))
        # A one-row CSV directly suitable for paper tables.
        new = not self.result_table_csv.exists()
        with open(self.result_table_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)

    def _trajectory_dicts(self) -> List[Dict[str, float]]:
        """Return trajectory rows using reference drone.py-style field names.

        The old LLM script stored compact numeric rows.  The plotting code below
        uses named rows like the reference drone.py, so this adapter keeps old
        runs compatible while enabling richer RGB/heatmap/3D figures.
        """
        rows: List[Dict[str, float]] = []
        for i, rec in enumerate(getattr(self, "trajectory_records", []) or []):
            if isinstance(rec, dict):
                d = dict(rec)
                # Normalize common aliases.
                d.setdefault("x", float(d.get("true_x", d.get("x", 0.0))))
                d.setdefault("y", float(d.get("true_y", d.get("y", 0.0))))
                d.setdefault("z", float(d.get("true_z", d.get("z", 0.0))))
                d.setdefault("slam_x", float(d.get("vslam_x", d.get("slam_x", d.get("x", 0.0)))))
                d.setdefault("slam_y", float(d.get("vslam_y", d.get("slam_y", d.get("y", 0.0)))))
                d.setdefault("slam_z", float(d.get("vslam_z", d.get("slam_z", d.get("z", 0.0)))))
                d.setdefault("step", int(d.get("step", i)))
                d.setdefault("time_sec", float(d.get("time_sec", i * self.dt)))
                d.setdefault("yaw", float(d.get("yaw", 0.0)))
                d.setdefault("slam_yaw", float(d.get("slam_yaw", d.get("yaw", 0.0))))
                d.setdefault("feature_count", float(d.get("feature_count", 0.0)))
                d.setdefault("mu_T", float(d.get("mu_T", 0.0)))
                d.setdefault("mu_L", float(d.get("mu_L", 0.0)))
                d.setdefault("mu_W", float(d.get("mu_W", 0.0)))
                d.setdefault("mu_A", float(d.get("mu_A", 0.0)))
                d.setdefault("localization_error_m", float(d.get("localization_error_m", d.get("localization_error", 0.0))))
                d.setdefault("speed_mps", float(d.get("speed_mps", d.get("commanded_speed_mps", 0.0))))
                d.setdefault("commanded_speed_mps", float(d.get("commanded_speed_mps", d.get("speed_mps", 0.0))))
                d.setdefault("raw_policy_speed_mps", float(d.get("raw_policy_speed_mps", d.get("speed_mps", 0.0))))
                d.setdefault("speed_reduction_mps", float(d.get("speed_reduction_mps", max(0.0, d.get("raw_policy_speed_mps", 0.0) - d.get("commanded_speed_mps", 0.0)))))
                d.setdefault("route_lateral_error_m", float(d.get("route_lateral_error_m", 0.0)))
                d.setdefault("route_yaw_error_rad", float(d.get("route_yaw_error_rad", 0.0)))
                d.setdefault("governor_override", int(d.get("governor_override", 0)))
                d.setdefault("abort", int(d.get("abort", 0)))
                d.setdefault("fallback_reason", str(d.get("fallback_reason", "accept")))
                d.setdefault("override_cause", str(d.get("override_cause", "accept")))
                d.setdefault("op_cbrs_potential", float(d.get("op_cbrs_potential", 0.0)))
                d.setdefault("vslam_risk", float(d.get("vslam_risk", 0.0)))
                d.setdefault("gate", float(d.get("gate", 1.0)))
                d.setdefault("target_index", int(d.get("target_index", d.get("target_idx", 0))))
                rows.append(d)
                continue

            arr = np.asarray(rec, dtype=np.float32).reshape(-1)
            def val(k: int, default: float = 0.0) -> float:
                return float(arr[k]) if k < len(arr) and np.isfinite(arr[k]) else float(default)
            x, y, z = val(0), val(1), val(2)
            vslam_risk = val(3)
            speed = val(4)
            gate = val(5, 1.0)
            sx, sy, sz = val(6, x), val(7, y), val(8, z)
            slam_quality = val(9, 1.0)
            loc_err = val(10)
            target_idx = int(round(val(11)))
            yaw = val(12)
            slam_yaw = val(13, yaw)
            feature_count = val(14, 0.0)
            mu_T = val(15, self._texture_mu(feature_count))
            mu_L = val(16, self._illum_mu(float(getattr(self, "illumination_lux", 450.0))))
            mu_W = val(17, self._wind_mu(float(getattr(self, "wind_mps", 0.5))))
            mu_A = val(18, 0.0)
            potential = val(19, 0.0)
            reward = val(20, 0.0)
            rows.append({
                "run_id": str(getattr(self, "run_id", "")),
                "episode": int(getattr(self, "metric_episode_id", 0)),
                "step": int(i),
                "time_sec": float(i * self.dt),
                "sim_env_id": "e2",
                "policy_name": str(getattr(self, "baseline", "ppo")),
                "slam_mode": str(getattr(self, "slam_mode", "proxy")),
                "x": x, "y": y, "z": z,
                "slam_x": sx, "slam_y": sy, "slam_z": sz,
                "yaw": yaw, "slam_yaw": slam_yaw,
                "target_index": target_idx,
                "feature_count": feature_count,
                "mu_T": mu_T, "mu_L": mu_L, "mu_W": mu_W, "mu_A": mu_A,
                "localization_error_m": loc_err,
                "raw_policy_speed_mps": speed,
                "speed_mps": speed,
                "commanded_speed_mps": speed,
                "speed_reduction_mps": 0.0,
                "route_lateral_error_m": 0.0,
                "route_yaw_error_rad": 0.0,
                "governor_override": 0,
                "abort": 0,
                "fallback_reason": "legacy",
                "override_cause": "legacy",
                "op_cbrs_potential": potential,
                "vslam_risk": vslam_risk,
                "slam_quality": slam_quality,
                "gate": gate,
                "reward": reward,
            })
        return rows

    def _write_episode_trajectory_csv(self, path: Path) -> None:
        rows = self._trajectory_dicts()
        if not rows:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames: List[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            print(f"[TRAJECTORY] Warning: failed to write {path}: {exc}")

    @staticmethod
    def _smooth_heatmap_array(H: np.ndarray, passes: int = 2) -> np.ndarray:
        H = np.asarray(H, dtype=np.float32)
        if H.size == 0:
            return H
        kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32)
        kernel /= float(kernel.sum())
        out = H.copy()
        for _ in range(max(1, int(passes))):
            P = np.pad(out, 1, mode="edge")
            out = (
                kernel[0, 0] * P[:-2, :-2] + kernel[0, 1] * P[:-2, 1:-1] + kernel[0, 2] * P[:-2, 2:] +
                kernel[1, 0] * P[1:-1, :-2] + kernel[1, 1] * P[1:-1, 1:-1] + kernel[1, 2] * P[1:-1, 2:] +
                kernel[2, 0] * P[2:, :-2] + kernel[2, 1] * P[2:, 1:-1] + kernel[2, 2] * P[2:, 2:]
            )
        return out.astype(np.float32)

    def _visible_feature_image_heatmap(self, pos: np.ndarray, yaw: float, feature_gain: float = 1.0, bins: int = 96, camera_mode: str = "front") -> Tuple[Optional[np.ndarray], int]:
        """Camera-frame feature heatmap, adapted from the reference drone.py.

        front = stereo forward camera, bottom/down = downward inspection camera.
        """
        if self.visual_feature_points is None or len(self.visual_feature_points) == 0:
            return None, 0
        pts = np.asarray(self.visual_feature_points, dtype=np.float32)
        pos = np.asarray(pos, dtype=np.float32).reshape(3)
        rel = pts - pos[None, :]
        c, s = math.cos(-float(yaw)), math.sin(-float(yaw))
        bx = c * rel[:, 0] - s * rel[:, 1]
        by = s * rel[:, 0] + c * rel[:, 1]
        bz = rel[:, 2]
        mode = str(camera_mode or "front").lower()
        H = np.zeros((int(bins), int(bins)), dtype=np.float32)
        max_depth = max(4.0, float(self.max_ray_range))
        if mode in ("bottom", "down", "downward", "below", "downcam"):
            depth_axis = -bz
            h_ang = np.arctan2(bx, np.maximum(depth_axis, 1e-6))
            v_ang = np.arctan2(by, np.maximum(depth_axis, 1e-6))
            depth = np.sqrt(bx * bx + by * by + depth_axis * depth_axis) + 1e-6
            visible = (
                (depth_axis > 0.15) &
                (depth < max_depth) &
                (np.abs(h_ang) <= 0.5 * float(self.camera_hfov)) &
                (np.abs(v_ang) <= 0.5 * float(self.camera_vfov))
            )
            u_all = np.tan(h_ang) / max(np.tan(0.5 * float(self.camera_hfov)), 1e-6)
            v_all = np.tan(v_ang) / max(np.tan(0.5 * float(self.camera_vfov)), 1e-6)
        else:
            depth_axis = bx
            depth = np.sqrt(bx * bx + by * by + bz * bz) + 1e-6
            h_ang = np.arctan2(by, np.maximum(depth_axis, 1e-6))
            v_ang = np.arctan2(bz, np.maximum(np.sqrt(bx * bx + by * by), 1e-6))
            visible = (
                (depth_axis > 0.15) &
                (depth < max_depth) &
                (np.abs(h_ang) <= 0.5 * float(self.camera_hfov)) &
                (np.abs(v_ang) <= 0.5 * float(self.camera_vfov))
            )
            u_all = np.tan(h_ang) / max(np.tan(0.5 * float(self.camera_hfov)), 1e-6)
            v_all = np.tan(v_ang) / max(np.tan(0.5 * float(self.camera_vfov)), 1e-6)
        visible_count = int(np.count_nonzero(visible))
        if np.any(visible):
            u = u_all[visible]
            v = v_all[visible]
            d = depth[visible]
            w = (1.0 - np.clip(d / max(max_depth, 1e-6), 0.0, 1.0))
            w *= (1.0 - 0.35 * np.clip(np.abs(u), 0.0, 1.0))
            w *= (1.0 - 0.25 * np.clip(np.abs(v), 0.0, 1.0))
            w *= max(float(feature_gain), 0.05)
            H0, _, _ = np.histogram2d(u, v, bins=int(bins), range=[[-1.0, 1.0], [-1.0, 1.0]], weights=w)
            H += H0.T.astype(np.float32)

        # Robust downward-camera fallback.  During early training the route may
        # pass above roof surfaces while some synthetic SLAM landmarks are still
        # close to UAV altitude.  Instead of returning an empty downcam heatmap,
        # project those landmarks onto the visible roof/ground plane below the UAV.
        if mode in ("bottom", "down", "downward", "below", "downcam") and visible_count < 18:
            projected_z = np.minimum(pts[:, 2], pos[2] - 2.0)
            depth2 = np.maximum(pos[2] - projected_z, 0.35)
            h2 = np.arctan2(bx, depth2)
            v2 = np.arctan2(by, depth2)
            rng2 = np.sqrt(bx * bx + by * by + depth2 * depth2) + 1e-6
            visible2 = (
                (rng2 < max_depth * 1.45) &
                (np.abs(h2) <= 0.56 * float(self.camera_hfov)) &
                (np.abs(v2) <= 0.56 * float(self.camera_vfov))
            )
            if np.any(visible2):
                u2 = np.tan(h2[visible2]) / max(np.tan(0.5 * float(self.camera_hfov)), 1e-6)
                v2n = np.tan(v2[visible2]) / max(np.tan(0.5 * float(self.camera_vfov)), 1e-6)
                d2 = rng2[visible2]
                w2 = (1.0 - np.clip(d2 / max(max_depth * 1.45, 1e-6), 0.0, 1.0))
                w2 *= (1.0 - 0.22 * np.clip(np.abs(u2), 0.0, 1.0))
                w2 *= (1.0 - 0.22 * np.clip(np.abs(v2n), 0.0, 1.0))
                w2 *= max(float(feature_gain), 0.05) * 0.82
                H2, _, _ = np.histogram2d(u2, v2n, bins=int(bins), range=[[-1.0, 1.0], [-1.0, 1.0]], weights=w2)
                H += H2.T.astype(np.float32)
                visible_count = max(visible_count, int(np.count_nonzero(visible2)))

        H = self._smooth_heatmap_array(H, passes=2)
        if float(H.max()) > 0.0:
            H = H / float(H.max())
        return H.astype(np.float32), visible_count

    def _camera_feature_rgb_frame(self, pos: np.ndarray, yaw: float, camera_mode: str = "bottom", bins: int = 256) -> np.ndarray:
        """Generate a camera-like RGB frame from visible scene features.

        This is only the safe fallback when Replicator cannot capture the camera.
        It is not a top-down map: it projects the scene footprint into the
        front/down camera image.  RGB frames intentionally do not draw magenta
        feature/heatmap dots; those are saved separately in the heatmap figures.
        """
        H, _cnt = self._visible_feature_image_heatmap(pos, yaw, feature_gain=1.0, bins=bins, camera_mode=camera_mode)
        if H is None:
            H = np.zeros((bins, bins), dtype=np.float32)
        H = np.power(np.clip(H, 0.0, 1.0), 0.65)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.cm as cm
            hot = (cm.magma(H)[:, :, :3] * 255).astype(np.uint8)
        except Exception:
            hot = (np.stack([H, 0.45 * H, 1.0 - 0.75 * H], axis=-1) * 255).astype(np.uint8)
        img = np.zeros((bins, bins, 3), dtype=np.uint8)
        mode = str(camera_mode or "bottom").lower()
        yy = np.linspace(0.0, 1.0, bins, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, bins, dtype=np.float32)[None, :]
        if mode in ("front", "stereo", "left"):
            sky = np.array([115, 142, 168], dtype=np.float32)
            ground = np.array([84, 80, 72], dtype=np.float32)
            plant = np.array([82, 91, 88], dtype=np.float32)
            horizon = 0.45 + 0.08 * math.sin(float(yaw))
            mask_sky = yy < horizon
            img[:, :, :] = (sky * (1 - yy) + ground * yy).astype(np.uint8)
            band = np.abs(yy - horizon) < 0.09
            img[band.repeat(bins, axis=1)] = plant.astype(np.uint8)
        else:
            # Fallback visual only.  It is no longer a fake central rectangle; it
            # projects actual collision-box roof footprints into the down camera
            # frame and adds a subtle roof/concrete texture.  Real RGB capture is
            # used whenever Replicator is available.
            gravel = np.array([74, 72, 66], dtype=np.float32)
            concrete = np.array([114, 112, 104], dtype=np.float32)
            roof = np.array([66, 83, 96], dtype=np.float32)
            metal = np.array([126, 128, 124], dtype=np.float32)
            pattern = 0.5 + 0.5 * np.sin(24.0 * xx + 15.0 * yy + 0.7 * math.sin(float(yaw)))
            img[:, :, :] = (gravel + 15.0 * pattern[..., None]).astype(np.uint8)

            pos = np.asarray(pos, dtype=np.float32).reshape(3)
            c, s = math.cos(-float(yaw)), math.sin(-float(yaw))
            tanx = max(math.tan(0.5 * float(self.camera_hfov)), 1e-6)
            tany = max(math.tan(0.5 * float(self.camera_vfov)), 1e-6)
            for bi, box in enumerate(getattr(self, "static_collision_boxes", [])[:220]):
                name = str(box.get("path", "")).lower()
                if any(k in name for k in ("fence", "smoke", "wire", "pole", "bulb", "safetyhalo")):
                    continue
                mn = np.asarray(box.get("mn", [0, 0, 0]), dtype=np.float32)
                mx = np.asarray(box.get("mx", [0, 0, 0]), dtype=np.float32)
                ztop = float(mx[2])
                if ztop >= float(pos[2]) - 0.10:
                    continue
                corners = np.array([[mn[0], mn[1], ztop], [mx[0], mn[1], ztop], [mx[0], mx[1], ztop], [mn[0], mx[1], ztop]], dtype=np.float32)
                rel = corners - pos[None, :]
                bx = c * rel[:, 0] - s * rel[:, 1]
                by = s * rel[:, 0] + c * rel[:, 1]
                depth = np.maximum(-rel[:, 2], 0.15)
                u = (bx / depth) / tanx
                v = (by / depth) / tany
                if np.all((np.abs(u) > 1.25) | (np.abs(v) > 1.25)):
                    continue
                px = np.clip(((u + 1.0) * 0.5 * (bins - 1)).astype(int), 0, bins - 1)
                py = np.clip(((v + 1.0) * 0.5 * (bins - 1)).astype(int), 0, bins - 1)
                x0, x1 = int(px.min()), int(px.max())
                y0, y1 = int(py.min()), int(py.max())
                if x1 <= x0 + 1 or y1 <= y0 + 1:
                    continue
                col = metal if ("tank" in name or "transformer" in name) else roof if (bi % 3 == 0) else concrete
                local = img[y0:y1+1, x0:x1+1].astype(np.float32)
                img[y0:y1+1, x0:x1+1] = (0.35 * local + 0.65 * col).astype(np.uint8)

        # RGB should look like a camera image.  Do not paint the feature heatmap
        # over RGB, because that produced the purple/magenta dots visible in
        # downcam_rgb_ep*.png.  The feature response is saved by
        # _write_downcam_heatmap_sequence_panel() and _write_heatmap().
        if bool(getattr(self.args, "rgb_feature_overlay", False)):
            alpha = np.clip(H * 0.18, 0.0, 0.18)[..., None]
            out = (img.astype(np.float32) * (1.0 - alpha) + hot.astype(np.float32) * alpha).astype(np.uint8)
            return out
        return img.astype(np.uint8)

    def _build_heatmap_array(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        rows = self._trajectory_dicts()
        if not rows:
            return None, None, None
        xs = np.array([float(r.get("x", 0.0)) for r in rows], dtype=np.float32)
        ys = np.array([float(r.get("y", 0.0)) for r in rows], dtype=np.float32)
        zs = np.array([float(r.get("z", 0.0)) for r in rows], dtype=np.float32)
        yaws = np.array([float(r.get("yaw", 0.0)) for r in rows], dtype=np.float32)
        features = np.array([float(r.get("feature_count", 0.0)) for r in rows], dtype=np.float32)
        tx = self.targets[:, 0] if getattr(self, "targets", None) is not None else np.array([], dtype=np.float32)
        ty = self.targets[:, 1] if getattr(self, "targets", None) is not None else np.array([], dtype=np.float32)
        if self.visual_feature_points is not None and len(self.visual_feature_points) > 0:
            fpts = np.asarray(self.visual_feature_points, dtype=np.float32)
            fx, fy = fpts[:, 0], fpts[:, 1]
        else:
            fpts = None
            fx, fy = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        all_x = np.concatenate([xs, tx.astype(np.float32), fx.astype(np.float32)])
        all_y = np.concatenate([ys, ty.astype(np.float32), fy.astype(np.float32)])
        x_min, x_max = float(np.nanmin(all_x) - 2.0), float(np.nanmax(all_x) + 2.0)
        y_min, y_max = float(np.nanmin(all_y) - 2.0), float(np.nanmax(all_y) + 2.0)
        if abs(x_max - x_min) < 1e-3:
            x_max += 1.0
        if abs(y_max - y_min) < 1e-3:
            y_max += 1.0
        H = np.zeros((self.heatmap_grid_size, self.heatmap_grid_size), dtype=np.float32)
        if fpts is not None and len(fpts) > 0:
            stride = max(1, len(xs) // 180)
            for i in range(0, len(xs), stride):
                pos = np.array([xs[i], ys[i], zs[i]], dtype=np.float32)
                rel = fpts - pos[None, :]
                c, ss = math.cos(-float(yaws[i])), math.sin(-float(yaws[i]))
                bx = c * rel[:, 0] - ss * rel[:, 1]
                by = ss * rel[:, 0] + c * rel[:, 1]
                bz = rel[:, 2]
                depth = np.sqrt(bx * bx + by * by + bz * bz) + 1e-6
                h_ang = np.arctan2(by, np.maximum(bx, 1e-6))
                v_ang = np.arctan2(bz, np.maximum(np.sqrt(bx * bx + by * by), 1e-6))
                mask = (
                    (bx > 0.15) &
                    (depth < max(4.0, float(self.max_ray_range))) &
                    (np.abs(h_ang) <= 0.5 * float(self.camera_hfov)) &
                    (np.abs(v_ang) <= 0.5 * float(self.camera_vfov))
                )
                if np.any(mask):
                    gain = 0.25 + 0.75 * np.clip(features[i] / max(float(np.nanmax(features)), 1.0), 0.0, 1.0)
                    h, _, _ = np.histogram2d(
                        fpts[mask, 0], fpts[mask, 1],
                        bins=self.heatmap_grid_size,
                        range=[[x_min, x_max], [y_min, y_max]],
                        weights=np.full(int(np.count_nonzero(mask)), float(gain), dtype=np.float32),
                    )
                    H += h.T.astype(np.float32)
        else:
            weights = np.maximum(features, 1e-3)
            h, xedges, yedges = np.histogram2d(xs, ys, bins=self.heatmap_grid_size, range=[[x_min, x_max], [y_min, y_max]], weights=weights)
            return self._smooth_heatmap_array(h.T, passes=2), xedges, yedges
        H = self._smooth_heatmap_array(H, passes=2)
        xedges = np.linspace(x_min, x_max, self.heatmap_grid_size + 1, dtype=np.float32)
        yedges = np.linspace(y_min, y_max, self.heatmap_grid_size + 1, dtype=np.float32)
        return H, xedges, yedges

    def _coverage_layout_metadata(self) -> Dict[str, Any]:
        pts = np.asarray(getattr(self, "inspection_points", self.targets), dtype=np.float32)
        route = np.asarray(getattr(self, "targets", pts), dtype=np.float32)[:, :2]
        xmin = float(np.nanmin(pts[:, 0]) - 2.8)
        xmax = float(np.nanmax(pts[:, 0]) + 2.8)
        ymin = float(np.nanmin(pts[:, 1]) - 2.8)
        ymax = float(np.nanmax(pts[:, 1]) + 2.8)
        low_boxes = []
        # Use a few actual plant/construction footprints as low-texture regions.
        for box in getattr(self, "static_collision_boxes", [])[:8]:
            mn = np.asarray(box.get("mn", [0, 0, 0]), dtype=np.float32)
            mx = np.asarray(box.get("mx", [0, 0, 0]), dtype=np.float32)
            sx, sy = float(mx[0] - mn[0]), float(mx[1] - mn[1])
            if sx > 2.0 and sy > 2.0:
                low_boxes.append((float(0.5 * (mn[0] + mx[0])), float(0.5 * (mn[1] + mx[1])), sx, sy))
        if not low_boxes:
            low_boxes = [(-16.0, 9.0, 9.0, 6.0), (0.0, 0.0, 12.0, 10.0), (15.0, 0.0, 12.0, 6.0)]
        return {
            "target_area": (xmin, ymin, xmax - xmin, ymax - ymin),
            "low_texture_boxes": low_boxes[:5],
            "fov_box": (xmin + 0.10 * (xmax - xmin), ymin + 0.12 * (ymax - ymin), 0.78 * (xmax - xmin), 0.34 * (ymax - ymin)),
            "route": route,
            "entry_point": np.asarray([route[0, 0], route[0, 1]], dtype=np.float32),
        }

    def _choose_visual_indices(self, rows: List[Dict[str, float]], max_frames: int = 12) -> List[int]:
        if not rows:
            return []
        steps = np.array([int(r.get("step", i)) for i, r in enumerate(rows)], dtype=np.int32)
        features = np.array([float(r.get("feature_count", 0.0)) for r in rows], dtype=np.float32)
        mu_t = np.array([float(r.get("mu_T", 0.0)) for r in rows], dtype=np.float32)
        mu_l = np.array([float(r.get("mu_L", 0.0)) for r in rows], dtype=np.float32)
        loc_err = np.array([float(r.get("localization_error_m", 0.0)) for r in rows], dtype=np.float32)
        def _norm(v: np.ndarray) -> np.ndarray:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            if hi - lo < 1e-6:
                return np.zeros_like(v, dtype=np.float32)
            return (v - lo) / (hi - lo + 1e-6)
        score = 0.55 * _norm(features) + 0.20 * np.maximum(0.0, 1.0 - mu_t) + 0.08 * np.maximum(0.0, 1.0 - mu_l) + 0.17 * _norm(loc_err)
        nframes = min(int(max_frames), len(rows))
        edges = np.linspace(0, len(rows), nframes + 1, dtype=int)
        chosen: List[int] = []
        feat_thr = max(1.0, 0.08 * float(np.nanmax(features)))
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            local = np.arange(a, b)
            non_empty = local[features[local] > feat_thr]
            cand = non_empty if len(non_empty) else local
            best = int(cand[int(np.nanargmax(score[cand]))])
            if best not in chosen:
                chosen.append(best)
        for idx in np.argsort(score)[::-1]:
            idx = int(idx)
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= nframes:
                break
        return sorted(chosen[:nframes], key=lambda i: steps[i])

    def _render_fuzzy_sequence_figure(self, episode_prefix: str, bundle: Dict[str, object], plt: Any) -> Dict[str, object]:
        rows = self._trajectory_dicts()
        if len(rows) < 4:
            return bundle
        try:
            steps = np.array([int(r.get("step", i)) for i, r in enumerate(rows)], dtype=np.int32)
            features = np.array([float(r.get("feature_count", 0.0)) for r in rows], dtype=np.float32)
            chosen = self._choose_visual_indices(rows, max_frames=12)
            vmax_features = max(float(np.nanmax(features)), 1.0)
            def _heat(idx: int, mode: str):
                r = rows[idx]
                H, cnt = self._visible_feature_image_heatmap(
                    np.array([float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("z", 0.0))], dtype=np.float32),
                    float(r.get("yaw", 0.0)),
                    feature_gain=0.65 + 0.75 * float(max(float(r.get("feature_count", 1.0)), 1.0)) / vmax_features,
                    bins=96,
                    camera_mode=mode,
                )
                if H is None:
                    H = np.zeros((96, 96), dtype=np.float32)
                return np.power(np.clip(H, 0.0, 1.0), 0.70), int(cnt)
            def _render_single(mode: str, title: str, out_name: str, key_name: str) -> None:
                ncols = 6 if len(chosen) > 6 else max(len(chosen), 1)
                nrows = int(math.ceil(len(chosen) / max(ncols, 1)))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.9 * nrows + 1.1), squeeze=False)
                fig.subplots_adjust(left=0.03, right=0.985, top=0.87, bottom=0.13, wspace=0.08, hspace=0.18)
                last_im = None
                for ax in axes.ravel():
                    ax.axis("off")
                for j, idx in enumerate(chosen):
                    ax = axes.ravel()[j]
                    H_plot, cnt = _heat(idx, mode)
                    last_im = ax.imshow(H_plot, cmap="magma", origin="lower", interpolation="bicubic", vmin=0.0, vmax=1.0)
                    ax.set_title(f"Step {int(steps[idx])}", fontsize=14, pad=6)
                    ax.text(0.03, 0.94, f"features={cnt}", transform=ax.transAxes, ha="left", va="top", color="white", fontsize=9.5,
                            bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.42, edgecolor="none"))
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                if last_im is not None:
                    cax = fig.add_axes([0.27, 0.055, 0.46, 0.022])
                    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
                    cbar.set_label("Normalized visual-feature response", fontsize=11)
                fig.suptitle(title, fontsize=18, y=0.965)
                path = self.figure_dir / f"{episode_prefix}_{out_name}.png"
                fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.10)
                plt.close(fig)
                bundle[key_name] = str(path)
            _render_single("front", "Front camera visual heatmap sequence around inspection objects", "front_visual_heatmap_sequence", "front_visual_heatmap_sequence_png")
            _render_single("bottom", "Downward inspection camera visual heatmap sequence", "bottom_visual_heatmap_sequence", "bottom_visual_heatmap_sequence_png")
            dual = chosen if len(chosen) <= 6 else [chosen[int(round(v))] for v in np.linspace(0, len(chosen) - 1, 6)]
            ncols = max(1, len(dual))
            fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 8.3), squeeze=False)
            fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.12, wspace=0.08, hspace=0.16)
            last_im = None
            for cidx, idx in enumerate(dual):
                for ridx, (mode, label) in enumerate([("front", "Front camera"), ("bottom", "Bottom camera")]):
                    ax = axes[ridx, cidx]
                    H_plot, cnt = _heat(idx, mode)
                    last_im = ax.imshow(H_plot, cmap="magma", origin="lower", interpolation="bicubic", vmin=0.0, vmax=1.0)
                    if ridx == 0:
                        ax.set_title(f"Step {int(steps[idx])}", fontsize=14, pad=6)
                    if cidx == 0:
                        ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")
                    ax.text(0.03, 0.94, f"n={cnt}", transform=ax.transAxes, ha="left", va="top", color="white", fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.42, edgecolor="none"))
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if last_im is not None:
                cax = fig.add_axes([0.28, 0.055, 0.44, 0.022])
                cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
                cbar.set_label("Normalized visual-feature response", fontsize=11)
            fig.suptitle("Dual-camera visual heatmap sequence: front view and downward inspection view", fontsize=17, y=0.96)
            seq_path = self.figure_dir / f"{episode_prefix}_visual_heatmap_sequence.png"
            fig.savefig(seq_path, dpi=240, bbox_inches="tight", pad_inches=0.10)
            plt.close(fig)
            bundle["visual_heatmap_sequence_png"] = str(seq_path)
            bundle["dual_camera_visual_heatmap_sequence_png"] = str(seq_path)
        except Exception as exc:
            print(f"[FIGURES] Warning: failed to render visual heatmap sequence: {exc}")
        return bundle

    def _render_rgb_sequence_figure(self, episode_prefix: str, bundle: Dict[str, object], plt: Any) -> Dict[str, object]:
        rows = self._trajectory_dicts()
        if len(rows) < 4:
            return bundle
        try:
            steps = np.array([int(r.get("step", i)) for i, r in enumerate(rows)], dtype=np.int32)
            chosen_front = self._choose_visual_indices(rows, max_frames=10)
            chosen_bottom = self._choose_visual_indices(rows, max_frames=12)
            if len(chosen_bottom) < 8:
                chosen_bottom = sorted(set(chosen_bottom + [int(i) for i in np.linspace(0, len(rows) - 1, min(12, len(rows)))]))[:12]
            def _rgb(idx: int, mode: str) -> np.ndarray:
                r = rows[idx]
                return self._camera_feature_rgb_frame(
                    np.array([float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("z", 0.0))], dtype=np.float32),
                    float(r.get("yaw", 0.0)), camera_mode=mode, bins=256,
                )
            def _render_single(mode: str, title: str, out_name: str, key_name: str, chosen: List[int]) -> None:
                ncols = 4 if len(chosen) > 4 else max(1, len(chosen))
                nrows = int(math.ceil(len(chosen) / max(ncols, 1)))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows + 0.9), squeeze=False)
                for ax in axes.ravel():
                    ax.axis("off")
                for j, idx in enumerate(chosen):
                    ax = axes.ravel()[j]
                    ax.imshow(_rgb(idx, mode), origin="lower")
                    ax.set_title(f"Step {int(steps[idx])}", fontsize=13)
                fig.suptitle(title, fontsize=17, y=0.98)
                fig.tight_layout()
                path = self.figure_dir / f"{episode_prefix}_{out_name}.png"
                fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
                plt.close(fig)
                bundle[key_name] = str(path)
            _render_single("front", "Front RGB-like camera frames with visual feature overlay", "front_rgb_sequence", "front_rgb_sequence_png", chosen_front)
            _render_single("bottom", "Downward RGB-like inspection RGB sequence over multiple trajectory poses", "bottom_rgb_sequence", "bottom_rgb_sequence_png", chosen_bottom)
        except Exception as exc:
            print(f"[FIGURES] Warning: failed to render RGB sequence: {exc}")
        return bundle

    def _render_episode_figures(self, episode_prefix: str, bundle: Dict[str, object], row: Dict[str, Any]) -> Dict[str, object]:
        rows = self._trajectory_dicts()
        if not self.save_figures or not rows:
            return bundle
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, FancyArrowPatch
            from matplotlib.lines import Line2D
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as exc:
            print(f"[FIGURES] matplotlib unavailable, skipping figure export: {exc}")
            return bundle
        try:
            steps = np.array([int(r.get("step", i)) for i, r in enumerate(rows)], dtype=np.int32)
            xs = np.array([float(r.get("x", 0.0)) for r in rows], dtype=np.float32)
            ys = np.array([float(r.get("y", 0.0)) for r in rows], dtype=np.float32)
            zs = np.array([float(r.get("z", 0.0)) for r in rows], dtype=np.float32)
            sxs = np.array([float(r.get("slam_x", r.get("x", 0.0))) for r in rows], dtype=np.float32)
            sys_ = np.array([float(r.get("slam_y", r.get("y", 0.0))) for r in rows], dtype=np.float32)
            szs = np.array([float(r.get("slam_z", r.get("z", 0.0))) for r in rows], dtype=np.float32)
            speed = np.array([float(r.get("speed_mps", 0.0)) for r in rows], dtype=np.float32)
            commanded = np.array([float(r.get("commanded_speed_mps", 0.0)) for r in rows], dtype=np.float32)
            potential = np.array([float(r.get("op_cbrs_potential", 0.0)) for r in rows], dtype=np.float32)
            features = np.array([float(r.get("feature_count", 0.0)) for r in rows], dtype=np.float32)
            mu_t = np.array([float(r.get("mu_T", 0.0)) for r in rows], dtype=np.float32)
            mu_a = np.array([float(r.get("mu_A", 0.0)) for r in rows], dtype=np.float32)
            loc_err = np.array([float(r.get("localization_error_m", 0.0)) for r in rows], dtype=np.float32)
            risk = np.array([float(r.get("vslam_risk", 0.0)) for r in rows], dtype=np.float32)
            low_tex = mu_t < 0.35
            layout = self._coverage_layout_metadata()
            plan = np.asarray(layout["route"], dtype=np.float32)
            entry = np.asarray(layout["entry_point"], dtype=np.float32)

            # Align SLAM path to the true world path for visual map display only.
            sxs_plot = np.array(sxs, copy=True); sys_plot = np.array(sys_, copy=True); szs_plot = np.array(szs, copy=True)
            if len(sxs_plot) > 4 and np.all(np.isfinite(sxs_plot)):
                try:
                    src = np.column_stack([sxs_plot, sys_plot]).astype(np.float64)
                    dst = np.column_stack([xs, ys]).astype(np.float64)
                    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
                    src0, dst0 = src - src_mean, dst - dst_mean
                    Hm = src0.T @ dst0
                    U, Svals, Vt = np.linalg.svd(Hm)
                    R = Vt.T @ U.T
                    if np.linalg.det(R) < 0:
                        Vt[-1, :] *= -1
                        R = Vt.T @ U.T
                    scale = float(np.clip(np.sum(Svals) / max(np.sum(src0 ** 2), 1e-9), 0.25, 4.0))
                    aligned = scale * (src0 @ R.T) + dst_mean
                    sxs_plot, sys_plot = aligned[:, 0].astype(np.float32), aligned[:, 1].astype(np.float32)
                    szs_plot = (szs_plot - szs_plot[0] + zs[0]).astype(np.float32)
                except Exception:
                    sxs_plot = sxs_plot - sxs_plot[0] + xs[0]
                    sys_plot = sys_plot - sys_plot[0] + ys[0]
                    szs_plot = szs_plot - szs_plot[0] + zs[0]

            # 1) Coverage plan.
            cov_path = self.figure_dir / f"{episode_prefix}_coverage_plan.png"
            fig, ax = plt.subplots(figsize=(11.5, 8.2))
            tx, ty, tw, th = layout["target_area"]
            ax.add_patch(Rectangle((tx, ty), tw, th, facecolor="#f3f3f3", edgecolor="black", linewidth=2.0, hatch=".", zorder=0))
            for cx, cy, w, h in layout["low_texture_boxes"]:
                ax.add_patch(Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h, facecolor="white", edgecolor="black", linewidth=1.6, zorder=2))
            fbx, fby, fbw, fbh = layout["fov_box"]
            ax.add_patch(Rectangle((fbx, fby), fbw, fbh, facecolor="#d7efb5", edgecolor="black", linestyle="--", linewidth=1.8, alpha=0.85, zorder=1))
            px = np.concatenate([[entry[0]], plan[:, 0]])
            py = np.concatenate([[entry[1]], plan[:, 1]])
            ax.plot(px, py, linestyle=(0, (6, 4)), linewidth=2.5, color="#1f4de3", marker="o", markersize=4.0, zorder=3, label="Coverage flight path")
            for a_i, b_i in zip(range(0, len(px) - 1), range(1, len(px))):
                if a_i % 3 == 0 or b_i == len(px) - 1:
                    ax.add_patch(FancyArrowPatch((px[a_i], py[a_i]), (px[b_i], py[b_i]), arrowstyle="-|>", mutation_scale=12, color="#1f4de3", linewidth=1.6, alpha=0.85, zorder=4))
            ax.scatter([entry[0]], [entry[1]], s=120, color="#1f4de3", zorder=4, label="Start")
            ax.scatter(plan[:, 0], plan[:, 1], s=60, facecolor="#1f4de3", edgecolor="white", linewidth=0.8, zorder=4, label="Inspection waypoints")
            ax.set_title("Boustrophedon coverage inspection path and low-texture regions")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.18)
            ax.set_xlim(tx - 0.6, tx + tw + 0.6); ax.set_ylim(ty - 0.6, ty + th + 0.6)
            ax.legend(handles=[
                Rectangle((0, 0), 1, 1, facecolor="#f3f3f3", edgecolor="black", hatch=".", label="Inspection target area"),
                Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", label="Low-texture area"),
                Rectangle((0, 0), 1, 1, facecolor="#d7efb5", edgecolor="black", linestyle="--", label="UAV field of view"),
                Line2D([0], [0], color="#1f4de3", linestyle=(0, (6, 4)), marker="o", label="Coverage flight path"),
            ], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
            fig.tight_layout(); fig.savefig(cov_path, dpi=240, bbox_inches="tight"); plt.close(fig)
            bundle["coverage_plan_png"] = str(cov_path)

            # 2) Top-view executed trajectory.
            traj2d_path = self.figure_dir / f"{episode_prefix}_trajectory2d.png"
            fig, ax = plt.subplots(figsize=(10.8, 8.1))
            ax.plot(plan[:, 0], plan[:, 1], linestyle=(0, (4, 4)), linewidth=1.9, color="#8e8e8e", label="Planned coverage path")
            ax.plot(xs, ys, linewidth=2.4, color="#0b57d0", label="True trajectory")
            ax.plot(sxs_plot, sys_plot, linewidth=2.0, color="#00a2ff", alpha=0.85, label="VSLAM-estimated trajectory")
            if np.any(low_tex):
                ax.scatter(xs[low_tex], ys[low_tex], s=24, color="#ef6c00", alpha=0.88, label="Low-texture traversal")
            ax.scatter(self.inspection_points[:, 0], self.inspection_points[:, 1], s=72, marker="x", color="black", linewidth=1.7, label="16 inspection points")
            ax.scatter(xs[0], ys[0], s=100, marker="o", color="#1f4de3", label="Start")
            ax.scatter(xs[-1], ys[-1], s=130, marker="*", color="#d32f2f", label="End")
            for cx, cy, w, h in layout["low_texture_boxes"]:
                ax.add_patch(Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h, facecolor="none", edgecolor="black", linewidth=1.1, alpha=0.45))
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(f"Top-view executed/VSLAM trajectory (e1, loops={int(getattr(self, 'effective_route_loops', 1))})")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.20)
            all_x = np.concatenate([xs, sxs_plot, plan[:, 0], self.inspection_points[:, 0]])
            all_y = np.concatenate([ys, sys_plot, plan[:, 1], self.inspection_points[:, 1]])
            x_pad = max(2.0, 0.08 * float(np.nanmax(all_x) - np.nanmin(all_x) + 1e-6))
            y_pad = max(2.0, 0.08 * float(np.nanmax(all_y) - np.nanmin(all_y) + 1e-6))
            ax.set_xlim(float(np.nanmin(all_x) - x_pad), float(np.nanmax(all_x) + x_pad))
            ax.set_ylim(float(np.nanmin(all_y) - y_pad), float(np.nanmax(all_y) + y_pad))
            ax.legend(loc="best")
            fig.tight_layout(); fig.savefig(traj2d_path, dpi=240, bbox_inches="tight"); plt.close(fig)
            bundle["trajectory_2d_png"] = str(traj2d_path)

            # 3) VSLAM map/trajectory style plot.
            vslam_path = self.figure_dir / f"{episode_prefix}_vslam_trajectory.png"
            fig, ax = plt.subplots(figsize=(7.6, 10.8))
            rng_map = np.random.default_rng(2026)
            cloud_x, cloud_y = [], []
            for cx, cy, w, h in layout["low_texture_boxes"]:
                for _ in range(360):
                    side = int(rng_map.integers(0, 4))
                    if side == 0:
                        x, y = rng_map.uniform(cx - 0.5*w, cx + 0.5*w), cy - 0.5*h
                    elif side == 1:
                        x, y = rng_map.uniform(cx - 0.5*w, cx + 0.5*w), cy + 0.5*h
                    elif side == 2:
                        x, y = cx - 0.5*w, rng_map.uniform(cy - 0.5*h, cy + 0.5*h)
                    else:
                        x, y = cx + 0.5*w, rng_map.uniform(cy - 0.5*h, cy + 0.5*h)
                    cloud_x.append(float(x) + rng_map.normal(0.0, 0.055)); cloud_y.append(float(y) + rng_map.normal(0.0, 0.055))
                for _ in range(180):
                    cloud_x.append(rng_map.uniform(cx - 0.62*w, cx + 0.62*w)); cloud_y.append(rng_map.uniform(cy - 0.62*h, cy + 0.62*h))
            if self.visual_feature_points is not None and len(self.visual_feature_points) > 0:
                pts = np.asarray(self.visual_feature_points, dtype=np.float32)
                if len(pts) > 5500:
                    pts = pts[rng_map.choice(len(pts), size=5500, replace=False)]
                cloud_x.extend(pts[:, 0].tolist()); cloud_y.extend(pts[:, 1].tolist())
            ax.scatter(cloud_x, cloud_y, s=1.5, color="black", alpha=0.55, linewidths=0, label="VSLAM map points")
            ax.plot(sxs_plot, sys_plot, color="#0037ff", linewidth=2.4, alpha=0.95, label="VSLAM trajectory")
            ax.scatter(sxs_plot, sys_plot, s=6, color="#0037ff", alpha=0.95, linewidths=0)
            ax.scatter(sxs_plot[0], sys_plot[0], s=70, color="#0037ff", zorder=6, label="Start")
            ax.scatter(sxs_plot[-1], sys_plot[-1], s=95, marker="*", color="#d7191c", zorder=6, label="End")
            ax.scatter(self.inspection_points[:, 0], self.inspection_points[:, 1], s=28, color="#e60000", alpha=0.88, label="Inspection points")
            for px_i, py_i in zip(self.inspection_points[:, 0], self.inspection_points[:, 1]):
                ax.scatter(rng_map.normal(float(px_i), 0.24, 45), rng_map.normal(float(py_i), 0.24, 45), s=4, color="#e60000", alpha=0.24, linewidths=0)
            ax.plot(plan[:, 0], plan[:, 1], linestyle=(0, (5, 4)), color="#0037ff", linewidth=1.0, alpha=0.45)
            ax.set_title("VSLAM map and lawnmower trajectory (e1)")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal", adjustable="box"); ax.set_facecolor("white"); ax.grid(False)
            bx = np.concatenate([sxs_plot, xs, plan[:, 0], self.inspection_points[:, 0], np.asarray(cloud_x, dtype=np.float32)])
            by = np.concatenate([sys_plot, ys, plan[:, 1], self.inspection_points[:, 1], np.asarray(cloud_y, dtype=np.float32)])
            x_pad = max(2.0, 0.08 * float(np.nanmax(bx) - np.nanmin(bx) + 1e-6)); y_pad = max(2.0, 0.08 * float(np.nanmax(by) - np.nanmin(by) + 1e-6))
            ax.set_xlim(float(np.nanmin(bx) - x_pad), float(np.nanmax(bx) + x_pad)); ax.set_ylim(float(np.nanmin(by) - y_pad), float(np.nanmax(by) + y_pad))
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            fig.tight_layout(); fig.savefig(vslam_path, dpi=260, bbox_inches="tight"); plt.close(fig)
            bundle["vslam_trajectory_png"] = str(vslam_path)

            # 4) 3D trajectory with both true and VSLAM paths.
            traj3d_path = self.figure_dir / f"{episode_prefix}_trajectory3d.png"
            fig = plt.figure(figsize=(9.6, 7.5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(xs, ys, zs, label="UAV true trajectory", linewidth=2.2, color="#0b57d0")
            ax.plot(sxs_plot, sys_plot, szs_plot, label="VSLAM-estimated trajectory", linewidth=1.9, color="#00a2ff", alpha=0.88)
            if np.any(low_tex):
                ax.scatter(xs[low_tex], ys[low_tex], zs[low_tex], s=14, color="#ef6c00", label="Low-texture zone", alpha=0.75)
            ax.scatter(self.inspection_points[:, 0], self.inspection_points[:, 1], self.inspection_points[:, 2], s=28, marker="x", color="black", label="16 inspection points")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
            ax.set_title("UAV/VSLAM inspection trajectory in 3D")
            ax.legend(loc="best")
            fig.tight_layout(); fig.savefig(traj3d_path, dpi=240, bbox_inches="tight"); plt.close(fig)
            bundle["trajectory_3d_png"] = str(traj3d_path)

            # 5) Decision dynamics.
            dyn_path = self.figure_dir / f"{episode_prefix}_decision_dynamics.png"
            fig, axes = plt.subplots(4, 1, figsize=(12.0, 10.4), sharex=True)
            axes[0].plot(steps, speed, linewidth=1.8, label="Adaptive speed")
            axes[0].plot(steps, commanded, linewidth=1.4, linestyle="--", label="Commanded speed")
            axes[0].set_ylabel("Speed (m/s)"); axes[0].legend(loc="best", frameon=True)
            ax_feat = axes[1]; ax_pot = ax_feat.twinx()
            ax_feat.plot(steps, features, linewidth=1.5, label="Visible features")
            ax_pot.plot(steps, potential, linewidth=1.7, linestyle="--", label="Potential")
            ax_feat.set_ylabel("Visual features"); ax_pot.set_ylabel("Potential")
            h1, l1 = ax_feat.get_legend_handles_labels(); h2, l2 = ax_pot.get_legend_handles_labels()
            ax_feat.legend(h1 + h2, l1 + l2, loc="best", frameon=True)
            axes[2].plot(steps, loc_err, linewidth=1.6, label="Localization error")
            axes[2].plot(steps, risk, linewidth=1.4, linestyle=":", label="VSLAM risk")
            axes[2].axhline(self.localization_threshold, linestyle="--", color="black", linewidth=1.1, label="Aloc threshold")
            axes[2].set_ylabel("Error / risk"); axes[2].legend(loc="best", frameon=True)
            axes[3].plot(steps, mu_t, linewidth=1.5, label=r"$\mu_T$ texture")
            axes[3].plot(steps, mu_a, linewidth=1.5, label=r"$\mu_A$ adherence")
            axes[3].set_xlabel("Step"); axes[3].set_ylabel("Membership"); axes[3].legend(loc="best", frameon=True)
            for ax_i in axes:
                ax_i.grid(alpha=0.22)
            fig.suptitle("Decision, perception, and localization dynamics along the inspection route", fontsize=15)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965]); fig.savefig(dyn_path, dpi=240, bbox_inches="tight"); plt.close(fig)
            bundle["decision_dynamics_png"] = str(dyn_path)

            # 6) Global perception heatmap.
            heatmap, xedges, yedges = self._build_heatmap_array()
            if heatmap is not None and xedges is not None and yedges is not None:
                heatmap_path = self.figure_dir / f"{episode_prefix}_perception_heatmap.png"
                fig, ax = plt.subplots(figsize=(9.4, 7.4))
                im = ax.imshow(heatmap, origin="lower", aspect="auto", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="magma", interpolation="bicubic")
                ax.plot(sxs_plot, sys_plot, linewidth=1.2, color="white", alpha=0.9, label="VSLAM trajectory")
                ax.scatter(self.inspection_points[:, 0], self.inspection_points[:, 1], s=24, marker="x", color="cyan", label="Inspection points")
                ax.set_xlim(float(xedges[0]), float(xedges[-1])); ax.set_ylim(float(yedges[0]), float(yedges[-1]))
                ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title("Global visual-feature / hotspot coverage heatmap")
                fig.colorbar(im, ax=ax, label="Useful visual feature density")
                ax.legend(loc="best"); fig.tight_layout(); fig.savefig(heatmap_path, dpi=240, bbox_inches="tight"); plt.close(fig)
                bundle["perception_heatmap_png"] = str(heatmap_path)
                npz_path = self.figure_dir / f"{episode_prefix}_perception_heatmap.npz"
                np.savez_compressed(npz_path, heatmap=heatmap, xedges=xedges, yedges=yedges)
                bundle["perception_heatmap_npz"] = str(npz_path)

            bundle = self._render_fuzzy_sequence_figure(episode_prefix, bundle, plt)
            bundle = self._render_rgb_sequence_figure(episode_prefix, bundle, plt)
        except Exception as exc:
            print(f"[FIGURES] Warning: failed to render paper figures: {exc}")
        return bundle

    def _save_trajectory(self, row: Dict[str, Any]) -> None:
        try:
            episode_prefix = f"episode_{self.metric_episode_id:04d}"
            traj_csv = self.trajectory_dir / f"{episode_prefix}_trajectory.csv"
            self._write_episode_trajectory_csv(traj_csv)
            bundle: Dict[str, object] = {
                "run_id": self.run_id,
                "episode": self.metric_episode_id,
                "sim_env_id": "e1",
                "policy_name": "LLM-VSLAM-Speed-Governor",
                "slam_mode": self.slam_mode,
                "trajectory_csv": str(traj_csv),
                "episode_metrics": row,
                "description": {
                    "rgb": "RGB sequence images are camera-frame visualizations generated from front/downward camera feature projection, not a top-down map.",
                    "heatmap": "Heatmaps use the same camera-frame visual-feature projection logic as the reference drone.py.",
                    "trajectory": "2D, 3D, and VSLAM map figures are exported separately for paper-style analysis."
                },
            }
            bundle = self._render_episode_figures(episode_prefix, bundle, row)
            traj_json = self.trajectory_dir / f"{episode_prefix}_trajectory_bundle.json"
            traj_json.write_text(json.dumps(bundle, indent=2))
            latest_json = self.trajectory_dir / "latest_trajectory_bundle.json"
            latest_json.write_text(json.dumps(bundle, indent=2))
        except Exception as exc:
            print(f"[FIG] trajectory/artifact save skipped: {exc}")

    # ----------------------------------------------------------------
    # Camera RGB + heatmap exports
    # ----------------------------------------------------------------
    def _capture_inspection_views(self, tag: str) -> None:
        color_path = self.figure_dir / f"downcam_rgb_{tag}.png"
        heat_path = self.figure_dir / f"downcam_heatmap_{tag}.png"
        front_path = self.figure_dir / f"front_rgb_{tag}.png"
        down_seq_path = self.figure_dir / f"downcam_rgb_sequence_{tag}.png"
        down_heat_seq_path = self.figure_dir / f"downcam_heatmap_sequence_{tag}.png"
        front_seq_path = self.figure_dir / f"front_rgb_sequence_{tag}.png"
        ok_down = False
        ok_front = False
        if self.capture_rgb:
            ok_down = self._capture_camera_rgb("/World/Drone/BottomInspectionCamera", color_path, width=960, height=720, warmup=8)
            ok_front = self._capture_camera_rgb("/World/Drone/StereoLeftCamera", front_path, width=960, height=720, warmup=8)
        if not ok_down:
            self._write_camera_like_rgb(color_path, camera_mode="bottom")
        if not ok_front:
            self._write_camera_like_rgb(front_path, camera_mode="front")

        # Export multi-frame camera sequences from the saved trajectory.  The old
        # artifact folder had only one downcam RGB image, which made it look like
        # the bottom camera was sampled once while the front camera had sequences.
        self._write_camera_sequence_panel(down_seq_path, camera_mode="bottom", max_frames=12, also_individual=True, prefix=f"downcam_rgb_{tag}")
        self._write_camera_sequence_panel(front_seq_path, camera_mode="front", max_frames=10, also_individual=False, prefix=f"front_rgb_{tag}")
        self._write_downcam_heatmap_sequence_panel(down_heat_seq_path, max_frames=12)
        self._write_heatmap(heat_path)
        print(f"[INSPECTION] saved front/down RGB, multi-frame downcam sequence, and heatmap for {tag}")

    def _write_camera_sequence_panel(self, path: Path, camera_mode: str = "bottom", max_frames: int = 12, also_individual: bool = False, prefix: str = "camera") -> None:
        rows = self._trajectory_dicts()
        if not rows:
            self._write_camera_like_rgb(path, camera_mode=camera_mode)
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            chosen = self._choose_visual_indices(rows, max_frames=max_frames)
            mode = str(camera_mode).lower()
            if mode in ("bottom", "down", "downward", "downcam"):
                uniform = [int(i) for i in np.linspace(0, len(rows) - 1, min(max_frames, len(rows)))]
                chosen = sorted(set(chosen + uniform))[:max_frames]
            ncols = 4 if len(chosen) > 4 else max(1, len(chosen))
            nrows = int(math.ceil(len(chosen) / max(ncols, 1)))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.75 * nrows + 0.6), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")

            camera_path = "/World/Drone/BottomInspectionCamera" if mode in ("bottom", "down", "downward", "downcam") else "/World/Drone/StereoLeftCamera"
            real_frames = 0
            old_pos = np.asarray(getattr(self, "pos", np.zeros(3, np.float32)), dtype=np.float32).copy()
            old_yaw = float(getattr(self, "yaw", 0.0))
            old_target = np.asarray(getattr(self, "target", old_pos), dtype=np.float32).copy()
            old_visible = True
            self._capture_replay_active = True
            if not bool(getattr(self, "show_capture_replay", False)):
                self._set_drone_visibility(False)
                old_visible = False
            for j, idx in enumerate(chosen):
                r = rows[idx]
                pos = np.array([float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("z", 0.0))], dtype=np.float32)
                yaw = float(r.get("yaw", 0.0))

                # Prefer the real Isaac/Replicator camera.  This prevents
                # downcam_rgb_ep*.png from becoming the old flat analytic
                # rectangle/feature-dot fallback.  Fallback is used only when
                # Isaac camera capture is unavailable or explicitly disabled.
                img = None
                if bool(getattr(self, "capture_rgb", False)) and omni is not None and getattr(self, "stage", None) is not None:
                    self._set_capture_pose(pos, yaw)
                    img = self._capture_camera_rgb_array(camera_path, width=960, height=720, warmup=8)
                if img is not None:
                    real_frames += 1
                else:
                    img = self._camera_feature_rgb_frame(pos, yaw, camera_mode=camera_mode, bins=384)

                ax = axes.ravel()[j]
                ax.imshow(img, origin="upper")
                tag = "real" if img is not None and img.shape[0] != 384 else "fallback"
                ax.set_title(f"Step {int(r.get('step', idx))}", fontsize=11)
                if also_individual:
                    self._write_png(self.figure_dir / f"{prefix}_{j:02d}.png", img)

            # Restore the simulated pose after replay capture.  This prevents
            # the main GUI from remaining at the last captured old trajectory pose
            # while Gym/SB3 is about to reset the episode.
            self.target = old_target.copy()
            self._set_capture_pose(old_pos, old_yaw, update=False)
            if not old_visible:
                self._set_drone_visibility(True)
            self._capture_replay_active = False
            self._stable_render_update(2)

            title = "Downward real RGB camera sequence" if mode in ("bottom", "down", "downward", "downcam") else "Front real RGB camera sequence"
            if real_frames == 0:
                title += " (fallback projection; enable GUI/Replicator if needed)"
            fig.suptitle(title, fontsize=15, y=0.985)
            fig.tight_layout()
            fig.savefig(path, dpi=210, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
        except Exception as exc:
            print(f"[CAMERA_SEQ] sequence skipped for {camera_mode}: {exc}")
            try:
                if 'old_target' in locals():
                    self.target = old_target.copy()
                self._set_capture_pose(old_pos, old_yaw, update=False)
                self._set_drone_visibility(True)
                self._capture_replay_active = False
                self._stable_render_update(2)
            except Exception:
                pass

    def _write_downcam_heatmap_sequence_panel(self, path: Path, max_frames: int = 12) -> None:
        rows = self._trajectory_dicts()
        if not rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            chosen = self._choose_visual_indices(rows, max_frames=max_frames)
            uniform = [int(i) for i in np.linspace(0, len(rows) - 1, min(max_frames, len(rows)))]
            chosen = sorted(set(chosen + uniform))[:max_frames]
            ncols = 4 if len(chosen) > 4 else max(1, len(chosen))
            nrows = int(math.ceil(len(chosen) / max(ncols, 1)))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.75 * nrows + 0.8), squeeze=False)
            last_im = None
            for ax in axes.ravel():
                ax.axis("off")
            for j, idx in enumerate(chosen):
                r = rows[idx]
                pos = np.array([float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("z", 0.0))], dtype=np.float32)
                yaw = float(r.get("yaw", 0.0))
                H, cnt = self._visible_feature_image_heatmap(pos, yaw, feature_gain=1.0, bins=128, camera_mode="bottom")
                if H is None:
                    H = np.zeros((128, 128), dtype=np.float32)
                H = np.power(np.clip(H, 0.0, 1.0), 0.70)
                ax = axes.ravel()[j]
                last_im = ax.imshow(H, cmap="magma", origin="lower", interpolation="bicubic", vmin=0.0, vmax=1.0)
                ax.set_title(f"Step {int(r.get('step', idx))}", fontsize=11)
                ax.text(0.03, 0.94, f"features={cnt}", transform=ax.transAxes, ha="left", va="top", color="white", fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.45, edgecolor="none"))
            if last_im is not None:
                cax = fig.add_axes([0.27, 0.045, 0.46, 0.022])
                cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
                cbar.set_label("Normalized down-camera visual feature response", fontsize=10)
            fig.suptitle("Downward camera heatmap sequence", fontsize=15, y=0.985)
            # Do not call tight_layout after adding a manual colorbar axes.
            # Matplotlib warns that these axes are incompatible with tight_layout.
            fig.subplots_adjust(left=0.03, right=0.985, bottom=0.10, top=0.90, wspace=0.08, hspace=0.22)
            fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
        except Exception as exc:
            print(f"[DOWNCAM_HEAT_SEQ] skipped: {exc}")

    def _set_capture_pose(self, pos: np.ndarray, yaw: float, update: bool = True) -> None:
        """Move drone/cameras to a recorded pose for RGB replay capture.

        The UAV mesh is usually hidden while this is called, so the GUI no
        longer shows multiple drone positions during reset/capture.
        """
        try:
            self.pos = np.asarray(pos, dtype=np.float32).reshape(3).copy()
            self.yaw = float(yaw)
            self._sync_scene()
            if update:
                self._stable_render_update(1)
        except Exception:
            pass

    def _capture_camera_rgb_array(self, camera_path: str, width: int = 640, height: int = 480, warmup: int = 4) -> Optional[np.ndarray]:
        try:
            # Isaac RTX/DLSS can warn or upscale when offscreen render products
            # are too small (for example 320x240). Force a safe minimum.
            width = int(max(int(width), 640))
            height = int(max(int(height), 480))
            if omni is None or getattr(self, "stage", None) is None:
                return None
            import omni.replicator.core as rep
            prim = self.stage.GetPrimAtPath(camera_path)
            if not prim or not prim.IsValid():
                print(f"[CAMERA_RGB] camera not found: {camera_path}")
                return None
            # Let camera transform and material changes reach Hydra before the
            # render product is created; otherwise Isaac can return an old or
            # partially initialized frame.
            for _ in range(2):
                omni.kit.app.get_app().update()
            rp = rep.create.render_product(camera_path, (int(width), int(height)))
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([rp])
            for _ in range(max(2, int(warmup))):
                try:
                    rep.orchestrator.step()
                except Exception:
                    pass
                omni.kit.app.get_app().update()
            arr = np.asarray(annot.get_data())
            try:
                annot.detach()
            except Exception:
                pass
            try:
                rp.destroy()
            except Exception:
                pass
            if arr is None or arr.size == 0 or arr.ndim != 3 or arr.shape[2] < 3:
                return None
            rgb = arr[:, :, :3].astype(np.uint8)
            # Reject all-white/all-black/constant bad captures; fall back to the
            # analytical camera projection in that case.
            if float(np.std(rgb)) < 2.0:
                return None
            return rgb
        except Exception as exc:
            print(f"[CAMERA_RGB] capture skipped for {camera_path}: {exc}")
            return None

    def _capture_camera_rgb(self, camera_path: str, out_png: Path, width: int = 640, height: int = 480, warmup: int = 4) -> bool:
        rgb = self._capture_camera_rgb_array(camera_path, width=width, height=height, warmup=warmup)
        if rgb is None:
            return False
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.image as mpimg
            out_png.parent.mkdir(parents=True, exist_ok=True)
            mpimg.imsave(str(out_png), rgb)
        except Exception:
            self._write_png(out_png, rgb)
        return True

    def _write_camera_like_rgb(self, path: Path, camera_mode: str = "bottom") -> None:
        rows = self._trajectory_dicts()
        if rows:
            r = rows[-1]
            pos = np.array([float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("z", 0.0))], dtype=np.float32)
            yaw = float(r.get("yaw", 0.0))
        else:
            pos = np.asarray(self.pos, dtype=np.float32)
            yaw = float(self.yaw)
        img = self._camera_feature_rgb_frame(pos, yaw, camera_mode=camera_mode, bins=512)
        self._write_png(path, img)

    def _write_synthetic_downcam_rgb(self, path: Path, grid: int = 512) -> None:
        # Kept for backward compatibility; now produces an actual camera-frame view.
        self._write_camera_like_rgb(path, camera_mode="bottom")

    def _write_heatmap(self, path: Path, grid: int = 160) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            heatmap, xedges, yedges = self._build_heatmap_array()
            if heatmap is None or xedges is None or yedges is None:
                heatmap = np.zeros((grid, grid), np.float32)
                xedges = np.linspace(-self.world_size, self.world_size, grid + 1)
                yedges = np.linspace(-self.world_size, self.world_size, grid + 1)
            fig, ax = plt.subplots(figsize=(6.2, 5.8))
            im = ax.imshow(heatmap, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="magma", interpolation="bicubic")
            rows = self._trajectory_dicts()
            if rows:
                sx = np.array([float(r.get("slam_x", r.get("x", 0.0))) for r in rows], dtype=np.float32)
                sy = np.array([float(r.get("slam_y", r.get("y", 0.0))) for r in rows], dtype=np.float32)
                ax.plot(sx, sy, color="white", lw=1.2, alpha=0.85, label="VSLAM trajectory")
            ax.plot(self.inspection_points[:, 0], self.inspection_points[:, 1], "*", ms=8, label="inspection pts")
            ax.set_title("Downward inspection heatmap: camera-visible VSLAM features")
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.legend(fontsize=7)
            fig.colorbar(im, ax=ax, label="normalized visual-feature intensity")
            fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
        except Exception:
            H = np.zeros((grid, grid), np.float32)
            rgb = (np.stack([H, H * 0.45, 1.0 - H], axis=-1) * 255).astype(np.uint8)
            self._write_png(path, rgb)

    @staticmethod
    def _write_png(path: Path, rgb: np.ndarray) -> None:
        arr = np.asarray(rgb, np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return
        h, w, _ = arr.shape
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(raw, 6))
        png += chunk(b"IEND", b"")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(png)


# =====================================================================
# Isaac binding + train/eval/demo
# =====================================================================
def _bind_omni() -> None:
    global omni, Gf, Sdf, UsdGeom, UsdLux, UsdShade
    import omni as _omni
    import omni.usd  # noqa: F401
    import omni.kit.app  # noqa: F401
    from pxr import Gf as _Gf, Sdf as _Sdf, UsdGeom as _UsdGeom, UsdLux as _UsdLux, UsdShade as _UsdShade
    omni = _omni
    Gf, Sdf, UsdGeom, UsdLux, UsdShade = _Gf, _Sdf, _UsdGeom, _UsdLux, _UsdShade


def _default_model_path(args: argparse.Namespace) -> str:
    """Source-domain model path used for e1 -> e2 transfer evaluation."""
    if args.model_path:
        return str(Path(args.model_path).expanduser())
    env_path = os.environ.get("UAV_SOURCE_MODEL", "")
    if env_path:
        return str(Path(env_path).expanduser())
    # Default to the GUI/e1 training folder used in the previous command.
    return str(Path("~/uav_results_e1_gui_llm/models/power_plant_ppo.zip").expanduser())


def evaluate(env: IndustrialE2EvalEnv, args: argparse.Namespace) -> None:
    baseline = str(getattr(args, "baseline", "ppo") or "ppo").lower()
    if baseline == "proposed":
        baseline = "ppo"

    model = None
    if baseline == "ppo":
        from stable_baselines3 import PPO
        model_path = _default_model_path(args)
        if not Path(model_path).expanduser().exists():
            raise FileNotFoundError(
                f"Source model not found: {model_path}\n"
                "Pass --model-path /path/to/power_plant_ppo.zip or set UAV_SOURCE_MODEL."
            )
        print(f"[E2_EVAL] loading source-domain model: {model_path}")
        model = PPO.load(model_path, device=args.device)
    else:
        print(f"[E2_EVAL] running baseline without PPO model: {baseline}")

    for ep in range(int(args.eval_episodes)):
        obs, _ = env.reset(seed=args.seed + ep)
        done = trunc = False
        while not (done or trunc):
            if model is not None:
                act, _ = model.predict(obs, deterministic=True)
            else:
                # Baseline speed proposal is computed inside env.step().
                act = np.array([0.0], np.float32)
            obs, _r, done, trunc, _info = env.step(act)
    print(f"[E2_EVAL] baseline={baseline} metrics -> {env.summary_json}")
    print(f"[E2_EVAL] paper row -> {env.result_table_csv}")


def demo(env: IndustrialE2EvalEnv, args: argparse.Namespace) -> None:
    obs, _ = env.reset(seed=args.seed)
    done = trunc = False
    while not (done or trunc):
        # Neutral action maps to mid speed; governor will slow in weak VSLAM.
        obs, _r, done, trunc, _info = env.step(np.array([0.0], np.float32))
    print("[E2_DEMO] rollout complete")
    print(f"[E2_DEMO] metrics -> {env.summary_json}")


def scene_only(env: IndustrialE2EvalEnv, args: argparse.Namespace) -> None:
    """Open only the e2 industrial scene/viewer for debugging."""
    env.reset(seed=args.seed)
    seconds = float(max(1.0, getattr(args, "scene_preview_seconds", 30.0)))
    print(f"[SCENE_ONLY] showing e2 industrial scene for {seconds:.1f}s; viewer_mode={args.viewer_mode}")
    t_end = time.time() + seconds
    while time.time() < t_end:
        env._flush_gui(frames=2, update_viewport=True)
        time.sleep(0.03)
    print("[SCENE_ONLY] finished")


def main() -> None:
    args = parse_args()
    # Safety: this file is intentionally evaluation-only.
    args.mode = "eval" if str(args.mode).lower() not in ("eval", "demo") else str(args.mode).lower()

    if SimulationApp is None:
        raise RuntimeError(
            "isaacsim.SimulationApp unavailable. Run under Isaac Lab:\n"
            "  cd ~/IsaacLab && ./isaaclab.sh -p source/standalone/uav_llm/uav_llm_e2.py --mode eval --model-path ~/uav_results_e1_gui_llm/models/power_plant_ppo.zip"
        )

    # Isaac Kit/SimulationApp can misinterpret script arguments if argv is left
    # populated after argparse. Clear it before constructing SimulationApp.
    sys.argv = [sys.argv[0]]

    app = SimulationApp({
        "headless": bool(args.headless),
        "renderer": str(args.renderer),
        "width": 1600,
        "height": 900,
    })
    _bind_omni()

    render_sim = (
        (not bool(args.headless) and (args.mode in ("demo", "eval") or bool(args.scene_only)))
        or args.slam_mode == "cuvslam"
        or bool(args.capture_rgb)
    )
    env = IndustrialE2EvalEnv(args, render_sim=render_sim)
    try:
        if bool(getattr(args, "scene_only", False)):
            scene_only(env, args)
        elif args.mode == "eval":
            evaluate(env, args)
        else:
            demo(env, args)
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
