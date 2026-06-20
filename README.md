# Robust Speed Control for UAV Infrastructure Inspection Under Visual Localization Degradation

**Risk-aware speed control, VSLAM-health monitoring, and command checking for GPS-denied UAV inspection**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-Required-green)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-Compatible-purple)](https://isaac-sim.github.io/IsaacLab/)
[![ROS 2](https://img.shields.io/badge/ROS%202-Logging%20%7C%20VSLAM-orange)](https://docs.ros.org/)
[![RL](https://img.shields.io/badge/RL-PPO%20Speed%20Policy-red)](#method-summary)
[![License](https://img.shields.io/badge/License-Research%20Use-lightgrey)](#license)

> **Main idea:** A learned policy proposes a forward speed, but a deterministic command-checking layer verifies VSLAM risk, corridor deviation, yaw error, rate limits, and fallback conditions before the final command is executed.

---

## Overview

<p align="justify">
This repository provides the code, evaluation data, figures, and paper graphs for <b>Robust Speed Control for UAV Infrastructure Inspection Under Visual Localization Degradation</b>. The work studies GPS-denied UAV inspection in infrastructure scenes where visual localization can degrade because of weak texture, smoke, repetitive structures, lighting changes, motion blur, yaw uncertainty, or intermittent VSLAM tracking loss. The goal is not open-ended UAV navigation; the route is already planned. The core problem is how to regulate forward speed safely when localization quality changes along that route.
</p>

<p align="justify">
The framework separates <b>policy proposal</b> from <b>command execution</b>. A PPO-based speed policy proposes a bounded nominal speed. A VSLAM-aware risk-speed rule then adjusts the proposal using feature support, inlier quality, pose uncertainty, yaw uncertainty, tracking-loss status, and mission risk. Finally, a deterministic governor checks corridor error, yaw error, localization risk, tracking loss, speed limits, and abort conditions before the command reaches the simulated autopilot interface.
</p>

<p align="justify">
The repository name includes historical script names such as <code>uav_llm.py</code> and <code>uav_gpt.py</code>. In the setting, language is not used at the control rate. Mission text is converted once per episode into a compact numeric risk context, and the controller receives only this bounded mission vector and scalar risk preference.
</p>

---

## System View

<p align="justify">
The system is designed for planned-route infrastructure inspection. The UAV follows fixed route waypoints, observes scene and localization-health signals, proposes speed through the learned policy, applies risk-aware speed correction, and executes only the checked command. The evaluation uses a source power-plant scene and a target industrial scene to test zero-shot source-to-target behavior under VSLAM degradation.
</p>

<p align="center">
  <img src="./fig2/power.png" width="46%" alt="Power plant source domain"/>
  <img src="./fig2/indust.png" width="46%" alt="Industrial target domain"/>
</p>

<p align="center"><b>Source and target inspection scenes used by the repository.</b></p>

---

## Method Summary

### 1. VSLAM-risk-aware speed proposal

<p align="justify">
The learned policy observes the online inspection state and mission context, then proposes a nominal speed. The risk-speed rule reduces this speed when visual localization becomes weak. The VSLAM risk score increases when feature support or inlier quality decreases, when position or yaw uncertainty rises, or when tracking loss occurs. A rate limiter prevents abrupt speed jumps.
</p>

```text
state x_t = [online observation o_t ; mission context m]
policy action a_t -> nominal speed v_t
VSLAM health h_t -> localization risk z_t
risk-speed rule -> adjusted speed v_hat_t
command gate -> checked command v_c_t
```

### 2. Deterministic command checking

<p align="justify">
The final command is not the raw policy output. The governor verifies route-corridor error, yaw error, tracking-loss flag, localization risk, speed bounds, and abort thresholds. If the command violates the safety envelope, the system uses a fallback command such as hover or abort/return behavior. This makes every correction auditable in the logs.
</p>

```text
if corridor OK and yaw OK and VSLAM OK and risk below threshold:
    execute checked speed
else:
    execute fallback speed / hover / abort command
```

### 3. Mission-risk encoding

<p align="justify">
Mission text uses a closed template, for example <code>E:low-texture G:coverage S:feature-slow O:slow-smooth</code>. The mission encoder maps this record to a numeric vector and scalar risk preference once per episode. This gives the controller mission awareness without running an LLM during real-time control.
</p>

### 4. Source-to-target validation

<p align="justify">
The source policy is trained in a smoke-degraded power-plant environment and evaluated zero-shot in an industrial target environment. The validation reports route progress, localization error, tracking loss, governor overrides, gate acceptance, abort incidence, drift return, and CVaR drift. This emphasizes drift-tail behavior instead of route completion alone.
</p>

---

## Repository Layout

```text
uav-llm-drl-main/
├── code/
│   ├── uav_e2_eval_baselines.py       <- target-domain evaluation loader for source-trained PPO
│   ├── uav_e2_fixed_governor.py       <- target fixed-speed baseline with command governor
│   ├── uav_e2_fixed_no_governor.py    <- target fixed-speed baseline without governor
│   ├── uav_fixed_governor.py          <- source fixed-speed baseline with governor
│   ├── uav_fixed_no_governor.py       <- source fixed-speed baseline without governor
│   ├── uav_fix.py                     <- source-domain corrected/proposed PPO + governor workflow
│   ├── uav_gpt.py                     <- mission-conditioned source workflow variant
│   ├── uav_llm_e2.py                  <- target-domain proposed PPO + governor evaluation
│   ├── uav_llm.py                     <- main source-domain mission-conditioned PPO workflow
│   ├── uav_pid_e2.py                  <- target-domain PID + governor baseline
│   └── uav_pid.py                     <- source-domain PID + governor baseline
│
├── data/
│   ├── e1.zip                         <- source-domain records and artifacts
│   ├── eval.zip                       <- evaluation logs/results archive
│   ├── other.zip                      <- auxiliary experiment files
│   ├── seeds.zip                      <- multi-seed logs and summaries
│   ├── speed.zip                      <- speed-control and sweep outputs
│   └── uav_llm.zip                    <- proposed method logs/checkpoints/artifacts
│
├── fig2/
│   ├── e1/
│   │   ├── 
│   │   ├── downcam_heatmap_ep0019.png
│   │   ├── downcam_heatmap_sequence_ep0020.png
│   │   ├── downcam_rgb_sequence_ep0020.png
│   │   ├── episode_0020_bottom_visual_heatmap_sequence.png
│   │   └── episode_0020_vslam_trajectory.png
│   ├── e2/
│   │   ├── downcam_heatmap_ep0020.png
│   │   ├── downcam_rgb_sequence_ep0020.png
│   │   ├── episode_0019_vslam_trajectory.png
│   │   ├── episode_0020_bottom_visual_heatmap_sequence.png
│   │   └── episode_0020_vslam_trajectory.png
│   ├── indust.png                     <- industrial target-domain visual
│   └── power.png                      <- power-plant source-domain visual
│
├── graphs/
│   ├── drift_speed_tradeoff.pdf       <- drift return, CVaR, and speed comparison
│   ├── override.pdf                   <- governor override and gate behavior
│   └── vslam_error_loss.pdf           <- ATE and tracking-loss comparison
│
└── README.md
```

---

## Code Tour

| Script | Main role | Typical use |
|---|---|---|
| `code/uav_llm.py` | Main source-domain PPO workflow with mission-risk context, VSLAM proxy, command checking, Isaac Sim scene generation, logging, and optional ROS 2 / cuVSLAM bridge hooks. | Train or evaluate the proposed policy in the source power-plant domain. |
| `code/uav_fix.py` | Corrected source-domain PPO/governor workflow used for stable  runs. | Re-run proposed source-domain training/evaluation with fixed settings. |
| `code/uav_llm_e2.py` | Target-domain proposed method evaluation script. | Load the source-trained policy and evaluate zero-shot in the industrial target domain. |
| `code/uav_e2_eval_baselines.py` | Target evaluation wrapper for the source-trained model and baseline comparison workflow. | Run target-domain evaluation with consistent logging. |
| `code/uav_fixed_no_governor.py` | Source fixed-speed baseline without command governor. | Isolate fixed-speed behavior without safety trimming. |
| `code/uav_fixed_governor.py` | Source fixed-speed baseline with deterministic governor. | Measure governor-only benefit without learned speed proposal. |
| `code/uav_e2_fixed_no_governor.py` | Target fixed-speed baseline without governor. | Target-domain no-governor baseline. |
| `code/uav_e2_fixed_governor.py` | Target fixed-speed baseline with governor. | Target-domain governor-only baseline. |
| `code/uav_pid.py` | Source PID speed scheduler with governor. | Compare learned speed proposal against a hand-tuned controller. |
| `code/uav_pid_e2.py` | Target PID speed scheduler with governor. | Zero-shot target PID baseline. |
| `code/uav_gpt.py` | Historical mission-conditioned workflow variant. | Ablation/debugging of mission-context encoding. |

---

## Data and Artifact Archives

<p align="justify">
The <code>data/</code> directory stores compressed experiment artifacts. These archives are useful when reproducing tables and figures without re-running all Isaac Sim episodes. Extract only the archives needed for the experiment you want to inspect.
</p>

```bash
# From repository root
mkdir -p extracted
unzip data/e1.zip -d extracted/e1
unzip data/eval.zip -d extracted/eval
unzip data/seeds.zip -d extracted/seeds
unzip data/speed.zip -d extracted/speed
unzip data/uav_llm.zip -d extracted/uav_llm
```

Suggested interpretation:

| Archive | Contents / purpose |
|---|---|
| `data/e1.zip` | Source-domain power-plant episodes, metrics, or visual artifacts. |
| `data/eval.zip` | Evaluation summaries and logs used for tables. |
| `data/seeds.zip` | Multi-seed records for robustness and variance checks. |
| `data/speed.zip` | Speed-control analysis, governor behavior, and sweep outputs. |
| `data/uav_llm.zip` | Proposed-method outputs, checkpoints, or mission-conditioned logs. |
| `data/other.zip` | Additional auxiliary files used during paper development. |

---

## Visual Results

### Source-domain inspection artifacts

<p align="center">
  <img src="./fig2/e1/downcam_rgb_sequence_ep0020.png" width="48%" alt="Source-domain downward RGB sequence"/>
</p>

<p align="justify">
These images show the source-domain visual stream and heatmap-style localization/feature-support artifacts from the power-plant inspection scene. They are useful for explaining why visual localization can degrade in smoke, repetitive structures, and weakly textured regions.
</p>

<p align="center">
  <img src="./fig2/e1/episode_0020_bottom_visual_heatmap_sequence.png" width="48%" alt="Source-domain visual feature heatmap sequence"/>
</p>

### Target-domain inspection artifacts

<p align="center">
  <img src="./fig2/e2/downcam_rgb_sequence_ep0020.png" width="48%" alt="Target-domain downward RGB sequence"/>
</p>

<p align="justify">
These images summarize the target industrial scene used for zero-shot evaluation. The target route is longer but visually easier because the scene contains more static and feature-rich structures than the smoke-degraded source scene.
</p>

<p align="center">
  <img src="./fig2/e2/episode_0020_bottom_visual_heatmap_sequence.png" width="48%" alt="Target-domain visual feature heatmap sequence"/>
</p>

### VSLAM trajectories

<p align="center">
  <img src="./fig2/e1/episode_0020_vslam_trajectory.png" width="48%" alt="Source-domain VSLAM trajectory"/>
</p>

<p align="center">
  <img src="./fig2/e2/episode_0020_vslam_trajectory.png" width="48%" alt="Target-domain VSLAM trajectory"/>
</p>

### Graph files

GitHub may open these PDF graphs as separate files rather than inline images:

- [`graphs/vslam_error_loss.pdf`](./graphs/vslam_error_loss.pdf) — target-domain ATE and tracking-loss comparison.
- [`graphs/drift_speed_tradeoff.pdf`](./graphs/drift_speed_tradeoff.pdf) — drift return, CVaR drift, and speed trade-off.
- [`graphs/override.pdf`](./graphs/override.pdf) — governor intervention and gate behavior.

---

## Requirements

### Recommended platform

- Ubuntu 20.04 / 22.04
- NVIDIA GPU with recent driver
- NVIDIA Isaac Sim and/or Isaac Lab environment
- Python 3.10+
- ROS 2 if using bridge or cuVSLAM-related topics
- Optional Hugging Face access if using the historical `llama_hf` mission encoder path

### Python packages

The scripts import standard scientific packages directly and load Isaac/Omniverse modules through Isaac Sim. A minimal Python layer is:

```bash
pip install numpy gymnasium stable-baselines3 torch transformers
```

For Isaac Sim / Isaac Lab execution, run the scripts through the Isaac Lab launcher so that `isaacsim`, `omni`, and `pxr` modules are available.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/szu-ai/uav-llm-drl.git
cd uav-llm-drl-main
```

### 2. Check the repository structure

```bash
find . -maxdepth 3 -type f | sort
```

### 3. Copy or reference the scripts from Isaac Lab

Option A: Run directly by absolute path through Isaac Lab:

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_llm.py --mode demo --headless
```

Option B: Copy the scripts into an Isaac Lab standalone folder:

```bash
mkdir -p ~/IsaacLab/source/standalone/uav_llm
cp /path/to/uav-llm-drl-main/code/*.py ~/IsaacLab/source/standalone/uav_llm/
cd ~/IsaacLab
./isaaclab.sh -p source/standalone/uav_llm/uav_llm.py --mode demo --headless
```

---

## Reproducing the Main Workflow

### A. Train the proposed source-domain policy

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_llm.py \
  --mode train \
  --headless \
  --device cuda \
  --seed 7 \
  --total-timesteps 500000 \
  --mission-text "E:low-texture G:coverage S:feature-slow O:slow-smooth" \
  --output-root ~/uav_results_e1_gui_llm
```

Expected checkpoint path:

```text
~/uav_results_e1_gui_llm/models/power_plant_ppo.zip
```

### B. Evaluate the proposed method in the source domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_llm.py \
  --mode eval \
  --headless \
  --device cuda \
  --eval-episodes 20 \
  --model-path ~/uav_results_e1_gui_llm/models/power_plant_ppo.zip \
  --output-root ~/uav_results_e1_eval
```

### C. Evaluate zero-shot in the target industrial domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_llm_e2.py \
  --mode eval \
  --headless \
  --device cuda \
  --eval-episodes 20 \
  --model-path ~/uav_results_e1_gui_llm/models/power_plant_ppo.zip \
  --output-root ~/uav_results_e2_eval
```

---

## Baseline Commands

### Fixed speed without governor — source domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_fixed_no_governor.py \
  --mode eval \
  --headless \
  --eval-episodes 20 \
  --fixed-speed 1.35 \
  --output-root ~/uav_results_fixed_no_gov_e1
```

### Fixed speed with governor — source domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_fixed_governor.py \
  --mode eval \
  --headless \
  --eval-episodes 20 \
  --fixed-speed 1.35 \
  --output-root ~/uav_results_fixed_gov_e1
```

### PID with governor — source domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_pid.py \
  --mode eval \
  --headless \
  --eval-episodes 20 \
  --output-root ~/uav_results_pid_e1
```

### Fixed speed without governor — target domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_e2_fixed_no_governor.py \
  --mode eval \
  --headless \
  --eval-episodes 20 \
  --fixed-speed 1.35 \
  --output-root ~/uav_results_fixed_no_gov_e2
```

### Fixed speed with governor — target domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_e2_fixed_governor.py \
  --mode eval \
  --headless \
  --eval-episodes 20 \
  --fixed-speed 1.35 \
  --output-root ~/uav_results_fixed_gov_e2
```

### PID with governor — target domain

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_pid_e2.py \
  --mode eval \
  --headless \
  --eval-episodes 20 \
  --output-root ~/uav_results_pid_e2
```

---

## Important Runtime Options

| Option | Meaning |
|---|---|
| `--mode train` | Train the source-domain PPO speed policy. |
| `--mode eval` | Load a checkpoint or run a baseline and collect metrics. |
| `--mode demo` | Run a visual/debug demonstration. |
| `--headless` | Run without GUI. Remove this flag for visual Isaac Sim rendering. |
| `--device cuda` | Use GPU for learning components where supported. |
| `--model-path` | Path to a trained PPO checkpoint. |
| `--output-root` | Directory for logs, metrics, figures, and checkpoints. |
| `--eval-episodes` | Number of evaluation episodes. |
| `--mission-text` | Closed mission template used by the mission-risk encoder. |
| `--slam-mode proxy` | Use the repeatable VSLAM proxy for controlled validation. |
| `--slam-mode cuvslam` | Use cuVSLAM-related odometry/bridge path when configured. |
| `--capture-rgb` | Save RGB/visual artifacts for paper figures. |
| `--disable-figures` | Disable figure generation for faster runs. |
| `--speed-min`, `--speed-max` | Speed bounds. |
| `--governor-alpha`, `--governor-beta` | VSLAM-risk and mission-risk speed reduction gains. |
| `--vslam-risk-limit` | Gate threshold for VSLAM risk. |
| `--corridor-limit`, `--yaw-limit-deg` | Route safety limits for command checking. |
| `--abort-risk-limit`, `--abort-corridor-limit`, `--abort-yaw-limit-deg` | Hard abort thresholds. |

---

## Metrics Reported by the Paper Workflow

| Metric | Meaning | Direction |
|---|---|---:|
| `Speed v_bar^p` | Mean policy-proposed speed before the final gate. | Context dependent |
| `ATE E^a` | Absolute trajectory error. | Lower is better |
| `Track loss ell` | Percentage of VSLAM tracking-loss events. | Lower is better |
| `Override O` | Percentage of commands modified by the governor. | Lower is usually better |
| `Gate g` | Percentage of hard-gate accepted commands. | Higher is usually better |
| `Abort A` | Abort or hard-safety event incidence. | Lower is better |
| `Drift D` | Episode-level drift return from corridor, yaw, and tracking loss. | Lower is better |
| `CVaR drift` | Tail-risk drift score over worst episodes. | Lower is better |

---

## Target-Domain Results from the  Evaluation

The paper reports the following zero-shot target-domain comparison for the seed-7 checkpoint:

| Method | Speed `v_bar^p` (m/s) | ATE `E^a` (m) ↓ | Track loss `ell` (%) ↓ | Override `O` (%) ↓ | Gate `g` (%) | Drift `D` ↓ | CVaR ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed speed without governor | 1.350 | 0.0350 | 0.111 | 0.000 | 100.000 | 5.520 | 6.215 |
| Fixed speed with governor | 1.140 | 0.0350 | 0.119 | 35.465 | 69.167 | 4.321 | 5.043 |
| PID speed control with governor | 1.110 | 0.0440 | 0.506 | 48.692 | 84.053 | 4.371 | 5.110 |
| Proposed PPO speed control with governor | 1.185 | **0.0341** | **0.103** | **23.772** | 83.943 | **4.060** | **4.488** |

<p align="justify">
These values show that the proposed policy-governor combination is not simply slower. It keeps a higher proposed speed than the fixed-governor and PID-governor baselines while reducing ATE, tracking loss, drift return, CVaR drift, and override rate in the target domain.
</p>

---

## Reproducing Paper Graphs

The paper graphs are stored as PDF files:

```text
graphs/vslam_error_loss.pdf
graphs/drift_speed_tradeoff.pdf
graphs/override.pdf
```

If you want GitHub to preview them inline as images, convert them to PNG:

```bash
mkdir -p graphs_png
pdftoppm -png -r 300 graphs/vslam_error_loss.pdf graphs_png/vslam_error_loss
pdftoppm -png -r 300 graphs/drift_speed_tradeoff.pdf graphs_png/drift_speed_tradeoff
pdftoppm -png -r 300 graphs/override.pdf graphs_png/override
```

---

## Suggested Experiment Order

1. Run `uav_llm.py --mode demo` to verify Isaac Sim and scene loading.
2. Run `uav_llm.py --mode train` for the source-domain PPO policy.
3. Run `uav_llm.py --mode eval` to evaluate source behavior.
4. Run `uav_llm_e2.py --mode eval` for target-domain zero-shot evaluation.
5. Run fixed-speed and PID baselines in both source and target domains.
6. Compare CSV logs using ATE, tracking loss, override rate, drift return, and CVaR drift.
7. Regenerate or update graphs in `graphs/`.

---

## Notes on VSLAM and ROS 2

<p align="justify">
The scripts support a repeatable simulation-side VSLAM proxy and include hooks for ROS 2 / cuVSLAM-style odometry and image bridge topics. The proxy is useful for controlled paper validation because it gives repeatable localization-health signals. It should not be interpreted as a replacement for a full field-ready SLAM backend.
</p>

Common VSLAM-related options include:

```text
--slam-mode proxy
--slam-mode gt
--slam-mode cuvslam
--cuvslam-odom-topic /visual_slam/tracking/odometry
--cuvslam-odom-udp-port 14555
--cuvslam-left-image-topic /visual_slam/image_0
--cuvslam-right-image-topic /visual_slam/image_1
--cuvslam-baseline 0.18
--cuvslam-img-width 640
--cuvslam-img-height 480
```

---

## Troubleshooting

### `isaacsim.SimulationApp unavailable`

Run the script through Isaac Lab instead of a normal Python interpreter:

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path/to/uav-llm-drl-main/code/uav_llm.py --mode demo --headless
```

### `ModuleNotFoundError: stable_baselines3`

Install Stable-Baselines3 inside the Isaac Lab Python environment:

```bash
cd ~/IsaacLab
./isaaclab.sh -p -m pip install stable-baselines3 gymnasium
```

### No GUI or black window

Use headless mode for server runs:

```bash
--headless
```

For visualization, remove `--headless`, check that the display server is available, and ensure that the NVIDIA driver supports Isaac Sim rendering.

### Graph PDFs do not show inside GitHub README

GitHub may not render PDF files inline. Link the PDFs directly or convert them to PNG using `pdftoppm`.

### Target evaluation cannot find the model

Pass the checkpoint explicitly:

```bash
--model-path ~/uav_results_e1_gui_llm/models/power_plant_ppo.zip
```

---

## Safety and Deployment Notice

<p align="justify">
This repository is a research prototype for controlled simulation and pre-HIL validation. It is not a field-ready autopilot system. Before real UAV deployment, the controller must be tested with hardware-in-the-loop, calibrated for the site and UAV platform, integrated with certified failsafe behavior, supervised by a safety pilot, and checked under local aviation and inspection regulations.
</p>

---

## Limitations

- Validation is performed in controlled Isaac Sim / pre-HIL settings.
- The VSLAM proxy provides repeatable localization-health signals but is not a new SLAM backend.
- The target domain is longer but visually easier than the source scene.
- Empirical CVaR is diagnostic and depends on the number of evaluation episodes.
- Mission text is closed-template and bounded; open-vocabulary mission understanding is outside the current validation.
- Final deployment requires hardware testing, site-specific tuning, and certified safety procedures.

---

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{borhan2026robustspeeduav,
  title   = {Robust Speed Control for UAV Infrastructure Inspection Under Visual Localization Degradation},
  author  = {Borhan, Uddin Md. and Raza, Arif and Lv, Bo and Li, Jianqiang and Chen, Jie},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2026},
  note    = {Code and resources available at https://github.com/szu-ai/uav-llm-drl}
}
```

---

## License

This repository is released for academic and research use. Add a project license file before public distribution if a specific open-source license is required.

---

## Contact

For questions about the repository or paper resources, please open an issue at:

```text
https://github.com/szu-ai/uav-llm-drl/issues
```
