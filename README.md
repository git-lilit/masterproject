# Offline RL Ensemble Methods for Biological Sequence Design

## Overview

This project investigates ensemble-based methods in Offline Reinforcement Learning (RL), focusing on their ability to mitigate distributional shift and stabilize value estimation. It provides a controlled experimental framework to compare different ensemble construction and aggregation strategies, along with a novel diversity mechanism for discrete-action settings that improves DQN-based ensembles.

The application domain is biological sequence design, where models must generalize beyond a fixed dataset without further environment interaction.

---

## Key Ideas

* **Offline RL challenge**: Learning from static datasets leads to extrapolation errors and instability due to distributional shift.
* **Ensemble methods**: Improve robustness by reducing overestimation and variance in value functions.
* **Contribution**: A diversity-promoting approach for discrete-action ensembles and a systematic comparison across methods.

---

## Repository Structure

```
rl/
│
├── main.py              # Entry point for training agents
│
├── agents/              # Agent implementations
│   ├── sac.py           # Soft Actor-Critic (continuous control)
│   ├── dqn.py           # Deep Q-Network (discrete control)
│   └── bc.py            # Behavior Cloning baseline
│
└── lib/                 # Core training components
    ├── replay_buffer.py # Dataset handling and sampling
    ├── models.py        # Neural network architectures
    └── ...              # Other utilities (training logic, helpers)
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

---

## Usage

Run training via:

```bash
python rl/main.py
```

Typical configurable elements (via flags or config inside `main.py`):

* Agent type (`sac`, `dqn`, `bc`)
* Ensemble size and aggregation method
* Dataset / task configuration

---

## Implemented Agents

* **SAC**: For continuous control tasks
* **DQN**: Supports ensemble variants and diversity enhancements
* **BC**: Supervised baseline using dataset actions

---

## Experiments

The framework enables:

* Controlled comparison of ensemble strategies
* Analysis of overestimation and stability
* Evaluation in biological sequence design tasks

---

## Extending the Project

To add a new method:

1. Implement the agent in `rl/agents/`
2. Integrate training logic using components in `rl/lib/`
3. Register the agent in `main.py`

---
