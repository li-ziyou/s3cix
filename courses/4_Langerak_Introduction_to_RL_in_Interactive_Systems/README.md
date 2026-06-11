# An Introduction to Reinforcement Learning in Interactive Systems

Tutorial materials for the **Summer School on Computational Interaction 2026**, Glasgow.

By Thomas Langerak, Aalto University — thomas.langerak@aalto.fi

---

## Contents

| File | Description |
|------|-------------|
| `demo.ipynb` | Main tutorial notebook |
| `Intro_RL_S3CIX26.pdf` | Accompanying slides |
| `notification_env/` | Custom Gymnasium environment |
| `config.yaml` | Environment hyperparameters |

---

## Setup

### Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Launch JupyterLab

```bash
uv run jupyter lab
```

Open `demo.ipynb` and select the **RL4HCI** kernel (top-right kernel picker).

---

## Environment: `NotificationScheduler-v0`

A POMDP Gymnasium environment simulating notification delivery timing.

**The agent** decides at every step whether to **wait** (protect the user's focus) or **deliver** (flush the notification queue and interrupt the user).

**Observation space** — `MultiDiscrete([3, 3, 4, 9, 4, 4])`:

| Index | Field | Values |
|-------|-------|--------|
| 0 | context | 0=idle, 1=focus, 2=meeting |
| 1 | time\_of\_day | 0=morning, 1=afternoon, 2=evening |
| 2 | time\_since\_delivery | 0=just, 1=short, 2=medium, 3=long |
| 3 | queue\_size | 0–8 |
| 4 | max\_urgency | 0=none, 1=low, 2=medium, 3=high |
| 5 | max\_importance | 0=none, 1=low, 2=medium, 3=high |

**Action space** — `Discrete(2)`: 0=wait, 1=deliver

**Hidden state**: user concentration (low / medium / high) — never observed by the agent.

```python
import notification_env
from notification_env import NotificationSchedulerEnv

env = NotificationSchedulerEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(1)  # deliver
```