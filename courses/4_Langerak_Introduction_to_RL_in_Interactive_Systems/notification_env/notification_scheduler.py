"""
NotificationScheduler-v0: A POMDP Gymnasium environment for notification delivery timing.

The agent must decide *when* to deliver batched notifications to a user,
balancing:
  - User productivity  (interrupting concentration is costly)
  - Information freshness (notifications decay in value over time)

Hidden state: user concentration (0=low, 1=medium, 2=high)
The agent only observes context, time of day, queue statistics, and delivery
history — it never directly observes concentration.

All hyperparameters are loaded from config.yaml at the repo root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import yaml
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml. Searches repo root by default."""
    if config_path is None:
        # Walk up from this file to find config.yaml
        here = Path(__file__).resolve().parent
        for candidate in [here, here.parent, here.parent.parent]:
            p = candidate / "config.yaml"
            if p.exists():
                config_path = str(p)
                break
    if config_path is None:
        raise FileNotFoundError("config.yaml not found. Expected at repo root.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Notification dtype
# ---------------------------------------------------------------------------

_NOTIF_DTYPE = np.dtype([
    ("urgency",     np.uint8),
    ("importance",  np.uint8),
    ("initial_val", np.float32),
    ("decay_rate",  np.float32),
    ("age",         np.uint32),
])

# Display labels
CONTEXT_LABELS       = ["idle", "focus", "meeting"]
TOD_LABELS           = ["morning", "afternoon", "evening"]
CONCENTRATION_LABELS = ["low", "medium", "high"]

_URGENCY_COLORS = {0: "#4caf50", 1: "#ff9800", 2: "#f44336"}
_URGENCY_LABELS = {0: "low", 1: "med", 2: "high"}
_CONC_COLORS    = ["#ef5350", "#ffa726", "#66bb6a"]



# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NotificationSchedulerEnv(gym.Env):
    """
    Gymnasium environment: NotificationScheduler-v0
    ================================================

    Observation (MultiDiscrete):
        context              0=idle, 1=focus, 2=meeting
        time_of_day          0=morning, 1=afternoon, 2=evening
        time_since_delivery  0=just, 1=short(1-5), 2=medium(6-15), 3=long(16+)
        queue_size           0-8
        max_urgency          0-3  (0=none in queue)
        max_importance       0-3  (0=none in queue)

    Action:
        0 = wait
        1 = deliver all queued notifications

    Hidden state:
        concentration  0/1/2  — accessible via env.concentration for educational use

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml. Defaults to auto-discovered repo root.
    render_mode : str, optional
        'human' or 'rgb_array'.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        super().__init__()

        cfg = _load_config(config_path)
        self._cfg = cfg

        # --- Episode ---
        self.EPISODE_LEN       = cfg["episode"]["length"]
        self.TOD_PERIOD        = cfg["episode"]["time_of_day_period"]

        # --- Queue ---
        self.MAX_QUEUE         = cfg["queue"]["max_size"]
        self.OVERFLOW_PENALTY  = cfg["queue"]["overflow_penalty"]

        # --- Notifications ---
        self.ARRIVAL_PROB      = cfg["notifications"]["arrival_prob"]
        self.VAL_BASE          = cfg["notifications"]["value_base"]
        self.VAL_URG_SCALE     = cfg["notifications"]["value_urgency_scale"]
        self.VAL_IMP_SCALE     = cfg["notifications"]["value_importance_scale"]
        self.DECAY_BASE        = cfg["notifications"]["decay_base"]
        self.DECAY_URG_SCALE   = cfg["notifications"]["decay_urgency_scale"]

        # --- Actions / rewards ---
        self.EMPTY_DELIVER_PEN = cfg["actions"]["empty_deliver_penalty"]
        self.PROD_BONUS        = cfg["rewards"]["productivity_bonus_per_concentration"]

        # --- Concentration ---
        cc = cfg["concentration"]["ceilings"]
        self._CONC_CEILING     = {0: cc["idle"], 1: cc["focus"], 2: cc["meeting"]}
        self.DRIFT_PROB        = cfg["concentration"]["drift_prob"]
        self.DROP2_PROB        = cfg["concentration"]["drop2_prob"]

        # --- Interruption ---
        self.COST_PER_LEVEL    = cfg["interruption"]["cost_per_level"]
        cm = cfg["interruption"]["context_multipliers"]
        self._CTX_MULT         = {0: cm["idle"], 1: cm["focus"], 2: cm["meeting"]}

        # --- Context ---
        self.CTX_TRANS_PROB    = cfg["context"]["transition_prob"]

        # Spaces
        self.observation_space = spaces.MultiDiscrete(
            [3, 3, 4, self.MAX_QUEUE + 1, 4, 4]
        )
        self.action_space = spaces.Discrete(2)
        self.render_mode = render_mode

        # Internal state (initialised in reset)
        self._queue     = np.zeros(self.MAX_QUEUE, dtype=_NOTIF_DTYPE)
        self._queue_len = 0
        self._step      = 0
        self._steps_since_delivery = self.EPISODE_LEN
        self._context   = 0
        self._concentration = 1
        self._np_random: np.random.Generator = np.random.default_rng()

        # History
        self.reward_history        : list[float] = []
        self.action_history        : list[int]   = []
        self.concentration_history : list[int]   = []
        self._cumulative_reward    : float       = 0.0

        self._fig = None

    # ------------------------------------------------------------------
    @property
    def concentration(self) -> int:
        """Hidden state — exposed for visualization/educational use."""
        return self._concentration

    @property
    def expected_concentration(self) -> float:
        """
        E[concentration | context] — estimated from the stationary distribution
        of the concentration Markov chain under the current context.
        Uses only observable information (context), not the hidden state.
        """
        ceiling = self._CONC_CEILING[self._context]
        # Build 3-state transition matrix and solve for stationary distribution
        T = np.zeros((3, 3))
        for c in range(3):
            if c < ceiling:
                T[c, c + 1] += self.DRIFT_PROB
                T[c, c]     += 1 - self.DRIFT_PROB
            elif c > ceiling:
                T[c, c - 1] += self.DRIFT_PROB
                T[c, c]     += 1 - self.DRIFT_PROB
            else:
                T[c, c] = 1.0
        A = (T.T - np.eye(3))
        A[-1, :] = 1.0
        b = np.zeros(3); b[-1] = 1.0
        pi = np.clip(np.linalg.solve(A, b), 0, 1)
        return float(pi @ np.array([0.0, 1.0, 2.0]))

    # ------------------------------------------------------------------
    def _time_of_day(self) -> int:
        return min(self._step // self.TOD_PERIOD, 2)

    def _tsd_bucket(self) -> int:
        s = self._steps_since_delivery
        if s == 0:   return 0
        if s <= 5:   return 1
        if s <= 15:  return 2
        return 3

    def _get_obs(self) -> np.ndarray:
        q = self._queue[:self._queue_len]
        max_urg = int(q["urgency"].max()) + 1 if self._queue_len > 0 else 0
        max_imp = int(q["importance"].max()) + 1 if self._queue_len > 0 else 0
        return np.array([
            self._context,
            self._time_of_day(),
            self._tsd_bucket(),
            self._queue_len,
            max_urg,
            max_imp,
        ], dtype=np.int64)

    def _get_info(self) -> dict:
        return {
            "concentration": self._concentration,
            "context":       self._context,
            "time_of_day":   self._time_of_day(),
            "queue_len":     self._queue_len,
            "step":          self._step,
        }

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self._queue     = np.zeros(self.MAX_QUEUE, dtype=_NOTIF_DTYPE)
        self._queue_len = 0
        self._step      = 0
        self._steps_since_delivery = self.EPISODE_LEN
        self._context   = int(self._np_random.integers(0, 3))
        self._concentration = int(self._np_random.integers(0, 3))

        self.reward_history        = []
        self.action_history        = []
        self.concentration_history = []
        self._cumulative_reward    = 0.0

        return self._get_obs(), self._get_info()

    def _compute_reward(self, action: int, queue: np.ndarray, queue_len: int,
                        concentration: int, context: int, overflow: bool) -> float:
        """
        Pure reward calculation given explicit state components.

        Used by both step() and simulate() so the reward logic lives in one place.
        Also used by build_reward_model() in value_iteration.py (with approximated
        concentration) so there is no separate reimplementation there.

        Parameters
        ----------
        action        : 0=wait, 1=deliver
        queue         : full queue array (only first queue_len slots are active)
        queue_len     : number of notifications currently in queue
        concentration : current concentration level (0/1/2)
        context       : current context (0=idle, 1=focus, 2=meeting)
        overflow      : whether a notification arrived but could not fit this step
        """
        reward = 0.0
        if overflow:
            reward += self.OVERFLOW_PENALTY
        if action == 0:
            reward += self.PROD_BONUS * concentration
        else:
            if queue_len == 0:
                reward += self.EMPTY_DELIVER_PEN
            else:
                q = queue[:queue_len]
                notification_value = float(np.maximum(0.0, q["initial_val"] - q["decay_rate"] * q["age"]).sum())
                interruption_cost  = (
                    (concentration + 1)
                    * self.COST_PER_LEVEL
                    * self._CTX_MULT[context]
                )
                reward += notification_value - interruption_cost
        return reward

    def simulate(self, action: int) -> tuple[float, np.ndarray]:
        """
        Return the reward and next observation for *action* without modifying
        the real environment state. Useful for agents that want to evaluate
        both actions before committing.

        Parameters
        ----------
        action : 0=wait, 1=deliver

        Returns
        -------
        reward   : immediate reward that would result from taking this action
        next_obs : observation that would follow (as a numpy array)
        """
        # Snapshot mutable state (shallow-copy the numpy arrays)
        queue      = self._queue.copy()
        queue_len  = self._queue_len
        conc       = round(self.expected_concentration)  # observable estimate, not hidden state
        context    = self._context
        tsd        = self._steps_since_delivery

        # --- Mirror step() logic on the snapshot ---
        # 1. Age notifications (use age+1 for value calculation; don't mutate array)
        aged_queue = queue.copy()
        aged_queue["age"][:queue_len] += 1

        # 2. Overflow: assume expected arrival (deterministic approximation)
        overflow = False   # can't know stochastic arrival without sampling

        # 3. Reward
        reward = self._compute_reward(action, aged_queue, queue_len, conc, context, overflow)

        # 4. Build next observation (deterministic part only; stochastic transitions
        #    use their expected/modal outcome)
        if action == 1:
            next_qs  = 0
            next_mu  = 0
            next_mi  = 0
            next_tsd = 0
        else:
            next_qs  = min(queue_len + 1, self.MAX_QUEUE)  # expected: one arrival
            q_active = aged_queue[:queue_len]
            next_mu  = (int(q_active["urgency"].max()) + 1) if queue_len > 0 else 0
            next_mi  = (int(q_active["importance"].max()) + 1) if queue_len > 0 else 0
            next_tsd = min(self._tsd_bucket() + 1, 3)

        next_obs = np.array([
            context,                # context: use current (modal — no transition)
            self._time_of_day(),    # time of day: deterministic
            next_tsd,
            next_qs,
            next_mu,
            next_mi,
        ], dtype=np.int64)

        return reward, next_obs

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Possibly generate a new notification
        overflow = False
        if self._np_random.random() < self.ARRIVAL_PROB:
            urgency    = int(self._np_random.integers(0, 3))
            importance = int(self._np_random.integers(0, 3))
            init_val   = self.VAL_BASE + urgency * self.VAL_URG_SCALE + importance * self.VAL_IMP_SCALE
            decay      = self.DECAY_BASE + urgency * self.DECAY_URG_SCALE
            if self._queue_len < self.MAX_QUEUE:
                self._queue[self._queue_len] = (urgency, importance, init_val, decay, 0)
                self._queue_len += 1
            else:
                overflow = True

        # 2. Age existing notifications
        for i in range(self._queue_len):
            self._queue[i]["age"] += 1

        # 3. Compute reward via shared function
        reward = self._compute_reward(
            action, self._queue, self._queue_len,
            self._concentration, self._context, overflow,
        )

        # 4. Apply action side-effects
        if action == 0:
            self._steps_since_delivery += 1
        else:
            if self._queue_len > 0:
                self._concentration = max(0, self._concentration - 1)
                if self._np_random.random() < self.DROP2_PROB:
                    self._concentration = max(0, self._concentration - 1)
            self._queue     = np.zeros(self.MAX_QUEUE, dtype=_NOTIF_DTYPE)
            self._queue_len = 0
            self._steps_since_delivery = 0

        # 5. Concentration drift (wait only)
        if action == 0:
            ceiling = self._CONC_CEILING[self._context]
            if self._np_random.random() < self.DRIFT_PROB:
                if self._concentration < ceiling:
                    self._concentration += 1
                elif self._concentration > ceiling:
                    self._concentration -= 1

        # 6. Context transition
        if self._np_random.random() < self.CTX_TRANS_PROB:
            choices = [c for c in range(3) if c != self._context]
            self._context = int(self._np_random.choice(choices))

        # 7. Advance step
        self._step += 1

        # 8. Record history
        self._cumulative_reward += reward
        self.reward_history.append(self._cumulative_reward)
        self.action_history.append(int(action))
        self.concentration_history.append(self._concentration)

        terminated = self._step >= self.EPISODE_LEN
        truncated  = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self.render_mode == "rgb_array":
            # Always fresh Agg figure — never reuse across steps
            fig = _render_state(
                step                  = self._step,
                queue                 = self._queue,
                queue_len             = self._queue_len,
                concentration         = self._concentration,
                context               = self._context,
                time_of_day           = self._time_of_day(),
                reward_history        = self.reward_history,
                action_history        = self.action_history,
                concentration_history = self.concentration_history,
                episode_len           = self.EPISODE_LEN,
                max_queue             = self.MAX_QUEUE,
                fig                   = None,
                use_agg               = True,
            )
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas: FigureCanvasAgg = fig.canvas  # type: ignore[assignment]
            canvas.draw()
            w, h = canvas.get_width_height()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            return buf.reshape(h, w, 4)[:, :, :3]
        else:
            fig = _render_state(
                step                  = self._step,
                queue                 = self._queue,
                queue_len             = self._queue_len,
                concentration         = self._concentration,
                context               = self._context,
                time_of_day           = self._time_of_day(),
                reward_history        = self.reward_history,
                action_history        = self.action_history,
                concentration_history = self.concentration_history,
                episode_len           = self.EPISODE_LEN,
                max_queue             = self.MAX_QUEUE,
                fig                   = self._fig,
                use_agg               = False,
            )
            self._fig = fig
            plt.pause(0.01)
            return None

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None


# ---------------------------------------------------------------------------
# Core rendering helper
# ---------------------------------------------------------------------------

def _render_state(
    *,
    step: int,
    queue: np.ndarray,
    queue_len: int,
    concentration: int,
    context: int,
    time_of_day: int,
    reward_history: list,
    action_history: list,
    concentration_history: list,
    episode_len: int,
    max_queue: int,
    fig,
    figsize=(14, 5),
    use_agg: bool = False,
) -> plt.Figure:
    """Three-panel visualisation: queue | user state | timeline.

    use_agg: if True, creates figures with the Agg backend directly so that
    buffer_rgba() works correctly regardless of the active pyplot backend.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as MplFigure

    if use_agg:
        fig = MplFigure(figsize=figsize)
        FigureCanvasAgg(fig)
    elif fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(figsize=figsize)

    fig.clf()

    # Layout: 1 row × 3 cols
    # Col 0 (wide): notification queue
    # Col 1 (narrow): user state card
    # Col 2 (wide): timeline — reward + concentration + delivery markers
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        hspace=0.3, wspace=0.4,
        width_ratios=[1.4, 0.8, 1.8],
    )

    ax_queue    = fig.add_subplot(gs[0, 0])
    ax_state    = fig.add_subplot(gs[0, 1])
    ax_timeline = fig.add_subplot(gs[0, 2])

    # ------------------------------------------------------------------ #
    # Panel 1: Notification queue
    # ------------------------------------------------------------------ #
    ax_queue.set_title("Notification Queue", fontsize=10, fontweight="bold")
    ax_queue.set_xlim(-0.5, max_queue - 0.5)
    ax_queue.set_ylim(0, 6.5)
    ax_queue.set_xlabel("Slot")
    ax_queue.set_ylabel("Value")
    ax_queue.set_xticks(range(max_queue))

    q_active = queue[:queue_len]
    for i in range(max_queue):
        if i < queue_len:
            notif   = q_active[i]
            cur_val = max(0.0, float(notif["initial_val"]) - float(notif["decay_rate"]) * int(notif["age"]))
            urg     = int(notif["urgency"])
            color   = _URGENCY_COLORS[urg]
            alpha   = max(0.25, cur_val / max(float(notif["initial_val"]), 1e-6))
            rect    = mpatches.FancyBboxPatch(
                (i - 0.4, 0), 0.8, max(cur_val, 0.05),
                boxstyle="round,pad=0.05",
                facecolor=color, alpha=alpha, edgecolor="black", linewidth=0.8,
            )
            ax_queue.add_patch(rect)
            ax_queue.text(i, cur_val + 0.18, _URGENCY_LABELS[urg], ha="center", fontsize=6)
        else:
            rect = mpatches.FancyBboxPatch(
                (i - 0.4, 0), 0.8, 0.15,
                boxstyle="round,pad=0.05",
                facecolor="#e0e0e0", alpha=0.5, edgecolor="#bdbdbd", linewidth=0.5,
            )
            ax_queue.add_patch(rect)

    legend_patches = [
        mpatches.Patch(color=_URGENCY_COLORS[0], label="low"),
        mpatches.Patch(color=_URGENCY_COLORS[1], label="med"),
        mpatches.Patch(color=_URGENCY_COLORS[2], label="high"),
    ]
    ax_queue.legend(handles=legend_patches, fontsize=6, loc="upper right", title="urgency")
    ax_queue.text(-0.45, 6.1, f"Queue: {queue_len}/{max_queue}", fontsize=8, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 2: User state card
    # ------------------------------------------------------------------ #
    ax_state.set_xlim(0, 1)
    ax_state.set_ylim(0, 1)
    ax_state.axis("off")

    # Background card
    card = mpatches.FancyBboxPatch(
        (0.03, 0.02), 0.94, 0.96,
        boxstyle="round,pad=0.02",
        facecolor="#fafafa", edgecolor="#cccccc", linewidth=1,
    )
    ax_state.add_patch(card)

    # Context
    ctx_colors = {"idle": "#90caf9", "focus": "#a5d6a7", "meeting": "#ffcc80"}
    ctx_label = CONTEXT_LABELS[context]
    ax_state.text(0.5, 0.91, "Context", ha="center", fontsize=7, color="#888888")
    ctx_bg = mpatches.FancyBboxPatch(
        (0.12, 0.76), 0.76, 0.14,
        boxstyle="round,pad=0.02",
        facecolor=ctx_colors.get(ctx_label, "#e0e0e0"), edgecolor="none",
    )
    ax_state.add_patch(ctx_bg)
    ax_state.text(0.5, 0.83, ctx_label.upper(), ha="center", va="center",
                  fontsize=10, fontweight="bold")

    # Time of day
    ax_state.text(0.5, 0.70, "Time of day", ha="center", fontsize=7, color="#888888")
    ax_state.text(0.5, 0.61, TOD_LABELS[time_of_day].capitalize(),
                  ha="center", fontsize=9, fontweight="bold")

    # Concentration bar (hidden)
    ax_state.text(0.5, 0.51, "Concentration (hidden)", ha="center", fontsize=6.5, color="#888888")
    conc_color = _CONC_COLORS[concentration]
    bar_w = 0.22
    for lvl in range(3):
        filled = lvl <= concentration
        rect = mpatches.FancyBboxPatch(
            (0.09 + lvl * (bar_w + 0.04), 0.36), bar_w, 0.12,
            boxstyle="round,pad=0.02",
            facecolor=conc_color if filled else "#e0e0e0",
            edgecolor="white", linewidth=0.8,
        )
        ax_state.add_patch(rect)
    ax_state.text(0.5, 0.29, CONCENTRATION_LABELS[concentration],
                  ha="center", fontsize=8, color=conc_color, fontweight="bold")

    # Step counter
    ax_state.text(0.5, 0.16, "Step", ha="center", fontsize=7, color="#888888")
    ax_state.text(0.5, 0.07, f"{step} / {episode_len}",
                  ha="center", fontsize=9, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 3: Timeline — cumulative reward + concentration + deliveries
    # ------------------------------------------------------------------ #
    ax_timeline.set_title("Timeline", fontsize=10, fontweight="bold")
    ax_timeline.set_xlabel("Step")
    ax_timeline.set_xlim(0, episode_len)

    # Twin axis: concentration on right
    ax_conc = ax_timeline.twinx()

    if reward_history:
        xs = list(range(1, len(reward_history) + 1))
        ax_timeline.plot(xs, reward_history, color="#1976d2", linewidth=1.8, label="cumulative reward")
        ax_timeline.fill_between(xs, reward_history, 0, alpha=0.12, color="#1976d2")

    if concentration_history:
        xs_c = list(range(1, len(concentration_history) + 1))
        ax_conc.plot(xs_c, concentration_history, color="#9e9e9e",
                     linewidth=1.2, linestyle="--", alpha=0.7, label="concentration")
        ax_conc.set_ylim(-0.5, 3.5)
        ax_conc.set_yticks([0, 1, 2])
        ax_conc.set_yticklabels(["low", "med", "high"], fontsize=7, color="#9e9e9e")

    # Mark deliveries as vertical lines
    if action_history:
        for x, a in enumerate(action_history, start=1):
            if a == 1:
                ax_timeline.axvline(x, color="#e53935", linewidth=1.0,
                                    linestyle=":", alpha=0.8)
        # Add a single legend entry for deliveries
        ax_timeline.axvline(-1, color="#e53935", linewidth=1.0,
                            linestyle=":", alpha=0.8, label="delivery")

    ax_timeline.set_ylabel("Cumulative reward", color="#1976d2", fontsize=8)
    ax_timeline.tick_params(axis="y", labelcolor="#1976d2")
    ax_timeline.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Compact legend
    lines1, labels1 = ax_timeline.get_legend_handles_labels()
    lines2, labels2 = ax_conc.get_legend_handles_labels()
    ax_timeline.legend(lines1 + lines2, labels1 + labels2,
                       fontsize=7, loc="upper left")

    # ------------------------------------------------------------------ #
    # Suptitle
    # ------------------------------------------------------------------ #
    cum = reward_history[-1] if reward_history else 0.0
    fig.suptitle(
        f"NotificationScheduler-v0  |  Step {step}/{episode_len}  |  Cumulative reward: {cum:.2f}",
        fontsize=11, fontweight="bold",
    )

    return fig


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register():
    """Register NotificationScheduler-v0 with Gymnasium."""
    if "NotificationScheduler-v0" not in gym.envs.registry:
        gym.register(
            id="NotificationScheduler-v0",
            entry_point="notification_env.notification_scheduler:NotificationSchedulerEnv",
        )
