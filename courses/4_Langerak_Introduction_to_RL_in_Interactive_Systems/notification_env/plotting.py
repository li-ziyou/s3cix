"""
All plotting and animation functions for NotificationScheduler-v0.
"""

from __future__ import annotations

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML


# ---------------------------------------------------------------------------
# Episode animation
# ---------------------------------------------------------------------------

def animate(frames: list, fps: int = 5) -> HTML:
    """
    Build an in-notebook animation from a list of rgb_array frames.

    Parameters
    ----------
    frames : list of (H, W, 3) uint8 arrays, one per step
    fps    : playback speed

    Returns
    -------
    IPython HTML object — display it directly in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 // fps, blit=True
    )
    plt.close(fig)
    return HTML(anim.to_jshtml(fps=fps))


# ---------------------------------------------------------------------------
# Value function visualisation
# ---------------------------------------------------------------------------

def plot_value_function(V: np.ndarray, state_to_index, title_suffix: str = "") -> None:
    """
    Plot a slice of the value function: V vs queue_size for each urgency level.

    Slice: context=focus (1), time_of_day=morning (0),
           time_since_delivery=long (3), max_importance=0.

    Parameters
    ----------
    V              : value array of shape (N_STATES,)
    state_to_index : function mapping state tuple to flat index
    title_suffix   : appended to the plot title
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ctx, tod, tsd, mi = 1, 0, 3, 0
    urgency_labels = ["no notifs", "low urgency", "med urgency", "high urgency"]
    colors = ["#bdbdbd", "#4caf50", "#ff9800", "#f44336"]

    for mu in range(4):
        vals = [V[state_to_index((ctx, tod, tsd, qs, mu, mi))] for qs in range(9)]
        ax.plot(range(9), vals, marker="o", markersize=4,
                label=urgency_labels[mu], color=colors[mu])

    ax.set_xlabel("Queue size")
    ax.set_ylabel("V(s)")
    title = "Value function: focus / morning / long since delivery"
    ax.set_title(title + (f" — {title_suffix}" if title_suffix else ""))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_convergence(deltas: list, title: str = "Convergence", ylabel: str = "Error") -> None:
    """
    Plot a convergence signal on a log scale.

    Works for both value iteration (max|ΔV| per sweep) and Q-learning
    (mean |TD error| per episode, optionally smoothed).

    Parameters
    ----------
    deltas : list of floats
    title  : plot title
    ylabel : y-axis label
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(deltas, color="#7b1fa2", linewidth=1.5)
    ax.set_xlabel("Iteration / Episode")
    ax.set_ylabel(f"{ylabel} (log scale)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
