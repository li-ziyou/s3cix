"""
Utility functions for running episodes with NotificationScheduler-v0.
"""

from __future__ import annotations

import numpy as np
from notification_env.notification_scheduler import NotificationSchedulerEnv


def run_episode_with_frames(
    policy_fn, seed: int = 0, env=None
) -> tuple[float, list, NotificationSchedulerEnv]:
    """
    Run one episode with rgb_array rendering, collecting one frame per step.

    Parameters
    ----------
    policy_fn : callable(obs) -> action
    seed      : environment reset seed
    env       : optional pre-built env (e.g. when the agent needs a reference
                to the same env it will run in); created internally if None

    Returns
    -------
    total_reward, frames, env
    """
    if env is None:
        env = NotificationSchedulerEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    total, frames, done = 0.0, [], False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        done = terminated or truncated
    return total, frames, env
