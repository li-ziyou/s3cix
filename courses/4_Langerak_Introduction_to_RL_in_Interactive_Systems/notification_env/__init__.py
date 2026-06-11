"""
notification_env — Gymnasium environment for notification scheduling.

Importing this package registers ``NotificationScheduler-v0`` with
Gymnasium so you can do::

    import notification_env          # registers the env
    import gymnasium as gym
    env = gym.make("NotificationScheduler-v0")

Or use the class directly::

    from notification_env import NotificationSchedulerEnv
"""

from notification_env.notification_scheduler import (
    NotificationSchedulerEnv,
    register,
)

# Auto-register on import
register()

__all__ = [
    "NotificationSchedulerEnv",
    "register",
]
