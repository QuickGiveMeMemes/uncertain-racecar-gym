from __future__ import annotations

from gymnasium.envs.registration import register, registry


def register_environments() -> None:
    if "UncertainRacecar-v0" not in registry:
        register(
            id="UncertainRacecar-v0",
            entry_point="uncertain_racecar_gym.env:UncertainRacecarEnv",
        )
