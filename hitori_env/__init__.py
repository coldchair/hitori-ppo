from gymnasium.envs.registration import register

register(
    id="hitori_env/Hitori-v2",
    entry_point="hitori_env.envs:HitoriEnv",
)
