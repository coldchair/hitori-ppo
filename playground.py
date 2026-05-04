import hitori_env
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

env = gym.make("hitori_env/Hitori-v2", render_mode="human")

# try:
#     check_env(env)
#     print("Environment passes all checks!")
# except Exception as e:
#     print(f"Environment has issues: {e}")


obs, info = env.reset(
    seed=42, options={"log_solution": True}
)  # Use seed for first reproducible puzzle

print(obs["game_grid"])


def coords_to_action(coords, grid_size=5):
    row, col = coords
    return row * grid_size + col


while True:
    env.render()
    action_mask = env.unwrapped.action_masks()
    ix = map(int, input("Enter action (row col): ").split())
    action = coords_to_action(ix)
    if action_mask[action] == 0:
        print("Invalid action, try again.")
        continue
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action {action=} REWARD={reward} {terminated=} {truncated=} {info=}")
    if terminated or truncated:
        new_game = input("Game over. Start a new game? (y/n): ")
        if new_game.lower() == "y":
            obs, info = env.reset()
            print(obs["game_grid"])
        else:
            break
        env.reset()
