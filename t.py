import gymnasium as gym
import numpy as np

initial_obs = []
for seed in [147, 577, 5362, 8699]:
    env = gym.make(
        "CarRacing-v2", render_mode="rgb_array", continuous=False, domain_randomize=True
    )
    i, _ = env.reset(seed=seed)
    initial_obs.append(i)
    # Visualize or log the initial state/map to confirm changes
    # env.render()  # Check visually if the map is different
    # env.close()


# Check if all initial observations are the same
all_same = print(list(np.array_equal(initial_obs[0], obs) for obs in initial_obs))

print("All initial observations are the same:", all_same)
