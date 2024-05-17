from typing import Any, Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

class ObsVisualize(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.obs_length = env.observation_space.shape[0]
        self.obs_buffer = np.zeros((1000, self.obs_length+2))
        self.buffer_idx = 0
        self.cum_reward = 0

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.obs_buffer[self.buffer_idx,:] = observation
        self.buffer_idx += 1
        return observation
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        # print(f'OBSERVED {self.buffer_idx} OBSERVATIONS')
        self.buffer_idx = 0
        self.cum_reward = 0
        return super().reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        self.cum_reward += rew
        self.obs_buffer[self.buffer_idx,:self.obs_length] = obs
        self.obs_buffer[self.buffer_idx,self.obs_length] = rew
        self.obs_buffer[self.buffer_idx,self.obs_length+1] = self.cum_reward
        self.buffer_idx += 1
        if term or trunc:
            print(f'OBSERVED {self.buffer_idx} OBSERVATIONS')
            self.plot_observations()
        return obs, rew, term, trunc, info
    
    def plot_observations(self):
        data = self.obs_buffer[:self.buffer_idx, :]
        column_mappings = self.env.unwrapped.range_mapping.copy()
        column_mappings['reward'] = (self.obs_length, self.obs_length+1)
        column_mappings['cum_reward'] = (self.obs_length+1, self.obs_length+2)

        n_subplots = len(column_mappings)
        rows = int(np.ceil(np.sqrt(n_subplots)))
        cols = int(np.ceil(n_subplots / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        axes = axes.flatten()

        # Loop over each range and corresponding axis
        for ax, (name, col_range) in zip(axes, column_mappings.items()):
            # Select columns for each range
            subset = data[:, col_range[0]:col_range[1]]

            # Plot each column in the range
            for col in range(subset.shape[1]):
                ax.plot(subset[:, col], label=str(col))

            # Setting some plot attributes
            ax.set_title(f'{name}: {col_range[0]} to {col_range[1] - 1}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.legend()

        for i in range(n_subplots, rows * cols):
            axes[i].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()