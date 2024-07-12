from typing import Any, Dict, Optional, Tuple, Callable
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np

from minimujo.utils.visualize.weighted_kde import WeightedKDEHeatmap

class LoggingWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, buffer_size: int = 1000):
        super().__init__(env)

        self.buffer_size = buffer_size
        self.buffer_idx = 0
        # self.obs_buffer = np.zeros((buffer_size, env.observation_space.shape[0]))
        # self.reward_buffer = np.zeros(buffer_size)
        # self.term_buffer = np.zeros(buffer_size)
        # self.trunc_buffer = np.zeros(buffer_size)
        self.heatmap_subscriptions = []

    def subscribe_heatmap(self, label: str, transform: Callable, channels: int = 1, decay: float = 1) -> None:
        heatmap = WeightedKDEHeatmap(channels=channels, decay=decay)
        self.heatmap_subscriptions.append({
            'label': label,
            'transform': transform,
            'heatmap': heatmap,
            'buffer': np.zeros((self.buffer_size, 2 + channels))
        })

    def process_subscriptions(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any]) -> None:
        for heatmap_sub in self.heatmap_subscriptions:
            heatmap_sub['buffer'][self.buffer_idx] = heatmap_sub['transform'](
                obs=obs, rew=rew, term=term, trunc=trunc, info=info, env=self.env, timestep=self.buffer_idx
            )

    def finish_subscriptions(self):
        for heatmap_sub in self.heatmap_subscriptions:
            heatmap: WeightedKDEHeatmap = heatmap_sub['heatmap']
            buffer: np.ndarray = heatmap_sub['buffer']
            print(buffer)
            heatmap.add_batch(buffer[:,:2], buffer[:,2])
            plt.imshow(heatmap.heatmap, origin='lower', cmap='viridis')
            plt.colorbar(label='Value')
            plt.title(heatmap_sub['label'])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        self.buffer_idx = 0
        return super().reset(seed=seed, options=options)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        self.process_subscriptions(obs, rew, term, trunc, info)
        self.buffer_idx += 1
        if term or trunc:
            self.finish_subscriptions()
        return obs, rew, term, trunc, info
