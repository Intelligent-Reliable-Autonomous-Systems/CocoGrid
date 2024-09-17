import gymnasium as gym
import numpy as np

from minimujo.state.minimujo_state import MinimujoState

POSE_SIZE = 13
OBJECT_SIZE = 16

def get_full_vector_observation_space(minimujo_state: MinimujoState, num_objects: int = -1):
    if num_objects < 0:
        num_objects = minimujo_state.objects.shape[0]
    walker_vec_size = sum(feature.shape[0] for feature in minimujo_state.walker)
    obj_size = POSE_SIZE + num_objects * OBJECT_SIZE + walker_vec_size
    return gym.spaces.Box(-np.inf, np.inf, shape=(obj_size,), dtype=np.float32)

def get_full_vector_observation(minimujo_state: MinimujoState, num_objects: int = -1):
    if num_objects < 0:
        num_objects = minimujo_state.objects.shape[0]
    target_size = num_objects * OBJECT_SIZE
    flat = minimujo_state.objects.flatten()[:target_size]
    object_arr = np.pad(flat, target_size - len(flat))

    return np.concatenate([minimujo_state.pose, object_arr, *minimujo_state.walker.values()])

def full_vector_back_to_state(grid, xy_scale, obs):
    return MinimujoState(grid, xy_scale, objects=np.reshape(obs[13:], (-1,16)), pose=obs[:13], walker={})