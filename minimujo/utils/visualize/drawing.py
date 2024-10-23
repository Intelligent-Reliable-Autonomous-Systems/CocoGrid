from typing import Tuple
import numpy as np
import gymnasium as gym

from minimujo.utils.gym_wrapper import unwrap_env
from minimujo.dmc_gym import MinimujoGym
from minimujo.box2d.gym import Box2DEnv

def get_camera_bounds(env: gym.Env) -> Tuple[float,float,float,float]:
    minimujo_env = unwrap_env(env, MinimujoGym)
    if minimujo_env is None:
        box2d_env = unwrap_env(env, Box2DEnv)
        assert box2d_env is not None, f"Could not get coordinate bounds. {env} was not a MinimujoGym or Box2DEnv"
        return 0, 0, box2d_env.arena_width, box2d_env.arena_height
    return 0, 0, minimujo_env.arena.maze_width, minimujo_env.arena.maze_height

def draw_rectangle(
    image: np.ndarray, 
    bounds: Tuple[float, float, float, float], 
    point1: Tuple[float, float], 
    point2: Tuple[float, float], 
    color: np.ndarray,
    thickness: int
):
    height, width = image.shape
    min_x, min_y, bound_width, bound_height = bounds
    screen1_x, screen1_y = (point1[0] - min_x) / bound_width, (point1[1] - min_y) / bound_height
    screen2_x, screen2_y = (point2[0] - min_x) / bound_width, (point2[1] - min_y) / bound_height
    topleft = max(0, min(screen1_x, screen2_x)), max(0, min(screen1_y, screen2_y))
    botright = min(width-1, max(screen1_x, screen2_x)), min(height-1, max(screen1_y, screen2_y))

    # draw horizontal top
    image[topleft[1]:topleft[1]+thickness, topleft[0]:botright[0]] = color
    # draw horizontal bottom
    image[botright[1]-thickness:botright[1], topleft[0]:botright[0]] = color
    # draw vertical left
    image[topleft[1]:botright[1], topleft[0]:botright[0]+thickness] = color
    # draw vertical right
    image[topleft[1]:botright[1], topleft[0]-thickness:botright[0]] = color
