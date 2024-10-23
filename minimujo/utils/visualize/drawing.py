from typing import Tuple
import numpy as np
import gymnasium as gym

from minimujo.dmc_gym import MinimujoGym
from minimujo.box2d.gym import Box2DEnv

def get_camera_bounds(env: gym.Env, norm_scale: bool = False) -> Tuple[float,float,float,float]:
    unwrapped_env = env.unwrapped
    if isinstance(unwrapped_env, MinimujoGym):
        scale = unwrapped_env.arena.xy_scale if norm_scale else 1
        return 0, -unwrapped_env.arena.maze_height / scale, unwrapped_env.arena.maze_width / scale, unwrapped_env.arena.maze_height / scale
    if isinstance(unwrapped_env, Box2DEnv):
        scale = unwrapped_env.xy_scale if norm_scale else 1
        return 0, -unwrapped_env.arena_height / scale, unwrapped_env.arena_width / scale, unwrapped_env.arena_height / scale
    else:
        raise Exception(f"To get camera bounds, env must be a MinimujoGym or Box2DEnv. Actual type: {type(unwrapped_env)}")

def draw_rectangle(
    image: np.ndarray, 
    bounds: Tuple[float, float, float, float], 
    point1: Tuple[float, float], 
    point2: Tuple[float, float], 
    color: np.ndarray,
    thickness: int
):
    cam_height, cam_width = image.shape[:2]
    min_x, min_y, bound_width, bound_height = bounds
    screen1_x, screen1_y = int((point1[0] - min_x) / bound_width * cam_width), int((1-(point1[1] - min_y) / bound_height) * cam_height)
    screen2_x, screen2_y = int((point2[0] - min_x) / bound_width * cam_height), int((1 - (point2[1] - min_y) / bound_height) * cam_height)
    topleft = max(0, min(screen1_x, screen2_x)), max(0, min(screen1_y, screen2_y))
    botright = min(cam_width-1, max(screen1_x, screen2_x)), min(cam_height-1, max(screen1_y, screen2_y))

    # breakpoint()
    # draw horizontal top
    image[topleft[1]:topleft[1]+thickness, topleft[0]:botright[0]] = color
    # draw horizontal bottom
    image[botright[1]-thickness:botright[1], topleft[0]:botright[0]] = color
    # draw vertical left
    image[topleft[1]:botright[1], topleft[0]:topleft[0]+thickness] = color
    # draw vertical right
    image[topleft[1]:botright[1], botright[0]-thickness:botright[0]] = color

    # image[0:100, 50:150] = color
