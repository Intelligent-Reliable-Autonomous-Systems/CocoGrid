from typing import Any, Callable, Dict, Tuple
import gymnasium as gym
from minigrid.core.world_object import Wall, Goal, Lava, Ball, Box, Key
import numpy as np
import pygame

from minimujo.box2d.observation import get_full_vector_observation, get_full_vector_observation_space
from minimujo.color import get_color_idx, get_color_rgba_255
from minimujo.state.minimujo_state import MinimujoState, MinimujoStateObserver
from minimujo.utils.minigrid import minigrid_tile_generator

try:
    import Box2D
    from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e

BOX_DAMPING = 1
BALL_DAMPING = 1
TIME_STEP = 0.06
MOVE_FORCE = 12
GRAB_FORCE = 12
MAX_SPEED = 5
MAX_GRAB_DISTANCE = 16
MAX_GRAB_INIT_DISTANCE = 4
K_DERIVATIVE = 0.6

class Box2DEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, minigrid_env: gym.Env, walker_type: str, get_task_function: Callable, xy_scale: float = 1, 
                 spawn_position=None, spawn_sampler=None,
                 seed: int = None, timesteps: int = 500, render_mode='rgb_array', render_width=64, **kwargs) -> None:
        super().__init__()

        self.minigrid_env = minigrid_env
        self.walker_type = walker_type
        self.xy_scale = xy_scale
        self.minigrid_seed = seed
        self.spawn_position = spawn_position
        self.spawn_sampler = spawn_sampler

        self.max_timesteps = timesteps
        self.get_task_function = get_task_function

        self.screen = None
        self.clock = None
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_width
        self.color_mapping = dict()

        def my_draw_polygon(polygon, body, color, pixel_per_meter, screen_height, surface):
            vertices = [(body.transform * v) * pixel_per_meter for v in polygon.vertices]
            vertices = [(v[0], v[1] + screen_height) for v in vertices]
            gfxdraw.aapolygon(surface, vertices, color)
            gfxdraw.filled_polygon(surface, vertices, color)
        polygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, color, pixel_per_meter, screen_height, surface):
            position = body.transform * circle.pos * pixel_per_meter
            position = [int(position[0]), int(screen_height + position[1])]
            radius = int(circle.radius * pixel_per_meter)
            gfxdraw.aacircle(surface, *position, radius, color)
            gfxdraw.filled_circle(surface, *position, radius, color)
        circleShape.draw = my_draw_circle

        self._skip_initializing = True
        self.world = None
        self._generate_arena()
        observation = get_full_vector_observation(self._get_state())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=observation.shape, dtype=observation.dtype)
        self.action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)

    def _generate_arena(self):
        if self.world is not None:
            # Explicit memory clean up to prevent memory leaks
            for body in self.world.bodies:
                self.world.DestroyBody(body)
            for joint in self.world.joints:
                self.world.DestroyJoint(joint)

            del self.agent
            del self.world
            del self._grid
            del self.color_mapping
        self.world = world(gravity=(0, 0), doSleep=True)

        self.minigrid_env.reset(seed=self.minigrid_seed)
        self._grid = MinimujoStateObserver.get_grid_state_from_minigrid(self.minigrid_env)
        
        self.arena_height = self.minigrid_env.grid.height * self.xy_scale
        self.arena_width = self.minigrid_env.grid.width * self.xy_scale
        self.color_mapping = dict()
        self.objects = []
        self.grabbed_object_idx = -1
        for x, y, tile in minigrid_tile_generator(self.minigrid_env):
            y = -y
            center_pos = ((x + .5) * self.xy_scale, (y - .5) * self.xy_scale)
            if isinstance(tile, Wall):
                wall = self.world.CreateStaticBody(
                    position=center_pos,
                    shapes=polygonShape(box=(self.xy_scale/2, self.xy_scale/2)),
                )
                self.color_mapping[wall] = (184, 184, 184, 255)
            elif isinstance(tile, Goal):
                goal = self.world.CreateStaticBody(
                    position=center_pos,
                    shapes=polygonShape(box=(self.xy_scale/2, self.xy_scale/2)),
                )
                for fixture in goal.fixtures:
                    fixture.sensor = True
                self.color_mapping[goal] = (0, 255, 0, 255)
            elif isinstance(tile, Lava):
                lava = self.world.CreateStaticBody(
                    position=center_pos,
                    shapes=polygonShape(box=(self.xy_scale/2, self.xy_scale/2)),
                )
                for fixture in lava.fixtures:
                    fixture.sensor = True
                self.color_mapping[lava] = (255, 165, 0, 255)
            elif isinstance(tile, Ball):
                ball = self.world.CreateDynamicBody(position=center_pos)
                circle = ball.CreateCircleFixture(radius=0.25, density=1, friction=0.3)
                ball.linearDamping = BALL_DAMPING
                self.color_mapping[ball] = get_color_rgba_255(tile.color)
                self.objects.append((ball, 0, get_color_idx(tile.color)))
            elif isinstance(tile, Box):
                box = self.world.CreateDynamicBody(position=center_pos)
                square = box.CreatePolygonFixture(box=(.25, .25), density=1, friction=0.3)
                box.linearDamping = BOX_DAMPING
                box.angularDamping = 1
                self.color_mapping[box] = get_color_rgba_255(tile.color)
                self.objects.append((box, 1, get_color_idx(tile.color)))

        if self.spawn_position is not None:
            pos = self.spawn_position
        elif self.spawn_sampler is not None:
            pos = self.spawn_sampler()
        else:
            agent_x, agent_y = self.minigrid_env.agent_pos
            pos = ((agent_x + .5) * self.xy_scale, -(agent_y + .5) * self.xy_scale)
        pos = tuple([float(x) for x in pos[:2]]) # box2d doesn't like numpy scalars
        self.agent = self.world.CreateDynamicBody(position=pos)
        # circle = self.agent.CreateCircleFixture(radius=0.4, density=1, friction=0.3)
        self.agent.CreatePolygonFixture(box=(.3, .3), density=1, friction=0.3)
        self.agent.linearDamping = 3
        self.agent.fixedRotation = True
        self.color_mapping[self.agent] = (115, 115, 115, 255)

    def reset(self, seed=None, options=None):
        if not self._skip_initializing:
            self._generate_arena()
        self._skip_initializing = False

        self._cum_reward = 0
        self.timesteps = 0
        self._task = Task(*self.get_task_function(self.minigrid_env))
        self._prev_state = self.state = self._get_state()
        return get_full_vector_observation(self.state), {}
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        agent_vel = self.agent.linearVelocity.tuple
        if agent_vel[0]**2 + agent_vel[1]**2 < MAX_SPEED:
            self.agent.ApplyForceToCenter((MOVE_FORCE * float(action[2]), -MOVE_FORCE * float(action[1])), True)

        if action[0] >= 0.5:
            self._do_grab()
        else:
            self.grabbed_object_idx = -1
        self.world.Step(TIME_STEP, 10, 10)

        self._prev_state = self.state
        self.state = self._get_state()
        rew, finished = self._task(self._prev_state, self.state)

        self.timesteps += 1
        return get_full_vector_observation(self.state), rew, finished or self.timesteps >= self.max_timesteps, False, {}
    
    @property
    def task(self):
        return self._task.description
    
    def _do_grab(self):
        def square_dist(x1, y1, x2, y2):
            return (x1 - x2)**2 + (y1 - y2)**2
        agent_pos = np.array(self.agent.position.tuple)
        if self.grabbed_object_idx >= 0:
            grabbed_obj = self.objects[self.grabbed_object_idx][0]
            obj_pos = grabbed_obj.position.tuple
            if square_dist(*agent_pos, *obj_pos) > MAX_GRAB_DISTANCE:
                self.grabbed_object_idx = -1
        if self.grabbed_object_idx == -1:
            for idx, (obj, object_id, _) in enumerate(self.objects):
                if object_id == 2: # Door
                    continue
                obj_pos = obj.position.tuple
                if square_dist(*agent_pos, *obj_pos) < MAX_GRAB_INIT_DISTANCE:
                    self.grabbed_object_idx = idx
        if self.grabbed_object_idx >= 0:
            grabbed_obj = self.objects[self.grabbed_object_idx][0]
            walker_facing_vec = self.agent.GetWorldVector((0,1)).tuple
            target_vecs = np.array((
                walker_facing_vec,
                (-walker_facing_vec[1], walker_facing_vec[0]),
                (-walker_facing_vec[0], -walker_facing_vec[1]),
                (walker_facing_vec[1], -walker_facing_vec[0]),
            ))
            obj_pos = np.array(grabbed_obj.position.tuple)
            actual_diff = obj_pos - agent_pos
            max_idx = np.argmax(np.dot(target_vecs, actual_diff))
            target_vec = target_vecs[max_idx]

            target_pos = np.array(agent_pos) + 1 * target_vec
            target_diff = (target_pos - obj_pos)
            magnitude = np.linalg.norm(target_diff)
            # if magnitude > 0.3:
            #     target_diff /= magnitude
            obj_vel = np.array(grabbed_obj.linearVelocity.tuple)
            force = GRAB_FORCE * target_diff - 5 * obj_vel
            # print(obj_vel, force)
            
            grabbed_obj.ApplyForceToCenter(tuple(force.astype(float)), True)
            
        # directions = 
        # self.agent.GetWorldVector((0,1))
    
    def _get_state(self):
        walker_pose = np.zeros(13, dtype=np.float32)
        walker_pose[:2] = self.agent.position.tuple
        walker_pose[3] = self.agent.angle
        walker_pose[7:9] = self.agent.linearVelocity.tuple
        walker_pose[9] = self.agent.angularVelocity

        object_array = np.zeros((len(self.objects), 16))
        for index, (obj, object_id, color_id) in enumerate(self.objects):
            object_array[index, 0] = object_id
            object_array[index, 1:3] = obj.position.tuple
            object_array[index, 4] = obj.angle
            object_array[index, 8:10] = obj.linearVelocity.tuple
            object_array[index, 11] = obj.angularVelocity
            object_array[index, 14:16] = (color_id, index == self.grabbed_object_idx)

        return MinimujoState(self._grid, self.xy_scale, object_array, walker_pose, {})

    def render(self, width=None, height=None):
        width = width or self.render_width
        height = height or self.render_height

        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode, width, height)

    def _render(self, mode: str, width: int, height: int):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((width, height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "world" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((width, height))

        pixel_per_meter = min(width / self.arena_width, height / self.arena_height) 
        for body in self.world.bodies:
            color = self.color_mapping.get(body, (255, 255, 255, 255))
            for fixture in body.fixtures:
                fixture.shape.draw(body, color, pixel_per_meter, height, self.surf)

        self.surf = pygame.transform.flip(self.surf, False, True)

        # font = pygame.font.Font(pygame.font.get_default_font(), 42)
        # text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        # text_rect = text.get_rect()
        # text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        # self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (width, height))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (width, height))
        else:
            return self.isopen
        
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

class Task:
    def __init__(self, task_function, description):
        self._task_function = task_function
        self._description = description
        self._cum_reward = 0
        self._terminated = False

    @property
    def description(self):
        return self._description
    
    @property
    def function(self):
        return self._task_function
    
    @property
    def terminated(self):
        return self._terminated
    
    @property
    def reward_total(self):
        return self._cum_reward
    
    def __call__(self, prev_state, next_state):
        rew, term = self._task_function(prev_state, next_state)
        self._cum_reward += rew
        self._terminated = self._terminated or term
        return rew, term

