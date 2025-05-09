from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
import numpy as np
import pygame
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Lava, Wall

from cocogrid.common.cocogrid_state import CocogridState
from cocogrid.common.color import get_color_rgba_255
from cocogrid.common.entity import ObjectEnum, get_color_id
from cocogrid.utils.minigrid import get_door_direction, minigrid_tile_generator

if TYPE_CHECKING:
    from cocogrid.box2d.box2d_agent import Box2DAgent
    from cocogrid.common.observation import ObservationSpecification

try:
    from Box2D.b2 import (
        circleShape,
        polygonShape,
        revoluteJointDef,
        world,
    )
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
    raise gym.error.DependencyNotInstalled('pygame is not installed, run `pip install "gymnasium[box2d]"`') from e

BOX_DAMPING = 1
BALL_DAMPING = 1
TIME_STEP = 0.06
MOVE_FORCE = 12
GRAB_FORCE = 12
GRAB_DAMPENING = 5
GRAB_VEL_ALIGN = 3
MAX_SPEED = 5
MAX_GRAB_DISTANCE = 16
MAX_GRAB_INIT_DISTANCE = 4
DOOR_UNLOCK_DISTANCE = 1


class Box2DEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        minigrid_env: gym.Env,
        agent: Box2DAgent,
        get_task_function: Callable,
        observation_spec: ObservationSpecification,
        xy_scale: float = 1,
        spawn_position=None,
        spawn_sampler=None,
        reset_options=None,
        seed: int = None,
        timesteps: int = 500,
        render_mode="rgb_array",
        render_width=64,
        **kwargs,
    ) -> None:
        super().__init__()

        self.minigrid_env = minigrid_env
        self.agent = agent
        self.xy_scale = xy_scale
        self.minigrid_seed = seed
        self._minigrid_options = reset_options or {}
        self.spawn_position = spawn_position
        self.spawn_sampler = spawn_sampler
        self._start_state = None  # if set, overrides the reset state

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
        self._generate_arena(minigrid_seed=self.minigrid_seed, minigrid_options=self._minigrid_options)

        self._observation_spec = observation_spec
        self.observation_space, self._observation_function = self._observation_spec.build_observation_space(
            self._get_state()
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def _generate_arena(self, minigrid_seed=None, minigrid_options=None):
        if self.world is not None:
            # Explicit memory clean up to prevent memory leaks
            for body in self.world.bodies:
                self.world.DestroyBody(body)
            for joint in self.world.joints:
                self.world.DestroyJoint(joint)

            self.agent.delete_body()
            del self.world
            del self._grid
            del self.color_mapping
        self.world = world(gravity=(0, 0), doSleep=True)

        self.minigrid_env.reset(seed=minigrid_seed, options=minigrid_options)
        self._grid = CocogridState.get_grid_state_from_minigrid(self.minigrid_env)

        self.arena_height = self.minigrid_env.grid.height * self.xy_scale
        self.arena_width = self.minigrid_env.grid.width * self.xy_scale
        self.color_mapping = dict()
        self.objects = []
        self.locks = {}
        self.door_dirs = {}
        self.grabbed_object_idx = -1
        for x, y, tile in minigrid_tile_generator(self.minigrid_env):
            y = -y
            center_pos = ((x + 0.5) * self.xy_scale, (y - 0.5) * self.xy_scale)
            if isinstance(tile, Wall):
                wall = self.world.CreateStaticBody(
                    position=center_pos,
                    shapes=polygonShape(box=(self.xy_scale / 2, self.xy_scale / 2)),
                )
                self.color_mapping[wall] = (184, 184, 184, 255)
            elif isinstance(tile, Goal):
                goal = self.world.CreateStaticBody(
                    position=center_pos,
                    shapes=polygonShape(box=(self.xy_scale / 2, self.xy_scale / 2)),
                )
                for fixture in goal.fixtures:
                    fixture.sensor = True
                self.color_mapping[goal] = (0, 255, 0, 255)
            elif isinstance(tile, Lava):
                lava = self.world.CreateStaticBody(
                    position=center_pos,
                    shapes=polygonShape(box=(self.xy_scale / 2, self.xy_scale / 2)),
                )
                for fixture in lava.fixtures:
                    fixture.sensor = True
                self.color_mapping[lava] = (255, 165, 0, 255)
            elif isinstance(tile, Ball):
                ball = self.world.CreateDynamicBody(position=center_pos)
                ball.CreateCircleFixture(radius=0.25, density=1, friction=0.3)
                ball.linearDamping = BALL_DAMPING
                self.color_mapping[ball] = get_color_rgba_255(tile.color)
                self.objects.append((ball, ObjectEnum.BALL.value, get_color_id(tile.color), 0))
            elif isinstance(tile, Box):
                box = self.world.CreateDynamicBody(position=center_pos)
                box.CreatePolygonFixture(box=(0.25, 0.25), density=1, friction=0.3)
                box.linearDamping = BOX_DAMPING
                box.angularDamping = 1
                self.color_mapping[box] = get_color_rgba_255(tile.color)
                self.objects.append((box, ObjectEnum.BOX.value, get_color_id(tile.color), 0))
            elif isinstance(tile, Key):
                key = self.world.CreateDynamicBody(position=center_pos)

                unit_size = 1 / 8
                bar_width = 1 / 24
                rectangles = [
                    (
                        (0, 0),
                        (unit_size, bar_width),
                    ),  # (center position offset, (half-width, half-height))
                    ((unit_size, unit_size), (bar_width, unit_size)),
                    ((-unit_size, unit_size), (bar_width, unit_size)),
                    ((0, 2 * unit_size), (unit_size, bar_width)),
                    ((0, -1.5 * unit_size), (bar_width, 1.5 * unit_size)),  # key stick
                    ((0.6 * unit_size, -1.3 * unit_size), (0.6 * unit_size, bar_width)),
                    (
                        (0.6 * unit_size, -2.75 * unit_size),
                        (0.6 * unit_size, bar_width),
                    ),
                ]

                # Attach each rectangle as a fixture to the body
                for center, half_size in rectangles:
                    shape = polygonShape(box=half_size)  # Create a rectangle shape
                    fixture = key.CreateFixture(shape=shape, density=1.4, friction=0.3)
                    fixture.shape.vertices = [(v[0] + center[0], v[1] + center[1]) for v in fixture.shape.vertices]

                key.linearDamping = BOX_DAMPING
                key.angularDamping = 1
                self.color_mapping[key] = get_color_rgba_255(tile.color)
                self.objects.append((key, ObjectEnum.KEY.value, get_color_id(tile.color), 0))
            elif isinstance(tile, Door):
                direction = get_door_direction(self.minigrid_env, x, -y)
                start_angle = [np.pi, -np.pi / 2, 0, np.pi / 2][direction]
                density = 1 / self.xy_scale
                # door = self.world.CreateStaticBody(
                #     position=center_pos,
                #     shapes=polygonShape(box=(self.xy_scale/4, self.xy_scale/2)),
                # )
                hinge_offset = self.xy_scale * 0.38
                hinge_base = self.world.CreateStaticBody(
                    position=(
                        center_pos[0] + hinge_offset * np.sin(-start_angle),
                        center_pos[1] + hinge_offset * np.cos(-start_angle),
                    )
                )
                hinge_base.CreateCircleFixture(radius=0.1 * self.xy_scale, density=density, friction=0.3)
                self.color_mapping[hinge_base] = get_color_rgba_255(tile.color)

                # for fixture in hinge_base.fixtures:
                #     fixture.sensor = True

                door_length = self.xy_scale * 0.42
                door = self.world.CreateDynamicBody(position=center_pos)
                door.CreatePolygonFixture(box=(0.1 * self.xy_scale, door_length), density=density, friction=10)
                door.angularDamping = 20
                door.angle = start_angle

                rjd = revoluteJointDef(
                    bodyA=hinge_base,
                    bodyB=door,
                    localAnchorA=(0, 0),
                    localAnchorB=(0, door_length),
                    enableMotor=False,
                    enableLimit=True,
                )
                rjd.upperAngle = np.pi / 2
                rjd.lowerAngle = -np.pi / 2
                hinge_base.joint = self.world.CreateJoint(rjd)
                self.color_mapping[door] = get_color_rgba_255(tile.color)
                self.objects.append(
                    (
                        door,
                        ObjectEnum.DOOR.value,
                        get_color_id(tile.color),
                        tile.is_locked * 2 + 1,
                    )
                )

                if tile.is_locked:
                    lock_offset = 0.2 * self.xy_scale
                    lock = self.world.CreateStaticBody(
                        position=(
                            center_pos[0] - lock_offset * np.cos(start_angle),
                            center_pos[1] - lock_offset * np.sin(start_angle),
                        ),
                        shapes=polygonShape(box=(0.05 * self.xy_scale, self.xy_scale * 0.4)),
                    )
                    lock.angle = start_angle
                    self.locks[door] = lock
                self.door_dirs[door] = start_angle

        if self.spawn_position is not None:
            pos = self.spawn_position
        elif self.spawn_sampler is not None:
            pos = self.spawn_sampler()
        else:
            agent_x, agent_y = self.minigrid_env.agent_pos
            pos = ((agent_x + 0.5) * self.xy_scale, -(agent_y + 0.5) * self.xy_scale)
        pos = tuple(float(x) for x in pos[:2])  # box2d doesn't like numpy scalars
        self.agent.construct_body(self.world, pos)

    def reset(self, seed=None, options=None):
        if not self._skip_initializing:
            self._generate_arena(
                minigrid_seed=seed or self.minigrid_seed,
                minigrid_options={**self._minigrid_options, **(options or {})},
            )
        self._skip_initializing = False
        if self._start_state is not None:
            self._set_state(self._start_state)

        self._cum_reward = 0
        self.timesteps = 0
        self._task = Task(*self.get_task_function(self.minigrid_env))
        self._prev_state = self.state = self._get_state()
        self._observation_spec.reset(self.state)
        return self._observation_function(self.state), {}

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        agent_vel = self.agent.body.linearVelocity.tuple
        if agent_vel[0] ** 2 + agent_vel[1] ** 2 < MAX_SPEED:
            self.agent.body.ApplyForceToCenter((MOVE_FORCE * float(action[2]), -MOVE_FORCE * float(action[1])), True)

        if action[0] >= 0.5:
            self._do_grab()
        elif self.grabbed_object_idx >= 0:
            self.objects[self.grabbed_object_idx] = (
                *self.objects[self.grabbed_object_idx][:3],
                0,
            )
            self.grabbed_object_idx = -1
        for idx, (obj, object_id, color, state) in enumerate(self.objects):
            if object_id != ObjectEnum.DOOR.value or state >= 2:  # Looking for doors that are unlocked
                continue
            is_closed = int(abs(obj.angle - self.door_dirs[obj]) < np.pi / 4)
            # print(obj.angle, self.door_dirs[obj], obj.angle - self.door_dirs[obj])
            if state & 1 != is_closed or True:
                self.objects[idx] = (*self.objects[idx][:3], is_closed)
        self.world.Step(TIME_STEP, 10, 10)

        self._prev_state = self.state
        self.state = self._get_state()
        rew, finished = self._task(self._prev_state, self.state)

        self.timesteps += 1
        return (
            self._observation_function(self.state),
            rew,
            finished or self.timesteps >= self.max_timesteps,
            False,
            {},
        )

    @property
    def task(self):
        return self._task.description

    @property
    def minigrid(self):
        return self.minigrid_env

    def _do_grab(self):
        def square_dist(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        agent_pos = np.array(self.agent.body.position.tuple)
        if self.grabbed_object_idx >= 0:
            (grabbed_obj, grabbed_type, grabbed_color, _) = self.objects[self.grabbed_object_idx]
            obj_pos = grabbed_obj.position.tuple
            if square_dist(*agent_pos, *obj_pos) > MAX_GRAB_DISTANCE:
                self.objects[self.grabbed_object_idx] = (
                    grabbed_obj,
                    grabbed_type,
                    grabbed_color,
                    0,
                )
                self.grabbed_object_idx = -1
            elif grabbed_type == 3:  # a key
                for idx, (obj, object_id, color, state) in enumerate(self.objects):
                    if (
                        object_id != ObjectEnum.DOOR.value or state & 2 == 0 or color != grabbed_color
                    ):  # skip if not a door that matches key color
                        continue
                    obj_pos = obj.position.tuple
                    key_pos = self.objects[self.grabbed_object_idx][0].position.tuple
                    if square_dist(*key_pos, *obj_pos) < DOOR_UNLOCK_DISTANCE * self.xy_scale:
                        # unlock
                        self.objects[idx] = (obj, object_id, color, state & 1)
                        self.world.DestroyBody(self.locks[obj])
        if self.grabbed_object_idx == -1:
            for idx, (obj, object_id, color, state) in enumerate(self.objects):
                if object_id == ObjectEnum.DOOR.value:  # Door
                    continue
                obj_pos = obj.position.tuple
                if square_dist(*agent_pos, *obj_pos) < MAX_GRAB_INIT_DISTANCE:
                    self.grabbed_object_idx = idx
                    self.objects[idx] = (obj, object_id, color, 1)
        if self.grabbed_object_idx >= 0:
            grabbed_obj = self.objects[self.grabbed_object_idx][0]
            walker_facing_vec = self.agent.body.GetWorldVector((0, 1)).tuple

            # get vectors in cardinal directions relative to facing direction
            target_vecs = np.array(
                (
                    walker_facing_vec,
                    (-walker_facing_vec[1], walker_facing_vec[0]),
                    (-walker_facing_vec[0], -walker_facing_vec[1]),
                    (walker_facing_vec[1], -walker_facing_vec[0]),
                )
            )

            # get the vector to the grabbed object
            obj_pos = np.array(grabbed_obj.position.tuple)
            actual_diff = obj_pos - agent_pos
            # get the target vector that most aligns with the object.
            max_idx = np.argmax(np.dot(target_vecs, actual_diff))
            target_vec = target_vecs[max_idx]

            target_pos = np.array(agent_pos) + target_vec  # the absolute target position
            target_diff = target_pos - obj_pos
            obj_vel = np.array(grabbed_obj.linearVelocity.tuple)

            force = (
                GRAB_FORCE * target_diff
                - GRAB_DAMPENING * obj_vel
                + GRAB_VEL_ALIGN * np.array(self.agent.body.linearVelocity.tuple)
            )
            grabbed_obj.ApplyForceToCenter(tuple(force.astype(float)), True)

    def _get_state(self):
        walker_pose = np.zeros(13, dtype=np.float32)
        walker_pose[:2] = self.agent.body.position.tuple
        walker_pose[3] = self.agent.body.angle
        walker_pose[7:9] = self.agent.body.linearVelocity.tuple
        walker_pose[9] = self.agent.body.angularVelocity

        object_array = np.zeros((len(self.objects), 16))
        for index, (obj, object_id, color_id, state) in enumerate(self.objects):
            object_array[index, 0] = object_id
            object_array[index, 1:3] = obj.position.tuple
            if object_id == ObjectEnum.DOOR.value:
                object_array[index, 4] = obj.angle - self.door_dirs[obj]
                object_array[index, 5] = self.door_dirs[obj] / np.pi
            else:
                object_array[index, 4] = obj.angle
            object_array[index, 8:10] = obj.linearVelocity.tuple
            object_array[index, 11] = obj.angularVelocity
            object_array[index, 14:16] = (color_id, state)

        # print('state', [obj[3] for obj in self.objects])

        return CocogridState(self._grid, self.xy_scale, object_array, walker_pose, {})

    def _set_state(self, state: CocogridState):
        self.agent.body.position = tuple(state.pose[:2].tolist())
        self.agent.body.angle = float(state.pose[4])
        self.agent.body.linearVelocity = tuple(state.pose[7:9].tolist())
        self.agent.body.angularVelocity = float(state.pose[9])

        objects = state.objects
        for obj_idx in range(objects.shape[0]):
            obj, _, _ = self.objects[obj_idx]
            obj.position = tuple(objects[obj_idx, 1:3].tolist())
            obj.angle = float(objects[obj_idx, 4])
            obj.linearVelocity = tuple(objects[obj_idx, 8:10].tolist())
            obj.angularVelocity = float(objects[obj_idx, 11])

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
            return None # reset() not called yet

        self.surf = pygame.Surface((width, height))

        pixel_per_meter = min(width / self.arena_width, height / self.arena_height)
        for body in self.world.bodies:
            if body == self.agent.body:
                self.agent.draw_body(pixel_per_meter, height, self.surf)
                continue
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
            return None
        if mode == "rgb_array" or mode == "state_pixels":
            return self._create_image_array(self.surf, (width, height))
        return self.isopen

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

    def override_reset_state(self, state: CocogridState):
        self._start_state = state


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
