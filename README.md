# Minimujo

A continuous control RL environment based on MuJoCo ([dm_control](https://github.com/google-deepmind/dm_controlhttps://)). Extends the discrete gridworld [Minigrid](https://github.com/Farama-Foundation/Minigridhttps://) to a continuous action and observation space. Supports many grid configurations, multiple objects, and custom tasks. It is easily extensible.

## Installation

```bash
$ git clone https://gitlab.engr.oregonstate.edu/jewettje/minimujo
$ cd minimujo
$ pip install -e .
$ export MUJOCO_GL=osmesa
$ export PYOPENGL_PLATFORM=osmesa
```

The OpenGL options for `$MUJOCO_GL` and `$PYOPENGL_PLATFORM` are `osmesa` (software rendering), `glfw` (windowed rendering), and `egl` (windowless hardware-accelerated rendering). `egl` is ideal with a GPU, but can be tricky to set up.

## Instantiation

A minimujo environment can be instantiated by importing `minimujo` and using `gymnasium.make`.

```python
import minimujo
import gymnasium
env = gymnasium.make('Minimujo-Empty-5x5-v0', timesteps=500)
```

Minimujo supports the following parameters:

- **`walker_type`** (default `'square'`)
  
  - What type of agent should be placed in the environment. More details about each below.
  - Values: `square`, `ball`, `ant`
- **`observation_type`** (default `'pos,vel,walker'`)
  
  - A comma-separated list of which features should be included in the observations. All the ones selected are concatenated into a vector. The camera is an image observation. If there is a vector and image observation, they are combined in a dict.
  - Values: `pos`, `vel`, `walker`, `top_camera`
- `xy_scale` (default `1`)
  
  - How big each grid unit should be. This changes the size of the walls, but not the size of the agent or objects.
  - Values: any float >= 1
- `timesteps` (default `500`)
  
  - How many steps to run before terminating the episode.
  - Values: any int >= 1
- `get_task_function` (default `minimujo.state.tasks.get_grid_goal_task`)
  
  - A function to get the task function for each episode. Takes a Minigrid environment as input and returns a task function, as described below.
  - Value: a function of the form `def my_get_task_function(minigrid: minigrid.minigrid_env.MiniGridEnv) -> Callable`
- `task_function` (default `None`)
  
  - This overrides `get_task_function` to use the same task_function for every episode. The task function takes in the previous and current arena states and outputs a tuple of (reward, terminate).
  - Value: a function of the form `def my_task_function(prev_state: MinimujoState, current_state: MinimujoState) -> Tuple[float, bool]`
- `spawn_position` (default `None`)
  
  - If specified, sets the position of the agent to a fixed position at the start of each episode.
  - Values: a (x,y) or (x,y,z) tuple.
- `spawn_sampler` (default `None')
  
  - If specified, this function will be called at the start of an episode to return a spawn position. `spawn_position` takes precedence.
- `random_spawn` (default `False`)
  
  - If `True`, spawns the agent in a random location (avoiding walls).
- `random_rotation` (default `False`)
  
  - If `True`, spawns the agent in a random orientation each episode. `spawn_position` and `spawn_sampler` take precedence.
- `cam_width` (default `320`)
  
  - Sets the pixel width of camera observations.
- `cam_height` (default `240`)
  
  - Sets the pixel height of camera observations.
- `image_observation_format` (default `'0-255'`)
  
  - What format should images be output in? Either in the int8 range or a float from 0 to 1.
  - Values: `'0-255'`, `'0-1'`

## Utility

For convenient testing, there is a utility command `python -m minimujo`. These are the utilities:

- `--list`, `-l`: list the available minimujo environment ids.
- `--gym`, `-g`: spawn an interactive window with keyboard input.
- `--minigrid`, `-m`: spawns an interactive window for a grid world
- `--detail`, `-d`: give details about the selected environment
- `--framerate`, `-f`: test the steps-per-second of the environment

And these are the parameters you can pass into the utility:

- `--env`, `-e`: Select the environment id
- `--walker-type`, `-w`: Select the `walker_type`
- `--obs-type`, `-o`: Select the `observation_type`
- `--scale`, `-s`: Select the `xy_scale`
- `--timesteps`, `-t`: Select the `timesteps`
- `--seed`: Select the `seed`
- `--episodes`: The number of episodes to run in `--gym` before exiting
- `--random-spawn`: Sets `random_spawn` to `True`
- `--random-rotate`: Sets `random_rotation` to `True`
- `--print-obs`: For debugging, print the observations in `--gym`
- `--print-reward`: For debugging, print the reward in `--gym`

Examples:

```bash
$ python -m minimujo -g -e Minimujo-Empty-5x5-v0 -w ball -t 200
```

```bash
$ python -m minimujo -d -e Minimujo-Empty-5x5-v0
```



