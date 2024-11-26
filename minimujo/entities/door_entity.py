import os

from dm_control import composer
from dm_control import mjcf
import numpy as np

from minimujo.color import get_color_idx, get_color_rgba

DOOR_IDX = 2
DOOR_QUATS = [
    [1, 0, 0, 0], # up
    [np.sin(np.pi/4), 0, 0, np.sin(np.pi/4)], # left
    [0, 0, 0, 1], # down
    [np.sin(np.pi/4), 0, 0, -np.sin(np.pi/4)] # right
]
DOOR_ANGLE = [0, 0.5, 1, -0.5]

class DoorEntity(composer.Entity):
    OPEN_ANGLE = np.pi / 4

    """A button Entity which changes colour when pressed with certain force."""
    def _build(self, color=None, is_locked=False, xy_scale=1, z_height=2):
        self.color = color
        rgba = get_color_rgba(color)
        self._color_idx = get_color_idx(color)
        self._is_locked = is_locked
        self._is_open = False
        self.default_is_locked = is_locked
        self._object_idx = DOOR_IDX

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/door.xml')
        self._mjcf_model = mjcf.from_path(asset_path)
        
        base_scale = 0.45

        hinge_base_body = self._mjcf_model.find('body', 'hinge_base')
        hinge_base_body.set_attributes(pos=[0, -0.85 * base_scale * xy_scale, 0])
        
        hinge_base_geom = self._mjcf_model.find('geom','hinge_base_geom')
        hinge_base_geom.set_attributes(size=[0.1*xy_scale], rgba=rgba)

        hinge_column_geom = self._mjcf_model.find('geom','hinge_column_geom')
        hinge_column_geom.set_attributes(size=[0.1*xy_scale], 
                                         fromto=[0, 0, 0, 0, 0, z_height],
                                         rgba=rgba)
        
        door_body = self._mjcf_model.find('body','door')
        door_body.set_attributes(pos=[0, 0.8 * base_scale * xy_scale, z_height / 2])

        door_geom = self._mjcf_model.find('geom','door_geom')
        door_geom.set_attributes(size=[0.1*xy_scale, 0.8 * base_scale * xy_scale, 0.48*z_height], rgba=rgba)

        lock_body = self._mjcf_model.find('body', 'lock_base')
        lock_body.set_attributes(pos=[0, -base_scale * xy_scale, 0])

        lock_front_geom = self._mjcf_model.find('geom','lock_front_geom')
        lock_front_geom.set_attributes(size=[0.05*xy_scale, base_scale * xy_scale, 0.48*z_height], 
                                       rgba=rgba, pos=[0.5 * base_scale * xy_scale, 0.96 * base_scale * xy_scale, z_height / 2])
        
        self._door_hinge = self._mjcf_model.find('joint', 'door_hinge')
        
        self._lock_slide = self._mjcf_model.find('joint', 'lock_slide')

        if not is_locked:
            lock_body.remove()

        self._keys = []
        self._num_activated_steps = 0
        self.position = np.zeros(3, dtype=float)
        self.orientation = DOOR_ANGLE[0]

    def set_position(self, physics, position, direction):
        self.position = position

        # direction indexes [up, left, down, right]
        self.orientation = DOOR_ANGLE[direction]
        quat = DOOR_QUATS[direction]

        self.set_pose(physics=physics, position=position, quaternion=quat)

    def register_key(self, key):
        if key.color == self.color:
            self._keys.append(key)

    def before_step(self, physics, random_state):
        # key_joints = physics.bind(self._root_joints)
        # key_joints.qvel += np.array([0, 0.1, 0])
        if self._is_locked:
            physics.bind(self._lock_slide).qpos = np.array([0])
            door_pos = self.get_pose(physics)[0].base
            for key in self._keys:
                key_pos = physics.bind(key._root_joints).qpos.base
                # key_joints = physics.bind(self.model).pos
                diff = door_pos - key_pos
                diff[2] = 0
                if np.linalg.norm(diff) < 1:
                    print("The key is in range!")
                    self._is_locked = False
                    # self.model.find('geom','lock_front_geom').remove()
                    physics.bind(self._lock_slide).qpos = np.array([-100])
                    break

        hinge_angle = physics.bind(self._door_hinge).qpos[0]
        self._is_open = np.abs(hinge_angle) >= DoorEntity.OPEN_ANGLE

        return super().before_step(physics, random_state)
    
    def reset(self):
        self._is_locked = self.default_is_locked
        self._keys = []

    @property
    def mjcf_model(self):
        return self._mjcf_model
    
    @property
    def is_open(self):
        return self._is_open

    @property
    def is_locked(self):
        return self._is_locked
    
    def get_object_state(self, physics):
        state: np.ndarray = np.zeros(16)
        state[0] = self._object_idx
        state[1:4] = self.position
        state[4] = physics.bind(self._door_hinge).qpos[0] # the hingle angle
        state[5] = self.orientation # the door orientation
        # slots 6-10 are empty, since the door is static
        # 2 spots empty for orientation, 3 for linear velocity
        state[11] = physics.bind(self._door_hinge).qvel[0]
        # slots 12-13 are empty
        state[14] = self._color_idx
        state[15] = self.is_locked * 2 + (not self.is_open)
        return state