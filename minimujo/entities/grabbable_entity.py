from dm_control import composer
import numpy as np

from minimujo.color import get_light_variation

class GrabbableEntity(composer.Entity):

    """A key Entity which attaches to agent model and can unlock doors."""
    def _build(self, grabber, body_id):
        self._grabber = grabber
        self._root_joints = None
        self.body_id = body_id
        self._is_grabbed = False
        self._position = np.array([-10000,-10000,0])

        self._color_geoms = self.mjcf_model.find_all('geom')
        self.light_rgba = get_light_variation(self.rgba)
    
    @property
    def root_joints(self):
        return self._root_joints
    
    @property
    def is_grabbed(self):
        return self._is_grabbed
    
    @property
    def position(self):
        return self._position
    
    def create_root_joints(self, attachment_frame):
        root_class = self.model.find('default', 'root')
        root_x = attachment_frame.add(
            'joint', name='key_x', type='slide', axis=[1, 0, 0], dclass=root_class)
        root_y = attachment_frame.add(
            'joint', name='key_y', type='slide', axis=[0, 1, 0], dclass=root_class)
        root_z = attachment_frame.add(
            'joint', name='key_z', type='slide', axis=[0, 0, 1], dclass=root_class)
        self._root_joints = [root_x, root_y, root_z]

    def set_pose(self, physics, position=None, quaternion=None):
        if position is not None:
            if self._root_joints is not None:
                physics.bind(self._root_joints).qpos = position
            else:
                super().set_pose(physics, position, quaternion=None)

    def before_substep(self, physics, random_state):
        # Reacts to grabber magnet using external force
        physics.named.data.xfrc_applied[self.body_id.full_identifier] = np.array([0,0,0,0,0,0], dtype=np.float64)
        self._position = physics.bind(self._root_joints).qpos.base
        was_grabbed = self._is_grabbed
        self._is_grabbed = self._grabber.is_being_grabbed(self, physics)
        if self._is_grabbed:
            physics.named.data.xfrc_applied[self.body_id.full_identifier] = self._grabber.get_magnet_force(self._position, physics)
        if self._is_grabbed != was_grabbed:
            physics.bind(self._color_geoms).rgba = (self.light_rgba if self._is_grabbed else self.rgba)

        return super().before_substep(physics, random_state)
    
    def get_object_state(self, physics):
        state: np.ndarray = np.zeros(16)
        bound_body = physics.bind(self.root_body)
        state[0] = self._object_idx
        state[1:4] = bound_body.xpos
        state[4:8] = bound_body.xquat
        state[8:14] = bound_body.cvel
        state[14] = self.color_idx
        state[15] = int(self._is_grabbed)
        return state