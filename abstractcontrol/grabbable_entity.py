from dm_control import composer
from dm_control import mjcf
import numpy as np
import mujoco

from abstractcontrol.color import getColorRGBA

class GrabbableEntity(composer.Entity):
    """A key Entity which attaches to agent model and can unlock doors."""
    def _build(self, grabber, body_id):
        self._grabber = grabber
        self._root_joints = None
        self.body_id = body_id
    
    @property
    def root_joints(self):
        return self._root_joints
    
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
        if self._grabber.is_being_grabbed(self, physics):
            key_pos = physics.bind(self._root_joints).qpos.base
            physics.named.data.xfrc_applied[self.body_id.full_identifier] = self._grabber.get_magnet_force(key_pos, physics)

        return super().before_substep(physics, random_state)