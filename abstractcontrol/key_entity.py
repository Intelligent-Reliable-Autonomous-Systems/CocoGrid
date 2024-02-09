from dm_control import composer
from dm_control import mjcf
import numpy as np

from abstractcontrol.color import getColorRGBA

class KeyEntity(composer.Entity):
    """A key Entity which attaches to agent model and can unlock doors."""
    def _build(self, color):
        self.color = color
        rgba = getColorRGBA(color)

        self.model = mjcf.from_path('abstractcontrol/key.xml')

        # key_box_geom = self.model.find('geom', 'key_box_geom')
        # key_box_geom.set_attributes(rgba=rgba)

        self._root_joints = None
        self._walker = None

        self.step_num = 0

    @property
    def mjcf_model(self):
        return self.model
    
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
                # position = physics.bind(self._walker.root_body).xpos.base + np.array([0,0,2])
                physics.bind(self._root_joints).qpos = position
                # physics.bind(self.model.find_all('joint')).qpos = 0.
            else:
                super().set_pose(physics, position, quaternion=None)

    def before_substep(self, physics, random_state):
        self.step_num += 1
        # key_joints = physics.bind(self._root_joints)
        # key_joints.qvel += np.array([0, 0.1, 0])
        if self._walker is not None and self.step_num % 100 == 0:
            walker_pos = physics.bind(self._walker.root_body).xpos
            key_joints = physics.bind(self._root_joints)
            key_pos = key_joints.qpos
            diff = (walker_pos - key_pos).base
            diff[2] = 0
            key_joints.qvel += diff

        return super().before_substep(physics, random_state)
    
    def register_walker(self, walker):
        self._walker = walker