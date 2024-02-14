from dm_control import composer
from dm_control import mjcf
import numpy as np
import mujoco

from abstractcontrol.color import getColorRGBA
from abstractcontrol.grabbable_entity import GrabbableEntity

class BoxEntity(GrabbableEntity):
    """A box Entity which can be grabbed."""
    def _build(self, grabber, color):

        self.color = color
        rgba = getColorRGBA(color)

        self.model = mjcf.from_path('abstractcontrol/box.xml')

        box_body = self.model.find('body', 'box_body')
        box_geom = self.model.find('geom', 'box_geom')
        box_geom.set_attributes(rgba=rgba)

        super()._build(grabber, box_body)

    @property
    def mjcf_model(self):
        return self.model
    
    # @property
    # def root_joints(self):
    #     return self._root_joints
    
    # def create_root_joints(self, attachment_frame):
    #     root_class = self.model.find('default', 'root')
    #     root_x = attachment_frame.add(
    #         'joint', name='key_x', type='slide', axis=[1, 0, 0], dclass=root_class)
    #     root_y = attachment_frame.add(
    #         'joint', name='key_y', type='slide', axis=[0, 1, 0], dclass=root_class)
    #     root_z = attachment_frame.add(
    #         'joint', name='key_z', type='slide', axis=[0, 0, 1], dclass=root_class)
    #     self._root_joints = [root_x, root_y, root_z]

    # def set_pose(self, physics, position=None, quaternion=None):
    #     if position is not None:
    #         if self._root_joints is not None:
    #             physics.bind(self._root_joints).qpos = position
    #         else:
    #             super().set_pose(physics, position, quaternion=None)

    # def before_substep(self, physics, random_state):
    #     # Reacts to grabber magnet using external force
    #     physics.named.data.xfrc_applied[self.key_box.full_identifier] = np.array([0,0,0,0,0,0], dtype=np.float64)
    #     if self.grabber.is_being_grabbed(self, physics):
    #         key_pos = physics.bind(self._root_joints).qpos.base
    #         physics.named.data.xfrc_applied[self.key_box.full_identifier] = self.grabber.get_magnet_force(key_pos, physics)

    #     return super().before_substep(physics, random_state)