import os

from dm_control import mjcf
import numpy as np

from minimujo.color import get_color_rgba
from minimujo.entities.grabbable_entity import GrabbableEntity


class KeyEntity(GrabbableEntity):
    """A key Entity which attaches to agent model and can unlock doors."""
    def _build(self, grabber, color):

        self.color = color
        rgba = get_color_rgba(color)

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/key.xml')
        self.model = mjcf.from_path(asset_path)

        offset = np.array([1.4/2, 1.4/2, 0])
        # self.key_box = self.model.find('geom', 'key_handle_geom1')
        for geom in self.model.find_all('geom'):
            pos = geom.pos
            size = geom.size
            geom.set_attributes(pos=(pos-offset)/2, size=size/2, rgba=rgba)

        key_body = self.model.find('body', 'key_body')
        # key_body.set_attributes(pos=[-1.4/2, -1.4/2, 0])
        print('')
        # self.key_box_geom = self.model.find('geom', 'key_box_geom')
        # self.key_box_geom.set_attributes(rgba=rgba)
        super()._build(grabber, key_body)

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