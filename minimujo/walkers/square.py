import os

from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.walkers.legacy_base import Walker
import numpy as np

class Square(Walker):

    def _build(self, *args, **kwargs):
        super()._build(*args, **kwargs)

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/square.xml')
        self._mjcf_root = mjcf.from_path(asset_path)

        self.y_control = self._mjcf_root.find('actuator', 'forward_backward')
        self.x_control = self._mjcf_root.find('actuator', 'left_right')

        self.quat = [1, 0, 0, 0]
        
    @property
    def mjcf_model(self):
        return self._mjcf_root
    
    def create_root_joints(self, attachment_frame):
        root_class = self._mjcf_root.find('default', 'root')
        root_x = attachment_frame.add(
            'joint', name='root_x', type='slide', axis=[1, 0, 0], dclass=root_class)
        root_y = attachment_frame.add(
            'joint', name='root_y', type='slide', axis=[0, 1, 0], dclass=root_class)
        root_z = attachment_frame.add(
            'joint', name='root_z', type='slide', axis=[0, 0, 1], dclass=root_class)
        self._root_joints = [root_x, root_y, root_z]

    def _build_observables(self):
        return SquareObservables(self)

    @composer.cached_property
    def actuators(self):
        return self._mjcf_root.find_all('actuator')
    
    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find('body', 'square_body')
    
    @composer.cached_property
    def observable_joints(self):
        return []
    
    @composer.cached_property
    def end_effectors(self):
        return [self._mjcf_root.find('body', 'square_body')]

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find('camera', 'egocentric')

    @composer.cached_property
    def ground_contact_geoms(self):
        return (self._mjcf_root.find('geom', 'square_geom'),)
    
    def before_substep(self, physics, random_state):
        x_ctrl = max(-1, min(1, physics.bind(self.x_control).ctrl))
        y_ctrl = max(-1, min(1, physics.bind(self.y_control).ctrl))

        position = physics.bind(self._root_joints).qpos.base

        position += 0.01 * np.array([x_ctrl, -y_ctrl, 0])

        physics.bind(self._root_joints).qpos = position 

        physics.bind(mjcf.get_attachment_frame(self.mjcf_model)).quat = self.quat

        return super().before_substep(physics, random_state)
    
class SquareObservables(composer.Observables):

    @property
    def proprioception(self):
        return []