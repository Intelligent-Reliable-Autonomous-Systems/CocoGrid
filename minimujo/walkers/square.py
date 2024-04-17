import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers.legacy_base import Walker
import numpy as np

class Square(Walker):

    def _build(self, *args, **kwargs):
        super()._build(*args, **kwargs)

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/square.xml')
        self._mjcf_root = mjcf.from_path(asset_path)

        self.y_control = self._mjcf_root.find('actuator', 'forward_backward')
        self.x_control = self._mjcf_root.find('actuator', 'left_right')

        self.upper_sensor = self._mjcf_root.find('sensor', 'upper')
        self.lower_sensor = self._mjcf_root.find('sensor', 'lower')
        self.right_sensor = self._mjcf_root.find('sensor', 'right')
        self.left_sensor = self._mjcf_root.find('sensor', 'left')
        self.sensors = [self.upper_sensor, self.lower_sensor, self.right_sensor, self.left_sensor]

        self.quat = [1, 0, 0, 0]
        self.force = 2000
        self.max_speed = 2
        
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
        y_ctrl = max(-1, min(1, -physics.bind(self.y_control).ctrl))
        # x_ctrl = 0
        # y_ctrl = -0

        position = physics.bind(self._root_joints).qpos.base
        position[2] = 0.3

        # print('before', position)

        # position += 0.01 * np.array([x_ctrl, -y_ctrl, 0])

        physics.bind(self._root_joints).qpos = position 

        sensor_contacts = physics.bind(self.sensors).sensordata > 0

        if x_ctrl != 0:
            if sensor_contacts[0]:
                y_ctrl = min(0, y_ctrl)
            if sensor_contacts[1]:
                y_ctrl = max(0, y_ctrl)
        if y_ctrl != 0:
            if sensor_contacts[2]:
                x_ctrl = min(0, x_ctrl)
            if sensor_contacts[3]:
                x_ctrl = max(0, x_ctrl)

        # physics.bind(self._root_joints).qvel = 0 * np.array([x_ctrl, -y_ctrl, 0])
        # physics.bind(mjcf.get_attachment_frame(self.mjcf_model)).quat = self.quat
        # physics.bind(self.root_body).quat = self.quat

        # physics.named.data.xfrc_applied[self.root_body.full_identifier] = np.array([10 * x_ctrl, -10 * y_ctrl,0,0,0,0], dtype=np.float64)

        physics.bind(self._root_joints).qfrc_applied = np.array([self.force * x_ctrl, self.force * y_ctrl, 0])
        velocity = physics.bind(self._root_joints).qvel.base
        vel_bounds = np.array([min(1, abs(x_ctrl)), min(1, abs(y_ctrl)), 1]) * self.max_speed
        velocity = np.clip(velocity, a_min=-vel_bounds, a_max=vel_bounds)
        # print(velocity)
        # velocity[0] = x_ctrl
        # velocity[1] = y_ctrl
        physics.bind(self._root_joints).qvel = velocity
        # print('after', position, velocity)



        return super().before_substep(physics, random_state)
    
class SquareObservables(composer.Observables):

    @composer.observable
    def collision_detection(self):
        def get_collision(physics):
            return (physics.bind(self._entity.sensors).sensordata > 0).astype(int)
        return observable.Generic(get_collision)

    @property
    def proprioception(self):
        return [self.collision_detection]