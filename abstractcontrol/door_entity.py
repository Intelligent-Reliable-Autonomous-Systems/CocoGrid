from dm_control import composer
from dm_control import mjcf
import numpy as np

from abstractcontrol.color import getColorRGBA

class DoorEntity(composer.Entity):
    """A button Entity which changes colour when pressed with certain force."""
    def _build(self, color=None, is_locked=False, xy_scale=1, z_height=2):
        self.color = color
        rgba = getColorRGBA(color)
        self.is_locked = is_locked
        self.default_is_locked = is_locked

        self.model = mjcf.from_path('abstractcontrol/door.xml')
        # test = mjcf.RootElement()
        base_scale = 0.45

        hinge_base_body = self.model.find('body', 'hinge_base')
        hinge_base_body.set_attributes(pos=[0, -0.85 * base_scale * xy_scale, 0])
        
        hinge_base_geom = self.model.find('geom','hinge_base_geom')
        hinge_base_geom.set_attributes(size=[0.1*xy_scale], rgba=rgba)

        hinge_column_geom = self.model.find('geom','hinge_column_geom')
        hinge_column_geom.set_attributes(size=[0.1*xy_scale], 
                                         fromto=[0, 0, 0, 0, 0, z_height],
                                         rgba=rgba)
        
        door_body = self.model.find('body','door')
        door_body.set_attributes(pos=[0, 0.8 * base_scale * xy_scale, z_height / 2])

        door_geom = self.model.find('geom','door_geom')
        door_geom.set_attributes(size=[0.1*xy_scale, 0.8 * base_scale * xy_scale, 0.48*z_height], rgba=rgba)

        lock_body = self.model.find('body', 'lock_base')
        lock_body.set_attributes(pos=[0, -base_scale * xy_scale, 0])

        lock_front_geom = self.model.find('geom','lock_front_geom')
        lock_front_geom.set_attributes(size=[0.05*xy_scale, base_scale * xy_scale, 0.48*z_height], 
                                       rgba=rgba, pos=[0.5 * base_scale * xy_scale, 0.96 * base_scale * xy_scale, z_height / 2])
        
        self._lock_slide = self.model.find('joint', 'lock_slide')

        if not is_locked:
            lock_body.remove()

        self._keys = []
        
        # self.model = mjcf.RootElement()
        # self.thigh = self.model.worldbody.add('body')
        # self.hip = self.thigh.add('joint', axis=[0, 0, 1])
        # self.thigh.add('geom', name='test', type='cylinder', fromto=[0, 0, 0, 0, 0, length], size=[length/4])
        # self._hinge_geom = self._mjcf_model.worldbody.add(
        #     'geom', name='hinge', type='cylinder', size=[0.05, 3], rgba=[1, 0, 0, 1])
        # self._door_body = self._mjcf_model.add('body', pos=[0,0,3])
        # self._door_geom = self._door_body.add('geom', name='door', type='box', 
        #                                        size=[0.2,0.2,0.2], rgba=[1, 0, 1, 1])
        # self._site = self._mjcf_model.worldbody.add(
        #     'site', type='cylinder', size=self._hinge_geom.size*1.01, rgba=[0, 1, 0, 0.2])
        # self._sensor = self._mjcf_model.sensor.add('touch', site=self._site)
        self._num_activated_steps = 0

    def register_key(self, key):
        if key.color == self.color:
            self._keys.append(key)

    def before_step(self, physics, random_state):
        # key_joints = physics.bind(self._root_joints)
        # key_joints.qvel += np.array([0, 0.1, 0])
        if self.is_locked:
            door_pos = self.get_pose(physics)[0].base
            for key in self._keys:
                key_pos = physics.bind(key._root_joints).qpos.base
                # key_joints = physics.bind(self.model).pos
                diff = door_pos - key_pos
                diff[2] = 0
                if np.linalg.norm(diff) < 3:
                    print("The key is in range!")
                    self.is_locked = False
                    # self.model.find('geom','lock_front_geom').remove()
                    physics.bind(self._lock_slide).qpos = np.array([-100])
                    break

        return super().before_step(physics, random_state)
    
    def reset(self):
        self.is_locked = self.default_is_locked
        self._keys = []

    @property
    def mjcf_model(self):
        return self.model