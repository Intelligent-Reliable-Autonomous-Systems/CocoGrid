from dm_control import composer
from dm_control import mjcf

class DoorEntity(composer.Entity):
    """A button Entity which changes colour when pressed with certain force."""
    def _build(self, xy_scale=1, z_height=2):
        self.model = mjcf.from_path('abstractcontrol/door.xml')

        base_scale = 0.45

        hinge_base_body = self.model.find('body', 'hinge_base')
        hinge_base_body.set_attributes(pos=[0, -base_scale * xy_scale, 0])
        
        hinge_base_geom = self.model.find('geom','hinge_base_geom')
        hinge_base_geom.set_attributes(size=[0.04*xy_scale])

        hinge_column_geom = self.model.find('geom','hinge_column_geom')
        hinge_column_geom.set_attributes(size=[0.04*xy_scale], 
                                         fromto=[0, 0, 0, 0, 0, z_height])
        
        door_body = self.model.find('body','door')
        door_body.set_attributes(pos=[0, 0.96 * base_scale * xy_scale, z_height / 2])

        door_geom = self.model.find('geom','door_geom')
        door_geom.set_attributes(size=[0.1*xy_scale, base_scale * xy_scale, 0.48*z_height])
        
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

    @property
    def mjcf_model(self):
        return self.model