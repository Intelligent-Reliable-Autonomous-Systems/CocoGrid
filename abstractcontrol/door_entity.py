from dm_control import composer
from dm_control import mjcf

class DoorEntity(composer.Entity):
    """A button Entity which changes colour when pressed with certain force."""
    def _build(self, target_force_range=(5, 10)):
        self._min_force, self._max_force = target_force_range
        length=3
        self.model = mjcf.from_path('abstractcontrol/door.xml')
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