from dm_control import composer
from dm_control import mjcf

from abstractcontrol.color import getColorRGBA

class KeyEntity(composer.Entity):
    """A key Entity which attaches to agent model and can unlock doors."""
    def _build(self, color):
        self.color = color
        rgba = getColorRGBA(color)

        self.model = mjcf.from_path('abstractcontrol/key.xml')

        key_box_geom = self.model.find('geom', 'key_box_geom')
        key_box_geom.set_attributes(rgba=rgba)

    @property
    def mjcf_model(self):
        return self.model