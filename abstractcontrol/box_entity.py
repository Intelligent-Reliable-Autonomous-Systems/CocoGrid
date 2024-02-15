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