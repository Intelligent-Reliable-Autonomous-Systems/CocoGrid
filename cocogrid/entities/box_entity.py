import os

from dm_control import mjcf

from cocogrid.color import get_color_rgba
from cocogrid.entities.grabbable_entity import GrabbableEntity
from cocogrid.entities import ObjectEnum, get_color_id

class BoxEntity(GrabbableEntity):
    """A box Entity which can be grabbed."""
    def _build(self, grabber, color):

        self.color = color
        self.rgba = get_color_rgba(color)
        self._color_idx = get_color_id(color)
        self._object_idx = ObjectEnum.BOX.value

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/box.xml')
        self.model = mjcf.from_path(asset_path)

        self.box_body = self.model.find('body', 'box_body')
        box_geom = self.model.find('geom', 'box_geom')
        box_geom.set_attributes(rgba=self.rgba)

        super()._build(grabber, self.box_body)

    @property
    def mjcf_model(self):
        return self.model
    
    @property
    def color_idx(self):
        return self._color_idx