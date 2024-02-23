import os

from dm_control import mjcf

from minimujo.color import get_color_rgba
from minimujo.entities.grabbable_entity import GrabbableEntity

class BoxEntity(GrabbableEntity):
    """A box Entity which can be grabbed."""
    def _build(self, grabber, color):

        self.color = color
        rgba = get_color_rgba(color)

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/box.xml')
        self.model = mjcf.from_path(asset_path)

        box_body = self.model.find('body', 'box_body')
        box_geom = self.model.find('geom', 'box_geom')
        box_geom.set_attributes(rgba=rgba)

        super()._build(grabber, box_body)

    @property
    def mjcf_model(self):
        return self.model