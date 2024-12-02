import os

from dm_control import mjcf

from minimujo.color import get_color_rgba
from minimujo.entities.grabbable_entity import GrabbableEntity
from minimujo.entities import ObjectEnum, get_color_id

class BallEntity(GrabbableEntity):
    """A ball Entity which can be grabbed."""
    def _build(self, grabber, color):

        self.color = color
        self.rgba = get_color_rgba(color)
        self._color_idx = get_color_id(color)
        self._object_idx = ObjectEnum.BALL.value

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/ball.xml')
        self.model = mjcf.from_path(asset_path)

        ball_body = self.model.find('body', 'ball_body')
        ball_geom = self.model.find('geom', 'ball_geom')
        ball_geom.set_attributes(rgba=self.rgba)

        super()._build(grabber, ball_body)

    @property
    def mjcf_model(self):
        return self.model
    
    @property
    def color_idx(self):
        return self._color_idx