import os

import numpy as np
from dm_control import mjcf

from cocogrid.common.color import get_color_rgba
from cocogrid.common.entity import ObjectEnum, get_color_id
from cocogrid.mujoco.entities.grabbable_entity import GrabbableEntity


class KeyEntity(GrabbableEntity):
    """A key Entity which attaches to agent model and can unlock doors."""
    def _build(self, grabber, color):

        self.color = color
        self.rgba = get_color_rgba(color)
        self._color_idx = get_color_id(color)
        self._object_idx = ObjectEnum.KEY.value

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/key.xml')
        self.model = mjcf.from_path(asset_path)

        offset = np.array([1.4/2, 1.4/2, 0])
        # self.key_box = self.model.find('geom', 'key_handle_geom1')
        for geom in self.model.find_all('geom'):
            pos = geom.pos
            size = geom.size
            geom.set_attributes(pos=(pos-offset)/2, size=size/2, rgba=self.rgba)

        key_body = self.model.find('body', 'key_body')

        super()._build(grabber, key_body)

    @property
    def mjcf_model(self):
        return self.model
    
    @property
    def color_idx(self):
        return self._color_idx