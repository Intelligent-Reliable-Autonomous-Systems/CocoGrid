import os

from dm_control import composer
from dm_control import mjcf
import numpy as np

from cocogrid.color import get_color_rgba, get_light_variation

class ContactTileEntity(composer.Entity):
    """A ball Entity which can be grabbed."""
    def _build(self, color, xy_scale=1):

        self.color = color
        self.rgba = get_color_rgba(color)
        self.light_rgba = get_light_variation(self.rgba)

        asset_path = os.path.join(os.path.dirname(__file__), 'assets/contact_tile.xml')
        self._model = mjcf.from_path(asset_path)

        self._body = self._model.find('body', 'contact_tile_body')
        self._body.set_attributes(pos=[-xy_scale*0, -xy_scale*0, 0.005])
        self._geom = self._model.find('geom', 'contact_tile_geom')
        self._geom.set_attributes(rgba=self.rgba, size=[xy_scale / 2, xy_scale / 2, 0.01])

        self._bounds = np.array([xy_scale / 2, xy_scale / 2, 1])
        self._walker = None
        self._num_activated_steps = 0
        self._is_activated = False

    @property
    def mjcf_model(self):
        return self._model
    
    def _update_activation(self, physics):
        self._is_activated = self.is_walker_in_bounds(physics)
        physics.bind(self._geom).rgba = (self.light_rgba if self._is_activated else self.rgba)
        self._num_activated_steps += int(self._is_activated)

    def initialize_episode(self, physics, random_state):
        self._num_activated_steps = 0
        self._update_activation(physics)

    def after_substep(self, physics, random_state):
        self._update_activation(physics)

    def register_walker(self, walker):
        self._walker = walker

    def is_walker_in_bounds(self, physics):
        if not self._walker:
            return False
        walker_pos = physics.bind(self._walker.root_body).xpos.base
        body_pos = physics.bind(self._body).xpos.base
        rel_pos = np.abs(walker_pos - body_pos)
        return (rel_pos <= self._bounds).all()

    @property
    def num_activated_steps(self):
        return self._num_activated_steps
    
    @property
    def is_activated(self):
        return self._is_activated