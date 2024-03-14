from dm_control import composer
from dm_control import mjcf
import mujoco
import numpy as np

class Grabber(composer.Entity):
    """A grabber which has an actuator to read the grab control."""
    def _build(self):
        self.walker = None
        self.target_dist = 1
        self.grab_force = 3
        self.max_grab_dist = 4
        self.max_grab_init_dist = 2

        self.model = mjcf.RootElement()

        body = self.model.worldbody.add('body', name='grabberbody')
        body.add('joint', name='grab', type='slide', axis=[1, 0, 0])
        self.grab_control = self.model.actuator.add('general', name="grab", joint="grab", ctrlrange=[0, 1], ctrllimited=True, gear="30")

        self.current_object = None
        self.candidate_object = None
        self.candidate_dist = np.inf

    @property
    def mjcf_model(self):
        return self.model
    
    def register_walker(self, walker):
        self.walker = walker
    
    def is_using_grab(self, physics):
        return physics.bind(self.grab_control).ctrl > 0.5
    
    def is_being_grabbed(self, object_entity, physics):
        if not self.is_using_grab(physics):
            self.current_object = None
            self.candidate_object = None
            return False
        if self.current_object is not None and object_entity is not self.current_object:
            return False
        if object_entity is self.candidate_object:
            self.current_object = self.candidate_object
            self.candidate_object = None
            self.candidate_dist = np.inf
        obj_pos = physics.bind(object_entity.root_joints).qpos.base
        walker_pos = physics.bind(self.walker.root_body).xpos
        dist = np.linalg.norm(obj_pos - walker_pos)
        if self.current_object is None:
            if dist < self.max_grab_init_dist and dist < self.candidate_dist:
                self.candidate_object = object_entity
                self.candidate_dist = dist
            return False
        else:
            if dist < self.max_grab_dist:
                return True
            self.current_object = None
            return False

        
    
    def get_magnet_force(self, pos, physics):
        xfrc_applied = np.array([0,0,0, 0,0,0], dtype=np.float64)
        if self.walker is None:
            return xfrc_applied
        
        walker_root = physics.bind(self.walker.root_body)
        walker_pos = walker_root.xpos
        walker_quat = walker_root.xquat

        walker_facing_vec = np.array([0,0,0], dtype=np.float64)
        # updates walker_facing_vec in-place
        mujoco.mju_rotVecQuat(res=walker_facing_vec, vec=np.array([0,1,0], dtype=np.float64), quat=walker_quat)

        # targets a spot behind the walker
        target_pos = walker_pos + self.target_dist * walker_facing_vec

        xfrc_applied[:3] = (target_pos - pos).base
        xfrc_applied[2] = 0

        magnitude = np.linalg.norm(xfrc_applied)
        if magnitude > 0.3:
            xfrc_applied /= magnitude
        return xfrc_applied * self.grab_force
            