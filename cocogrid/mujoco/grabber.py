from dm_control import composer
from dm_control import mjcf
import mujoco
import numpy as np

class Grabber(composer.Entity):
    K_DERIVATIVE = 0.2

    """A grabber which has an actuator to read the grab control."""
    def _build(self):
        self.walker = None
        self.target_dist = 1
        self.grab_force = 10
        self.max_grab_dist = 4
        self.max_grab_init_dist = 2

        self.model = mjcf.RootElement()

        body = self.model.worldbody.add('body', name='grabberbody')
        body.add('joint', name='grab', type='slide', axis=[1, 0, 0])
        self.grab_control = self.model.actuator.add('general', name="grab", joint="grab", ctrlrange=[0, 1], ctrllimited=True, gear="30")

        self.current_object = None
        self.candidate_object = None
        self.candidate_dist = np.inf
        self.grab_dir = np.zeros(3, dtype=np.float64)
        self.cardinal_dirs = np.array([
            (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)
        ])

    @property
    def mjcf_model(self):
        return self.model
    
    def register_walker(self, walker):
        self.walker = walker
    
    def is_using_grab(self, physics):
        return physics.bind(self.grab_control).ctrl > 0.5
    
    def is_being_grabbed(self, object_entity, physics):
        # if not holding grab, not grabbed
        if not self.is_using_grab(physics):
            self.current_object = None
            self.candidate_object = None
            return False
        # if is holding a different object
        if self.current_object is not None and object_entity is not self.current_object:
            return False
        
        obj_pos = physics.bind(object_entity.root_joints).qpos.base
        walker_pos = physics.bind(self.walker.root_body).xpos
        dist = np.linalg.norm(obj_pos - walker_pos)

        # if was marked as a candidate object, grab object and reset candidate
        if object_entity is self.candidate_object:
            self.current_object = self.candidate_object
            diff = np.array(obj_pos - walker_pos)
            diff[2] = 0
            max_idx = np.argmax(np.dot(self.cardinal_dirs, diff))
            # diff /= np.linalg.norm(diff)
            # angle = np.arctan2(*diff)
            # c, s = np.cos(angle), np.sin(angle)
            self.grab_dir = self.cardinal_dirs[max_idx]
            self.candidate_object = None
            self.candidate_dist = np.inf
        # if not holding an object, set as candidate object for next step. 
        # This is so that if multiple objects are in grabbing range, they all have a chance to be grabbed.
        if self.current_object is None:
            if dist < self.max_grab_init_dist and dist < self.candidate_dist:
                self.candidate_object = object_entity
                self.candidate_dist = dist
            return False
        else:
            # keep holding object while in grab distance
            if dist < self.max_grab_dist:
                return True
            self.current_object = None
            return False
        
    def get_walker_facing_vec(self, physics):
        walker_root = physics.bind(self.walker.root_body)
        walker_quat = walker_root.xquat
        walker_facing_vec = np.array([0,0,0], dtype=np.float64)
        # updates walker_facing_vec in-place
        mujoco.mju_rotVecQuat(res=walker_facing_vec, vec=np.array([0,1,0], dtype=np.float64), quat=walker_quat)
        walker_facing_vec[2] = 0
        return walker_facing_vec
    
    def get_magnet_force(self, pos, vel, physics):
        xfrc_applied = np.array([0,0,0, 0,0,0], dtype=np.float64)
        if self.walker is None:
            return xfrc_applied
        
        walker_root = physics.bind(self.walker.root_body)
        walker_pos = walker_root.xpos
        actual_diff = (pos - walker_pos).base

        walker_facing_vec = self.get_walker_facing_vec(physics)
        target_vecs = np.array((
            walker_facing_vec,
            (-walker_facing_vec[1], walker_facing_vec[0], 0),
            -walker_facing_vec,
            (walker_facing_vec[1], -walker_facing_vec[0], 0),
        ))
        max_idx = np.argmax(np.dot(target_vecs, actual_diff))
        target_vec = target_vecs[max_idx]

        # targets a spot behind the walker
        target_pos = walker_pos + self.target_dist * target_vec
        target_diff = (target_pos - pos)
        magnitude = np.linalg.norm(target_diff)
        if magnitude > 0.3:
            target_diff /= magnitude

        xfrc_applied[:3] = target_diff - Grabber.K_DERIVATIVE * vel
        xfrc_applied[2] = 0

        return xfrc_applied * self.grab_force
            