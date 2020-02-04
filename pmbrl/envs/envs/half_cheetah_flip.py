import os

import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


class SparseHalfCheetahFlipEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_x_torso = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/half_cheetah.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = obs[12] 
        done = False
        return obs, reward, done, {}

    def _get_obs(self):
        z_position = self.sim.data.qpos.flat[1:2]
        y_rotation = self.sim.data.qpos.flat[2:3]
        other_positions = self.sim.data.qpos.flat[3:]
        velocities = self.sim.data.qvel.flat

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        average_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_rotation_sin, y_rotation_cos = np.sin(y_rotation), np.cos(y_rotation)

        obs = np.concatenate(
            [
                average_velocity,
                z_position,
                y_rotation_sin,
                y_rotation_cos,
                other_positions,
                velocities,
            ]
        )
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
