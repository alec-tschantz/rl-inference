import os

import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


class SparseHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_x_torso = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/half_cheetah.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        x_pos = self.get_body_com("torso")[0]
        if x_pos <= 5.0:
            reward = 0.0
        else:
            reward = 1.0
        done = False
        return obs, reward, done, {"x_pos": x_pos}

    def _get_obs(self):
        """ https://github.com/openai/vime/blob/master/envs/half_cheetah_env_x.py """
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                self.get_body_com("torso").flat,
            ]
        )

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
