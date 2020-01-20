import roboschool
import gym


class SparseHalfCheetah(object):
    def __init__(self):
        self._env = gym.make("RoboschoolHalfCheetah-v1")
        self.threshold = 0

    def reset(self):
        return self._env.reset()

    def step(self, a):
        s, r, d, i = self._env.step(a)
        
        """
        pos_x = self._env.body_xyz[0]
        if pos_x > self.threshold:
            r = 1.
        else:
            r = 0.
        """

        return s, r, d, i

    def render(self, mode="human"):
        return self._env.render(mode)

    def close(self):
        return self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space
