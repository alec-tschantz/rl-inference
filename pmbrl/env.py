import gym

from pmbrl.envs import (
    SparseMountainCarEnv,
    SparseCartpoleSwingupEnv,
    SparseHalfCheetahEnv,
    SparseAntEnv,
)

SPARSE_MOUNTAIN_CAR = "SparseMountainCar"
SPARSE_CARTPOLE_SWINGUP = "SparseCartpoleSwingup"
SPARSE_HALF_CHEETAH = "SparseHalfCheetah"
SPARSE_ANT = "SparseAnt"


class GymEnv(object):
    def __init__(self, env_name, max_episode_len, action_repeat=1, seed=None):
        if env_name == SPARSE_MOUNTAIN_CAR:
            self._env = SparseMountainCarEnv()
        elif env_name == SPARSE_CARTPOLE_SWINGUP:
            self._env = SparseCartpoleSwingupEnv()
        elif env_name == SPARSE_HALF_CHEETAH:
            self._env = SparseHalfCheetahEnv()
        elif env_name == SPARSE_ANT:
            self._env = SparseAntEnv()
        else:
            self._env = gym.make(env_name)
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.done = False
        if seed is not None:
            self._env.seed(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        state = self._env.reset()
        self.done = False
        return state

    def step(self, action):
        reward = 0

        for _ in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break

        return state, reward, done, info

    def sample_action(self):
        return self._env.action_space.sample()

    def render(self):
        pass

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space
