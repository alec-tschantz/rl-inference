# pylint: disable=not-callable
# pylint: disable=no-member

import gym
import torch

from ebme.envs import (
    SparseMountainCar,
    SparseCartpoleSwingup,
    SparseDoublePendulum,
    SparseHalfCheetah,
    SparseBipedalWalker,
    const,
)


class TorchEnv(object):
    def __init__(
        self, env_name, max_episode_len, action_repeat=1, device="cpu", seed=None
    ):

        if env_name == const.SPARSE_MOUNTAIN_CAR:
            self._env = SparseMountainCar()
        elif env_name == const.SPARSE_CARTPOLE_SWINGUP:
            self._env = SparseCartpoleSwingup()
        elif env_name == const.SPARSE_DOUBLE_PENDULUM:
            self._env = SparseDoublePendulum()
        elif env_name == const.SPARSE_HALF_CHEETAH:
            self._env = SparseHalfCheetah()
        elif env_name == const.SPARSE_BIPEDAL_WALKER:
            self._env = SparseBipedalWalker()
        else:
            self._env = gym.make(env_name)

        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.done = False
        self.device = device
        if seed is not None:
            self._env.seed(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        state = self._env.reset()
        self.done = False
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def step(self, action):
        action = action.cpu().detach().numpy()
        reward = 0

        for _ in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break

        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return state, reward, done

    def sample_action(self):
        return torch.from_numpy(self._env.action_space.sample()).to(self.device)

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def state_dims(self):
        return self._env.observation_space.shape

    @property
    def action_dims(self):
        return self._env.action_space.shape
