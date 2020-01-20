# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from .torch_env import TorchEnv


class Wrapper(TorchEnv):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def sample_action(self):
        return self.env.sample_action()

    @property
    def state_dims(self):
        return self.env.state_dims

    @property
    def action_dims(self):
        return self.env.action_dims


class NoisyEnv(Wrapper):
    def __init__(self, env, stdev):
        self.stdev = stdev
        super().__init__(env)

    def add_noise(self, state):
        state += torch.normal(0, self.stdev, size=state.size()).to(state.device)
        return state

    def reset(self, filename=""):
        state = self.env.reset()
        return self.add_noise(state)

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.add_noise(state), reward, done