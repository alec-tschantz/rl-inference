# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import gym
import torch


class GymEnv(object):
    def __init__(self, env_name, max_episode_len, action_repeat=1, seed=None):

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
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break

        return state, reward, done

    def sample_action(self):
        return self._env.action_space.sample()

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

    @property
    def steps(self):
        return self.t


class Wrapper(GymEnv):
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
        state += np.random.normal(0, self.stdev, state.shape)
        return state

    def reset(self):
        state = self.env.reset()
        return self.add_noise(state)

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.add_noise(state), reward, done
