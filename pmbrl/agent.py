# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy


class Agent(object):
    def __init__(self, env, planner):
        self.env = env
        self.planner = planner

    def get_seed_episodes(self, buffer, n_episodes):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done = self.env.step(action)
                buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, action_noise=0.0):
        total_reward = 0
        total_steps = 0
        done = False

        with torch.no_grad():
            state = self.env.reset()
            while not done:
                action = self.planner(state)
                action = action.cpu().detach().numpy()

                if action_noise > 0:
                    action = action + np.random.normal(0, action_noise, action.shape)

                next_state, reward, done = self.env.step(action)
                total_reward += reward
                total_steps += 1

                if buffer is not None:
                    buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break

        self.env.close()
        stats = self.planner.get_stats()

        if buffer is not None:
            return total_reward, total_steps, buffer, stats
        else:
            return total_reward, total_steps, stats
