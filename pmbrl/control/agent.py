# pylint: disable=not-callable
# pylint: disable=no-member

from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from pmbrl import tools


class Agent(object):
    def __init__(self, env, planner, logger=None):
        self.env = env
        self.planner = planner
        self.logger = logger

    def get_seed_episodes(self, buffer, n_episodes):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)
                buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, action_noise=None):
        self.logger.log("=== Collecting data ===")
        total_reward = 0
        total_steps = 0
        trajectory = []
        actions = []
        done = False

        with torch.no_grad():
            state = self.env.reset()
            while not done:
                action = self.planner(state)
                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                trajectory.append(state)
                actions.append(action)

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                total_steps += 1

                if self.logger is not None and total_steps % self.logger.log_every == 0:
                    self.logger.log(
                        "> Step {} [reward {:.2f}]".format(
                            total_steps, total_reward
                        )
                    )

                if buffer is not None:
                    buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break

        self.env.close()
        trajectory = np.vstack(trajectory)
        actions = np.vstack(actions)
        return total_reward, total_steps, trajectory, actions

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action
