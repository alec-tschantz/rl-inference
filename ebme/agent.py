# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn

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
                buffer.add(state, action, reward, next_state, done)
                state = deepcopy(next_state)
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, render=False, action_noise=0):
        trajectory = []
        actions = []
        total_reward = 0
        done = False

        with torch.no_grad():
            state = self.env.reset()
            while not done:
                action = self.planner(state)
                if action_noise > 0:
                    action = action + torch.normal(
                        mean=0, std=action_noise, size=action.size()
                    ).to(action.device)
                trajectory.append(state)
                actions.append(action)
                next_state, reward, done = self.env.step(action)

                if render:
                    self.env.render()

                total_reward += reward.item()
                if buffer is not None:
                    buffer.add(state, action, reward, next_state, done)
                state = deepcopy(next_state)
                if done:
                    break

        self.env.close()

        trajectory = torch.stack(trajectory)
        actions = torch.stack(actions)
        if buffer is not None:
            return total_reward, trajectory, actions, buffer
        else:
            return total_reward, trajectory, actions
