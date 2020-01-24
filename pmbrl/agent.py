# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy


class Agent(object):
    def __init__(self, env, planner, logdir):
        self.env = env
        self.planner = planner
        self.logdir = logdir

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

    def run_episode(
        self,
        buffer=None,
        use_exploration=True,
        use_reward=True,
        render=False,
        episode=None,
        action_noise=0.0,
    ):
        total_reward = 0
        total_steps = 0
        done = False

        if render:
            filename = self.logdir + "/episode_{}.mp4".format(episode)
            self.env.setup_render(filename)

        with torch.no_grad():
            state = self.env.reset()
            while not done:
                action = self.planner(
                    state, use_exploration=use_exploration, use_reward=use_reward
                )
                action = action.cpu().detach().numpy()

                if action_noise > 0:
                    action = action + np.random.normal(0, action_noise, action.shape)

                next_state, reward, done = self.env.step(action)
                total_reward += reward
                total_steps += 1

                if render:
                    self.env.render()

                if buffer is not None:
                    buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break

        self.env.close()
        stats = self.planner.get_stats(
            use_exploration=use_exploration, use_reward=use_reward
        )

        if buffer is not None:
            return total_reward, total_steps, buffer, stats
        else:
            return total_reward, total_steps, stats
