# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch


class Normalizer(object):
    def __init__(self):
        self.state_mean = None
        self.state_sk = None
        self.state_stdev = None
        self.action_mean = None
        self.action_sk = None
        self.action_stdev = None
        self.state_delta_mean = None
        self.state_delta_sk = None
        self.state_delta_stdev = None
        self.count = 0

    @staticmethod
    def update_mean(mu_old, addendum, n):
        mu_new = mu_old + (addendum - mu_old) / n
        return mu_new

    @staticmethod
    def update_sk(sk_old, mu_old, mu_new, addendum):
        sk_new = sk_old + (addendum - mu_old) * (addendum - mu_new)
        return sk_new

    def update(self, state, action, state_delta):
        self.count += 1

        if self.count == 1:
            self.state_mean = state.copy()
            self.state_sk = np.zeros_like(state)
            self.state_stdev = np.zeros_like(state)
            self.action_mean = action.copy()
            self.action_sk = np.zeros_like(action)
            self.action_stdev = np.zeros_like(action)
            self.state_delta_mean = state_delta.copy()
            self.state_delta_sk = np.zeros_like(state_delta)
            self.state_delta_stdev = np.zeros_like(state_delta)
            return

        state_mean_old = self.state_mean.copy()
        action_mean_old = self.action_mean.copy()
        state_delta_mean_old = self.state_delta_mean.copy()

        self.state_mean = self.update_mean(self.state_mean, state, self.count)
        self.action_mean = self.update_mean(self.action_mean, action, self.count)
        self.state_delta_mean = self.update_mean(
            self.state_delta_mean, state_delta, self.count
        )

        self.state_sk = self.update_sk(
            self.state_sk, state_mean_old, self.state_mean, state
        )
        self.action_sk = self.update_sk(
            self.action_sk, action_mean_old, self.action_mean, action
        )
        self.state_delta_sk = self.update_sk(
            self.state_delta_sk,
            state_delta_mean_old,
            self.state_delta_mean,
            state_delta,
        )

        self.state_stdev = np.sqrt(self.state_sk / self.count)
        self.action_stdev = np.sqrt(self.action_sk / self.count)
        self.state_delta_stdev = np.sqrt(self.state_delta_sk / self.count)

    @staticmethod
    def setup_vars(x, mean, stdev):
        mean, stdev = mean.copy(), stdev.copy()
        mean = torch.from_numpy(mean).float().to(x.device)
        stdev = torch.from_numpy(stdev).float().to(x.device)
        return mean, stdev

    def _normalize(self, x, mean, stdev):
        mean, stdev = self.setup_vars(x, mean, stdev)
        n = x - mean
        n = n / torch.clamp(stdev, min=1e-8)
        return n

    def normalize_states(self, states):
        return self._normalize(states, self.state_mean, self.state_stdev)

    def normalize_actions(self, actions):
        return self._normalize(actions, self.action_mean, self.action_stdev)

    def normalize_state_deltas(self, state_deltas):
        return self._normalize(
            state_deltas, self.state_delta_mean, self.state_delta_stdev
        )

    def denormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(
            state_deltas_means, self.state_delta_mean, self.state_delta_stdev
        )
        return state_deltas_means * stdev + mean

    def denormalize_state_delta_vars(self, state_delta_vars):
        _, stdev = self.setup_vars(
            state_delta_vars, self.state_delta_mean, self.state_delta_stdev
        )
        return state_delta_vars * (stdev ** 2)

    def renormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(
            state_deltas_means, self.state_delta_mean, self.state_delta_stdev
        )
        return (state_deltas_means - mean) / torch.clamp(stdev, min=1e-8)

    def renormalize_state_delta_vars(self, state_delta_vars):
        _, stdev = self.setup_vars(
            state_delta_vars, self.state_delta_mean, self.state_delta_stdev
        )
        return state_delta_vars / (torch.clamp(stdev, min=1e-8) ** 2)

