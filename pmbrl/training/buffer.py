# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np


class Buffer(object):
    def __init__(
        self,
        state_size,
        action_size,
        ensemble_size,
        normalizer,
        buffer_size=10 ** 6,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.device = device

        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.state_deltas = np.zeros((buffer_size, state_size))

        self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, action, reward, next_state):
        idx = self._total_steps % self.buffer_size
        state_delta = next_state - state

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_deltas[idx] = state_delta
        self._total_steps += 1

        self.normalizer.update(state, action, state_delta, reward)

    def get_train_batches(self, batch_size):
        size = len(self)
        indices = [
            np.random.permutation(range(size)) for _ in range(self.ensemble_size)
        ]
        indices = np.stack(indices).T

        for i in range(0, size, batch_size):
            j = min(size, i + batch_size)

            if (j - i) < batch_size and i != 0:
                return

            batch_size = j - i

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)

            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(
                self.ensemble_size, batch_size, self.state_size
            )

            yield states, actions, rewards, state_deltas

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def save_data(self, filepath):
        np.savez_compressed(
            filepath,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            state_deltas=self.state_deltas,
            total_steps=self._total_steps,
        )

    def load_data(self, filepath):
        data = np.load(filepath)
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.state_deltas = data["state_deltas"]
        self._total_steps = int(data["total_steps"])

    def __len__(self):
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        return self._total_steps
