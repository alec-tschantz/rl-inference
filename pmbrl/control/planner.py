# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn

from pmbrl.control.measures import InformationGain


class Planner(nn.Module):
    def __init__(
        self,
        ensemble,
        reward_model,
        action_size,
        ensemble_size,
        plan_horizon,
        optimisation_iters,
        n_candidates,
        top_candidates,
        use_reward=True,
        use_exploration=True,
        use_reward_ensemble=False,
        use_mean=False,
        use_kl_div=False,
        use_normalized=True,
        expl_scale=1.0,
        reward_scale=1.0,
        reward_prior=1.0,
        rollout_clamp=None,
        log_stats=False,
        device="cpu",
    ):
        super().__init__()
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.action_size = action_size
        self.ensemble_size = ensemble_size

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates

        self.use_reward = use_reward
        self.use_exploration = use_exploration
        self.use_reward_ensemble = use_reward_ensemble
        self.use_mean = use_mean
        self.use_kl_div = use_kl_div
        self.use_normalized = use_normalized
        self.expl_scale = expl_scale
        self.reward_scale = reward_scale
        self.reward_prior = reward_prior
        self.rollout_clamp = rollout_clamp
        self.log_stats = log_stats
        self.device = device

        self.prior = (
            torch.ones((self.plan_horizon * self.ensemble_size * self.n_candidates))
            .float()
            .to(self.device)
        ) * self.reward_prior

        self.measure = InformationGain(self.ensemble, normalized=use_normalized)
        self.kl_loss = torch.nn.MSELoss(reduction="none")
        self.rewards_trial = []
        self.bonuses_trial = []
        self.to(device)

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        state_size = state.size(0)

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        for _ in range(self.optimisation_iters):
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )
            states, delta_vars, delta_means = self.perform_rollout(state, actions)

            returns = torch.zeros(self.n_candidates).float().to(self.device)
            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
                expl_bonus = expl_bonus.sum(dim=0)
                returns += expl_bonus
                if self.log_stats:
                    self.bonuses_trial.append(expl_bonus)

            if self.use_reward:
                if self.use_reward_ensemble:
                    states = states.permute(1, 0, 2, 3)
                    states = states.contiguous().view(
                        self.ensemble_size, -1, state_size
                    )
                else:
                    states = states.view(-1, state_size)
                rewards = self.reward_model(states)

                if self.use_kl_div:
                    rewards = (
                        -(self.kl_loss(rewards, self.prior) / 2.0) * self.reward_scale
                    )
                else:
                    rewards = rewards * self.reward_scale

                rewards = rewards.view(
                    self.plan_horizon, self.ensemble_size, self.n_candidates
                )
                rewards = rewards.mean(dim=1).sum(dim=0)
                returns += rewards

                if self.log_stats:
                    self.rewards_trial.append(rewards)

            action_mean, action_std_dev = self._fit_gaussian(actions, returns)

        return action_mean[0].squeeze(dim=0)

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T

        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1)
        states[0] = current_state

        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], actions[t])
            if self.rollout_clamp is not None:
                delta_mean = delta_mean.clamp(
                    -self.rollout_clamp,  # pylint: disable=invalid-unary-operand-type
                    self.rollout_clamp,
                )
            if self.use_mean:
                states[t + 1] = states[t] + delta_mean
            else:
                states[t + 1] = states[t] + self.ensemble.sample(delta_mean, delta_var)
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means

    def return_stats(self):
        if self.log_stats:
            if self.use_exploration:
                info_stats = self._create_stats(self.bonuses_trial)
            else:
                info_stats = {"max": 0, "min": 0, "mean": 0, "std": 0}
            if self.use_reward:
                reward_stats = self._create_stats(self.rewards_trial)
            else:
                reward_stats = {"max": 0, "min": 0, "mean": 0, "std": 0}
        else:
            reward_stats = {"max": 0, "min": 0, "mean": 0, "std": 0}
            info_stats = {"max": 0, "min": 0, "mean": 0, "std": 0}
        self.rewards_trial = []
        self.bonuses_trial = []
        return reward_stats, info_stats

    def set_use_reward(self, use_reward):
        self.use_reward = use_reward

    def set_use_exploration(self, use_exploration):
        self.use_exploration = use_exploration

    def _fit_gaussian(self, actions, returns):
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.action_size
        )
        action_mean, action_std_dev = (
            best_actions.mean(dim=1, keepdim=True),
            best_actions.std(dim=1, unbiased=False, keepdim=True),
        )
        return action_mean, action_std_dev

    def _create_stats(self, arr):
        tensor = torch.stack(arr)
        tensor = tensor.view(-1)

        return {
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }

