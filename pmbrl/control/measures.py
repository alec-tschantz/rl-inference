# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np
from scipy.special import psi, gamma


class InformationGain(object):
    def __init__(self, model, normalized=True):
        self.model = model
        self.normalized = True

    def __call__(self, delta_means, delta_vars):
        """
        delta_means   (plan_horizon, ensemble_size, n_candidates, n_dim)
        delta_vars    (plan_horizon, ensemble_size, n_candidates, n_dim)
        """

        plan_horizon = delta_means.size(0)
        n_candidates = delta_means.size(2)

        if self.normalized:
            delta_means = self.model.normalizer.renormalize_state_delta_means(
                delta_means
            )
            delta_vars = self.model.normalizer.renormalize_state_delta_vars(delta_vars)

        delta_states = self.model.sample(delta_means, delta_vars)
        info_gains = (
            torch.zeros(plan_horizon, n_candidates).float().to(delta_means.device)
        )

        for t in range(plan_horizon):
            ent_avg = self.entropy_of_average(delta_states[t])
            avg_ent = self.average_of_entropy(delta_vars[t])
            info_gains[t, :] = ent_avg - avg_ent

        return info_gains

    def entropy_of_average(self, samples):
        """
        samples (ensemble_size, n_candidates, n_dim) 
        """
        samples = samples.permute(1, 0, 2)
        n_samples = samples.size(1)
        dims = samples.size(2)
        k = 3

        distances_yy = self.batched_cdist_l2(samples, samples)
        y, _ = torch.sort(distances_yy, dim=1)
        v = self.volume_of_the_unit_ball(dims)
        h = (
            np.log(n_samples - 1)
            - psi(k)
            + np.log(v)
            + dims * torch.sum(torch.log(y[:, k - 1]), dim=1) / n_samples
            + 0.5
        )
        return h

    def batched_cdist_l2(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = (
            torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2)
            .add_(x1_norm)
            .clamp_min_(1e-30)
            .sqrt_()
        )
        return res

    def volume_of_the_unit_ball(self, dim):
        return np.pi ** (dim / 2) / gamma(dim / 2 + 1)

    def average_of_entropy(self, delta_vars):
        return torch.mean(self.gaussian_diagonal_entropy(delta_vars), dim=0)

    def gaussian_diagonal_entropy(self, delta_vars):
        min_variance = 1e-8
        return 0.5 * torch.sum(
            torch.log(2 * np.pi * np.e * torch.clamp(delta_vars, min=min_variance)),
            dim=len(delta_vars.size()) - 1,
        )
