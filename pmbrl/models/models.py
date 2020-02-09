# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def swish(x):
    return x * torch.sigmoid(x)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, act_fn="swish"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.act_fn_name = act_fn
        self.act_fn = self._get_act_fn(self.act_fn_name)
        self.reset_parameters()

    def forward(self, x):
        op = torch.baddbmm(self.biases, x, self.weights)
        op = self.act_fn(op)
        return op

    def reset_parameters(self):
        weights = torch.zeros(self.ensemble_size, self.in_size, self.out_size).float()
        biases = torch.zeros(self.ensemble_size, 1, self.out_size).float()

        for weight in weights:
            self._init_weight(weight, self.act_fn_name)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _init_weight(self, weight, act_fn_name):
        if act_fn_name == "swish":
            nn.init.xavier_uniform_(weight)
        elif act_fn_name == "linear":
            nn.init.xavier_normal_(weight)

    def _get_act_fn(self, act_fn_name):
        if act_fn_name == "swish":
            return swish
        elif act_fn_name == "linear":
            return lambda x: x


class EnsembleModel(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        ensemble_size,
        normalizer,
        act_fn="swish",
        device="cpu",
    ):
        super().__init__()

        self.fc_1 = EnsembleDenseLayer(
            in_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_2 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_3 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_4 = EnsembleDenseLayer(
            hidden_size, out_size * 2, ensemble_size, act_fn="linear"
        )

        self.ensemble_size = ensemble_size
        self.normalizer = normalizer
        self.device = device
        self.max_logvar = -1
        self.min_logvar = -5
        self.device = device
        self.to(device)

    def forward(self, states, actions):
        norm_states, norm_actions = self._pre_process_model_inputs(states, actions)
        norm_delta_mean, norm_delta_var = self._propagate_network(
            norm_states, norm_actions
        )
        delta_mean, delta_var = self._post_process_model_outputs(
            norm_delta_mean, norm_delta_var
        )
        return delta_mean, delta_var

    def loss(self, states, actions, state_deltas):
        states, actions = self._pre_process_model_inputs(states, actions)
        delta_targets = self._pre_process_model_targets(state_deltas)
        delta_mu, delta_var = self._propagate_network(states, actions)
        loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        loss = loss.mean(-1).mean(-1).sum()
        return loss

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.fc_1(inp)
        op = self.fc_2(op)
        op = self.fc_3(op)
        op = self.fc_4(op)

        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = (
            self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar
        )
        delta_var = torch.exp(delta_logvar)

        return delta_mean, delta_var

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)
        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, delta_var):
        delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
        delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var


class RewardModel(nn.Module):
    def __init__(self, in_size, hidden_size, act_fn="relu", device="cpu"):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.device = device
        self.act_fn = getattr(F, act_fn)
        self.reset_parameters()
        self.to(device)

    def forward(self, states, actions):
        inp = torch.cat((states, actions), dim=-1)
        reward = self.act_fn(self.fc_1(inp))
        reward = self.act_fn(self.fc_2(reward))
        reward = self.fc_3(reward).squeeze(dim=1)
        return reward

    def loss(self, states, actions, rewards):
        r_hat = self(states, actions)
        return F.mse_loss(r_hat, rewards)

    def reset_parameters(self):
        self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_3 = nn.Linear(self.hidden_size, 1)
        self.to(self.device)
