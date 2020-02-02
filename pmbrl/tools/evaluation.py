# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np


def evaluate_trajectory(
    ensemble, trajectory, actions, steps, device, rollout_delta_clamp=None
):
    trajectory = torch.from_numpy(trajectory).float().to(device)
    actions = torch.from_numpy(actions).float().to(device)

    pred_states = []
    pred_delta_vars = []

    traj_steps = trajectory.size(0)
    if traj_steps < steps:
        start_idx = 0
        end_idx = traj_steps
    else:
        start_idx = np.random.choice(range(0, traj_steps - steps))
        end_idx = start_idx + steps

    """ convert to (ensemble_size, batch_size, state_size) """
    state = trajectory[start_idx].unsqueeze(0).unsqueeze(0)
    state = state.repeat(ensemble.ensemble_size, 1, 1)

    for t in range(start_idx, end_idx - 1):
        action = actions[t].unsqueeze(0).unsqueeze(0)
        action = action.repeat(ensemble.ensemble_size, 1, 1)
        delta_mean, delta_var = ensemble(state, action)
        if rollout_delta_clamp is not None:
            delta_mean = delta_mean.clamp(
                -rollout_delta_clamp,  # pylint: disable=invalid-unary-operand-type
                rollout_delta_clamp,
            )
        state = state + ensemble.sample(delta_mean, delta_var)
        pred_states.append(state)
        pred_delta_vars.append(delta_var)

    pred_states = torch.stack(pred_states)
    pred_delta_vars = torch.stack(pred_delta_vars)

    """ convert to (T, ensemble_size, state_size) """
    pred_states = pred_states.squeeze(2)
    pred_delta_vars = pred_delta_vars.squeeze(2)

    pred_states = pred_states.cpu().detach().numpy()
    pred_delta_vars = pred_delta_vars.cpu().detach().numpy()
    trajectory = trajectory[start_idx : end_idx - 1].cpu().detach().numpy()

    return pred_states, pred_delta_vars, trajectory
