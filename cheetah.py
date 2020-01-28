# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from pmbrl import Experiment

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Args:
    logdir = "log-cheetah"
    env_name = "SparseHalfCheetah"
    record_states = True
    max_episode_len = 100
    action_repeat = 3
    ensemble_size = 20
    hidden_size = 256
    reward_size = 256
    buffer_size = 10 ** 6

    reward_prior = 1.0 * action_repeat

    optimisation_iters = 7
    plan_horizon = 13
    n_candidates = 700
    top_candidates = 70
    use_exploration = True
    use_reward = True
    use_mean = False
    expl_scale = 0.1
    reward_scale = 1.0

    learning_rate = 1e-3
    epsilon = 1e-4
    n_train_epochs = 5
    batch_size = 32
    grad_clip_norm = 1000
    n_seed_episodes = 1
    n_episodes = 100
    log_every = 10
    save_every = 25
    verbosity = True
    device = DEVICE


if __name__ == "__main__":
    args = Args()
    exp = Experiment(args)
    exp.run()