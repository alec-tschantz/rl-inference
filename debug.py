# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from pmbrl import Experiment

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Args:
    logdir = "log-test"
    env_name = "SparseCartpoleSwingup"
    record_states = False
    max_episode_len = 100
    action_repeat = 1
    ensemble_size = 5
    hidden_size = 32
    reward_size = 32
    buffer_size = 10 ** 5

    reward_prior = 1.0 * action_repeat

    optimisation_iters = 2
    plan_horizon = 5
    n_candidates = 50
    top_candidates = 10
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
    n_episodes = 10
    log_every = 10
    save_every = 5
    verbosity = True
    device = DEVICE


if __name__ == "__main__":
    args = Args()
    exp = Experiment(args)
    exp.run()
