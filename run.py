# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from pmbrl import Experiment, tools

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MountainCarArgs:
    logdir = "log-mountain"
    env_name = "SparseMountainCar"
    record_states = False
    max_episode_len = 500
    action_repeat = 1
    ensemble_size = 20
    hidden_size = 200
    reward_size = 200
    buffer_size = 10 ** 6

    reward_prior = 1.0 * action_repeat

    optimisation_iters = 5
    plan_horizon = 20
    n_candidates = 500
    top_candidates = 50
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


class CartPoleArgs:
    logdir = "log-cartpole"
    env_name = "SparseCartpoleSwingup"
    record_states = False
    max_episode_len = 500
    action_repeat = 2
    ensemble_size = 20
    hidden_size = 200
    reward_size = 200
    buffer_size = 10 ** 6

    reward_prior = 1.0 * action_repeat

    optimisation_iters = 5
    plan_horizon = 20
    n_candidates = 500
    top_candidates = 50
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
    n_episodes = 200
    log_every = 10
    save_every = 25
    verbosity = True
    device = DEVICE


class HalfCheetahArgs:
    logdir = "log-cheetah"
    env_name = "SparseHalfCheetah"
    record_states = True
    max_episode_len = 500
    action_repeat = 2
    ensemble_size = 20
    hidden_size = 200
    reward_size = 200
    buffer_size = 10 ** 6

    reward_prior = 1.0 * action_repeat

    optimisation_iters = 5
    plan_horizon = 20
    n_candidates = 500
    top_candidates = 50
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
    n_episodes = 200
    log_every = 10
    save_every = 25
    verbosity = True
    device = DEVICE


if __name__ == "__main__":

    N_SEEDS = 5

    for seed in range(N_SEEDS):
        tools.log("\n RUNNING HALF CHEETAH SEED {}".format(seed))
        args = HalfCheetahArgs()
        args.logdir = "log-cheetah-{}".format(seed)
        exp = Experiment(args, seed)
        exp.run()

        tools.log("\n RUNNING MOUNTAIN CAR SEED {}".format(seed))
        args = CartPoleArgs()
        args.logdir = "log-mountain-{}".format(seed)
        exp = Experiment(args, seed)
        exp.run()

        tools.log("\n RUNNING CARTPOLE SEED {}".format(seed))
        args = MountainCarArgs()
        args.logdir = "log-cartpole-{}".format(seed)
        exp = Experiment(args, seed)
        exp.run()

