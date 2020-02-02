import pprint


def get_config(args):
    if args.config_name == "mountain_car":
        config = MountainCarConfig()
    elif args.config_name == "cup_catch":
        config = CupCatchConfig()
    elif args.config_name == "debug":
        config = DebugConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))

    config.set_logdir(args.logdir)
    config.set_seed(args.seed)
    return config


class Config(object):
    def __init__(self):
        self.logdir = "logs"
        self.seed = 0
        self.log_every = 20
        self.traj_eval_steps = 12
        self.plot_trajectory = True

        self.env_name = None
        self.max_episode_len = 500
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 5
        self.hidden_size = 200
        self.max_logvar = -1
        self.min_logvar = -5
        self.act_fn = "swish"
        self.rollout_delta_clamp = 20

        self.n_episodes = 10
        self.n_seed_episodes = 1
        self.n_train_epochs = 50
        self.signal_noise = None
        self.batch_size = 50
        self.learning_rate = 1e-3
        self.epsilon = 1e-4
        self.grad_clip_norm = 100

        self.plan_horizon = 10
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False
        self.use_kl_div = False
        self.expl_scale = 0.1
        self.reward_scale = 1.0
        self.reward_prior = 1.0

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_seed(self, seed):
        self.seed = seed

    def __repr__(self):
        return pprint.pformat(vars(self))


class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.log_every = 5
        self.env_name = "Pendulum-v0"
        self.max_episode_len = 100

        self.ensemble_size = 3
        self.hidden_size = 100

        self.n_episodes = 5
        self.n_seed_episodes = 1
        self.n_train_epochs = 5
        self.batch_size = 5

        self.plan_horizon = 3
        self.optimisation_iters = 5
        self.n_candidates = 100
        self.top_candidates = 10


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mountain_car"
        self.seed = 0
        self.log_every = 20
        self.traj_eval_steps = 20
        self.plot_trajectory = False

        self.env_name = "SparseMountainCar"
        self.max_episode_len = 500
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 20
        self.hidden_size = 200
        self.max_logvar = -1
        self.min_logvar = -5
        self.rollout_delta_clamp = 20

        self.n_episodes = 50
        self.n_seed_episodes = 1
        self.n_train_epochs = 5
        self.batch_size = 32

        self.plan_horizon = 20
        self.optimisation_iters = 10
        self.n_candidates = 1000
        self.top_candidates = 100

        self.expl_scale = 0.1


class CupCatchConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "catch"
        self.seed = 0
        self.log_every = 50
        self.traj_eval_steps = 12
        self.plot_trajectory = False

        self.env_name = "DeepMindCatch"
        self.max_episode_len = 1000
        self.action_repeat = 4

        self.ensemble_size = 10
        self.hidden_size = 200
        self.max_logvar = -1
        self.min_logvar = -5
        self.rollout_delta_clamp = 20

        self.n_episodes = 50
        self.n_seed_episodes = 1
        self.n_train_epochs = 5

        self.plan_horizon = 12
        self.optimisation_iters = 10
        self.n_candidates = 1000
        self.top_candidates = 100

        self.use_reward = True
        self.use_exploration = True
