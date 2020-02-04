import pprint

MOUNTAIN_CAR_CONFIG = "mountain_car"
CUP_CATCH_CONFIG = "cup_catch"
DEBUG_CONFIG = "debug"


def get_config(args):
    if args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == CUP_CATCH_CONFIG:
        config = CupCatchConfig()
    elif args.config_name == DEBUG_CONFIG:
        config = DebugConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))
    config.set_seed(args.seed)
    return config


class Config(object):
    def __init__(self):
        self.logdir = "log"
        self.seed = 0
        self.traj_eval_steps = 30
        self.plot_trajectory = False

        self.env_name = None
        self.max_episode_len = 500
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 10
        self.hidden_size = 200
        self.max_logvar = -1
        self.min_logvar = -5
        self.act_fn = "swish"
        self.rollout_clamp = None

        self.n_episodes = 40
        self.n_seed_episodes = 5
        self.n_train_epochs = 5
        self.signal_noise = None
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epsilon = 1e-4
        self.grad_clip_norm = 1000

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False
        self.use_kl_div = False
        self.expl_scale = 1.0
        self.reward_scale = 1.0
        self.reward_prior = 1.0

    def set_seed(self, seed):
        self.seed = seed

    def __repr__(self):
        return pprint.pformat(vars(self))


class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Pendulum-v0"
        self.max_episode_len = 100
        self.plot_trajectory = True

        self.ensemble_size = 5
        self.hidden_size = 64

        self.n_episodes = 10
        self.n_seed_episodes = 1
        self.n_train_epochs = 5
        self.batch_size = 32

        self.plan_horizon = 5
        self.optimisation_iters = 5
        self.n_candidates = 100
        self.top_candidates = 10


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mountain_car"
        self.env_name = "SparseMountainCar"
        self.max_episode_len = 500


class CupCatchConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "catch"

        self.env_name = "DeepMindCatch"
        self.max_episode_len = 1000
        self.action_repeat = 4

        self.n_episodes = 50
        self.n_seed_episodes = 1
        self.n_train_epochs = 5

        self.plan_horizon = 12

