import pprint

MOUNTAIN_CAR_CONFIG = "mountain_car"
CUP_CATCH_CONFIG = "cup_catch"
HALF_CHEETAH_RUN_CONFIG = "half_cheetah_run"
HALF_CHEETAH_FLIP_CONFIG = "half_cheetah_flip"
LUNAR_LANDER = "lunar_lander"
DEBUG_CONFIG = "debug"


def get_config(args):
    if args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == CUP_CATCH_CONFIG:
        config = CupCatchConfig()
    elif args.config_name == HALF_CHEETAH_RUN_CONFIG:
        config = HalfCheetahRunConfig()
    elif args.config_name == HALF_CHEETAH_FLIP_CONFIG:
        config = HalfCheetahFlipConfig()
    elif args.config_name == LUNAR_LANDER:
        config = LunarLanderConfig()
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
        self.n_episodes = 50
        self.n_seed_episodes = 5
        self.record_every = 1

        self.env_name = None
        self.max_episode_len = 500
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 10
        self.hidden_size = 200

        self.n_train_epochs = 100
        self.batch_size = 50
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.grad_clip_norm = 1000

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False

        self.expl_scale = 1.0
        self.reward_scale = 1.0

    def set_seed(self, seed):
        self.seed = seed

    def __repr__(self):
        return pprint.pformat(vars(self))


class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Pendulum-v0"
        self.n_episodes = 5
        self.max_episode_len = 100
        self.hidden_size = 64
        self.plan_horizon = 5


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mountain_car"
        self.env_name = "SparseMountainCar"
        self.max_episode_len = 500
        self.n_train_epochs = 20
        self.n_seed_episodes = 1


class CupCatchConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "catch"
        self.env_name = "DeepMindCatch"
        self.max_episode_len = 1000
        self.action_repeat = 4


class LunarLanderConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "lunar_lander"
        self.env_name = "LunarLander"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 1000
        self.action_repeat = 5

        self.ensemble_size = 5
        self.hidden_size = 200

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.use_exploration = False
        self.use_mean = True
        self.expl_scale = 0.1


class HalfCheetahRunConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "half_cheetah_run"
        self.env_name = "HalfCheetahRun"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1


class HalfCheetahFlipConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "half_cheetah_flip"
        self.env_name = "HalfCheetahFlip"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1
