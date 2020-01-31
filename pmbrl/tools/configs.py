import pprint


class DebugConfig(object):
    def __init__(self):
        self.logdir = "log"
        self.exp_name = None
        self.log_every = 20
        self.eval_steps = 20

        self.env_name = "SparseMountainCar"
        self.max_episode_len = 200
        self.action_repeat = 3
        self.action_noise = None

        self.ensemble_size = 5
        self.hidden_size = 100
        self.max_logvar = -1
        self.min_logvar = -5
        self.act_fn = "swish"

        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.n_train_epochs = 5
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.epsilon = 1e-4
        self.grad_clip_norm = 100

        self.plan_horizon = 7
        self.optimisation_iters = 5
        self.n_candidates = 200
        self.top_candidates = 20

        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False
        self.use_kl_div = False
        self.expl_scale = 1.0
        self.reward_scale = 1.0
        self.reward_prior = 1.0

    def __repr__(self):
        return pprint.pformat(vars(self))


class MountainCarConfig(object):
    def __init__(self):
        self.env_name = "SparseMountainCar"
        self.max_episode_len = 500
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 20
        self.hidden_size = 200
        self.max_logvar = -1
        self.min_logvar = -5
        self.act_fn = "swish"

        self.n_episodes = 10
        self.n_seed_episodes = 5
        self.n_train_epochs = 10
        self.batch_size = 20
        self.learning_rate = 1e-3
        self.epsilon = 1e-4
        self.grad_clip_norm = 100
        self.log_every = 20

        self.plan_horizon = 5
        self.optimisation_iters = 2
        self.n_candidates = 20
        self.top_candidates = 5

        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False
        self.use_kl_div = False
        self.expl_scale = 1.0
        self.reward_scale = 1.0
        self.reward_prior = 1.0

    def __repr__(self):
        return pprint.pformat(vars(self))
