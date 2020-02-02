import pprint


class DebugConfig(object):
    def __init__(self):
        self.logdir = "logs"
        self.experiment_id = "test"
        self.log_every = 5
        self.traj_eval_steps = 10

        self.env_name = "Pendulum-v0"
        self.max_episode_len = 20
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 3
        self.hidden_size = 50
        self.max_logvar = -1
        self.min_logvar = -5
        self.act_fn = "swish"
        self.rollout_delta_clamp = 5

        self.n_episodes = 10
        self.n_seed_episodes = 1
        self.n_train_epochs = 10
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.epsilon = 1e-4
        self.grad_clip_norm = 100

        self.plan_horizon = 5
        self.optimisation_iters = 2
        self.n_candidates = 100
        self.top_candidates = 10

        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False
        self.use_kl_div = False
        self.expl_scale = 0.1
        self.reward_scale = 1.0
        self.reward_prior = 1.0

    def __repr__(self):
        return pprint.pformat(vars(self))


class MountainCarConfig(object):
    def __init__(self):
        self.logdir = "logs"
        self.experiment_id = "mountain_car"
        self.log_every = 20
        self.traj_eval_steps = 50

        self.env_name = "SparseMountainCar"
        self.max_episode_len = 500
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 20
        self.hidden_size = 200
        self.max_logvar = -1
        self.min_logvar = -5
        self.act_fn = "swish"
        self.rollout_delta_clamp = 5

        self.n_episodes = 20
        self.n_seed_episodes = 1
        self.n_train_epochs = 5
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.epsilon = 1e-4
        self.grad_clip_norm = 100

        self.plan_horizon = 20
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

    def __repr__(self):
        return pprint.pformat(vars(self))
