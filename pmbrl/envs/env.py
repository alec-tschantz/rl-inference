import gym
import roboschool

SPARSE_MOUNTAIN_CAR = "SparseMountainCar"
SPARSE_CARTPOLE_SWINGUP = "SparseCartpoleSwingup"
SPARSE_HALF_CHEETAH = "SparseHalfCheetah"
SPARSE_ANT = "SparseAnt"
DM_CARTPOLE = "DeepMindCartpole"
DM_PENDULUM = "DeepMindPendulum"
DM_CATCH = "DeepMindCatch"
DM_CHEETAH = "DeepMindCheetah"


class GymEnv(object):
    def __init__(self, env_name, max_episode_len, action_repeat=1, seed=None):
        self._env = self._get_env_object(env_name)
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.done = False
        if seed is not None:
            self._env.seed(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        state = self._env.reset()
        self.done = False
        return state

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break
        return state, reward, done, info

    def sample_action(self):
        return self._env.action_space.sample()

    def render(self, mode="human"):
        self._env.render(mode)

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def _get_env_object(self, env_name):
        if env_name == SPARSE_MOUNTAIN_CAR:
            from pmbrl.envs.envs.mountain_car import SparseMountainCarEnv

            return SparseMountainCarEnv()
        elif env_name == SPARSE_CARTPOLE_SWINGUP:
            from pmbrl.envs.envs.cartpole import SparseCartpoleSwingupEnv

            return SparseCartpoleSwingupEnv()
        elif env_name == SPARSE_HALF_CHEETAH:
            from pmbrl.envs.envs.half_cheetah import SparseHalfCheetahEnv

            return SparseHalfCheetahEnv()
        elif env_name == SPARSE_ANT:
            from pmbrl.envs.envs.ant import SparseAntEnv

            return SparseAntEnv()
        elif env_name == DM_CARTPOLE:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            domain = "cartpole"
            task = "swingup_sparse"
            return DeepMindWrapper(domain=domain, task=task)
        elif env_name == DM_PENDULUM:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            domain = "pendulum"
            task = "swingup"
            return DeepMindWrapper(domain=domain, task=task)
        elif env_name == DM_CATCH:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            domain = "ball_in_cup"
            task = "catch"
            return DeepMindWrapper(domain=domain, task=task)
        elif env_name == DM_CHEETAH:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            domain = "cheetah"
            task = "run"
            return DeepMindWrapper(domain=domain, task=task)

        else:
            return gym.make(env_name)
