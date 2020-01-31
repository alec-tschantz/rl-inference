from dm_control import suite  # pylint: disable=import-error
import gym.spaces as spaces
import numpy as np


class DeepMindWrapper(object):
    def __init__(self, domain, task, camera_id="side", max_step=-1):
        self.domain = domain
        self.task = task
        self.camera_id = camera_id
        self.max_step = max_step
        self.env = suite.load(domain_name=domain, task_name=task)
        self.control_min = self.env.action_spec().minimum[0]
        self.control_max = self.env.action_spec().maximum[0]
        self.control_shape = self.env.action_spec().shape
        self._action_space = spaces.Box(
            self.control_min, self.control_max, self.control_shape
        )
        total_size = 0
        for _, j in self.env.observation_spec().items():
            total_size += j.shape[0] if len(j.shape) > 0 else 1
        self._observation_space = spaces.Box(-np.inf, np.inf, (total_size,))
        self.step_count = 0
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 67,
        }

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def physics(self):
        return self.env.physics

    def reset(self):
        obs = self.env.reset().observation
        l = []
        for _, j in obs.items():
            l.append(j if len(j.shape) > 0 else j.reshape(1))
        return np.concatenate(l)

    def step(self, action):
        ret = self.env.step(action)
        l = []
        for _, j in ret.observation.items():
            l.append(j if len(j.shape) > 0 else j.reshape(1))
        state = np.concatenate(l)
        reward = ret.reward
        done = (ret.step_type == 2) or (self.step_count == self.max_step)
        info = {}
        self.step_count += 1
        if done:
            self.step_count = 0
        return state, reward, done, info

    def seed(self, seed):
        self.env = suite.load(
            domain_name=self.domain, task_name=self.task, task_kwargs={"random": seed}
        )

    def close(self):
        pass
