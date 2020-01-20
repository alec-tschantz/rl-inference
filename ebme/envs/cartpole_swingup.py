import numpy as np
import gym, gym.spaces, gym.utils, gym.utils.seeding
import roboschool
from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv


class SparseCartpoleSwingup(RoboschoolMujocoXmlEnv):
    def __init__(self, swingup=True):
        self.swingup = swingup
        RoboschoolMujocoXmlEnv.__init__(
            self, "inverted_pendulum.xml", "cart", action_dim=1, obs_dim=5
        )
        self.threshold = 0.8

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=9.8, timestep=0.0165, frame_skip=1)

    def robot_specific_reset(self):
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        u = self.np_random.uniform(low=-0.1, high=0.1)
        self.j1.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert np.isfinite(a).all()
        self.slider.set_motor_torque(100 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        self.theta, theta_dot = self.j1.current_position()
        x, vx = self.slider.current_position()
        assert np.isfinite(x)
        return np.array([x, vx, np.cos(self.theta), np.sin(self.theta), theta_dot])

    def step(self, a):
        self.apply_action(a)
        self.scene.global_step()
        state = self.calc_state()
        reward = 0.0
        if self.swingup:
            if np.cos(self.theta) > self.threshold:
                reward = 1.0
            done = False
        else:
            reward = 1.0
            done = np.abs(self.theta) > 0.2
        self.rewards = [float(reward)]
        self.frame += 1
        self.done += done
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0.0, 1.2, 1.0, 0.0, 0, 0.5)