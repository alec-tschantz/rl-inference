# pylint: disable=not-callable
# pylint: disable=no-member

import os
import time
import gym
import roboschool
import torch
import numpy as np

from pmbrl.env import GymEnv
from pmbrl.normalizer import Normalizer
from pmbrl.buffer import Buffer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.planner import Planner
from pmbrl.agent import Agent
from pmbrl import tools


class Experiment(object):
    def __init__(self, args, seed=0):
        tools.log("=== Loading experiment ===")
        tools.log("Using: {}".format(args.device))
        tools.log(args.__dict__)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        os.makedirs(args.logdir, exist_ok=True)
        
        self._build_experiment(args)

    def run(self):
        self.collect_seed_episodes()
        for episode in range(self.args.n_episodes):
            self.run_episode(episode)
            if episode % self.args.save_every == 0:
                self.checkpoint(episode)

    def run_episode(self, episode):
        tools.log("\n=== Episode {} ===".format(episode))
        start_time = time.process_time()

        self.train_model()

        tools.log("=== Collecting data ===")
        reward, steps, _buffer, stats, infos = self.agent.run_episode(
            buffer=self.buffer, verbosity=self.args.verbosity
        )
        self.buffer = _buffer

        message = "Episode {}: [reward {:.2f} | steps {:.2f}]"
        tools.log(message.format(episode, reward, steps))

        episode_time = time.process_time() - start_time
        message = "Total episode time: {:.2f}"
        tools.log(message.format(episode_time))

        self.metrics["rewards"].append(reward)
        self.metrics["steps"].append(steps)
        self.metrics["episode_time"].append(episode_time)

        self._log_stats(stats, self.normalizer, infos)

    def checkpoint(self, episode):
        tools.save_model(
            self.args.logdir, self.ensemble, self.reward_model, self.optim, episode
        )
        tools.save_metrics(self.args.logdir, self.metrics, episode)
        tools.save_buffer(self.args.logdir, self.buffer, episode)

    def collect_seed_episodes(self):
        self.buffer = self.agent.get_seed_episodes(
            self.buffer, self.args.n_seed_episodes
        )
        message = "Collected {} seed episodes [{} frames]"
        tools.log(message.format(self.args.n_seed_episodes, self.buffer.total_steps))

    def train_model(self):
        message = "Training on {} data points"
        tools.log(message.format(self.buffer.total_steps))

        for epoch in range(self.args.n_train_epochs):
            e_losses = []
            r_losses = []
            for (
                states,
                actions,
                rewards,
                delta_states,
            ) in self.buffer.get_train_batches(self.args.batch_size):
                self.ensemble.train()
                self.reward_model.train()

                self.optim.zero_grad()
                e_loss = self.ensemble.loss(states, actions, delta_states)
                r_loss = self.reward_model.loss(states, rewards)
                e_losses.append(e_loss.item())
                r_losses.append(r_loss.item())
                (e_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.params, self.args.grad_clip_norm, norm_type=2
                )
                self.optim.step()

            if epoch > 0 and epoch % self.args.log_every == 0:
                message = "> Epoch {} [ensemble {:.2f} | reward {:.2f}]"
                tools.log(message.format(epoch, e_loss.item(), r_loss.item()))

        message = "Summed losses: [ensemble {:.2f} | reward {:.2f}]"
        tools.log(message.format(sum(e_losses), sum(r_losses)))
        self.metrics["ensemble_loss"].append(sum(e_losses))
        self.metrics["reward_loss"].append(sum(r_losses))

    def _build_experiment(self, args):
        self.args = args
        self.env = self._build_env(args)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.normalizer = self._build_normalizer(args)
        self.buffer = self._build_buffer(
            args, self.normalizer, self.state_size, self.action_size
        )
        self.ensemble = self._build_ensemble(
            args, self.normalizer, self.state_size, self.action_size
        )
        self.reward_model = self._build_reward_model(args, self.state_size)
        self.params = list(self.ensemble.parameters()) + list(
            self.reward_model.parameters()
        )
        self.optim = self._build_optim(args, self.params)
        self.planner = self._build_planner(
            args, self.ensemble, self.reward_model, self.action_size
        )
        self.agent = self._build_agent(self.env, self.planner)
        self.metrics = self._build_metrics()

    def _build_env(self, args):
        return GymEnv(
            args.env_name, args.max_episode_len, action_repeat=args.action_repeat
        )

    def _build_normalizer(self, args):
        return Normalizer()

    def _build_buffer(self, args, normalizer, state_size, action_size):
        return Buffer(
            state_size,
            action_size,
            args.ensemble_size,
            normalizer,
            buffer_size=args.buffer_size,
            device=args.device,
        )

    def _build_ensemble(self, args, normalizer, state_size, action_size):
        return EnsembleModel(
            state_size + action_size,
            state_size,
            args.hidden_size,
            args.ensemble_size,
            normalizer,
            device=args.device,
        )

    def _build_reward_model(self, args, state_size):
        return RewardModel(state_size, args.reward_size, device=args.device)

    def _build_optim(self, args, params):
        return torch.optim.Adam(params, lr=args.learning_rate, eps=args.epsilon)

    def _build_planner(self, args, ensemble, reward_model, action_size):
        return Planner(
            ensemble,
            reward_model,
            action_size,
            args.ensemble_size,
            plan_horizon=args.plan_horizon,
            optimisation_iters=args.optimisation_iters,
            n_candidates=args.n_candidates,
            top_candidates=args.top_candidates,
            reward_prior=args.reward_prior,
            use_reward=args.use_reward,
            use_exploration=args.use_exploration,
            use_mean=args.use_mean,
            expl_scale=args.expl_scale,
            reward_scale=args.reward_scale,
            device=args.device,
        )

    def _build_agent(self, env, planner):
        return Agent(env, planner)

    def _build_metrics(self):
        return {
            "ensemble_loss": [],
            "reward_loss": [],
            "rewards": [],
            "steps": [],
            "episode_time": [],
            "information_mean": [],
            "reward_mean": [],
            "state_info": [],
        }

    def _log_stats(self, stats, normalizer, infos):
        tools.log("=== Log ===")
        info_stats, reward_stats = stats
        message = (
            "Information gain: [max {:.2f} | min {:.2f} | mean {:.2f} | std {:.2f}]"
        )
        tools.log(
            message.format(
                info_stats["max"],
                info_stats["min"],
                info_stats["mean"],
                info_stats["std"],
            )
        )
        message = "Reward: [max {:.2f} | min {:.2f} | mean {:.2f} | std {:.2f}]"
        tools.log(
            message.format(
                reward_stats["max"],
                reward_stats["min"],
                reward_stats["mean"],
                reward_stats["std"],
            )
        )
        message = "Reward structure: [max {:.2f} | min {:.2f} | mean {:.2f}]"
        tools.log(
            message.format(
                normalizer.max_reward, normalizer.min_reward, normalizer.reward_mean
            )
        )
        self.metrics["information_mean"].append(info_stats["mean"])
        self.metrics["reward_mean"].append(reward_stats["mean"])

        if self.args.record_states:
            states = []
            for info in infos:
                states.append(info["x_pos"])
            states = np.array(states)
            message = "Cheetah info: [max {:.2f} | min {:.2f} | mean {:.2f}]"
            tools.log(message.format(states.max(), states.min(), states.mean()))
            self.metrics['state_info'].append(states.max())

