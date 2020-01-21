""" 
scp -r pmbrl at449@allocortex.inf.susx.ac.uk:/its/home/at449/ 
scp -r at449@allocortex.inf.susx.ac.uk:/its/home/at449/pmbrl pmbrl
nohup python main.py &
ps -ef | grep "main.py"
kill PID 
"""
# pylint: disable=not-callable
# pylint: disable=no-member

import gym
import roboschool
import torch
import numpy as np
import argparse
import time

from pmbrl.env import GymEnv, NoisyEnv
from pmbrl.normalizer import TransitionNormalizer
from pmbrl.buffer import Buffer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.planner import Planner
from pmbrl.agent import Agent
from pmbrl import tools


def main(args):
    tools.log(" === Loading experiment ===")
    tools.log(args)

    env = GymEnv(args.env_name, args.max_episode_len, action_repeat=args.action_repeat)
    state_size = env.state_dims[0]
    action_size = env.action_dims[0]

    if args.env_std > 0.0:
        env = NoisyEnv(env, args.env_std)

    normalizer = TransitionNormalizer()
    buffer = Buffer(
        state_size,
        action_size,
        args.ensemble_size,
        normalizer,
        buffer_size=args.buffer_size,
        device=DEVICE,
    )

    ensemble = EnsembleModel(
        state_size + action_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        normalizer,
        device=DEVICE,
    ).to(DEVICE)
    reward_model = RewardModel(state_size, args.hidden_size).to(DEVICE)
    params = list(ensemble.parameters()) + list(reward_model.parameters())
    optim = torch.optim.Adam(params, lr=args.learning_rate, eps=args.epsilon)

    planner = Planner(
        ensemble,
        reward_model,
        action_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        use_exploration=args.use_exploration,
        use_reward=args.use_reward,
        expl_scale=args.expl_scale,
        device=DEVICE,
    ).to(DEVICE)
    agent = Agent(env, planner)

    if tools.logdir_exists(args.logdir):
        tools.log("Loading existing _logdir_ at {}".format(args.logdir))
        normalizer = tools.load_normalizer(args.logdir)
        buffer = tools.load_buffer(args.logdir, buffer)
        buffer.set_normalizer(normalizer)
        metrics = tools.load_metrics(args.logdir)
        model_dict = tools.load_model_dict(args.logdir, metrics["last_save"])
        ensemble.load_state_dict(model_dict["ensemble"])
        ensemble.set_normalizer(normalizer)
        reward_model.load_state_dict(model_dict["reward"])
        optim.load_state_dict(model_dict["optim"])
    else:
        tools.init_dirs(args.logdir)
        metrics = tools.build_metrics()
        buffer = agent.get_seed_episodes(buffer, args.n_seed_episodes)
        message = "Collected seeds: [{} episodes] [{} frames]"
        tools.log(message.format(args.n_seed_episodes, buffer.total_steps))

    for episode in range(metrics["episode"], args.n_episodes):
        tools.log("\n === Episode {} ===".format(episode))
        start_time_episode = time.process_time()
        start_time_training = time.process_time()

        tools.log("Training on {} data points".format(buffer.total_steps))
        for epoch in range(args.n_train_epochs):
            e_losses = []
            r_losses = []

            for (states, actions, rewards, delta_states) in buffer.get_train_batches(
                args.batch_size
            ):
                ensemble.train()
                reward_model.train()
                optim.zero_grad()

                e_loss = ensemble.loss(states, actions, delta_states)
                r_loss = reward_model.loss(states, rewards)
                e_losses.append(e_loss.item())
                r_losses.append(r_loss.item())
                (e_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm, norm_type=2)
                optim.step()

            if epoch % args.log_every == 0 and epoch > 0:
                message = "> Epoch {} [ ensemble {:.2f} | rew {:.2f}]"
                tools.log(
                    message.format(epoch, sum(e_losses) / epoch, sum(r_losses) / epoch)
                )
        metrics["ensemble_loss"].append(sum(e_losses))
        metrics["reward_loss"].append(sum(r_losses))
        message = "Losses: [ensemble {} | reward {}]"
        tools.log(message.format(sum(e_losses), sum(r_losses)))
        end_time_training = time.process_time() - start_time_training
        tools.log("Total training time: {:.2f}".format(end_time_training))

        if args.do_noise_exploration:
            start_time_expl = time.process_time()
            expl_reward, expl_steps, buffer, stats = agent.run_episode(
                buffer=buffer, action_noise=args.action_noise
            )
            metrics["train_rewards"].append(expl_reward)
            metrics["train_steps"].append(expl_steps)
            message = "Exploration: [reward {:.2f} | steps {:.2f} ]"
            tools.log(message.format(expl_reward, expl_steps))
            end_time_expl = time.process_time() - start_time_expl
            tools.log("Total exploration time: {:.2f}".format(end_time_expl))
            info_stats, reward_stats = stats
            print("Info stats: \n {}".format(info_stats))
            print("Reward stats: \n {}".format(reward_stats))

        start_time = time.process_time()
        reward, steps, buffer, stats = agent.run_episode(buffer=buffer)
        metrics["test_rewards"].append(reward)
        metrics["test_steps"].append(steps)
        message = "Exploitation: [reward {:.2f} | steps {:.2f} ]"
        tools.log(message.format(reward, steps))
        end_time = time.process_time() - start_time
        tools.log("Total exploitation time: {:.2f}".format(end_time))
        info_stats, reward_stats = stats
        print("Info stats: \n {}".format(info_stats))
        print("Reward stats: \n {}".format(reward_stats))

        end_time_episode = time.process_time() - start_time_episode
        tools.log("Total episode time: {:.2f}".format(end_time_episode))
        metrics["episode"] += 1
        metrics["total_steps"].append(buffer.total_steps)
        metrics["episode_time"].append(end_time_episode)

        if episode % args.save_every == 0:
            metrics["episode"] += 1
            metrics["last_save"] = episode
            tools.save_model(args.logdir, ensemble, reward_model, optim, episode)
            tools.save_normalizer(args.logdir, normalizer)
            tools.save_buffer(args.logdir, buffer)
            tools.save_metrics(args.logdir, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--logdir", type=str, default="log-cheetah")
    parser.add_argument("--env_name", type=str, default="RoboschoolHalfCheetah-v1")
    parser.add_argument("--max_episode_len", type=int, default=5000)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--env_std", type=float, default=0.01)
    parser.add_argument("--action_noise", type=float, default=0.3)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int, default=30)
    parser.add_argument("--n_candidates", type=int, default=500)
    parser.add_argument("--optimisation_iters", type=int, default=5)
    parser.add_argument("--top_candidates", type=int, default=50)
    parser.add_argument("--n_seed_episodes", type=int, default=5)
    parser.add_argument("--n_train_epochs", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--use_reward", type=bool, default=True)
    parser.add_argument("--use_exploration", type=bool, default=True)
    parser.add_argument("--do_noise_exploration", type=bool, default=False)
    parser.add_argument("--expl_scale", type=int, default=1)

    args = parser.parse_args()
    main(args)

    """
    parser.add_argument("--logdir", type=str, default="log-cheetah")
    parser.add_argument("--env_name", type=str, default="RoboschoolHalfCheetah-v1")
    parser.add_argument("--max_episode_len", type=int, default=10)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--env_std", type=float, default=0.02)
    parser.add_argument("--action_noise", type=float, default=0.3)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int, default=2)
    parser.add_argument("--n_candidates", type=int, default=20)
    parser.add_argument("--optimisation_iters", type=int, default=2)
    parser.add_argument("--top_candidates", type=int, default=10)
    parser.add_argument("--n_seed_episodes", type=int, default=5)
    parser.add_argument("--n_train_epochs", type=int, default=10)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--use_reward", type=bool, default=True)
    parser.add_argument("--use_exploration", type=bool, default=False)
    parser.add_argument("--expl_scale", type=int, default=1)
    """
