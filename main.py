# pylint: disable=not-callable
# pylint: disable=no-member

import gym
import roboschool
import torch
import numpy as np
import argparse
import time

from pmbrl.env import GymEnv, NoisyEnv
from pmbrl.normalizer import Normalizer
from pmbrl.buffer import Buffer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.planner import Planner
from pmbrl.agent import Agent
from pmbrl import tools
from pmbrl.envs import rate_buffer


def build_experiment(args):
    env = GymEnv(args.env_name, args.max_episode_len, action_repeat=args.action_repeat)
    state_size = env.state_dims[0]
    action_size = env.action_dims[0]

    if args.env_std > 0.0:
        env = NoisyEnv(env, args.env_std)

    norm = Normalizer()
    buffer = Buffer(
        state_size,
        action_size,
        args.ensemble_size,
        norm,
        buffer_size=args.buffer_size,
        device=DEVICE,
    )

    ensemble = EnsembleModel(
        state_size + action_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        norm,
        device=DEVICE,
    ).to(DEVICE)
    reward_model = RewardModel(state_size, args.hidden_size, norm).to(DEVICE)
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
        expl_scale=args.expl_scale,
        device=DEVICE,
    ).to(DEVICE)
    agent = Agent(env, planner, args.logdir)
    return (norm, buffer, ensemble, reward_model, params, optim, agent)


def init_experiment(args, norm, buffer, ensemble, reward_model, optim, agent):
    if tools.logdir_exists(args.logdir):
        tools.log("Loading existing _logdir_ at {}".format(args.logdir))
        norm = tools.load_normalizer(args.logdir)
        buffer = tools.load_buffer(args.logdir, buffer)
        buffer.set_normalizer(norm)
        metrics = tools.load_metrics(args.logdir)
        model_dict = tools.load_model_dict(args.logdir, metrics["last_save"])
        ensemble.load_state_dict(model_dict["ensemble"])
        ensemble.set_normalizer(norm)
        reward_model.set_normalizer(norm)
        reward_model.load_state_dict(model_dict["reward"])
        optim.load_state_dict(model_dict["optim"])
    else:
        tools.init_dirs(args.logdir)
        metrics = tools.build_metrics()
        buffer = agent.get_seed_episodes(buffer, args.n_seed_episodes)
        message = "Collected seeds: [{} episodes] [{} frames]"
        tools.log(message.format(args.n_seed_episodes, buffer.total_steps))
    return metrics


def train(args, buffer, ensemble, reward_model, optim, params, metrics):
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


def run_trial(
    agent, buffer, render=False, episode=None, use_exploration=True, use_reward=True
):

    reward, steps, buffer, stats = agent.run_episode(
        buffer=buffer,
        render=render,
        episode=episode,
        use_exploration=use_exploration,
        use_reward=use_reward,
    )
    message = ":::::: [reward {:.2f} | steps {:.2f} ] ::::::"
    tools.log(message.format(reward, steps))
    info_stats, reward_stats = stats
    tools.log("> Info stats: \n {}".format(info_stats))
    tools.log("> Reward stats: \n {}".format(reward_stats))
    return reward, steps


def main(args):
    tools.log(" === Loading experiment ===")
    tools.log(args)

    norm, buffer, ensemble, reward_model, params, optim, agent = build_experiment(args)
    metrics = init_experiment(args, norm, buffer, ensemble, reward_model, optim, agent)

    if args.env_name == "SparseAntEnv":
        use_reward = False
        record_coverage = True
    else:
        use_reward = args.use_reward
        record_coverage = False


    for episode in range(metrics["episode"], args.n_episodes):
        tools.log("\n === Episode {} ===".format(episode))
        start_time_episode = time.process_time()
        start_time_training = time.process_time()
        tools.log("Training on {} data points".format(buffer.total_steps))
        train(args, buffer, ensemble, reward_model, optim, params, metrics)
        end_time_training = time.process_time() - start_time_training
        tools.log("Total training time: {:.2f}".format(end_time_training))

        render = False
        if episode % args.render_every == 0:
            render = True

        start_time = time.process_time()
        reward, steps = run_trial(
            agent,
            buffer,
            render=render,
            episode=episode,
            use_exploration=args.use_exploration,
            use_reward=use_reward,
        )
        metrics["test_rewards"].append(reward)
        metrics["test_steps"].append(steps)
        end_time = time.process_time() - start_time
        tools.log("Total exploitation time: {:.2f}".format(end_time))

        end_time_episode = time.process_time() - start_time_episode
        tools.log("Total episode time: {:.2f}".format(end_time_episode))
        metrics["episode"] += 1
        metrics["total_steps"].append(buffer.total_steps)
        metrics["episode_time"].append(end_time_episode)

        if record_coverage:
            coverage = rate_buffer(buffer=buffer)
            tools.log("Coverage: {:.2f}".format(coverage))
            metrics["coverage"].append(coverage)

        if episode % args.save_every == 0:
            metrics["episode"] += 1
            metrics["last_save"] = episode
            tools.save_model(args.logdir, ensemble, reward_model, optim, episode)
            tools.save_normalizer(args.logdir, norm)
            tools.save_buffer(args.logdir, buffer)
            tools.save_metrics(args.logdir, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(DEVICE))

    parser.add_argument("--logdir", type=str, default="log-cheetah")
    parser.add_argument("--env_name", type=str, default="SparseHalfCheetah")
    parser.add_argument("--max_episode_len", type=int, default=100)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--env_std", type=float, default=0.01)
    parser.add_argument("--ensemble_size", type=int, default=15)
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int, default=15)
    parser.add_argument("--n_candidates", type=int, default=1000)
    parser.add_argument("--optimisation_iters", type=int, default=10)
    parser.add_argument("--top_candidates", type=int, default=100)
    parser.add_argument("--n_seed_episodes", type=int, default=1)
    parser.add_argument("--n_train_epochs", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--use_reward", type=bool, default=True)
    parser.add_argument("--use_exploration", type=bool, default=True)
    parser.add_argument("--expl_scale", type=float, default=0.1)
    parser.add_argument("--render_every", type=int, default=1)

    args = parser.parse_args()
    main(args)
