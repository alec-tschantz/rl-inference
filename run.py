# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np
import argparse

from ebme.envs import TorchEnv, NoisyEnv, const
from ebme.normalizer import TransitionNormalizer
from ebme.buffer import Buffer
from ebme.models import EnsembleModel, RewardModel
from ebme.planner import Planner
from ebme.agent import Agent


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    normalizer = TransitionNormalizer()

    env = TorchEnv(
        args.env_name,
        args.max_episode_len,
        action_repeat=args.action_repeat,
        device=DEVICE,
    )

    if args.env_std > 0:
        env = NoisyEnv(env, args.env_std)

    state_size = env.state_dims[0]
    action_size = env.action_dims[0]

    buffer = Buffer(
        state_size,
        action_size,
        args.ensemble_size,
        normalizer,
        buffer_size=args.buffer_size,
        device=DEVICE,
    )

    dynamics_model = EnsembleModel(
        state_size + action_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        normalizer,
        device=DEVICE,
    ).to(DEVICE)

    reward_model = RewardModel(state_size, args.hidden_size).to(DEVICE)

    planner = Planner(
        dynamics_model,
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
    )

    agent = Agent(env, planner)
    params = list(dynamics_model.parameters()) + list(reward_model.parameters())
    optim = torch.optim.Adam(params, lr=args.learning_rate, eps=args.epsilon)

    buffer = agent.get_seed_episodes(buffer, args.n_seed_episodes)

    for episode in range(args.n_episodes):
        for epoch in range(args.n_train_epochs):
            d_losses = []
            r_losses = []

            for (states, actions, rewards, delta_states) in buffer.get_train_batches(
                args.batch_size
            ):

                optim.zero_grad()
                d_loss = dynamics_model.loss(states, actions, delta_states)
                r_loss = reward_model.loss(states, rewards)

                d_losses.append(d_loss.item())
                r_losses.append(r_loss.item())
                (d_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm, norm_type=2)
                optim.step()

            if epoch % args.log_every == 0:
                print(
                    "> Epoch {} [dynamics loss {:.2f} | reward loss {:.2f}]".format(
                        epoch, np.mean(d_losses), np.mean(r_losses)
                    )
                )

        reward, trajectory, _, buffer = agent.run_episode(
            buffer=buffer, render=args.render
        )

        print(
            "Episode {} [reward {} / steps {}]".format(
                episode, reward, trajectory.shape[0]
            )
        )


if __name__ == "__main__":
    print("Loading experiment ({})".format(DEVICE))
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="BipedalWalker-v2")
    parser.add_argument("--max_episode_len", type=int, default=500)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--env_std", type=float, default=0.0)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int, default=12)
    parser.add_argument("--n_candidates", type=int, default=1000)
    parser.add_argument("--optimisation_iters", type=int, default=10)
    parser.add_argument("--top_candidates", type=int, default=100)
    parser.add_argument("--n_seed_episodes", type=int, default=5)
    parser.add_argument("--n_train_epochs", type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    parser.add_argument("--action_noise", type=float, default=0.3)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--use_reward", type=bool, default=True)
    parser.add_argument("--use_exploration", type=bool, default=False)
    parser.add_argument("--expl_scale", type=int, default=1)
    parser.add_argument("--render", type=bool, default=False)
    args = parser.parse_args()

    main(args)

# scp -r ebme at449@allocortex.inf.susx.ac.uk:/its/home/at449/
