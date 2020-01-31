# pylint: disable=not-callable
# pylint: disable=no-member

""" __tasks__: seeds, logging, recoding """

import argparse
import torch

from pmbrl.envs import GymEnv
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Planner, Agent
from pmbrl import tools

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    print("\n=== Loading experiment ===")
    print("Using [{}]".format(DEVICE))
    print(args)

    env = GymEnv(args.env_name, args.max_episode_len, action_repeat=args.action_repeat)
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    normalizer = Normalizer()
    buffer = Buffer(
        state_size, action_size, args.ensemble_size, normalizer, device=DEVICE
    )

    ensemble = EnsembleModel(
        state_size + action_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        normalizer,
        max_logvar=args.max_logvar,
        min_logvar=args.min_logvar,
        act_fn=args.act_fn,
        device=DEVICE,
    )
    reward_model = RewardModel(state_size, args.hidden_size, device=DEVICE)
    trainer = Trainer(
        ensemble,
        reward_model,
        buffer,
        n_train_epochs=args.n_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
    )

    planner = Planner(
        ensemble,
        reward_model,
        action_size,
        args.ensemble_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        use_reward=args.use_reward,
        use_exploration=args.use_exploration,
        use_mean=args.use_mean,
        use_kl_div=args.use_kl_div,
        expl_scale=args.expl_scale,
        reward_scale=args.reward_scale,
        reward_prior=args.reward_prior,
        device=DEVICE,
    )

    agent = Agent(env, planner)

    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "Collected [{} episodes] [{} frames]"
    print(msg.format(args.n_seed_episodes, buffer.total_steps))

    for episode in range(args.n_episodes):
        print("\n === Episode {} ===".format(episode))
        trainer.train()
        reward, steps = agent.run_episode(buffer)
        print("Reward {:.2f} | Steps {:.2f}]".format(reward, steps))


if __name__ == "__main__":
    args = tools.DebugConfig()
    main(args)
