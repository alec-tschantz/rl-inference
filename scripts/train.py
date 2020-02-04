# pylint: disable=not-callable
# pylint: disable=no-member

import sys
import time
import pathlib
import argparse

import torch
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pmbrl.envs import GymEnv
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Planner, Agent
from pmbrl import utils

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    logger = utils.Logger(args.logdir, args.seed)
    logger.log("\n=== Loading experiment [device: {}] ===".format(DEVICE))
    logger.log(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    env = GymEnv(
        args.env_name,
        args.max_episode_len,
        action_repeat=args.action_repeat,
        seed=args.seed,
    )
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    normalizer = Normalizer()
    buffer = Buffer(
        state_size,
        action_size,
        args.ensemble_size,
        normalizer,
        signal_noise=args.signal_noise,
        device=DEVICE,
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
        logger=logger,
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
        rollout_clamp=args.rollout_clamp,
        device=DEVICE,
    )
    agent = Agent(env, planner, logger=logger)

    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes | {} frames]"
    logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))

    for episode in range(args.n_episodes):
        logger.log("\n\n=== Episode {} ===".format(episode))
        start_time = time.time()

        e_loss, r_loss = trainer.train()
        logger.log_losses(e_loss, r_loss)

        reward, steps, trajectory, actions = agent.run_episode(
            buffer, action_noise=args.action_noise
        )
        logger.log_episode(reward, steps)

        if args.plot_trajectory:
            path = logger.img_path + "trajectory_{}.png".format(episode)
            utils.log_trajectory_predictions(
                ensemble,
                trajectory,
                actions,
                args.traj_eval_steps,
                path,
                rollout_clamp=args.rollout_clamp,
                device=DEVICE,
            )

        total_time = time.time() - start_time
        logger.log_time(total_time)
        logger.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = utils.get_config(args)
    main(config)
