# pylint: disable=not-callable
# pylint: disable=no-member

import argparse

import torch
import numpy as np
from tqdm import tqdm

from pmbrl.envs import GymEnv
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Planner, Agent
from pmbrl import tools

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    logger = tools.Logger(args.logdir, args.experiment_id, log_every=args.log_every)
    logger.log("\n=== Loading experiment [{}] ===".format(DEVICE))
    logger.log(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    env = GymEnv(args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=args.seed)
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
        rollout_delta_clamp=args.rollout_delta_clamp,
        device=DEVICE,
    )
    agent = Agent(env, planner, logger=logger)

    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes] [{} frames]"
    logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))

    for episode in tqdm(range(args.n_episodes)):
        logger.log("\n\n=== Episode {} ===".format(episode))

        e_loss, r_loss = trainer.train()
        logger.log_scalar("Loss/Ensemble", e_loss, episode)
        logger.log_scalar("Loss/Reward", r_loss, episode)

        reward, steps, trajectory, actions = agent.run_episode(
            buffer, action_noise=args.action_noise
        )
        logger.log_scalar("Evaluation/Reward", reward, episode)
        logger.log_scalar("Evaluation/Steps", steps, episode)

        pred_states, pred_delta_vars, traj = tools.evaluate_trajectory(
            ensemble,
            trajectory,
            actions,
            args.traj_eval_steps,
            device=DEVICE,
            rollout_delta_clamp=args.rollout_delta_clamp,
        )
        fig = tools.plot_trajectory_evaluation(pred_states, pred_delta_vars, traj)
        logger.log_figure("Predictions/Trajectory", fig, episode)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="mountain_car")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--experiment_id", type=str, default="test")
    args = parser.parse_args()
    config = tools.get_config(args)
    main(config)
