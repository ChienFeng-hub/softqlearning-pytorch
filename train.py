import os
import sys
import random
import datetime
import wandb
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import argparse
import toy_envs
from agents import *
from modules import flatten_cfg, outputdir_make_and_add
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg : DictConfig) -> None:
    # parse args
    cfg = flatten_cfg(cfg) # flatten the nested Dict structure from hydra
    args = argparse.Namespace(**cfg)

    # logger init
    save_path = os.path.join('ckpts', args.env, args.algo, args.description)
    os.makedirs(save_path, exist_ok=True)
    outputdir = outputdir_make_and_add(outputdir=save_path, title=f'seed{args.seed}')
    args.save_path = outputdir
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if args.wandb:
        wandb.init(
            project=args.project,
            name=f"{args.env}_{args.algo}_{args.description}_{t0}_seed{args.seed}",
            config=args,
            sync_tensorboard=True,
        )
    logger = SummaryWriter(log_dir=args.save_path)

    # determinism
    random.seed(args.seed) # if you're using random
    np.random.seed(args.seed) # if you're using numpy
    torch.manual_seed(args.seed) # torch.cuda.manual_seed_all(SEED) is not required
    if args.deterministic:
        print("Enable deterministic")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # environment init
    train_envs = gym.make_vec(args.env, num_envs=1)
    test_envs = gym.make_vec(args.env, num_envs=args.test_num)
    train_envs = RescaleAction(train_envs, min_action=-1, max_action=1) # rescale tanh action (-1~1) to env action space
    test_envs = RescaleAction(test_envs, min_action=-1, max_action=1) # rescale tanh action (-1~1) to env action space
    args.state_sizes = train_envs.observation_space.shape[1]
    args.action_sizes = train_envs.action_space.shape[1]
    print("Args:", args)
    print("Observation space:", train_envs.observation_space)
    print("Action space:", train_envs.action_space)
    
    # model
    AGENT = getattr(sys.modules[__name__], args.algo.upper())
    agent = AGENT(args, logger)
    agent.train(train_envs, test_envs)


if __name__ == '__main__':
    main()