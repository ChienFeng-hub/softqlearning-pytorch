from collections.abc import MutableMapping
import os
import torch
import numpy as np


def flatten_cfg(cfg):
    items = []
    for key, value in cfg.items():
        if isinstance(value, MutableMapping):
            items.extend(flatten_cfg(value).items())
        else:
            items.append((key, value))
    return dict(items)


def log(result, global_step, logger):
    for k, v in result.items():
        logger.add_scalar(k, v, global_step)


def outputdir_make_and_add(outputdir, title=None):
    #creates outputdir
    os.makedirs(outputdir,exist_ok=True)
    folder_num = len(next(os.walk(outputdir))[1]) #counts how many folders already there 
    if folder_num == 0:
        folder_num = 1
    elif folder_num == 1 and next(os.walk(outputdir))[1][0][0] == ".":
        folder_num = 1
    else:
        folder_num = max([int(i.split('-')[0]) for i in next(os.walk(outputdir))[1] if i[0] != '.'],default=0) + 1 # this looks for max folder num and adds one... this works even if title is used (because we index at 1) (dot check to ignore .ipynb) 
        #currently returns error when a subfolder contains anything other than a number (exept dot handle) 
        #so essentially this assumes the outputdir structure with numbers (and possible titles). will need to fix if i want to use it later for something else
    
    if title == None:
        outputdir += '/' + str(folder_num) #adds one
    else:
        outputdir += '/' + str(folder_num) + f'-({title})' #adds one and appends title
    os.makedirs(outputdir,exist_ok=True)
    return outputdir


def evaluate(envs, agent, deterministic=False):
    with torch.no_grad():
        num_envs = envs.unwrapped.num_envs
        rewards = np.zeros((num_envs,))
        dones = np.zeros((num_envs,)).astype(bool)
        s, _ = envs.reset(seed=range(num_envs))
        while not all(dones):
            a = agent.act(s, deterministic=deterministic)
            a = a.cpu().detach().numpy()
            s_, r, terminated, truncated, _ = envs.step(a)
            done = terminated | truncated
            rewards += r * (1-dones)
            dones |= done
            s = s_
    return rewards.mean()


def train_loop(agent, args, buffer, train_envs, test_envs, logger):
    # warm-up buffer
    s, _ = train_envs.reset(seed=args.seed)
    while len(buffer) < args.warmup_steps:
        a = train_envs.action_space.sample()
        s_, r, terminated, truncated, _ = train_envs.step(a)
        done = terminated or truncated
        buffer.store(s, a, r, s_, done*1.)
        s = s_
        if done:
            s, _ = train_envs.reset(seed=args.seed)
    print("Buffer preload: ", len(buffer))

    # training loop
    best_test_rewards = -np.inf
    best_train_rewards = -np.inf
    rewards = 0
    episode = 0
    s, _ = train_envs.reset(seed=args.seed)
    for t in range(args.steps+1):
        a = agent.act(s)
        assert torch.any(torch.isnan(a)) == False
        a = a.cpu().detach().numpy()
        s_, r, terminated, truncated, _ = train_envs.step(a)
        done = terminated or truncated
        buffer.store(s, a, r, s_, done*1.)
        result = agent.update()
        rewards += r
        s = s_

        if t % args.eval_every == 0:
            test_rewards = evaluate(test_envs, agent)
            # print(f'test reward: {test_rewards} at step {t}')
            if test_rewards > best_test_rewards:
                best_test_rewards = test_rewards
                torch.save(agent.actor, os.path.join(args.save_path, 'actor.pt'))
                print(f"save agent to: {args.save_path} with best return {best_test_rewards} at step {t}")
            log({
                **result,
                "Test/return": test_rewards,
                "Test/best_return": best_test_rewards,
                "Steps": t,
            }, t, logger)
        if done:
            best_train_rewards = max(best_train_rewards, rewards)
            log({
                "Train/return": rewards,
                "Train/best_return": best_train_rewards,
                "Episodes": episode,
            }, episode, logger)
            s, _ = train_envs.reset(seed=args.seed)
            rewards = 0
            episode += 1
