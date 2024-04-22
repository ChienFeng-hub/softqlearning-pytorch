from copy import deepcopy
import numpy as np
import torch
from torch import nn
from modules.stochastic_policy import StochasticNNPolicy
from modules.value_function import NNQFunction
from modules.buffer import ReplayBuffer
from modules.train_loop import train_loop
from modules.kernel import RBF_kernel


class SQL():
    def __init__(
        self,
        args,
        logger,
    ):
        self.args = args
        self.logger = logger
        self.buffer = ReplayBuffer(int(args.buffer_size), args.state_sizes, args.action_sizes)
        self.device = args.device
        self.gamma = args.gamma
        self.tau = args.tau
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_scale = args.reward_scale
        self.state_sizes = args.state_sizes
        self.action_sizes = args.action_sizes
        self.n_particles = args.n_particles
        
        actor = StochasticNNPolicy(
            state_sizes=args.state_sizes,
            action_sizes=args.action_sizes,
            hidden_sizes=[args.hidden_sizes] * args.hidden_layers,
            device=args.device
        ).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

        critic1 = NNQFunction(
            state_sizes=args.state_sizes,
            action_sizes=args.action_sizes,
            hidden_sizes=[args.hidden_sizes] * args.hidden_layers,
            device=args.device
        ).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

        critic2 = NNQFunction(
            state_sizes=args.state_sizes,
            action_sizes=args.action_sizes,
            hidden_sizes=[args.hidden_sizes] * args.hidden_layers,
            device=args.device
        ).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
        
        self.actor = actor
        self.actor_optim = actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_optim = critic2_optim


    def act(self, s, deterministic=False):
        act = self.actor(obs=s)
        return act

    def update_critic(self, batch, mseloss=nn.MSELoss()):
        (s, a, r, s_n, d) = batch

        with torch.no_grad():
            # get next Q
            next_actions = torch.FloatTensor(s.shape[0], self.n_particles, self.action_sizes).uniform_(-1, 1)
            next_actions = next_actions.reshape(-1, self.action_sizes)
            sn_repeat = s_n[:, None, :].repeat(1, self.n_particles, 1).reshape(-1, self.state_sizes)
            next_q = torch.min(self.critic1_old(sn_repeat, next_actions), self.critic2_old(sn_repeat, next_actions))
            next_q = next_q.reshape(s.shape[0], self.n_particles, 1)

            # Equation 10:
            next_value = torch.logsumexp(next_q, dim=1) - np.log(self.n_particles) # logmeanexp = logsumexp - log(N)
            next_value += self.action_sizes * np.log(2) # importance sampling weights

            # target Q
            target_q = self.reward_scale * r + (1-d) * self.gamma * next_value

        # update critic1
        current_q1 = self.critic1(s, a)
        c1_loss = mseloss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        c1_loss.backward()
        self.critic1_optim.step()

        # update critic2
        current_q2 = self.critic2(s, a)
        c2_loss = mseloss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        c2_loss.backward()
        self.critic2_optim.step()

        result = {
            "Loss/critic1_loss": c1_loss.item(),
            "Loss/critic2_loss": c2_loss.item(),
            "Q/current_q1": current_q1.mean().item(),
            "Q/current_q2": current_q2.mean().item(),
            "Q/target_q": target_q.mean().item(),
        }
        return result

    def update_actor(self, batch):
        (s, a, r, s_n, d) = batch

        # According to haarnoja/softqlearning:
        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        s_repeat = s[:, None, :].repeat(1, self.n_particles, 1).reshape(-1, self.state_sizes)
        updated_actions = self.act(s=s_repeat).reshape(-1, self.n_particles, self.action_sizes)
        fixed_actions = self.act(s=s_repeat).reshape(-1, self.n_particles, self.action_sizes)
        fixed_actions.detach().requires_grad_()

        # get svgd target
        current_q1 = self.critic1(s_repeat, fixed_actions.reshape(-1, self.action_sizes))
        current_q2 = self.critic2(s_repeat, fixed_actions.reshape(-1, self.action_sizes))
        current_q = torch.min(current_q1, current_q2)
        current_q = current_q.reshape(-1, self.n_particles)

        # Target log-density. Q_soft in Equation 13:
        squash_correction = torch.sum(torch.log(1 - fixed_actions.pow(2) + self.eps), dim=-1)
        log_p = current_q + squash_correction

        grad_log_p = torch.autograd.grad(log_p.sum(), fixed_actions)[0]
        grad_log_p = grad_log_p.reshape(-1, self.n_particles, 1, self.action_sizes).detach() # (N, n_particles, 1, action_sizes)
        grad_log_p.detach_()

        kappa, kappa_grad = RBF_kernel(fixed_actions, updated_actions)

        # Stein Variational Gradient in Equation 13:
        action_gradients = torch.mean(kappa * grad_log_p + kappa_grad, dim=1) # (N, n_particles, action_sizes)
        
        self.actor_optim.zero_grad()
        updated_actions.backward(gradient=-action_gradients)
        self.actor_optim.step()
        
        result = {
            "Loss/action_gradient": action_gradients.mean().item(),
        }
        return result

    def update(self):
        batch = self.buffer.sample(self.args.batch_size, device=self.device)
        (s, a, r, s_n, d) = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
        
        # update critic & policy
        result1 = self.update_critic(batch=(s, a, r, s_n, d))
        result2 = self.update_actor(batch=(s, a, r, s_n, d))

        # sync weights
        self.soft_update(self.critic1_old, self.critic1)
        self.soft_update(self.critic2_old, self.critic2)
        return {**result1, **result2}
    
    def train(self, train_envs, test_envs):
        train_loop(self, self.args, self.buffer, train_envs, test_envs, self.logger)

    def soft_update(self, tgt: nn.Module, src: nn.Module):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1-self.tau) * tgt_param.data)