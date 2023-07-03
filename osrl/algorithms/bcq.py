import copy
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

from osrl.common.net import (VAE, EnsembleDoubleQCritic, LagrangianPIDController,
                             MLPGaussianPerturbationActor)

class BCQL(nn.Module):
    """
        Batch-Constrained deep Q-learning with PID Lagrangian (BCQL)

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes
            of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes
            of the layers in the critic network.
        vae_hidden_sizes (int): Number of hidden units in the VAE.
        sample_action_num (int): Number of action samples to draw.
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks.
        phi (float): Scale parameter for the Gaussian perturbation
            applied to the actor's output.
        lmbda (float): Weight of the Lagrangian term.   [没用]
        beta (float): Weight of the KL divergence term.
        PID (list): List of three floats containing the coefficients
            of the PID controller.[没用]
        num_q (int): Number of Q networks in the ensemble.
        num_qc (int): Number of cost Q networks in the ensemble.
        cost_limit (int): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0').
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 vae_hidden_sizes: int = 64,
                 sample_action_num: int = 10,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 phi: float = 0.05,
                 # lmbda: float = 0.75,
                 beta: float = 0.5,
                 # PID: list = [0.1, 0.003, 0.001],
                 num_q: int = 1,
                 num_qc: int = 1,
                 cost_limit: int = 10,
                 episode_len: int = 300,
                 device: str = "cpu"):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.latent_dim = self.action_dim * 2
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.vae_hidden_sizes = vae_hidden_sizes
        self.sample_action_num = sample_action_num
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.beta = beta
        self.num_q = num_q
        self.num_qc = num_qc
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device

        ################ create actor critic model ###############
        self.actor = MLPGaussianPerturbationActor(self.state_dim, self.action_dim,
                                                  self.a_hidden_sizes, nn.Tanh, self.phi,
                                                  self.max_action).to(self.device)
        self.critic = EnsembleDoubleQCritic(self.state_dim,
                                            self.action_dim,
                                            self.c_hidden_sizes,
                                            nn.ReLU,
                                            num_q=self.num_q).to(self.device)
        self.vae = VAE(self.state_dim, self.action_dim, self.vae_hidden_sizes,
                       self.latent_dim, self.max_action, self.device).to(self.device)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module
        towards the parameters of source module.
        """
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def vae_loss(self, observations, actions):
        """
        Calculate CVAE Loss

        Args:
            observations:
            actions:

        Returns:
            loss_vae:
            stats_vae:
        """
        recon, mean, std = self.vae(observations, actions)
        recon_loss = nn.functional.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        loss_vae = recon_loss + self.beta * KL_loss

        self.vae_optim.zero_grad()
        loss_vae.backward()
        self.vae_optim.step()
        stats_vae = {"loss/loss_vae": loss_vae.item()}
        return loss_vae, stats_vae

    def critic_loss(self, observations, next_observations, actions, rewards, done):
        _, _, q1_list, q2_list = self.critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]

            # generated 10 actions
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num,
                                               0).to(self.device)
            generated_action = self.vae.decode(obs_next)

            # action net include perturbation
            act_targ_next = self.actor_old(obs_next, generated_action)
            q1_targ, q2_targ, _, _ = self.critic_old.predict(obs_next, act_targ_next)

            # soft clipped double q-learning
            q_targ = self.lmbda * torch.min(
                q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)

            # take max over each action sampled from the generation and perturbation model
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]
            backup = rewards + self.gamma * (1 - done) * q_targ


        loss_critic = self.critic.loss(backup, q1_list) + self.critic.loss(
            backup, q2_list)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        stats_critic = {"loss/critic_loss": loss_critic.item()}
        return loss_critic, stats_critic

    def actor_loss(self, observations):
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False

        actions = self.actor(observations, self.vae.decode(observations))
        q1_pi, q2_pi, _, _ = self.critic.predict(observations, actions)  # [batch_size]
        qc1_pi, qc2_pi, _, _ = self.cost_critic.predict(observations, actions)
        qc_pi = torch.min(qc1_pi, qc2_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        with torch.no_grad():
            multiplier = self.controller.control(qc_pi).detach()
        qc_penalty = ((qc_pi - self.qc_thres) * multiplier).mean()
        loss_actor = -q_pi.mean() + qc_penalty

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        stats_actor = {
            "loss/actor_loss": loss_actor.item(),
            "loss/qc_penalty": qc_penalty.item(),
            "loss/lagrangian": multiplier.item()
        }

        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        for p in self.vae.parameters():
            p.requires_grad = True
        return loss_actor, stats_actor























