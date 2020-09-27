from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm,ExpAlgorithmIter,ExpAlgorithmFin, ExpAlgorithmFin2,ExpAlgorithmFin3


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _do_training_fit(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step_fit(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(ptu.get_numpy(self.agent.z_means))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def _take_step_fit(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        #self.qf1_optimizer.step()
        #self.qf2_optimizer.step()
        self.context_optimizer.step()



    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot


class ExpSACIter(ExpAlgorithmIter):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            nets_exp,
            encoder,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            use_info_in_context=False,
            entropy_weight=1e-2,
            intrinsic_reward_weight=1e-1,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            agent_exp=nets_exp[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            encoder=encoder,
            **kwargs
        )
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.entropy_weight = entropy_weight
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.qf_exp_criterion = nn.MSELoss()
        self.vf_exp_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.qf1_exp, self.qf2_exp, self.vf_exp = nets_exp[1:]
        self.target_vf = self.vf.copy()
        self.target_exp_vf = self.vf_exp.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.policy_exp_optimizer = optimizer_class(
            self.exploration_agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_exp_optimizer = optimizer_class(
            self.qf1_exp.parameters(),
            lr=qf_lr,
        )
        self.qf2_exp_optimizer = optimizer_class(
            self.qf2_exp.parameters(),
            lr=qf_lr,
        )
        self.vf_exp_optimizer = optimizer_class(
            self.vf_exp.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return  [self.context_encoder, self.agent.policy] + [self.qf1, self.qf2, self.vf, self.target_vf] + [self.exploration_agent.policy] + [self.qf1_exp, self.qf2_exp, self.vf_exp, self.target_exp_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def unpack_batch_context(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        info = batch['env_infos'][None,...]
        return [o, a, r, no, t,info]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices,sequence=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch_context(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-2], dim=2)
        else:
            context = torch.cat(context[:-3], dim=2)
        return context, context_unbatched

    ##### Training #####
    def _do_training(self, indices, num_iter):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch,context_unbatched = self.sample_context(indices,False)
        _,context_unbatched = self.sample_context(indices,True)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))
        self.exploration_agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            if num_iter<500:
                self._take_step(indices, context,context_unbatched)
            else:
                self._take_step_exp(indices, context, context_unbatched)

            # stop backprop
            self.agent.detach_z()

    def _min_q_exp(self, obs, actions):
        #print(obs.shape,actions.shape)
        self.qf1_exp.inner_reset(num_tasks=obs.shape[0])
        self.qf2_exp.inner_reset(num_tasks=obs.shape[0])
        q1 = self.qf1_exp(torch.cat([obs, actions],dim=2))
        q2 = self.qf2_exp(torch.cat([obs, actions],dim=2))
        min_q = torch.min(q1, q2)
        return min_q

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _update_target_network_exp(self):
        ptu.soft_update_from_to(self.vf_exp, self.target_exp_vf, self.soft_target_tau)

    def _take_step(self, indices, context,context_unbatched):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        rewards_traj = rewards
        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def _take_step_exp(self, indices, context,context_unbatched):


        obs, actions, rewards, next_obs, terms, er = context_unbatched
        self.exploration_agent.reset_RNN(num_tasks=obs.shape[0])
        self.qf1_exp.inner_reset(num_tasks=obs.shape[0])
        self.qf2_exp.inner_reset(num_tasks=obs.shape[0])
        self.vf_exp.inner_reset(num_tasks=obs.shape[0])
        self.target_exp_vf.inner_reset(num_tasks=obs.shape[0])
        policy_outputs,er = self.exploration_agent(obs, context)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        rewards = rewards.view(t*b,-1)
        #obs = obs.view(t * b, -1)
        #actions = actions.view(t * b, -1)
        #next_obs = next_obs.view(t * b, -1)
        rew = er * self.intrinsic_reward_weight + rewards
        #print(z_mean.shape, z_mean_next.shape, obs.shape, t, b)
        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1_exp(torch.cat([obs, actions],dim=2))
        q2_pred = self.qf2_exp(torch.cat([obs, actions],dim=2))
        v_pred = self.vf_exp(obs)
        # get targets for use in V and Q updates

        with torch.no_grad():
            target_v_values = self.target_exp_vf(next_obs)

        # KL constraint on z if probabilistic


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_exp_optimizer.zero_grad()
        self.qf2_exp_optimizer.zero_grad()
        rewards = rew.detach()
        rewards_flat = rewards.view(t * b, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_exp_optimizer.step()
        self.qf2_exp_optimizer.step()


        # compute min Q on the new actions
        new_actions = new_actions.view(t , b, -1)
        min_q_new_actions = self._min_q_exp(obs, new_actions)

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_exp_criterion(v_pred, v_target.detach())
        self.vf_exp_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_exp_optimizer.step()
        self._update_target_network_exp()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_exp_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_exp_optimizer.step()

        # save some statistics for eval

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            qf1_exp=self.qf1_exp.state_dict(),
            qf2_exp=self.qf2_exp.state_dict(),
            policy_exp=self.exploration_agent.state_dict(),
            vf_exp=self.vf_exp.state_dict(),
            target_vf_exp=self.target_exp_vf.state_dict(),
        )
        return snapshot

class ExpSACRew(ExpAlgorithmIter):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            nets_exp,
            encoder,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            use_info_in_context=False,
            entropy_weight=1e-2,
            intrinsic_reward_weight=1e-1,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            agent_exp=nets_exp[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            encoder=encoder,
            **kwargs
        )
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.entropy_weight = entropy_weight
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.qf_exp_criterion = nn.MSELoss()
        self.vf_exp_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.pred_loss = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.qf1_exp, self.qf2_exp, self.vf_exp, self.rew_predictor = nets_exp[1:]
        self.target_vf = self.vf.copy()
        self.target_exp_vf = self.vf_exp.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.policy_exp_optimizer = optimizer_class(
            self.exploration_agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_exp_optimizer = optimizer_class(
            self.qf1_exp.parameters(),
            lr=qf_lr,
        )
        self.qf2_exp_optimizer = optimizer_class(
            self.qf2_exp.parameters(),
            lr=qf_lr,
        )
        self.vf_exp_optimizer = optimizer_class(
            self.vf_exp.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=context_lr,
        )
        self.predictor_optimizer = optimizer_class(
            self.rew_predictor.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return  [self.context_encoder, self.agent.policy] + [self.qf1, self.qf2, self.vf, self.target_vf] + [self.exploration_agent.policy] + [self.qf1_exp, self.qf2_exp, self.vf_exp, self.target_exp_vf,self.rew_predictor]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def unpack_batch_context(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        info = batch['env_infos'][None,...]
        return [o, a, r, no, t,info]


    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices,sequence=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch_context(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-2], dim=2)
        else:
            context = torch.cat(context[:-3], dim=2)
        return context, context_unbatched

    def pred_context(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0],1,context[2].shape[2])
        tmp = torch.cat([r_0,context[2]],dim=1)[:,:-1,:]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([context[0],context[1],tmp], dim=2)
        return contextr

    ##### Training #####
    def _do_training(self, indices, num_iter):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch,context_unbatched = self.sample_context(indices,False)
        _,context_unbatched = self.sample_context(indices,True)
        context_pred = self.pred_context(context_unbatched)
        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))
        self.exploration_agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            if num_iter<500:
                self._take_step(indices, context,context_unbatched,context_pred)
            else:
                self._take_step_exp(indices, context, context_unbatched,context_pred)

            # stop backprop
            self.agent.detach_z()

    def _min_q_exp(self, obs, actions):
        #print(obs.shape,actions.shape)
        self.qf1_exp.inner_reset(num_tasks=obs.shape[0])
        self.qf2_exp.inner_reset(num_tasks=obs.shape[0])
        q1 = self.qf1_exp(torch.cat([obs, actions],dim=2))
        q2 = self.qf2_exp(torch.cat([obs, actions],dim=2))
        min_q = torch.min(q1, q2)
        return min_q

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _update_target_network_exp(self):
        ptu.soft_update_from_to(self.vf_exp, self.target_exp_vf, self.soft_target_tau)

    def _take_step(self, indices, context,context_unbatched,context_pred):

        num_tasks = len(indices)
        self.rew_predictor.inner_reset(context_pred.shape[0])
        rew_pred = self.rew_predictor(context_pred)
        rew = context_unbatched[2].contiguous()
        #print(rew_pred.shape)
        rew = rew.view(rew_pred.shape[0],-1)
        loss = self.pred_loss(rew,rew_pred)
        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        rewards_traj = rewards
        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def _take_step_exp(self, indices, context,context_unbatched,context_pred):


        obs, actions, rewards, next_obs, terms, er = context_unbatched
        pred_rew = self.rew_predictor(context_pred)
        self.exploration_agent.reset_RNN(num_tasks=obs.shape[0])
        self.qf1_exp.inner_reset(num_tasks=obs.shape[0])
        self.qf2_exp.inner_reset(num_tasks=obs.shape[0])
        self.vf_exp.inner_reset(num_tasks=obs.shape[0])
        self.target_exp_vf.inner_reset(num_tasks=obs.shape[0])
        self.rew_predictor.inner_reset(context_pred.shape[0])
        policy_outputs,er = self.exploration_agent(obs, context,cal_rew=False)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        rewards = rewards.contiguous()
        rewards = rewards.view(t*b,-1)
        #obs = obs.view(t * b, -1)
        #actions = actions.view(t * b, -1)
        #next_obs = next_obs.view(t * b, -1)
        #er = er.view(t * b, -1)
        rew = (rewards-pred_rew)**2 * self.intrinsic_reward_weight + rewards
        rew = rew.detach()
        #print(z_mean.shape, z_mean_next.shape, obs.shape, t, b)
        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1_exp(torch.cat([obs, actions],dim=2))
        q2_pred = self.qf2_exp(torch.cat([obs, actions],dim=2))
        v_pred = self.vf_exp(obs)
        # get targets for use in V and Q updates

        with torch.no_grad():
            target_v_values = self.target_exp_vf(next_obs)

        # KL constraint on z if probabilistic


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_exp_optimizer.zero_grad()
        self.qf2_exp_optimizer.zero_grad()
        rewards = rew.detach()
        rewards_flat = rewards.view(t * b, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_exp_optimizer.step()
        self.qf2_exp_optimizer.step()


        # compute min Q on the new actions
        new_actions = new_actions.view(t , b, -1)
        min_q_new_actions = self._min_q_exp(obs, new_actions)

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_exp_criterion(v_pred, v_target.detach())
        self.vf_exp_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_exp_optimizer.step()
        self._update_target_network_exp()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_exp_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_exp_optimizer.step()

        # save some statistics for eval

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            qf1_exp=self.qf1_exp.state_dict(),
            qf2_exp=self.qf2_exp.state_dict(),
            policy_exp=self.exploration_agent.state_dict(),
            vf_exp=self.vf_exp.state_dict(),
            target_vf_exp=self.target_exp_vf.state_dict(),
        )
        return snapshot

class ExpSACFin(ExpAlgorithmFin):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            nets_exp,
            encoder,
            latent_dim,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            use_info_in_context=False,
            entropy_weight=1e-2,
            intrinsic_reward_weight=1e-1,
            use_kl_div_intrinsic=False,
            gradient_from_Q=False,
            prediction_reward_scale=1,
            intrinsic_reward_decay = 1,
            kl_min_weight = 1e-3,
            pie_hidden_dim = 5,
            consider_dynamics=0,
            prediction_transition_scale=1,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=None,
            agent_exp=nets_exp[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            encoder=encoder,

            **kwargs
        )
        self.use_kl_div_intrinsic = use_kl_div_intrinsic
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.entropy_weight = entropy_weight
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.latent_dim = latent_dim
        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.qf_exp_criterion = nn.MSELoss()
        self.vf_exp_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.pred_loss = nn.MSELoss()
        self.kl_lambda = kl_lambda
        self.prediction_reward_scale = prediction_reward_scale
        self.prediction_transition_scale = prediction_transition_scale

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.gradient_from_Q = gradient_from_Q
        self.intrinsic_reward_decay = intrinsic_reward_decay
        self.consider_dynamics = consider_dynamics


        self.qf1_exp, self.qf2_exp, self.vf_exp, self.rew_decoder, self.transition_decoder = nets_exp[1:]
        self.target_exp_vf = self.vf_exp.copy()


        self.policy_exp_optimizer = optimizer_class(
            self.exploration_agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_exp_optimizer = optimizer_class(
            self.qf1_exp.parameters(),
            lr=qf_lr,
        )
        self.qf2_exp_optimizer = optimizer_class(
            self.qf2_exp.parameters(),
            lr=qf_lr,
        )
        self.vf_exp_optimizer = optimizer_class(
            self.vf_exp.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=context_lr,
        )
        self.rew_optimizer = optimizer_class(
            self.rew_decoder.parameters(),
            lr=context_lr,
        )
        self.transition_optimizer = optimizer_class(
            self.transition_decoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return  [self.context_encoder] + [self.exploration_agent.policy] + [self.qf1_exp, self.qf2_exp, self.vf_exp, self.target_exp_vf,self.rew_decoder,self.transition_decoder]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            sr = batch['sparse_rewards'][None, ...]
        else:
            sr=None
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t,sr]

    def unpack_batch_context(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        sparse_r = batch['sparse_rewards'][None, ...]
        r = batch['rewards'][None, ...]
        if not sparse_reward:
            sparse_r = r
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        info = batch['env_infos'][None,...]
        #print(o[0,:5],a[0,:5],r[0],sparse_r[0],no[0,:5])
        return [o, a, sparse_r, no, t,info,r]


    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        if self.use_per:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)[0]) for idx in indices]
        else:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for
                       idx in indices]
        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices,sequence=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch_context(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-3], dim=2)
        else:
            context = torch.cat(context[:-4], dim=2)
        return  context,context_unbatched

    def pred_context(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0],1,context[2].shape[2])
        tmp = torch.cat([r_0,context[2]],dim=1)
        a_0 = ptu.zeros(context[1].shape[0], 1, context[1].shape[2])
        tmp2 = torch.cat([a_0, context[1]], dim=1)
        tmp3 = torch.cat([torch.unsqueeze(context[0][:,0,:],1),context[3]],dim=1)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([tmp3,tmp2,tmp], dim=2)
        return contextr

    ##### Training #####
    def _do_training(self, indices, mode):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        _,context_unbatched = self.sample_context(indices,True)
        context_pred = self.pred_context(context_unbatched)
        context = self.sample_sac(indices)
        # zero out context and hidden encoder state


        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):


            self._take_step_exp(indices, context_unbatched,context_pred,context)

            # stop backprop


    def _min_q_exp(self, obs, actions,z_mean_prev, z_var_prev):
        #print(obs.shape,actions.shape)

        q1 = self.qf1_exp(torch.cat([obs, actions,z_mean_prev, z_var_prev],dim=1))
        q2 = self.qf2_exp(torch.cat([obs, actions,z_mean_prev, z_var_prev],dim=1))
        min_q = torch.min(q1, q2)
        return min_q


    def _update_target_network_exp(self):
        ptu.soft_update_from_to(self.vf_exp, self.target_exp_vf, self.soft_target_tau)

    def compute_kl(self,means,vars):
        std_mean = ptu.zeros(means.size())
        std_var = ptu.ones(means.size())
        tem = vars / std_var
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((std_mean - means) / std_var) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        kl_div = torch.mean(kl_div)
        return kl_div

    def compute_intrinsic(self,z_mean_prev, z_var_prev,z_mean_post,z_var_post):
        tem = z_var_post / z_var_prev
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((z_mean_prev - z_mean_post) / z_var_prev) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        return kl_div

    def _take_step(self, indices,context_pred):


        z_s = self.context_encoder.forward_seq(context_pred[:,:-1,:])
        z_mean = z_s[:,:self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:,self.latent_dim:])
        #print(z_mean.shape,z_var.shape)
        z_dis = torch.distributions.Normal(z_mean,torch.sqrt(z_var))
        z_sample = z_dis.rsample()
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        t,b,_ = obs.size()
        obs = obs.view(t*b,-1)
        actions = actions.view(t * b, -1)
        rewards = rewards.view(t * b, -1)
        rewards = rewards * self.prediction_reward_scale
        z_sample = z_sample.view(t*b,-1)
        rew_pred = self.rew_decoder.forward(z_sample,obs,actions)
        self.context_optimizer.zero_grad()
        self.rew_optimizer.zero_grad()
        loss = self.pred_loss(rewards,rew_pred)
        loss.backward(retain_graph=True)
        kl_div = self.compute_kl(z_mean,z_var)
        kl_loss = kl_div * self.kl_lambda
        kl_loss.backward()
        self.context_optimizer.step()
        self.rew_optimizer.step()



        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            z_mean = np.mean(np.abs(ptu.get_numpy(z_mean)))
            z_sig = np.mean(ptu.get_numpy(z_var))
            self.eval_statistics['Z mean train'] = z_mean
            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
            self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics['reward prediction loss'] = ptu.get_numpy(loss)


    def _take_step_exp(self, indices,context_unbatched,context_pred,context):

        self.context_optimizer.zero_grad()
        if self.prediction:
            t, b, _ = context_pred.size()
            z_s = self.context_encoder.forward_seq(context_pred)
            z_s = z_s.view(t,b,-1)
            z_s_pre = z_s[:,:-1,:]
            z_s_pre = z_s_pre.contiguous()
            z_s_pre = z_s_pre.view(t*(b-1),-1)
            z_mean = z_s_pre[:, :self.latent_dim]
            z_var = torch.nn.functional.softplus(z_s_pre[:, self.latent_dim:])
            # print(z_mean.shape,z_var.shape)
            z_dis = torch.distributions.Normal(z_mean, torch.sqrt(z_var))
            z_sample = z_dis.rsample()
            #z_sample = z_sample[-1:,:].repeat(t*(b-1),1)
            obs, actions, rewards, next_obs, terms, sparse_r = context
            if self.sparse_rewards:
                rewards = sparse_r
            #obs, actions, rewards, next_obs, terms = context
            #_,cu = self.sample_context(indices,False)
            #obs, actions, _, next_obs, terms, info,rewards = cu
            t, b, _ = obs.size()
            obs = obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            rewards = rewards.view(t * b, -1)
            next_obs = next_obs.view(t * b, -1)
            z_sample = z_sample.view(t * b, -1)
            rew_pred = self.rew_decoder.forward(z_sample, obs, actions)
            self.rew_optimizer.zero_grad()
            loss = self.pred_loss(rewards, rew_pred)* self.prediction_reward_scale
            loss.backward(retain_graph=True)
            kl_div = self.compute_kl(z_mean, z_var)
            kl_loss = kl_div * self.kl_lambda
            kl_loss.backward(retain_graph=True)
            self.rew_optimizer.step()
            if self.consider_dynamics:
                self.transition_optimizer.zero_grad()
                trans_pred = self.transition_decoder.forward(z_sample, obs, actions)
                trans_loss = self.pred_loss(next_obs, trans_pred)* self.prediction_transition_scale
                trans_loss.backward(retain_graph=True)
                self.transition_optimizer.step()
            if self.eval_statistics is None:
                # eval should set this to None.
                # this way, these statistics are only computed for one batch.
                self.eval_statistics = OrderedDict()
                z_mean = np.mean(np.abs(ptu.get_numpy(z_mean)))
                z_sig = np.mean(ptu.get_numpy(z_var))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
                self.eval_statistics['reward prediction loss'] = ptu.get_numpy(loss)
                if self.consider_dynamics:
                    self.eval_statistics['transisition prediction loss'] = ptu.get_numpy(trans_loss)
        #obs, actions, _, next_obs, terms, info, agent_rew = context_unbatched
        obs, actions, agent_rew, next_obs, terms, sr = context
        if self.sparse_rewards:
            pred_rewardss=  sr
        else:
            pred_rewardss = agent_rew
        obs = obs.contiguous()
        t,b,_ = context_pred.size()
        if not self.prediction:
            z_s = self.context_encoder.forward_seq(context_pred)
            z_s = z_s.view(t,b,-1)
        z_mean = z_s[:, :,:self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:,:, self.latent_dim:])
        #print(z_mean.shape,z_var.shape)
        z_mean_prev, z_var_prev,z_mean_post,z_var_post = z_mean[:,:-1,:],z_var[:,:-1,:],z_mean[:,1:,:],z_var[:,1:,:]
        z_mean_prev, z_var_prev, z_mean_post, z_var_post = z_mean_prev.contiguous(), z_var_prev.contiguous(), z_mean_post.contiguous(), z_var_post.contiguous()
        z_mean_prev, z_var_prev,z_mean_post,z_var_post = z_mean_prev.view(t*(b-1),-1), z_var_prev.view(t*(b-1),-1),z_mean_post.view(t*(b-1),-1),z_var_post.view(t*(b-1),-1)
        policy_outputs,_ = self.exploration_agent(obs.view(t*(b-1),-1), z_mean_prev,z_var_prev)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        agent_rew = agent_rew.contiguous()
        pred_rewardss = pred_rewardss.contiguous()
        agent_rew = agent_rew.view(t*b,-1)
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        pred_rewardss = pred_rewardss.view(t * b, -1)
        #info = info.view(t * b, -1)
        if self.intrinsic_reward_weight>0:
            if self.use_kl_div_intrinsic:
                intrinsic_reward = self.compute_intrinsic(z_mean_prev, z_var_prev, z_mean_post, z_var_post).detach()
            else:
                if not self.prediction:
                    z_dis = torch.distributions.Normal(z_mean_prev, torch.sqrt(z_var_prev))
                    z_sample = z_dis.rsample()
                pred_rew = self.rew_decoder.forward(z_sample, obs, actions)
                intrinsic_reward = (pred_rew - pred_rewardss) ** 2
                if self.consider_dynamics:
                    pred_trans = self.transition_decoder.forward(z_sample, obs, actions)
                    intrinsic_reward = intrinsic_reward + torch.mean((pred_trans - next_obs) ** 2,dim=1,keepdim=True)
            intrinsic_reward = intrinsic_reward.view(t * b, -1)
            if self.intrinsic_reward_decay !=1:
                intrinsic_reward = intrinsic_reward * torch.unsqueeze(ptu.from_numpy(self.intrinsic_reward_decay **np.linspace(0,t*b-1,t*b)),1)
            rew = intrinsic_reward * self.intrinsic_reward_weight + agent_rew
        else:
            rew = agent_rew
        rew = rew.detach()
        #print(z_mean.shape, z_mean_next.shape, obs.shape, t, b)
        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1_exp(torch.cat([obs, actions,z_mean_prev, z_var_prev],dim=1))
        q2_pred = self.qf2_exp(torch.cat([obs, actions,z_mean_prev, z_var_prev],dim=1))
        v_pred = self.vf_exp(torch.cat([obs,z_mean_prev.detach(), z_var_prev.detach()],dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            target_v_values = self.target_exp_vf(torch.cat([next_obs,z_mean_post,z_var_post],dim=1))

        # KL constraint on z if probabilistic

        if not self.gradient_from_Q:
            self.context_optimizer.step()
        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_exp_optimizer.zero_grad()
        self.qf2_exp_optimizer.zero_grad()
        rewards = rew.detach()
        rewards_flat = rewards.view(t * b, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        if self.gradient_from_Q:
            kl_div = self.compute_kl(z_mean_prev, z_var_prev)
            kl_loss = kl_div * self.kl_lambda
            kl_loss.backward(retain_graph=True)
        self.qf1_exp_optimizer.step()
        self.qf2_exp_optimizer.step()
        if self.gradient_from_Q:
            self.context_optimizer.step()


        # compute min Q on the new actions
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q_exp(obs, new_actions,z_mean_prev.detach(), z_var_prev.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_exp_criterion(v_pred, v_target.detach())
        self.vf_exp_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_exp_optimizer.step()
        self._update_target_network_exp()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_exp_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_exp_optimizer.step()

        # save some statistics for eval

        if self.eval_statistics_2 is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics_2 = OrderedDict()

            self.eval_statistics_2['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics_2['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics_2['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.gradient_from_Q:
                self.eval_statistics_2['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics_2['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            context_encoder=self.context_encoder.state_dict(),
            qf1_exp=self.qf1_exp.state_dict(),
            qf2_exp=self.qf2_exp.state_dict(),
            policy_exp=self.exploration_agent.state_dict(),
            vf_exp=self.vf_exp.state_dict(),
            target_vf_exp=self.target_exp_vf.state_dict(),
        )
        return snapshot

class ExpSACFin2(ExpAlgorithmFin2):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            nets_exp,
            encoder,
            latent_dim,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            use_info_in_context=False,
            entropy_weight=1e-2,
            intrinsic_reward_weight=1e-1,
            use_kl_div_intrinsic=False,
            gradient_from_Q=False,
            prediction_reward_scale=1,
            intrinsic_reward_decay = 1,
            kl_min_weight=5,
            pie_hidden_dim=15,
            consider_dynamics=0,
            prediction_transition_scale=1,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            agent_exp=nets_exp[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            encoder=encoder,
            **kwargs
        )
        self.use_kl_div_intrinsic = use_kl_div_intrinsic
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.entropy_weight = entropy_weight
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.latent_dim = latent_dim
        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.qf_exp_criterion = nn.MSELoss()
        self.vf_exp_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.pred_loss = nn.MSELoss()
        self.kl_lambda = kl_lambda
        self.prediction_reward_scale = prediction_reward_scale
        self.consider_dynamics = consider_dynamics
        self.prediction_transition_scale = prediction_transition_scale

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.gradient_from_Q = gradient_from_Q
        self.intrinsic_reward_decay = intrinsic_reward_decay
        self.kl_min_weight = kl_min_weight

        self.qf1, self.qf2, self.vf = nets[1:]
        self.qf1_exp, self.qf2_exp, self.vf_exp, self.rew_decoder, self.transition_decoder = nets_exp[1:]
        self.target_exp_vf = self.vf_exp.copy()
        self.target_vf = self.vf.copy()


        self.policy_exp_optimizer = optimizer_class(
            self.exploration_agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_exp_optimizer = optimizer_class(
            self.qf1_exp.parameters(),
            lr=qf_lr,
        )
        self.qf2_exp_optimizer = optimizer_class(
            self.qf2_exp.parameters(),
            lr=qf_lr,
        )
        self.vf_exp_optimizer = optimizer_class(
            self.vf_exp.parameters(),
            lr=vf_lr,
        )
        self.policy_optimizer = optimizer_class(
            self.agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=context_lr,
        )
        self.rew_optimizer = optimizer_class(
            self.rew_decoder.parameters(),
            lr=context_lr,
        )
        self.transition_optimizer = optimizer_class(
            self.transition_decoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return  [self.context_encoder] + [self.exploration_agent.policy] + [self.qf1_exp, self.qf2_exp, self.vf_exp, self.target_exp_vf,self.rew_decoder,self.transition_decoder] + [self.agent.policy,self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            sr = batch['sparse_rewards'][None, ...]
        else:
            sr = batch['rewards'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t, sr]

    def unpack_batch_context(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        sparse_r = batch['sparse_rewards'][None, ...]
        r = batch['rewards'][None, ...]
        if not sparse_reward:
            sparse_r = r
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        info = batch['env_infos'][None, ...]
        # print(o[0,:5],a[0,:5],r[0],sparse_r[0],no[0,:5])
        return [o, a, sparse_r, no, t, info, r]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        if self.use_per:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)[0]) for
                       idx in indices]
        else:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for
                       idx in indices]
        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        return unpacked

    def sample_context(self, indices, sequence=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx
                   in indices]
        context = [self.unpack_batch_context(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-3], dim=2)
        else:
            context = torch.cat(context[:-4], dim=2)
        return context, context_unbatched

    def pred_context(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0], 1, context[2].shape[2])
        tmp = torch.cat([r_0, context[2]], dim=1)
        a_0 = ptu.zeros(context[1].shape[0], 1, context[1].shape[2])
        tmp2 = torch.cat([a_0, context[1]], dim=1)
        tmp3 = torch.cat([torch.unsqueeze(context[0][:, 0, :], 1), context[3]], dim=1)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([tmp3, tmp2, tmp], dim=2)
        return contextr

    def sample_exp(self, indices,sequence=True):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.exp_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch_context(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-3], dim=2)
        else:
            context = torch.cat(context[:-4], dim=2)
        return  context,context_unbatched


    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        #_,exp_context_unbatched = self.sample_exp(indices,True)
        #exp_context_pred = self.pred_context(exp_context_unbatched)
        _, context_unbatched = self.sample_context(indices, False)
        context_pred = self.pred_context(context_unbatched)
        context = self.sample_sac(indices)
        # zero out context and hidden encoder state


        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):

            #self._take_step(indices, context_unbatched,context_pred)
            self._take_step_exp(indices, context_unbatched,context_pred,context)

            # stop backprop


    def _min_q_exp(self,  obs,actions,z_mean,z_var):
        #print(obs.shape,actions.shape)

        q1 = self.qf1_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        q2 = self.qf2_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        min_q = torch.min(q1, q2)
        return min_q


    def _min_q(self, obs, actions,z):
        #print(obs.shape,actions.shape)

        q1 = self.qf1(torch.cat([obs, actions,z],dim=1))
        q2 = self.qf2(torch.cat([obs, actions,z],dim=1))
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network_exp(self):
        ptu.soft_update_from_to(self.vf_exp, self.target_exp_vf, self.soft_target_tau)

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def compute_kl(self,means,vars):
        std_mean = ptu.zeros(means.size())
        std_var = ptu.ones(means.size())
        tem = vars / std_var
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((std_mean - means) / std_var) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        kl_div = torch.mean(kl_div)
        return kl_div

    def compute_intrinsic(self,z_mean_prev, z_var_prev,z_mean_post,z_var_post):
        tem = z_var_post / z_var_prev
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((z_mean_prev - z_mean_post) / z_var_prev) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        return kl_div

    def _take_step(self, indices, context_unbatched,context_pred,context):
        t,b,_ = context_pred.size()
        num_tasks = len(indices)
        z_s = self.context_encoder.forward_seq(context_pred)
        z_s = z_s.view(t,b,-1)
        z_mean = z_s[:,:-1,:self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:,:-1,self.latent_dim:])
        z_mean_post = z_s[:, 1:, self.latent_dim]
        z_var_post = torch.nn.functional.softplus(z_s[:, 1:, self.latent_dim:])
        z_dis = torch.distributions.Normal(z_mean,torch.sqrt(z_var))
        z_sample = z_dis.rsample()

        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        t,b,_ = obs.size()
        obs = obs.view(t*b,-1)
        actions = actions.view(t * b, -1)
        rewards = rewards.view(t * b, -1)
        z_sample = z_sample.view(t * b, -1)


        # run inference in networks
        policy_outputs = self.agent(obs, z_sample.detach())
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, z_sample)
        q2_pred = self.qf2(obs, actions, z_sample)
        v_pred = self.vf(obs, z_sample.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, z_sample)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        kl_div = self.compute_kl(z_mean,z_var)
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)

        kl_min_loss = self.compute_intrinsic(z_mean.contiguous().view(t*b,-1),z_var.contiguous().view(t*b,-1),z_mean_post.contiguous().view(t*b,-1),z_var_post.contiguous().view(t*b,-1))

        kl_min_loss = torch.mean(kl_min_loss) * self.kl_min_weight
        kl_min_loss.backward(retain_graph=True)



        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, z_sample.detach())

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
                self.eval_statistics['KL Min Loss'] = ptu.get_numpy(kl_min_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))



    def _take_step_exp(self, indices,context_unbatched,context_pred,context):

        t, b, _ = context_pred.size()
        b = b - 1
        context_pred_pre = context_pred [:,:-1,:]
        #context_pred = context_pred.contiguous()
        z_s = self.context_encoder.forward_seq(context_pred_pre)
        #z_s = z_s.view(t, b, -1)
        z_mean = z_s[:, :self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:, self.latent_dim:])
        # print(z_mean.shape,z_var.shape)
        z_dis = torch.distributions.Normal(z_mean, torch.sqrt(z_var))
        z_sample = z_dis.rsample()
        z_sample_pearl = z_sample

        obs, actions, agent_rew, next_obs, terms, sr = context
        if self.sparse_rewards:
            pred_rewardss = sr
        else:
            pred_rewardss = agent_rew

        t, b, _ = obs.size()
        #agent_rew = agent_rew.contiguous()
        #pred_rewardss = pred_rewardss.contiguous()
        agent_rew = agent_rew.view(t * b, -1)
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        pred_rewardss = pred_rewardss.view(t * b, -1)

        rewards_flat = agent_rew.detach()

        q1_pred = self.qf1(torch.cat([obs, actions, z_sample_pearl], dim=1))
        q2_pred = self.qf2(torch.cat([obs, actions, z_sample_pearl], dim=1))
        v_pred = self.vf(torch.cat([obs, z_sample_pearl.detach()], dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            target_v_values = self.target_vf(torch.cat([next_obs, z_sample_pearl], dim=1))

        # KL constraint on z if probabilistic


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        self.context_optimizer.zero_grad()
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)

        kl_div = self.compute_kl(z_mean, z_var)
        kl_loss = kl_div * self.kl_lambda
        kl_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        policy_outputs, _ = self.agent(obs, z_sample_pearl.detach())

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q(obs, new_actions, z_sample_pearl.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        rew_pred = self.rew_decoder.forward(z_sample_pearl.detach(), obs, actions)
        self.rew_optimizer.zero_grad()
        rew_loss = self.pred_loss(pred_rewardss, rew_pred) * self.prediction_reward_scale
        rew_loss.backward()
        self.rew_optimizer.step()
        if self.consider_dynamics:
            self.transition_optimizer.zero_grad()
            trans_pred = self.transition_decoder.forward(z_sample_pearl.detach(), obs, actions)
            trans_loss = self.pred_loss(next_obs, trans_pred) * self.prediction_transition_scale
            trans_loss.backward()
            self.transition_optimizer.step()

        policy_outputs, _ = self.exploration_agent(obs, z_mean.detach(), z_var.detach())

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        context_post = context_pred[:, 1:, :]
        context_post = context_post.contiguous()
        z_s_post = self.context_encoder.forward_seq(context_post)
        z_mean_post = z_s_post[:, :self.latent_dim]
        z_var_post = torch.nn.functional.softplus(z_s_post[:, self.latent_dim:])
        if self.intrinsic_reward_weight > 0:
            if self.use_kl_div_intrinsic:


                intrinsic_reward = self.compute_intrinsic(z_mean, z_var, z_mean_post, z_var_post).detach()
            else:

                pred_rew = self.rew_decoder.forward(z_sample.detach(), obs, actions)
                intrinsic_reward = (pred_rew - pred_rewardss) ** 2
                if self.consider_dynamics:
                    pred_trans = self.transition_decoder.forward(z_sample.detach(), obs, actions)
                    intrinsic_reward = intrinsic_reward + torch.mean((pred_trans - next_obs) ** 2, dim=1, keepdim=True)
            intrinsic_reward = intrinsic_reward.view(t * b, -1)
            if self.intrinsic_reward_decay != 1:
                intrinsic_reward = intrinsic_reward * torch.unsqueeze(
                    ptu.from_numpy(self.intrinsic_reward_decay ** np.linspace(0, t * b - 1, t * b)), 1)
            rew = intrinsic_reward * self.intrinsic_reward_weight + agent_rew
        else:
            rew = agent_rew
        rew = rew.detach()
        # print(z_mean.shape, z_mean_next.shape, obs.shape, t, b)
        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred_exp = self.qf1_exp(torch.cat([obs, actions, z_mean.detach(), z_var.detach()], dim=1))
        q2_pred_exp = self.qf2_exp(torch.cat([obs, actions, z_mean.detach(), z_var.detach()], dim=1))
        v_pred_exp = self.vf_exp(torch.cat([obs, z_mean.detach(), z_var.detach()], dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            #print(next_obs.shape,z_mean_post.shape)
            target_v_values = self.target_exp_vf(torch.cat([next_obs, z_mean_post, z_var_post], dim=1))

        # KL constraint on z if probabilistic

        self.qf1_exp_optimizer.zero_grad()
        self.qf2_exp_optimizer.zero_grad()
        rewards_flat = rew
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss_exp = torch.mean((q1_pred_exp - q_target) ** 2) + torch.mean((q2_pred_exp - q_target) ** 2)
        qf_loss_exp.backward()

        self.qf1_exp_optimizer.step()
        self.qf2_exp_optimizer.step()


        # compute min Q on the new actions
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q_exp(obs, new_actions, z_mean.detach(), z_var.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss_exp = self.vf_exp_criterion(v_pred_exp, v_target.detach())
        self.vf_exp_optimizer.zero_grad()
        vf_loss_exp.backward()
        self.vf_exp_optimizer.step()
        self._update_target_network_exp()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss_exp = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss_exp = policy_loss_exp + policy_reg_loss

        self.policy_exp_optimizer.zero_grad()
        policy_loss_exp.backward()
        self.policy_exp_optimizer.step()

        if self.eval_statistics_2 is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics_2 = OrderedDict()

            self.eval_statistics_2['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics_2['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics_2['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics_2['QF Loss Exp'] = np.mean(ptu.get_numpy(qf_loss_exp))
            self.eval_statistics_2['VF Loss Exp'] = np.mean(ptu.get_numpy(vf_loss_exp))
            self.eval_statistics_2['Policy Loss Exp'] = np.mean(ptu.get_numpy(
                policy_loss_exp
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions Exp',
                ptu.get_numpy(q1_pred_exp),
            ))


            self.eval_statistics_2['KL Divergence'] = ptu.get_numpy(kl_div)
            self.eval_statistics_2['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions Exp',
                ptu.get_numpy(v_pred_exp),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics_2['Z mean train'] = np.mean(ptu.get_numpy(z_mean))
            self.eval_statistics_2['Z variance train'] = np.mean(ptu.get_numpy(z_var))
            self.eval_statistics_2['reward prediction loss'] = ptu.get_numpy(rew_loss)
            if self.consider_dynamics:
                self.eval_statistics_2['transisition prediction loss'] = ptu.get_numpy(trans_loss)


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            context_encoder=self.context_encoder.state_dict(),
            qf1_exp=self.qf1_exp.state_dict(),
            qf2_exp=self.qf2_exp.state_dict(),
            policy_exp=self.exploration_agent.state_dict(),
            vf_exp=self.vf_exp.state_dict(),
            target_vf_exp=self.target_exp_vf.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
        )
        return snapshot

class ExpSACFin3(ExpAlgorithmFin3):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            nets_exp,
            encoder,
            latent_dim,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            use_info_in_context=False,
            entropy_weight=1e-2,
            intrinsic_reward_weight=1e-1,
            use_kl_div_intrinsic=False,
            gradient_from_Q=False,
            prediction_reward_scale=1,
            intrinsic_reward_decay = 1,
            kl_min_weight=5,
            pie_hidden_dim=15,
            consider_dynamics=0,
            prediction_transition_scale=1,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            agent_exp=nets_exp[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            encoder=encoder,
            **kwargs
        )
        self.use_kl_div_intrinsic = use_kl_div_intrinsic
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.entropy_weight = entropy_weight
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.latent_dim = latent_dim
        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.qf_exp_criterion = nn.MSELoss()
        self.vf_exp_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.pred_loss = nn.MSELoss()
        self.kl_lambda = kl_lambda
        self.prediction_reward_scale = prediction_reward_scale
        self.consider_dynamics = consider_dynamics
        self.prediction_transition_scale = prediction_transition_scale

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.gradient_from_Q = gradient_from_Q
        self.intrinsic_reward_decay = intrinsic_reward_decay
        self.kl_min_weight = kl_min_weight

        self.qf1, self.qf2, self.vf = nets[1:]
        self.qf1_exp, self.qf2_exp, self.vf_exp, self.rew_decoder, self.transition_decoder = nets_exp[1:]
        self.target_exp_vf = self.vf_exp.copy()
        self.target_vf = self.vf.copy()


        self.policy_exp_optimizer = optimizer_class(
            self.exploration_agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_exp_optimizer = optimizer_class(
            self.qf1_exp.parameters(),
            lr=qf_lr,
        )
        self.qf2_exp_optimizer = optimizer_class(
            self.qf2_exp.parameters(),
            lr=qf_lr,
        )
        self.vf_exp_optimizer = optimizer_class(
            self.vf_exp.parameters(),
            lr=vf_lr,
        )
        self.policy_optimizer = optimizer_class(
            self.agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=context_lr,
        )
        self.rew_optimizer = optimizer_class(
            self.rew_decoder.parameters(),
            lr=context_lr,
        )
        self.transition_optimizer = optimizer_class(
            self.transition_decoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return  [self.context_encoder] + [self.exploration_agent.policy] + [self.qf1_exp, self.qf2_exp, self.vf_exp, self.target_exp_vf,self.rew_decoder,self.transition_decoder] + [self.agent.policy,self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            sr = batch['sparse_rewards'][None, ...]
        else:
            sr = batch['rewards'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t, sr]

    def unpack_batch_context(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        sparse_r = batch['sparse_rewards'][None, ...]
        r = batch['rewards'][None, ...]
        if not sparse_reward:
            sparse_r = r
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        info = batch['env_infos'][None, ...]
        # print(o[0,:5],a[0,:5],r[0],sparse_r[0],no[0,:5])
        return [o, a, sparse_r, no, t, info, r]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        if self.use_per:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)[0]) for
                       idx in indices]
        else:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for
                       idx in indices]
        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        return unpacked

    def sample_context(self, indices, sequence=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx
                   in indices]
        context = [self.unpack_batch_context(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-3], dim=2)
        else:
            context = torch.cat(context[:-4], dim=2)
        return context, context_unbatched

    def pred_context(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0], 1, context[2].shape[2])
        tmp = torch.cat([r_0, context[2]], dim=1)
        a_0 = ptu.zeros(context[1].shape[0], 1, context[1].shape[2])
        tmp2 = torch.cat([a_0, context[1]], dim=1)
        tmp3 = torch.cat([torch.unsqueeze(context[0][:, 0, :], 1), context[3]], dim=1)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([tmp3, tmp2, tmp], dim=2)
        return contextr

    def sample_exp(self, indices,sequence=True):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.exp_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch_context(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-3], dim=2)
        else:
            context = torch.cat(context[:-4], dim=2)
        return  context,context_unbatched


    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        #_,exp_context_unbatched = self.sample_exp(indices,True)
        #exp_context_pred = self.pred_context(exp_context_unbatched)
        _, context_unbatched = self.sample_context(indices, False)
        context_pred = self.pred_context(context_unbatched)
        context = self.sample_sac(indices)
        # zero out context and hidden encoder state


        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):

            #self._take_step(indices, context_unbatched,context_pred)
            self._take_step_exp(indices, context_unbatched,context_pred,context)

            # stop backprop


    def _min_q_exp(self,  obs,actions,z_mean,z_var):
        #print(obs.shape,actions.shape)

        q1 = self.qf1_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        q2 = self.qf2_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        min_q = torch.min(q1, q2)
        return min_q


    def _min_q(self, obs, actions,z):
        #print(obs.shape,actions.shape)

        q1 = self.qf1(torch.cat([obs, actions,z],dim=1))
        q2 = self.qf2(torch.cat([obs, actions,z],dim=1))
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network_exp(self):
        ptu.soft_update_from_to(self.vf_exp, self.target_exp_vf, self.soft_target_tau)

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def compute_kl(self,means,vars):
        std_mean = ptu.zeros(means.size())
        std_var = ptu.ones(means.size())
        tem = vars / std_var
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((std_mean - means) / std_var) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        kl_div = torch.mean(kl_div)
        return kl_div

    def compute_intrinsic(self,z_mean_prev, z_var_prev,z_mean_post,z_var_post):
        tem = z_var_post / z_var_prev
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((z_mean_prev - z_mean_post) / z_var_prev) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        return kl_div

    def _take_step(self, indices, context_unbatched,context_pred,context):
        t,b,_ = context_pred.size()
        num_tasks = len(indices)
        z_s = self.context_encoder.forward_seq(context_pred)
        z_s = z_s.view(t,b,-1)
        z_mean = z_s[:,:-1,:self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:,:-1,self.latent_dim:])
        z_mean_post = z_s[:, 1:, self.latent_dim]
        z_var_post = torch.nn.functional.softplus(z_s[:, 1:, self.latent_dim:])
        z_dis = torch.distributions.Normal(z_mean,torch.sqrt(z_var))
        z_sample = z_dis.rsample()

        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        t,b,_ = obs.size()
        obs = obs.view(t*b,-1)
        actions = actions.view(t * b, -1)
        rewards = rewards.view(t * b, -1)
        z_sample = z_sample.view(t * b, -1)


        # run inference in networks
        policy_outputs = self.agent(obs, z_sample.detach())
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, z_sample)
        q2_pred = self.qf2(obs, actions, z_sample)
        v_pred = self.vf(obs, z_sample.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, z_sample)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        kl_div = self.compute_kl(z_mean,z_var)
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)

        kl_min_loss = self.compute_intrinsic(z_mean.contiguous().view(t*b,-1),z_var.contiguous().view(t*b,-1),z_mean_post.contiguous().view(t*b,-1),z_var_post.contiguous().view(t*b,-1))

        kl_min_loss = torch.mean(kl_min_loss) * self.kl_min_weight
        kl_min_loss.backward(retain_graph=True)



        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, z_sample.detach())

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
                self.eval_statistics['KL Min Loss'] = ptu.get_numpy(kl_min_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))



    def _take_step_exp(self, indices,context_unbatched,context_pred,context):

        t, b, _ = context_pred.size()
        b = b - 1
        context_pred_pre = context_pred [:,:-1,:]
        #context_pred = context_pred.contiguous()
        z_s = self.context_encoder.forward_seq(context_pred_pre)
        #z_s = z_s.view(t, b, -1)
        z_mean = z_s[:, :self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:, self.latent_dim:])
        # print(z_mean.shape,z_var.shape)
        z_dis = torch.distributions.Normal(z_mean, torch.sqrt(z_var))
        z_sample = z_dis.rsample()
        z_sample_pearl = z_sample




        obs, actions, agent_rew, next_obs, terms, sr = context
        if self.sparse_rewards:
            pred_rewardss = sr
        else:
            pred_rewardss = agent_rew

        rew_pred = self.rew_decoder.forward(z_sample_pearl.detach(), obs, actions)
        self.rew_optimizer.zero_grad()
        rew_loss = self.pred_loss(pred_rewardss, rew_pred) * self.prediction_reward_scale
        rew_loss.backward()
        self.rew_optimizer.step()
        if self.consider_dynamics:
            self.transition_optimizer.zero_grad()
            trans_pred = self.transition_decoder.forward(z_sample_pearl.detach(), obs, actions)
            trans_loss = self.pred_loss(next_obs, trans_pred) * self.prediction_transition_scale
            trans_loss.backward()
            self.transition_optimizer.step()


        if self.intrinsic_reward_weight > 0:
            if self.use_kl_div_intrinsic:
                intrinsic_reward = self.compute_intrinsic(z_mean, z_var, z_mean_post, z_var_post).detach()
            else:

                pred_rew = self.rew_decoder.forward(z_sample.detach(), obs, actions)
                intrinsic_reward = (pred_rew - pred_rewardss) ** 2
                if self.consider_dynamics:
                    pred_trans = self.transition_decoder.forward(z_sample.detach(), obs, actions)
                    intrinsic_reward = intrinsic_reward + torch.mean((pred_trans - next_obs) ** 2, dim=1, keepdim=True)
            intrinsic_reward = intrinsic_reward.view(t * b, -1)
            if self.intrinsic_reward_decay != 1:
                intrinsic_reward = intrinsic_reward * torch.unsqueeze(
                    ptu.from_numpy(self.intrinsic_reward_decay ** np.linspace(0, t * b - 1, t * b)), 1)
            rew = intrinsic_reward * self.intrinsic_reward_weight + agent_rew
        else:
            rew = agent_rew
        rew = rew.detach()
        agent_rew = rew

        t, b, _ = obs.size()
        #agent_rew = agent_rew.contiguous()
        #pred_rewardss = pred_rewardss.contiguous()
        agent_rew = agent_rew.view(t * b, -1)
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        pred_rewardss = pred_rewardss.view(t * b, -1)

        rewards_flat = agent_rew.detach()

        q1_pred = self.qf1(torch.cat([obs, actions, z_sample_pearl], dim=1))
        q2_pred = self.qf2(torch.cat([obs, actions, z_sample_pearl], dim=1))
        v_pred = self.vf(torch.cat([obs, z_sample_pearl.detach()], dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            target_v_values = self.target_vf(torch.cat([next_obs, z_sample_pearl], dim=1))

        # KL constraint on z if probabilistic


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        self.context_optimizer.zero_grad()
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)

        kl_div = self.compute_kl(z_mean, z_var)
        kl_loss = kl_div * self.kl_lambda
        kl_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        policy_outputs, _ = self.agent(obs, z_sample_pearl.detach())

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q(obs, new_actions, z_sample_pearl.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()





        if self.eval_statistics_2 is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics_2 = OrderedDict()

            self.eval_statistics_2['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics_2['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics_2['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics_2['QF Loss Exp'] = np.mean(ptu.get_numpy(qf_loss_exp))
            self.eval_statistics_2['VF Loss Exp'] = np.mean(ptu.get_numpy(vf_loss_exp))
            self.eval_statistics_2['Policy Loss Exp'] = np.mean(ptu.get_numpy(
                policy_loss_exp
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions Exp',
                ptu.get_numpy(q1_pred_exp),
            ))


            self.eval_statistics_2['KL Divergence'] = ptu.get_numpy(kl_div)
            self.eval_statistics_2['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions Exp',
                ptu.get_numpy(v_pred_exp),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics_2['Z mean train'] = np.mean(ptu.get_numpy(z_mean))
            self.eval_statistics_2['Z variance train'] = np.mean(ptu.get_numpy(z_var))
            self.eval_statistics_2['reward prediction loss'] = ptu.get_numpy(rew_loss)
            if self.consider_dynamics:
                self.eval_statistics_2['transisition prediction loss'] = ptu.get_numpy(trans_loss)


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            context_encoder=self.context_encoder.state_dict(),
            qf1_exp=self.qf1_exp.state_dict(),
            qf2_exp=self.qf2_exp.state_dict(),
            policy_exp=self.exploration_agent.state_dict(),
            vf_exp=self.vf_exp.state_dict(),
            target_vf_exp=self.target_exp_vf.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
        )
        return snapshot