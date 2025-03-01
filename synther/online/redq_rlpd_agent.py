import numpy as np
import torch
from redq.algos.core import (ReplayBuffer,
                             soft_update_model1_with_model2)
from redq.algos.redq_sac import REDQSACAgent
from synther.online.conditional_nets import Curiosity
from torch import Tensor
from tqdm import trange


def combine_two_tensors(tensor1, tensor2):
    return Tensor(np.concatenate([tensor1, tensor2], axis=0))


class REDQRLPDCondAgent(REDQSACAgent):

    def __init__(self, cond_hidden_size, diffusion_buffer_size=int(1e6), diffusion_sample_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=diffusion_buffer_size)
        self.diffusion_sample_ratio = diffusion_sample_ratio

        self.cond_net = Curiosity(input_size=self.obs_dim, 
                                    hidden_size=cond_hidden_size, 
                                    output_size=self.act_dim).to(self.device)
        self.cond_optimizer = torch.optim.Adam(self.cond_net.parameters(), lr=self.lr)
    
    def get_current_num_data(self):
        # used to determine whether we should get action from policy or take random starting actions
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            if self.get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False,
                                             return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def train(self, logger):
        # Put conditional net in training mode
        self.cond_net.train()
        # this function is called after each datapoint collected.
        # when we only have very limited data, we don't make updates
        num_update = 0 if self.get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        for i_update in range(num_update):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)
            
            """Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """policy and alpha loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                # get policy loss
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                # get alpha loss
                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])

            """update networks"""
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                self.policy_optimizer.step()

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)


            """Update Cond Net"""
            # Only on last iteration for now
            # And only with real data
            if i_update == num_update - 1:
                if self.cond_optimizer is not None:
                    cond_obs_tensor, cond_obs_next_tensor, cond_acts_tensor, _, _ = self.sample_real_data(self.batch_size)
                    self.cond_optimizer.zero_grad()
                    cond_loss = self.cond_net.forward_loss(cond_obs_tensor, cond_obs_next_tensor, cond_acts_tensor)
                    cond_loss.backward()
                    self.cond_optimizer.step()
                else:
                    cond_loss = Tensor([0])
            else:
                cond_loss = Tensor([0])

            # by default only log for the last update out of <num_update> updates
            if i_update == num_update - 1:
                logger.store(LossCond=cond_loss.cpu().item(), LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

        # if there is no update, log 0 to prevent logging problems
        if num_update == 0:
            logger.store(LossCond=0, LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    def sample_data(self, batch_size):
        diffusion_batch_size = int(batch_size * self.diffusion_sample_ratio)
        online_batch_size = int(batch_size - diffusion_batch_size)
        # Sample from the diffusion buffer
        if self.diffusion_buffer.size < diffusion_batch_size:
            return super().sample_data(batch_size)
        diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
        online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size)
        obs_tensor = combine_two_tensors(online_batch['obs1'], diffusion_batch['obs1']).to(self.device)
        obs_next_tensor = combine_two_tensors(online_batch['obs2'], diffusion_batch['obs2']).to(self.device)
        acts_tensor = combine_two_tensors(online_batch['acts'], diffusion_batch['acts']).to(self.device)
        rews_tensor = combine_two_tensors(online_batch['rews'], diffusion_batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = combine_two_tensors(online_batch['done'], diffusion_batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def sample_real_data(self, batch_size):
        return super().sample_data(batch_size)

    def reset_diffusion_buffer(self):
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                             size=self.diffusion_buffer.max_size)

