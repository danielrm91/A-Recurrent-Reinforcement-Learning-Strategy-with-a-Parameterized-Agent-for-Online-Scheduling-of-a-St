

'''
PPO class to train and learn the policy
'''

import gym
import gym_kondili
import time
import csv
import numpy as np
import torch
import random
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

#OBS:
#    [H, R1, R2, D, HA, intBC, intAB, impE, P1, P2, tR1, tR2, H,  D,  t ]
#    [0  1   2   3  4   5      6      7     8   9   10   11   12  13  14]

class PPO:
    def __init__(self, policy_class, value_class, env, **hyperparameters):
        self.max_vals_obs = torch.tensor([100,80,50,200,300,300,300,300,200,200,4,4,2,2,30], dtype=torch.float32)
        self._init_hyperparameters(hyperparameters)
        self.env = env
        self.obs_dim =      15
        self.sen_dim =      250
        self.out_ac1_dim =  9  #Discrete actor, Action 9 is idle state
        self.out_ac2_dim =  8  #Continuous actor
        # Initialize Actor and Critic
        self.actor = policy_class(self.obs_dim, self.sen_dim, self.out_ac1_dim, self.out_ac2_dim)
        self.critic = value_class(self.obs_dim, 1)
        # Initialize optimizers for Actor and Critic
        self.actor_optim = Adam(self.actor.parameters(), lr= self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        #Initialize the covariance matrix used to query the actor for actions
        #self.cov_var = torch.full(size=(self.out_ac2_dim,),fill_value=0.04)
        #self.cov_mat = torch.diag(self.cov_var)
        
        # This logger will help for printing the summary
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0, 
            'batch_lens': [],
            'batch_rews': [],
            'actord_losses': [],
            'actorc_losses': [],
        }
        
    def learn(self, total_timesteps):
        print(f"Learning...Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            
            #Variance annealing
            t_it = total_timesteps / self.timesteps_per_batch                       # Number of total iterations in this training
            var =  self.var * (1e-5/self.var)**(i_so_far/t_it)                      # 8e-6 min value of var                DONT FORGET TO SET THIS
            self.cov_var = torch.full(size=(self.out_ac2_dim,),fill_value=var)      # fill value: variance, eg variance=0.01, std dev 0.1
            self.cov_mat = torch.diag(self.cov_var)                                 # square matrix with diagonal equal to variance, rest 0s

            #Learning rate annealing
            new_lr = self.lr * (1e-6/self.lr)**(i_so_far/t_it)                      # 5e-8 min value of lr   DONT FORGET TO SET THIS
            self.actor_optim.param_groups[0]["lr"] = new_lr
            self.critic_optim.param_groups[0]["lr"] = new_lr

            #Temperature annealing
            self.temperature = self.temp * (0.1/self.temp)**(i_so_far/t_it)                   # initial * (final/initial)***(t/T)

            #Entropy annealing
            self.ent_coeff = self.ent_init * (0.01/self.ent_init)**(i_so_far/t_it)  # initial * (final/initial)***(t/T)            

            #Calling Rollout

            batch_obs, batch_mask_vec, batch_dacts, batch_cacts, batch_log_probsd, batch_log_probsc, batch_lens, \
                 batch_vals, batch_rews, done_flags = self.rollout() 

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            #Normalize the obs by ividing by max values
            batch_obs_n = np.divide(batch_obs, self.max_vals_obs)
            batch_obs_n = batch_obs_n.reshape(-1,self.obs_dim)
            
            V = self.evaluate_V(batch_obs_n)

            #Minibatch construction
            b_size = batch_obs_n.size(0)
            idxs = np.arange(b_size)
            mini_batchsize = b_size//self.num_minibatches

            # Generalized Advantage Estimation function 
            A_k = self.GAE(batch_rews, batch_vals, done_flags)
            V = V.squeeze()
            batch_rtgs = A_k + V.detach()

            # For a more stable convergence
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Loop for updating the network n number of epochs
            for _ in range(self.n_updates_per_iteration):
                np.random.shuffle(idxs)
                for start in range(0, b_size, mini_batchsize):
                    end = start + mini_batchsize
                    id = idxs[start:end]
                    min_batch_obs_n         = batch_obs_n[id,:]
                    min_batch_mask_vec      = batch_mask_vec[id,:]
                    min_batch_dacts         = batch_dacts[id,:]
                    min_batch_cacts         = batch_cacts[id,:]
                    min_batch_log_probsd    = batch_log_probsd[id]
                    min_batch_log_probsc    = batch_log_probsc[id]
                    min_batch_rtgs          = batch_rtgs[id]
                    min_A_k                 = A_k[id]

                # Update Continuous Agent
                curr_log_probsc, entropyc = self.evaluate_c(min_batch_obs_n, min_batch_mask_vec, min_batch_cacts)
                logratios_c = curr_log_probsc - min_batch_log_probsc
                ratioc = torch.exp(logratios_c)
                surr1c = ratioc * min_A_k
                surr2c = torch.clamp(ratioc, 1-self.clip, 1+self.clip) * min_A_k
                actorc_loss = (-torch.min(surr1c, surr2c)).mean()- entropyc.mean() * self.ent_coeff
                #gp = open('actorc_loss','a')
                #writer = csv.writer(gp, lineterminator='\n')
                #writer.writerow([np.array(actorc_loss.detach())])
                #gp.close()
                self.logger['actorc_losses'].append(actorc_loss.detach())
                #Code to see the names of the layers
                #for name, param in self.actor.named_parameters(): 
                #    if param.requires_grad: 
                #        print(name,param.data.shape)
                #for i, param in enumerate(self.actor.parameters()):
                #         print(i,param.data.shape)
                # Deactivate the gradients from the Discrete Actor
                for i, param in enumerate(self.actor.parameters()):
                    if i in [6,7,8,9,10,11]:
                        param.requires_grad = False
                # Optimize the whole network (with the frozen params)
                self.actor_optim.zero_grad()
                actorc_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                # Reactivate Everything
                for i, param in enumerate(self.actor.parameters()):
                    param.requires_grad = True
                    
                # Update Discrete Agent
                curr_log_probsd, entropyd = self.evaluate_d(min_batch_obs_n, min_batch_mask_vec, min_batch_dacts)
                logratios_d = curr_log_probsd - min_batch_log_probsd
                ratiod = torch.exp(logratios_d)
                surr1d = ratiod * min_A_k
                surr2d = torch.clamp(ratiod, 1-self.clip, 1+self.clip) * min_A_k
                actord_loss = (-torch.min(surr1d, surr2d)).mean() - entropyd.mean() * self.ent_coeff
                #gl = open('actord_loss','a')
                #writer = csv.writer(gl, lineterminator='\n')
                #writer.writerow([np.array(actord_loss.detach())])
                #gl.close()
                self.logger['actord_losses'].append(actord_loss.detach())
                # Deactivate the gradients from the Continuous Actor
                for i, param in enumerate(self.actor.parameters()):
                    if i in [12,13,14,15,16,17]:
                        param.requires_grad = False
                # Optimize the whole network (with the frozen params)
                self.actor_optim.zero_grad()
                actord_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                # Reactivate everything
                for i, param in enumerate(self.actor.parameters()):
                    param.requires_grad = True
                
                # Update the Critic 
                V = self.evaluate_V(min_batch_obs_n)
                critic_loss = nn.MSELoss()(V, min_batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                
            # Print a summary
            self._log_summary()
            # Save the model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor_exp3.pth')
                torch.save(self.critic.state_dict(), './ppo_critic_exp3.pth')
    
    def rollout(self):
        batch_obs =         []
        batch_dacts =       []
        batch_cacts =       []
        batch_log_probsd =  []
        batch_log_probsc =  []
        batch_rews =        []
        batch_lens =        []
        batch_mask_vec =    []
        done_flags =        []
        batch_vals =        []
        t = 0
        # Episode run
        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []
            done = False
            prev = torch.zeros([self.obs_size])
            mask_vec = self.env.reset_m(prev)
            times = [0,0,0,0,0,0,0,0,0]
            for ep_t in range(self.max_timesteps_per_episode):    #+1
                t += 1     
                batch_obs.append(torch.clone(prev))
                batch_mask_vec.append(mask_vec)
                daction, caction, log_probd, log_probc = self.get_action(prev, mask_vec)
                val = self.critic(prev)
                call = torch.clone(prev)
                obs1, rew, done, times, mask_vec = self.env.step1(daction, caction, call, times) 
                ep_dones.append(done)
                prev = obs1  #obs1.reshape([1,self.obs_size])
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_dacts.append(daction)
                batch_cacts.append(caction)
                batch_log_probsd.append(log_probd)
                batch_log_probsc.append(log_probc)   
                if done:
                    break
            batch_vals.append(ep_vals)
            done_flags.append(ep_dones)
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)
        # Reshape data as tensors in the appropiate shape
        batch_obs       = torch.concat(batch_obs)
        batch_obs       = batch_obs.reshape([-1,self.obs_dim]) #shape [310,15]
        batch_mask_vec   = np.array(batch_mask_vec)
        batch_mask_vec  = torch.tensor(batch_mask_vec, dtype=bool)
        batch_dacts     = np.array(batch_dacts)
        batch_dacts     = torch.tensor(batch_dacts, dtype=torch.int)
        batch_dacts     = batch_dacts.reshape([-1,1])
        batch_cacts     = np.array(batch_cacts)
        batch_cacts     = torch.tensor(batch_cacts, dtype=torch.float)
        batch_log_probsd = torch.tensor(batch_log_probsd, dtype=torch.float)
        batch_log_probsc = torch.tensor(batch_log_probsc, dtype=torch.float)
        #batch_rtgs = self.compute_rtgs(batch_rews)
        # Save the episodic returns and lenghts
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        return batch_obs, batch_mask_vec, batch_dacts, batch_cacts, batch_log_probsd, batch_log_probsc, batch_lens, batch_vals, batch_rews, done_flags
        
    #def compute_rtgs(self, batch_rews):
    #    batch_rtgs = []
    #    for idx, ep_rews in enumerate(reversed(batch_rews)):
    #        discounted_reward = 0
    #        for rew in reversed(ep_rews):
    #            discounted_reward = rew + discounted_reward * self.gamma
    #            batch_rtgs.insert(0, discounted_reward)
    #    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    #    return batch_rtgs

    def GAE(self, rewards, values, dones):
        advs = []
        for ep_rews, ep_vals, ep_done in zip (rewards,values,dones):
            advantages = []
            last_adv = 0
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_done[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]
                
                advantage = delta + self.gamma * self.lam * (1-ep_done[t]) * last_adv
                last_adv = advantage
                advantages.insert(0,advantage)
            advs.extend(advantages)
        return torch.tensor(advs, dtype=torch.float)
    
    def get_action(self, obs2, mask_vec):
        # Normalize the obs by dividing by max values
        obs_n = np.divide(obs2,self.max_vals_obs)
        # Get action
        act_d, act_c = self.actor(obs_n, mask_vec)
        # Apply Temperature
        act_d = act_d/self.temperature
        act_d = F.softmax(act_d, dim=-1)
        cat = Categorical(act_d)
        dist = MultivariateNormal(act_c, self.cov_mat)
        # Sample an action from the distributions
        action_d = cat.sample()
        action_c = dist.sample()
        # Calculate the log probability
        log_prob_d = cat.log_prob(action_d)
        log_prob_c = dist.log_prob(action_c)
        # Detach and make np arrays
        action_d = action_d.detach().numpy()
        action_c = action_c.detach().numpy()
        log_prob_d = log_prob_d.detach()
        log_prob_c = log_prob_c.detach()     
        return action_d, action_c, log_prob_d, log_prob_c
    
    def evaluate_V(self, batch_obsv):
        V = self.critic(batch_obsv).squeeze()
        return V
    
    def evaluate_d(self, batch_obsd, batch_mask_vec, batch_dacts):
        act_d, _ = self.actor(batch_obsd, batch_mask_vec)
        act_d = act_d/self.temperature
        act_d = F.softmax(act_d, dim=-1)
        meand = torch.tensor([])
        mean_en = torch.tensor([])
        for i in range(len(batch_obsd)):
            cat = Categorical(act_d[i])
            meand = torch.cat((meand, cat.log_prob(batch_dacts[i])))
            cat_ent = torch.tensor([cat.entropy().detach()])
            mean_en = torch.cat((mean_en,cat_ent))
        return meand, mean_en
        
    def evaluate_c(self, batch_obsc, batch_mask_vec, batch_cacts):
        _, act_c = self.actor(batch_obsc, batch_mask_vec)
        batch_cacts_s = batch_cacts.squeeze()
        dist = MultivariateNormal(act_c, self.cov_mat)
        meanc = dist.log_prob(batch_cacts_s)
        ent_c = dist.entropy()
        return meanc, ent_c
    
    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch =          310
        self.max_steps_per_episodes =        30
        self.n_updates_per_iteration =        5
        self.obs_size =                      15
        self.lr =                          1e-3
        self.gamma =                       0.99
        self.clip =                        0.2
        self.save_freq =                     10
        self.seed =                         None
        self.var =                          0.04
        self.temp =                         0.1
        self.ent_init =                    0.001
        self.num_minibatches =              5
        self.max_grad_norm =                0.5
        self.lam =                          0.98
        self.target_kl =                    0.02
        
        for param, val in hyperparameters.items():
            exec('self.' + param + '=' + str(val))

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            print(f'Succesfully set used to {self.seed}')
    
    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t,2))
        
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews2 = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actord_loss = np.mean([losses.float().mean() for losses in self.logger['actord_losses']])
        avg_actorc_loss = np.mean([losses.float().mean() for losses in self.logger['actorc_losses']])
        
        # Rounding to decimal places
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews2, 2))       
        avg_actord_loss = str(round(avg_actord_loss, 5))
        avg_actorc_loss = str(round(avg_actorc_loss, 5))

        # Save values of rewards
        avg_ep_rews1 = (round(avg_ep_rews2,2))
        f = open('rewards_exp3_2', 'a')
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow([avg_ep_rews1])
        f.close()
        
        # Print 
        print(flush=True)
        print(f"----------------- Iteration #{i_so_far}-----------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Discrete Loss: {avg_actord_loss}", flush=True) 
        print(f"Average Continuous Loss: {avg_actorc_loss}", flush=True)
        print(f"Timesteps so Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t}", flush=True)
        print(f"--------------------------------------------------------", flush=True)
        print(flush=True)
        
        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actord_losses'] = []
        self.logger['actorc_losses'] = []