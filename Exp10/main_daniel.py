'''
this file is based on the work on 
https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
without some functions that are not convenient for the purpose
'''

import gym
import gym_kondili
import sys
import torch

from arguments_daniel import get_args
from eval_policy_daniel_iterative import eval_policy
from ppo_daniel import PPO                     #import this class from ppo
from network_daniel import RNNactor  #import this class from network
from network_daniel import RNNcritic #import this class from network

def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training",flush=True)
    #Create a model for PPO
    model = PPO(policy_class=RNNactor, value_class=RNNcritic, env=env, **hyperparameters)
    
    #load existing model
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Succesfully loaded.",flush=True)
    elif actor_model != '' or critic_model != '':
        print(f"Error: Either specify both actor/critic models or non at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)
    model.learn(total_timesteps = 3_360_000)

def test(env, actor_model):
    print(f"Testing {actor_model}", flush=True)
    if actor_model =='':
        print(f"Didn't specify model file. Exiting", flush=True)
        sys.exit(0)
    seq_len = 4
    obs_dim = 27
    sen_dim = 250
    actd_dim = 11
    actc_dim = 10
    policy = RNNactor(obs_dim, sen_dim, actd_dim, actc_dim, seq_len)
    policy.load_state_dict(torch.load(actor_model))
    eval_policy(policy=policy, env=env)
    
def main(args):
    hyperparameters = {
        'timesteps_per_batch':          480,       #610 for 30 steps,  1220 for 60, 2900
        'seq_len':                      4,         #new hyperparameter
        'max_timesteps_per_episode':    48,
        'gamma':                        0.99,
        'n_updates_per_iteration':      5,        #8 #6       
        'lr':                           1e-4,      #1e-6  #1e-2
        'clip':                         0.2,       #0.2
        'ent_init':                     0.01,      #0.001
        'max_grad_norm':                0.5,
        'var' :                         1e-2,
        'temp':                         1,
        'num_minibatches' :             5,
        'lam':                          0.98,
        'target_kl':                    0.02,  
    }
    env = gym.make('kondili-v0')
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)
        
if __name__ == '__main__':
    args = get_args()
    main(args)
