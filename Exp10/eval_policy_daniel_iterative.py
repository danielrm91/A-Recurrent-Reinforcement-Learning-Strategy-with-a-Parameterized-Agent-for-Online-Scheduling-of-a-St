'''
Use it for evaluate the policy
'''
import numpy as np
import torch
import csv
import pandas as pd
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from scipy import stats as st

def _log_summary(ep_len, ep_ret, ep_num):
    #Round the values
    print('ep_len',ep_len)
    print('ep_rt', ep_ret)
    if isinstance(ep_ret, np.ndarray):
            ep_ret = torch.tensor(ep_ret, dtype=torch.float)
            ep_ret = str(torch.round(ep_ret, decimals=3))
    else:
        ep_ret = str(torch.round(ep_ret, decimals=3))
    
    #Print statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"-----------------------------------------------------------", flush=True)
    print(flush=True)
    
def rollout(policy, env, render):
    #Rollout 
    max_timesteps_per_episode = 144
    prev = env.reset(seq_len=10)       # CREATE OBS SEQUENCE, seq_len, input_len
    mask_vec = env.reset_m(prev[-1])
    done = False
    t = 0
    ep_len = 0
    ep_ret = 0
    times = [0,0,0,0,0,0,0,0,0,0,0]
    out_ac2_dim = 10
    e = []
    h_, c_ = (torch.zeros((3,1,25)),torch.zeros((3,1,25)))
    
    for ep_t in range(max_timesteps_per_episode+1):  
        max_vals_obs = torch.tensor([5,8,6,8,3,4,30,30,20,20,20,20,30,20,20,20,20,20,1,1,1,1,1,1,25,40,20], dtype=torch.float32)
        t += 1
        obs_normalized = np.divide(prev,max_vals_obs)    
        daction, caction, h_, c_ = get_action(policy, obs_normalized, mask_vec, h_,c_)
        if daction[0][0] != 10:
            a = torch.clone(prev[-1])
            a = a.numpy()
            b = [daction[0][0]] #this is a numpy
            c = [caction[0][0][daction[0][0]]] #[0.75*(caction[daction[0][0]]) + 0.25]
            d = np.concatenate((a,b,c))
        else: #if the action is 10
            a = torch.clone(prev[-1])
            a = a.numpy()
            b = [daction[0][0]] #this is a numpy
            c = [np.array(0)]
            d = np.concatenate((a,b,c))
        d = np.round(d, 2)
        e.append(d)                         
        calle = torch.clone(prev[-1])
        caction = caction[0][0].reshape([1,10])  #it was just caction.reshape([1,10])
        obs3, rew, done, times, mask_vec = env.step1(daction[0][0], caction, calle, times)
        ep_ret += rew
        obs3 = obs3.reshape([1,27])
        prev = torch.cat((prev,obs3),0)    
        prev = prev[1:10+1] 

    #Last observation withs actions 10 and 0
    a = torch.clone(prev[-1])
    a = a.numpy()
    b = [np.array(0)] #this is a numpy
    c = [np.array(0)]
    d = np.concatenate((a,b,c))
    d = np.round(d, 3)
    e.append(d)

    df = pd.DataFrame(e) #convert to a dataframe
    df.columns = ['U1','U2','U3','U4','U5','U6','S1','S2','S3','S4','S5',
                   'S6','Int1','Int2','P1','P2','P3','WS','TU1','TU2',
                   'TU3','TU4','TU5','TU6','CW','LPS','HPS','Disc','Cont']
    df.to_csv("test_file.csv", index=False) #save to file
    ep_len = t
    yield ep_len, ep_ret
    
def get_action(policy, obs_n, mask_vec,h,c):
    d_iter = []
    c_iter = []  
    # Get action
    act_d, act_c, h,c = policy(obs_n, mask_vec,h,c)
    cat = Categorical(act_d)

    #dist = MultivariateNormal(act_c, cov_mat)
    #action_d = cat.sample()
    #action_c = dist.sample()

    action_d = torch.argmax(cat.probs)

    #print('sample', action_d, action_c)
    #print('mean', action_d, act_c)
    action_d = torch.reshape(action_d,(1,))
    action_d = action_d.detach().numpy()
    d_iter.append(action_d)
    daction = d_iter
    action_c = act_c.detach().numpy()
    c_iter.append(action_c)
    caction = c_iter
    return daction, caction, h, c
    
def eval_policy(policy, env, render=False):
    open('test_file.csv','w').close
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy,env,render)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
