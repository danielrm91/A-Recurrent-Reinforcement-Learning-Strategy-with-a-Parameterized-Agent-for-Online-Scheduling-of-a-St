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
import torch.nn.functional as F

def _log_summary(ep_len, ep_ret, ep_num):
    #Round the values
    ep_len = str(round(ep_len,2))
    #if isinstance(ep_ret, np.ndarray):
    #        ep_ret = torch.tensor(ep_ret, dtype=torch.float)
    #        ep_ret = str(torch.round(ep_ret, decimals=3))
    #else:
    #    ep_ret = str(torch.round(ep_ret, decimals=3))
    
    #Print statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"-----------------------------------------------------------", flush=True)
    print(flush=True)
    
def rollout(policy, env, render):
    #Rollout 
    max_timesteps_per_episode = 30
    prev = torch.zeros([4,15])      # CREATE OBS SEQUENCE, seq_len, input_len
    mask_vec = env.reset_m(prev[-1])
    done = False
    t = 0
    ep_len = 0
    ep_ret = 0
    times = [0,0,0,0,0,0,0,0,0]
    e = []
    h_, c_ = (torch.zeros((3,1,25)),torch.zeros((3,1,25)))
    
    for ep_t in range(max_timesteps_per_episode+1):  
        max_vals_obs = torch.tensor([100,80,50,200,300,300,300,300,200,200,4,4,2,2,30], dtype=torch.float32)
        t += 1
        obs_normalized = np.divide(prev,max_vals_obs)
        daction, caction, h_, c_ = get_action(policy, obs_normalized, mask_vec, h_,c_)
        #daction, caction = get_action1(ep_t)
        if daction[0][0] != 8:
            a = torch.clone(prev[-1])
            a = a.numpy()
            b = [daction[0][0]] #this is a numpy
            c = [caction[0][0][daction[0][0]]] #[0.75*(caction[daction[0][0]]) + 0.25] 
            d = np.concatenate((a,b,c))
        else: #if the action is 8
            a = torch.clone(prev[-1])
            a = a.numpy()
            b = [daction[0][0]] #this is a numpy
            c = [np.array(0)]
            d = np.concatenate((a,b,c))
        d = np.round(d, 2)
        e.append(d)                         
        calle = torch.clone(prev[-1])
        caction = caction[0][0].reshape([1,8])  #it was just caction.reshape([1,10])  
        obs3, rew, done, times, mask_vec = env.step1(daction[0][0], caction, calle, times)
        ep_ret += rew
        obs3 = obs3.reshape([1,15])
        prev = torch.cat((prev,obs3),0)    
        prev = prev[1:4+1] 

    #Last observation withs actions 10 and 0
    a = torch.clone(prev[-1])
    a = a.numpy()
    b = [np.array(0)] #this is a numpy
    c = [np.array(0)]
    d = np.concatenate((a,b,c))
    d = np.round(d, 3)
    e.append(d)

    df = pd.DataFrame(e) #convert to a dataframe
    df.columns = ['H','R1','R2','D','HA','intBC','intAB','impE','P1','P2','tR1','tR2','H','D','Times','Disc','Cont']
    df.to_csv("test_file_kondili.csv", index=False) #save to file
    ep_len = t
    yield ep_len, ep_ret
    
def get_action(policy, obs_n, mask_vec,h,c):
    d_iter = []
    c_iter = []  
    # Get action
    act_d, act_c, h,c = policy(obs_n, mask_vec,h,c)

    act_d = act_d/0.001
    act_d = F.softmax(act_d,dim=-1)
    cat = Categorical(act_d)
    action_d = torch.argmax(cat.probs)

    ##################################################   Erase when not using samples from the covariance or the model
    #cov_var = torch.full(size=(8,),fill_value=1e-5)      # fill value: variance, eg variance=0.01, std dev 0.1
    #cov_mat = torch.diag(cov_var) 
    #act_d = act_d/0.001
    #act_d = F.softmax(act_d,dim=-1)
    #cat = Categorical(act_d)
    #dist = MultivariateNormal(act_c, cov_mat)
    #action_d = cat.sample()
    #action_c = dist.sample()
    #################################################

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
    open('test_file_kondili.csv','w').close
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy,env,render)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

