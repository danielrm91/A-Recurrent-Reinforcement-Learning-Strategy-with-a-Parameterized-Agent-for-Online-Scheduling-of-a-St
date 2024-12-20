import gym 
import numpy as np
from gym import spaces
import csv
import random
import torch

class KondiliEnv(gym.Env):           #action    #accumulated times
    def __init__(self):   
        self.observation_space = spaces.Dict({"agent" : spaces.Box(0, 1, shape=(2,), dtype=int),
                                              "target": spaces.Box(0, 1, shape=(2,), dtype=int),})
        self.action_space = spaces.Discrete(4)
        self.max_steps = 48
        self.max_val = 1.0
        
    def step1(self, d_action, c_action, observations, times):  
        self.d_action = d_action
        self.c_action = c_action[0]
        obs = observations  
        t = times   
        self.r = 0
        self.done = False
        self.available_actions = np.array([1,1,1,1,1,1,1,1,1,1,1], dtype=np.int32)
        
        #capacities       
        unit1  =  5
        unit2  =  8
        unit3  =  6
        unit4  =  8 
        unit5  =  3
        unit6  =  4
        state1 =  30
        state2 =  30 
        state3 =  20
        state4 =  20
        state5 =  20
        state6 =  20
        int1   =  30
        int2   =  20

        #time units: 1 unit = 1 hours
        task1t  =  4 #12 #2 
        task2t  =  2 #6 #1 
        task3t  =  2 #6 #1 
        task4t  =  4 #12 #2 
        task5t  =  4 #12 #2 
        task6t  =  4 #12 #2
        task7t  =  8 #24 #4
        task8t  =  4 #12 #2
        task9t  =  4 #12 #2
        task10t =  6 #18 #3

        #Utilities kg/(min*kg) in batch
        ut1 = 2
        ut2 = 2
        ut3 = 3
        ut4 = 2
        ut5 = 4
        ut6 = 3
        ut7 = 4
        ut8 = 3
        ut9 = 3
        ut10 = 3

        #Maximum limit kg/min
        cw = 25
        lps = 40
        hps = 20
        
        #continuous
        a = np.clip(self.c_action[0], 0.0, 1.00)   #task1                
        b = np.clip(self.c_action[1], 0.0, 1.00)   #task2     
        c = np.clip(self.c_action[2], 0.0, 1.00)   #task3   
        d = np.clip(self.c_action[3], 0.0, 1.00)   #task4   
        e = np.clip(self.c_action[4], 0.0, 1.00)   #task5    
        f = np.clip(self.c_action[5], 0.0, 1.00)   #task6     
        g = np.clip(self.c_action[6], 0.0, 1.00)   #task7  
        h = np.clip(self.c_action[7], 0.0, 1.00)   #task8                
        i = np.clip(self.c_action[8], 0.0, 1.00)   #task9                
        j = np.clip(self.c_action[9], 0.0, 1.00)   #task10      

        #print('cactionfromenv', [a,b,c,d,e,f,g,h,i,j])         

############################################## Move products to storage #####################################
        #Task 1 at Unit 1
        if obs[0] != 0 and t[0] != 0:
            t[0] += 1
            obs[18] -= 1
        if t[0] > task1t:
            obs[6] += obs[0]
            obs[25] -= obs[0]*ut1
            obs[0],t[0] = [0,0]
            
        #Task 2 at Unit 2
        if obs[1] != 0 and t[1] != 0:
            t[1] += 1
            obs[19] -= 1
            if t[1] > task2t:
                obs[7] += obs[1]
                obs[24] -= obs[1]*ut2
                obs[1],t[1] = [0,0]
            
        #Task 3 at Unit 3
        if obs[2] != 0 and t[2] != 0:
            t[2] += 1
            obs[20] -= 1
            if t[2] > task3t:
                obs[12] += obs[2]
                obs[25] -= obs[2]*ut3
                obs[2],t[2] = [0,0]
        
        #Task 4 at Unit 1
        if obs[0] != 0 and t[3] != 0:
            t[3] += 1
            obs[18] -= 1
            if t[3] > task4t:
                obs[8] += obs[0]
                obs[26] -= obs[0]*ut4
                obs[0],t[3] = [0,0]
        
        #Task 5 at Unit 4
        if obs[3] != 0 and t[4] != 0:
            t[4] += 1
            obs[21] -= 1
            if t[4] > task5t:
                obs[14] += obs[3]
                obs[25] -= obs[3]*ut5
                obs[3],t[4] = [0,0]
            
        #Task 6 at Unit 4
        if obs[3] != 0 and t[5] != 0:
            t[5] += 1
            obs[21] -= 1
            if t[5] > task6t:
                obs[15] += obs[3]*0.98
                obs[17] += obs[3]*0.02
                obs[26] -= obs[3]*ut6
                obs[3],t[5] = [0,0]
            
        #Task 7 at Unit 5
        if obs[4] != 0 and t[6] != 0:
            t[6] += 1
            obs[22] -= 1
            if t[6] > task7t:
                obs[10] += obs[4]
                obs[24] -= obs[4]*ut7
                obs[4],t[6] = [0,0]
            
        #Task 8 at Unit 6
        if obs[5] != 0 and t[7] != 0:
            t[7] += 1
            obs[23] -= 1
            if t[7] > task8t:
                obs[13] += obs[5]*0.9
                obs[9] += obs[5]*0.1
                obs[25] -= obs[5]*ut8
                obs[5],t[7] = [0,0]   

        #Task 9 at Unit 5
        if obs[4] != 0 and t[8] != 0:
            t[8] += 1
            obs[22] -= 1
            if t[8] > task9t:
                obs[11] += obs[4]
                obs[24] -= obs[4]*ut9
                obs[4],t[8] = [0,0]

        #Task 10 at Unit 6
        if obs[5] != 0 and t[9] != 0:
            t[9] += 1
            obs[23] -= 1
            if t[9] > task10t:
                obs[16] += obs[5]
                obs[24] -= obs[5]*ut10
                obs[5],t[9] = [0,0]
        
############################################## overloading storage ###################################        
        if obs[6] > state1:             
            obs[6] = state1
            
        if obs[7] > state2:
            obs[7] = state2 

        if obs[8] > state3:
            obs[8] = state3

        if obs[9] > state4:
            obs[9] = state4

        if obs[10] > state5:
            obs[10] = state5

        if obs[11] > state6:
            obs[11] = state6

        if obs[12] > int1:
            obs[12] = int1
        
        if obs[13] > int2:
            obs[13] = int2
            
############################################## New actions ##########################################         
        #choosing task1 in unit 1  maxr=30, minr=-30
        if self.d_action == [0] and a > 0:
            k = 0
            k_ = 10
            if obs[0] == 0:
                self.r += k + a*k_
                if obs[25] + a*unit1*ut1 < lps:
                    self.r += k + a*k_
                    obs[25] += a*unit1*ut1
                    if obs[6] <= state1 - a*unit1 :
                        obs[0] = a*unit1
                        t[0] = 1
                        obs[18] = task1t
                        self.r += k + a*k_
                        if obs[4] > 0:
                                self.r = 1.5 * self.r
                    else:
                        self.r -= 2*(k + a*k_) + 20*(-(state1 - a*unit1) + obs[6])
                        obs[25] -= a*unit1*ut1 
                else:
                    self.r -= k + a*k_ + 20*((obs[25] + a*unit1*ut1) - lps)
            else:
                self.r -= 5

        #choosing task2 in unit 2
        if self.d_action == [1] and b > 0:
            l = 0
            l_ = 80
            if obs[1] == 0:
                self.r += l + b*l_
                if obs[24] + b*unit2*ut2 < cw:
                    self.r += l + b*l_
                    obs[24] += b*unit2*ut2
                    if obs[6] >= b*unit2:
                        self.r += l + b*l_
                        if obs[7] <= state2 - b*unit2:
                            obs[1] = b*unit2
                            obs[6] -= b*unit2
                            t[1] = 1
                            obs[19] = task2t
                            self.r += l + b*l_
                            if obs[4] > 0:
                                self.r = 1.5 * self.r
                        else:
                            self.r -= 3*(l + b*l_) + 20*(-(state2 - b*unit2) + obs[7])
                            obs[24] -= b*unit2*ut2
                    else:
                        self.r -= 2*(l + b*l_) + 20*((b*unit2) - obs[6])
                        obs[24] -= b*unit2*ut2
                else:
                    self.r -= l + b*l_ + 20*((obs[24] + b*unit2*ut2) - cw)
            else:
                self.r -= 5

        #chossing task3 in unit 3
        if self.d_action == [2] and c > 0:
            m = 0
            m_ = 80
            if obs[2] == 0:
                self.r += m + c*m_
                if obs[25] + c*unit3*ut3 < lps:
                    self.r += m + c*m_
                    obs[25] += c*unit3*ut3
                    if obs[7] >= c*unit3:
                        self.r += m + c*m_
                        if obs[12] <= int1 - c*unit3:
                            obs[2] = c*unit3
                            obs[7] -= c*unit3
                            t[2] = 1
                            obs[20] = task3t
                            self.r += m + c*m_
                            if obs[5] > 0:
                                self.r = 1.5 * self.r
                        else:
                            self.r -= 3*(m + c*m_) + 20*(-(int1 - c*unit3) + obs[12])
                            obs[25] -= c*unit3*ut3
                    else:
                        self.r -= 2*(m + c*m_) + 20*((c*unit3) - obs[7])
                        obs[25] -= c*unit3*ut3
                else:
                    self.r -= m + c*m_ + 20*((obs[25] + c*unit3*ut3) - lps)
            else:
                self.r -= 5                  

        #choosing task 4 in unit 1
        if self.d_action == [3] and d > 0:
            n = 0
            n_ = 40
            if obs[0] == 0:
                self.r += n + d*n_
                if obs[26] + d*unit1*ut4 < hps:
                    self.r += n + d*n_
                    obs[26] += d*unit1*ut4
                    if obs[12] >= d*unit1:
                        self.r += n + d*n_
                        if obs[8] <= state3 - d*unit1:
                            obs[0] = d*unit1
                            obs[12] -= d*unit1
                            t[3] = 1
                            obs[18] = task4t
                            self.r += n + d*n_
                            if obs[4] > 0:
                                self.r = 1.5 * self.r
                        else:
                            self.r -= 3*(n + d*n_) + 20*(-(state3 - d*unit1) + obs[8])
                            obs[26] -= d*unit1*ut4
                    else:
                        self.r -= 2*(n + d*n_) + 20*((d*unit1) - obs[12])
                        obs[26] -= d*unit1*ut4
                else:
                    self.r -= n + d*n_ + 20*((obs[26] + d*unit1*ut4) - hps)
            else:
                self.r -= 5 

        #choosing task 5 in unit 4
        if self.d_action == [4] and e > 0:
            o = 0
            o_ = 80
            if obs[3] == 0:
                self.r += o + e*o_
                if obs[25] + e*unit4*ut5 < lps:
                    self.r += o + e*o_
                    obs[25] += e*unit4*ut5
                    if obs[8] >= e*unit4:
                        self.r += o + e*o_
                        obs[3] = e*unit4
                        obs[8] -= e*unit4
                        t[4] = 1
                        obs[21] = task5t    
                        if obs[5] > 0:
                            self.r = 1.5 * self.r
                        if obs[15] > 0:
                            self.r += obs[15] * self.r
                    else:
                        self.r -= 2*(o + e*o_) + 20*((e*unit4) - obs[8]  )
                        obs[25] -= e*unit4*ut5 
                else:
                    self.r -= o + e*o_ + 20*((obs[25] + e*unit4*ut5) - lps)
            else:
                self.r -= 5 

        #choosing task 6 in unit 4
        if self.d_action == [5] and f > 0:
            p = 0
            p_ = 80
            if obs[3] == 0:
                self.r += p + f*p_
                if obs[26] + f*unit4*ut6 < hps:
                    self.r += p + f*p_
                    obs[26] += f*unit4*ut6
                    if obs[8] >= f*unit4:
                        self.r += p + f*p_
                        obs[3] = f*unit4
                        obs[8] -= f*unit4
                        t[5] = 1
                        obs[21] = task6t     
                        if obs[5] > 0:
                            self.r = 1.5 * self.r
                        if obs[14] > 0:
                            self.r += obs[14] * self.r
                    else:
                        self.r -= 2*(p + f*p_) + 20*((f*unit4) - obs[8])
                        obs[26] -= f*unit4*ut6
                else:
                    self.r -= p + f*p_ + 20*((obs[26] + f*unit4*ut6) - hps)
            else:
                self.r -= 5 

        #choosing task 7 in unit 5
        if self.d_action == [6] and g > 0:
            q = 0
            q_ = 20 +10+10+10
            if obs[4] == 0:
                self.r += q + g*q_
                if obs[24] + g*unit5*ut7 < cw:
                    self.r += q + g*q_
                    obs[24] += g*unit5*ut7
                    if obs[9] >= g*unit5*0.05:
                        self.r += q + g*q_
                        if obs[10] <= state5 - g*unit5:
                            obs[9] -= 0.05*g*unit5
                            obs[4] = g*unit5
                            t[6] = 1
                            obs[22] = task7t
                            self.r += q + g*q_
                            if obs[0] > 0:
                                self.r = 1.5 * self.r
                        else:
                            self.r -= 3*(q + g*q_) + 20*(-(state5 - g*unit5) + obs[10])
                            obs[24] -= g*unit5*ut7
                    else:
                        self.r -= 2*(q + g*q_) + 20*((g*unit5*0.05) - obs[9])
                        obs[24] -= g*unit5*ut7
                else:
                    self.r -= q + g*q_ + 20*((obs[24] + g*unit5*ut7) - cw)
            else:
                self.r -= 5 

        #choosing task 8 in unit 6
        if self.d_action == [7] and h > 0:
            w = 0
            w_ = 30 +10+10+10
            if obs[5] == 0:
                self.r += w + h*w_
                if obs[25] + h*unit6*ut8 < lps:
                    self.r += w + h*w_
                    obs[25] += h*unit6*ut8
                    if obs[10] >= h*unit6:
                        self.r += w + h*w_
                        if obs[13] <= int2 - h*unit6*0.9:
                            self.r += w + h*w_
                            if obs[9] <= state4 - h*unit6*0.1:
                                obs[10] -= h*unit6
                                obs[5] = h*unit6
                                t[7] = 1
                                obs[23] = task8t
                                self.r += w + h*w_
                                if obs[2] > 0:
                                    self.r = 1.5 * self.r
                            else:
                                self.r -= 4*(w + h*w_) + 20*(-(state4 - h*unit6*0.1) + obs[9])
                                obs[25] -= h*unit6*ut8
                        else:
                            self.r -= 3*(w + h*w_) - 20*((int2 - h*unit6*0.9) - obs[13])
                            obs[25] -= h*unit6*ut8
                    else:
                        self.r -= 2*(w + h*w_) + 20*((h*unit6) - obs[10])
                        obs[25] -= h*unit6*ut8
                else:
                    self.r -= w + h*w_ + 20*((obs[25] + h*unit6*ut8) - lps)
            else:
                self.r -= 5 

        #choosing task 9 in unit 5
        if self.d_action == [8] and i > 0:
            s = 0
            s_ = 60
            if obs[4] == 0:
                self.r += s + i*s_
                if obs[24] + i*unit5*ut9 < cw:
                    obs[24] += i*unit5*ut9
                    self.r += s + i*s_
                    if obs[12] >= i*unit5*0.5:
                        self.r += s + i*s_
                        if obs[13] >= i*unit5*0.5:
                            self.r += s + i*s_
                            if obs[11] <= state6 - i*unit5:
                                obs[12] -= i*unit5*0.5
                                obs[13] -= i*unit5*0.5
                                obs[4] = i*unit5
                                t[8] = 1
                                obs[22] = task9t
                                self.r += s + i*s_
                                if obs[0] > 0:
                                    self.r = 1.5 * self.r
                            else:
                                self.r -= 4*(s + i*s_) + 20*(-(state6 - i*unit5) + obs[11])
                                obs[24] -= i*unit5*ut9
                        else:
                            self.r -= 3*(s + i*s_) + 20*((i*unit5*0.5) - obs[13])
                            obs[24] -= i*unit5*ut9
                    else:
                        self.r -= 2*(s + i*s_) + 20*((i*unit5*0.5) - obs[12] )
                        obs[24] -= i*unit5*ut9
                else:
                    self.r += s + i*s_ + 20*((obs[24] + i*unit5*ut9) - cw)
            else:
                self.r -= 5 

        #choosing task 10 in unit 6
        if self.d_action == [9] and j > 0:
            u = 0
            u_ = 70
            if obs[5] == 0:
                self.r += u + j*u_
                if obs[24] + j*unit6*ut10 < cw:
                    self.r += u + j*u_
                    obs[24] += j*unit6*ut10
                    if obs[11] >= j*unit6:
                        obs[5] = j*unit6
                        obs[11] -= j*unit6
                        t[9] = 1
                        obs[23] = task10t    
                        self.r += u + j*u_ 
                        if obs[3] > 0:
                            self.r = 1.5 * self.r
                    else:
                        self.r -= 2*(u + j*u_) + 20*((j*unit6) - obs[11])
                        obs[24] -= j*unit6*ut10
                else:
                    self.r -= u + j*u_ + 20*((obs[24] + j*unit6*ut10) - cw)
            else:
                self.r -= 5 
        ################################# Masking ####################################
        if t[10] == 0:
            self.available_actions = np.array([1,0,0,0,0,0,1,0,0,0,0], dtype=np.int32)    
        if t[10] == 1:
            self.available_actions = np.array([1,0,0,0,0,0,1,0,0,0,1], dtype=np.int32) 
        if 2 <= t[10] < 24:   #Campaign 1      
            self.available_actions = np.array([1,1,1,0,0,0,1,1,0,0,1], dtype=np.int32)
        if 72 <= t[10] < 48:  #Campaign 2
            self.available_actions = np.array([0,0,0,1,1,1,0,0,1,1,1], dtype=np.int32)
        
        #Prevent actions that will not produce int1 or int2 in the process
        if t[10] >= 24 - (task1t + task2t + 1):
            self.available_actions[0] = 0 
        if t[10] >= 24 - (task2t + 1):
            self.available_actions[1] = 0
        if t[10] >= 24 - (task7t + 1):
            self.available_actions[6] = 0
        #Prevent actions that will not produce P1 P2 P3
        if t[10] >= 48 - (task9t + 1):
            self.available_actions[8] = 0
        if t[10] >= 48 - (task4t + 1):
            self.available_actions[3] = 0


        #Restriction to prevent actions below 25% capacity
        # Prevent T2
        #if obs[6] < 2.0:       
        #    self.available_actions[1] = 0 
        # Prevent T3
        #if obs[7] < 1.5:                      
        #    self.available_actions[2] = 0
        # Prevent T4
        #if obs[12] < 1.25:                      
        #    self.available_actions[3] = 0
        # Prevent T5
        #if obs[8] < 2.0:                      
        #    self.available_actions[4] = 0  
        # Prevent T6
        #if obs[8] < 2.0:
        #    self.available_actions[5] = 0
        # Prevent T8
        #if obs[10] < 1.0:
        #    self.available_actions[7] = 0
        # Prevent T9
        #if obs[12] < 0.75 or obs[13] < 0.75:
        #    self.available_actions[8] = 0
        # Prevent T10
        #if obs[11] < 1.0:
        #    self.available_actions[9] = 0
 
        # Prevent Actions from occupied machines    
        if obs[18] > 1:                      
            self.available_actions[0] = 0  
            self.available_actions[3] = 0
        if obs[19] > 1:
            self.available_actions[1] = 0
        if obs[20] > 1:
            self.available_actions[2] = 0
        if obs[21] > 1:
            self.available_actions[4] = 0
            self.available_actions[5] = 0
        if obs[22] > 1:                      
            self.available_actions[6] = 0    
            self.available_actions[8] = 0
        if obs[23] > 1:
            self.available_actions[7] = 0
            self.available_actions[9] = 0
        ##############################################################################    
        if t[10] == self.max_steps:  #for the final step we see the product 
            self.done = True     
        ##############################################################################
        #reward factor to promote reduction in makespan
        fr = 1 #((144 - t[10])*0.208 + 70)/100            

        #reward = self.r * fr + 25*obs[3] + 2*obs[7] #2
        reward = self.r * fr + 25*obs[3] + 2*obs[7] #2
        
        t[10] += 1
        #print('dac ',self.d_action,'cac ', [a,b,c,d,e,f,g,h,i,j],'reward ',reward) #'dac ',self.d_action,'cac ', [a,b,c,d,e,f,g,h,i,j]

        #Reward normalization (first attempt)
        #reward = (reward - 3000) * 0.02    #considering that the min and max rewards are 3000 and 5000. the new range is 0-100
        
        return obs, reward, self.done, t, self.available_actions
        ##############################################################################        
    def reset(self,seq_len):
        initial_states = torch.tensor([0,0,0,0,0,0,0,0,0,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
        self.done = False     
        self.observation = initial_states.repeat(seq_len,1)
        return self.observation
    
    def reset_m(self,obs):
        self.available_inactions = np.array([0,0,0,0,0,0,1,0,0,0,0], dtype=np.int32)  
        return self.available_inactions
    
    def action_mask(self):
        mask_vec = np.array(self.available_actions, dtype=bool)
        return mask_vec
    
    def render (self):
        pass
