import gym 
import numpy as np
from gym import spaces
import csv
import torch
from scipy.stats import truncnorm

class KondiliEnv(gym.Env):           #action    #accumulated times
    def __init__(self):   
        self.observation_space = spaces.Dict({"agent" : spaces.Box(0, 1, shape=(2,), dtype=int),
                                              "target": spaces.Box(0, 1, shape=(2,), dtype=int),})
        self.action_space = spaces.Discrete(4)
        self.max_steps = 30
        self.max_val = 1.0
        
    def step1(self, d_action, c_action, observations, times):  
        self.d_action = d_action
        self.c_action = c_action[0]
        obs = observations  
        t = times
        self.r = 0
        self.done = False
        self.available_actions = np.array([1,1,1,1,1,1,1,1,1], dtype=np.int32)
        
        #capacities       
        r1cap  =   80
        r2cap  =   50
        discap =  200
        hcap   =  100 
        hacap  =  300 #100
        iecap  =  300 #120
        ibccap =  300 #150
        iabcap =  300 #200 
        #times
        ht   =  2 #1
        r1t  =  4 #2
        r2t  =  4 #2
        r3t  =  2 #1
        dist =  2 #1
        
        #continuous
        a = np.clip(self.c_action[0], 0.0, 1.0)   #Heater                   100
        b = np.clip(self.c_action[1], 0.0, 1.0)   #reaction1 in reactor 1    80 
        c = np.clip(self.c_action[2], 0.0, 1.0)   #reaction2 in reactor 1    80 
        d = np.clip(self.c_action[3], 0.0, 1.0)   #reaction1 in reactor 2    50
        e = np.clip(self.c_action[4], 0.0, 1.0)   #reaction2 in reactor 2    50
        f = np.clip(self.c_action[5], 0.0, 1.0)   #reaction3 in reactor 1    80  
        g = np.clip(self.c_action[6], 0.0, 1.0)   #reaction3 in reactor 2    50
        h = np.clip(self.c_action[7], 0.0, 1.0)   #Distiller                200

############################################## Move products to storage #####################################
        #For uncertainty in output of reaction 2, 
        # sampling from gaussian truncated distribution  
        value = 0.6  #Activate this to take off the uncertainty   

        #heater
        if obs[0] != 0:
            t[6] += 1
            obs[12] -= 1
        if t[6] > ht:
            obs[4] += obs[0]
            obs[0],t[6] = [0,0]
            
        #reaction 1 in reactor 1
        if obs[1] != 0 and t[0] != 0:
            t[0] += 1
            obs[10] -= 1
            if t[0] > r1t:
                obs[5] += obs[1]
                obs[1],t[0] = [0,0]
            
        #reaction 2 in reactor 1 with output uncertainty
        if obs[1] != 0 and t[1] != 0:
            t[1] += 1
            obs[10] -= 1
            if t[1] > r2t:
                obs[6] += obs[1]*(value)   #0.6
                obs[8] += obs[1]*(1-value) #0.4
                obs[1],t[1] = [0,0]
        
        #reaction 3 in reactor 1
        if obs[1] != 0 and t[2] != 0:
            t[2] += 1
            obs[10] -= 1
            if t[2] > r3t:
                obs[7] += obs[1]
                obs[1],t[2] = [0,0]
        
        #reaction 1 in reactor 2
        if obs[2] != 0 and t[3] != 0:
            t[3] += 1
            obs[11] -= 1
            if t[3] > r1t:
                obs[5] += obs[2]
                obs[2],t[3] = [0,0]
            
        #reaction 2 in reactor 2 with output uncertainty
        if obs[2] != 0 and t[4] != 0:
            t[4] += 1
            obs[11] -= 1
            if t[4] > r2t:
                obs[6] += obs[2]*(value)   #0.6
                obs[8] += obs[2]*(1-value) #0.4
                obs[2],t[4] = [0,0]
            
        #reaction 3 in reactor 2
        if obs[2] != 0 and t[5] != 0:
            t[5] += 1
            obs[11] -= 1
            if t[5] > r3t:
                obs[7] += obs[2]
                obs[2],t[5] = [0,0]
            
        #distiller
        if obs[3] != 0:
            t[7] += 1
            obs[13] -= 1
            if t[7] > dist:
                obs[9] += obs[3]*0.9
                obs[6] += obs[3]*0.1
                obs[3],t[7] = [0,0]            
        
############################################## overloading storage ###################################        
        if obs[4] > hacap:             
            obs[4] = hacap
            
        if obs[5] > ibccap:
            obs[5] = ibccap 
            
        if obs[6] > iabcap:
            obs[6] = iabcap
        
        if obs[7] > iecap:
            obs[7] = iecap
            
############################################## New actions ##########################################         
        #choosing the heater
        if self.d_action == [0] and a > 0:
            wa = 100*a*0.5
            if obs[0] == 0:
                self.r += wa
                if obs[4] + a*hcap <= hacap:
                    obs[0] = a*hcap
                    t[6] = 1
                    obs[12] = ht
                    self.r += wa 
                else:
                    self.r -= wa + ( obs[4] + a*hcap - hacap) #/ hcap)
            else:
                self.r = -100

        #choosing reaction1 in reactor 1
        if self.d_action == [1] and b > 0:
            wb = 80* b *1.25
            if obs[1] == 0:
                self.r += wb
                if obs[5] + b*r1cap <= ibccap:
                    obs[1] = b*r1cap
                    t[0] = 1
                    obs[10] = r1t
                    self.r += wb
                else:
                    self.r -= wb + (obs[5] + b*r1cap - ibccap) #/ r1cap)
            else:
                self.r = -100

        #chossing reaction2 in reactor 1
        if self.d_action == [2] and c > 0:
            wc = 80 * c * 1.5625
            if obs[1] == 0:                                          #if the reactor is empty
                self.r += wc
                if obs[4] >= c*r1cap*0.4 and obs[5] >= c*r1cap*0.6:  #if there is enough material 
                    self.r += wc
                    if obs[6] + c*r1cap*0.6 <= iabcap :
                        obs[4] -= c*r1cap*0.4
                        obs[5] -= c*r1cap*0.6
                        obs[1] = c*r1cap
                        t[1] = 1
                        obs[10] = r2t
                        self.r += wc
                    else:
                        self.r -= 2*wc +  (obs[6] + c*r1cap*0.6 - iabcap) #( r1cap*0.6))
                else:
                    self.r -= wc + np.min(( (c*r1cap*0.4 - obs[4]) , (c*r1cap*0.6 - obs[5]) )) #( (c*r1cap*0.4 - obs[4])/(r1cap*0.4) , (c*r1cap*0.6 - obs[5])/(r1cap*0.6) )
            else:
                self.r = -100                   

        #choosing reaction 3 in reactor 1
        if self.d_action == [5] and f > 0:          
            wf = 80 * f * 3.75
            if obs[1] == 0:
                self.r += wf
                if obs[6] +1e-2>= f*r1cap*0.8 :
                    self.r += wf
                    if obs[7] + f*r1cap <= iecap :
                        obs[6] -= f*r1cap*0.8
                        obs[1] = f*r1cap
                        t[2] = 1
                        obs[10] = r3t
                        self.r += wf
                    else:
                        self.r -= 2*wf + (obs[7] + f*r1cap - iecap)#/r1cap)
                else:
                    self.r -= wf + (f*r1cap*0.8 - obs[6])#/(r1cap*0.8)
            else:
                self.r = -100

        #choosing reaction1 in reactor 2
        if self.d_action == [3] and d > 0:
            wd = 50* d * 2
            if obs[2] == 0:
                self.r += wd
                if obs[5] + d*r2cap <= ibccap:
                    obs[2] = d*r2cap
                    t[3] = 1
                    obs[11] = r1t
                    self.r += wd 
                else:
                    self.r -= wd + (obs[5] + d*r2cap - ibccap)#/r2cap)
            else:
                self.r = -100

        #chossing reaction2 in reactor 2
        if self.d_action == [4] and e > 0:
            we = 50 * e * 2.5
            if obs[2] == 0:
                self.r += we
                if obs[4] >= e*r2cap*0.4 and obs[5] >= e*r2cap*0.6:  
                    self.r += we
                    if obs[6] + e*r2cap*0.6 <= iabcap:
                        obs[4] -= e*r2cap*0.4
                        obs[5] -= e*r2cap*0.6
                        obs[2] = e*r2cap
                        t[4] = 1
                        obs[11] = r2t
                        self.r += we
                    else:
                        self.r -= 2*we + (e*r2cap*0.6 + obs[6] - iabcap)#/(r2cap*0.6))
                else:
                    self.r -= we + np.min(( (e*r2cap*0.4 - obs[4]) , (e*r2cap*0.6 - obs[5])  )) #( (e*r2cap*0.4 - obs[4])/(r2cap*0.4) , (e*r2cap*0.6 - obs[5])/(r2cap*0.6)  )
            else:
                self.r = -100   

        #choosing reaction 3 in reactor 2
        if self.d_action == [6] and g > 0:
            wg = 50 * g * 6
            if obs[2] == 0: 
                self.r += wg
                if obs[6] >= g*r2cap*0.8:
                    self.r += wg
                    if obs[7] + g*r2cap <= iecap:
                        obs[6] -= g*r2cap*0.8
                        obs[2] = g*r2cap
                        t[5] = 1
                        obs[11] = r3t
                        self.r += wg
                    else:
                        self.r -= 2*wg + (g*r2cap + obs[7] - iecap)#/r2cap)
                else:
                    self.r -= wg + (g*r2cap*0.8 - obs[6])#/(r2cap*0.8)
            else:
                self.r = -100

        #choosing the distiller
        if self.d_action == [7] and h > 0:
            wh = 200 * h * 8
            if obs[3] == 0:
                self.r += wh
                if obs[7] >= h*discap:
                    self.r += wh
                    if obs[6] + h*discap*0.1 <= iabcap : 
                        obs[7] -= h*discap
                        obs[3] = h*discap
                        t[7] = 1
                        obs[13] = dist
                        self.r += wh
                    else:
                        self.r -= 2*wh + (h*discap*0.1 + obs[6] - iabcap)#/(discap*0.1))
                else:
                    self.r -= wh + (h*discap - obs[7])#/discap
            else:
                self.r = -100
        ################################# Masking ####################################
        #if t[8] == 0:
        #    self.available_actions = np.array([1,0,0,0,0,0,0,0,0], dtype=np.int32)    #[0,1,0,0,0,0,0,0,0]
        #if t[8] == 1:
        #    self.available_actions = np.array([0,0,1,0,0,0,0,0,0], dtype=np.int32)
        #Prevent action 0
        #if obs[4] > 32:
        #    self.available_actions[0] = 0
        #Prevent Action 0
        '''
        tot_h = obs[0] + obs[4] +  obs[8]
        if t[1] != 0:
            tot_h+= 0.4*obs[1]
        if t[4] != 0:
            tot_h += 0.4*obs[2]
        if  tot_h> 170:
            self.available_actions[0] = 0
        # Prevent Action 2
        if obs[4] < 32 or obs[5] < 48:       #si no hay suficiente HotA ni IntBC para reactor 1 al 100% (80)
            self.available_actions[2] = 0
        # Prevent Action 4
        if obs[4] < 20 or obs[5] < 30:       #si no hay suficiente HotA ni IntBC para reactor 2 al 100% (50)
            self.available_actions[4] = 0 
        # Prevent Action 5
        if obs[6] < 64:                      #si no hay suficiente IntAB para reactor 1 al 100%         (80)
            self.available_actions[5] = 0
        # Prevent Action 6
        if obs[6] < 40:                      #si no hay suficiente IntAB para reactor 2 al 100%         (50)
            self.available_actions[6] = 0
        # Prevent Action 7
        if obs[7] < 10:                      #si no hay suficiente impE para separador
            self.available_actions[7] = 0  
        '''
        # Prevent Actions from occupied machines    
        if obs[10] > 1:                      #si el reactor 1 estara ocupado en el proximo state
            self.available_actions[1] = 0    #si es 0 o 1 significa que esta desocupado o lo estara
            self.available_actions[2] = 0
            self.available_actions[5] = 0
        if obs[11] > 1:                      #si el reactor 2 estara ocupado en el proximo state
            self.available_actions[3] = 0    #si es 0 o 1 significa que esta desocupado o lo estara
            self.available_actions[4] = 0
            self.available_actions[6] = 0
        
        if t[8] == self.max_steps:  #for the final step we see the product 
            self.done = True     
        t[8] += 1
        obs[-1] = t[8]    
        # Time restrictions
        if t[8] >= 27:
            self.available_actions[5] = 0
            self.available_actions[6] = 0
        if t[8] >= 23:
            self.available_actions[2] = 0
            self.available_actions[4] = 0  
        if t[8] >= 19:
            self.available_actions[1] = 0
            self.available_actions[3] = 0 
        if t[8] >= 17:
            self.available_actions[0] = 0
        if t[8] >= 29:
            self.available_actions[7] = 0                   
        ##############################################################################    
        #if t[8] == self.max_steps:  #for the final step we see the product 
        #    self.done = True           
        #t_factor = (self.max_steps-t[8])/self.max_steps

        reward = self.r #+ t_factor*obs[4]/100 + t_factor*1.7*obs[5]/100 + 2*((1/0.4)*obs[8])/100 + 2.5*obs[7]/100 \
            #+ 5*(1/0.9)*(obs[9])/100 + 5*(obs[3])/100 #+ 25*obs[3] + 2*obs[7] #2
        #t[8] += 1
        #obs[-1] = t[8]
        return obs, reward, self.done, t, self.available_actions
        ##############################################################################        
    def reset(self,seq_len):
        initial_states = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
        self.done = False     #np.array([0,0,0,0,0,0,0,0,0,0,80,150,100,0,0,0,0,0,0,0,0,100,0])
        self.observation = initial_states.repeat(seq_len,1)
        return self.observation
    
    def reset_m(self,obs):
        self.available_inactions = np.array([1,1,1,1,1,1,1,1,1], dtype=np.int32)  
        self.available_inactions = np.array([1,0,0,0,0,0,0,0,0], dtype=np.int32) 
        if obs[4] < 32 or obs[5] < 48:
            self.available_inactions[2] = 0
        if obs[4] < 20 or obs[5] < 30:
            self.available_inactions[4] = 0 
        if obs[6] < 64:
            self.available_inactions[5] = 0
        if obs[6] < 40:
            self.available_inactions[6] = 0
        if obs[7] < 10:
            self.available_inactions[7] = 0 
        return self.available_inactions
    
    
    def action_mask(self):
        mask_vec = np.array(self.available_actions, dtype=bool)
        return mask_vec
    
    def render (self):
        pass
