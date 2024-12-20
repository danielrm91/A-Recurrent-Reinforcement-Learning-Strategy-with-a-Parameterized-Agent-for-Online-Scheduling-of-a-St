'''
this file contains the hybrid RNN for the actor and the RNN 
for the critic 
'''
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class RNNactor(nn.Module):
    def __init__(self, in_dim_sen, out_dim_sen, out_ac1_dim, out_ac2_dim, seq_len):
        super(RNNactor,self).__init__()
        self.hidden_size = 25
        self.seq_len = seq_len
        self.in_dim_sen = in_dim_sen
        self.out_ac1_dim = out_ac1_dim
        # State Encoding Network (SEN)
        self.rnnlayer1 = nn.LSTM(in_dim_sen, hidden_size=25,num_layers=3,batch_first=True)
        self.layer2 = nn.Linear(100,250)
        self.layer3 = nn.Linear(250, out_dim_sen)
        # Discrete Actor
        self.layer4 = nn.Linear(out_dim_sen, 250)
        self.layer5 = nn.Linear(250,250)
        self.layer6 = nn.Linear(250,out_ac1_dim)
        # Continuous Actor
        self.layer7 = nn.Linear(out_dim_sen, 250)
        self.layer8 = nn.Linear(250,250)
        self.layer9 = nn.Linear(250, out_ac2_dim)
    
    def forward (self, obs, mask_vec, h, c):
        # Convert obs to tensor if its a np arrray
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        if isinstance(mask_vec, np.ndarray):
            mask_vec = torch.tensor(mask_vec, dtype=torch.bool)   
        obs = obs.reshape(-1, self.seq_len, self.in_dim_sen)
        # State Encoder Network (SEN)
        activation1, (h1,c1) = self.rnnlayer1(obs,(h,c))
        activation2 = F.tanh(self.layer2(activation1.contiguous().view(-1,100)))
        sen = F.tanh(self.layer3(activation2))
        # Discrete Actor
        activation3 = F.tanh(self.layer4(sen))
        activation4 = F.tanh(self.layer5(activation3))
        activation5 = F.tanh(self.layer6(activation4))
        logits = torch.where(mask_vec, activation5, torch.tensor(-1e+8))
        output1 =  logits #F.softmax(logits, dim=1) # logits
        # Continuous Actor
        activation6 = F.tanh(self.layer7(sen))
        activation7 = F.tanh(self.layer8(activation6))
        output2 = torch.sigmoid(self.layer9(activation7))
        return output1, output2, h1, c1
        
class RNNcritic(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len):
        super(RNNcritic, self).__init__()
        self.hidden_size = 25
        self.seq_len = seq_len
        self.in_dim = in_dim
        # Critic Network
        self.rnnlayer1 = nn.LSTM(in_dim, hidden_size=25, num_layers=3, batch_first=True)
        self.layer2 = nn.Linear(100,250)
        self.layer3 = nn.Linear(250,250)
        self.layer4 = nn.Linear(250, out_dim)

    def forward(self, obs,  h, c):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # Critic Network
        obs = obs.reshape(-1, self.seq_len, self.in_dim)
        activation1, (h,c) = self.rnnlayer1(obs, (h,c))
        activation2 = F.tanh(self.layer2(activation1.contiguous().view(-1,100)))
        activation3 = F.tanh(self.layer3(activation2))
        output = self.layer4(activation3)
        
        return output
        
