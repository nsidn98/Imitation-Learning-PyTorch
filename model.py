import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

##########################################################################################
class QNetwork(nn.Module):
    '''
    Q(s,a)
    Input:  State: [b, 45*80]
            Dim:   [b, 3]
            Action:[b,2]
    Output: Q1(s,a) --> [b,2]
    '''
    def __init__(self, input_size, num_actions, init_w=3e-3, hidden_arr=[500,100,50,9]):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(input_size,hidden_arr[0])
        self.linear2 = nn.Linear(hidden_arr[0],hidden_arr[1])
        self.linear3 = nn.Linear(hidden_arr[1],hidden_arr[2])
        self.linear4 = nn.Linear(hidden_arr[2],hidden_arr[3])
        self.linear5 = nn.Linear(hidden_arr[3]+3+num_actions,1)
        self.apply(weights_init_)

    def forward(self,state, dim, action):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        z = torch.cat([x,dim,action],dim=1)
        out = self.linear5(z)
        return out 
        
##########################################################################################
class StochasticPolicy(nn.Module):
    '''
    Pi(s)
    Input: State --> [b,2,90,128]
    Output: mean,log_std [b,2,2]
            action A(s)
    '''
    def __init__(self, num_inputs, num_actions, init_w=3e-3, hidden_arr=[500,100,50,9]):
        super(StochasticPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs,hidden_arr[0])
        self.linear2 = nn.Linear(hidden_arr[0],hidden_arr[1])

        # for mean
        self.linear3_mean = nn.Linear(hidden_arr[1],hidden_arr[2])
        self.linear4_mean = nn.Linear(hidden_arr[2],hidden_arr[3])
        self.linear5_mean = nn.Linear(hidden_arr[3]+3,num_actions)
        self.linear5_mean.weight.data.uniform_(-init_w,init_w)
        self.linear5_mean.bias.data.uniform_(-init_w,init_w)
        
        # for log_std
        self.linear3_std = nn.Linear(hidden_arr[1],hidden_arr[2])
        self.linear4_std = nn.Linear(hidden_arr[2],hidden_arr[3])
        self.linear5_std = nn.Linear(hidden_arr[3]+3,num_actions)
        self.linear5_std.weight.data.uniform_(-init_w,init_w)
        self.linear5_std.bias.data.uniform_(-init_w,init_w)
        
        self.apply(weights_init_)

    def forward(self, state, dim):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = F.relu(self.linear3_mean(x))
        mean = F.relu(self.linear4_mean(mean))
        mean = torch.cat([mean,dim],dim=1)
        mean = self.linear5_mean(mean)

        std = F.relu(self.linear3_std(x))
        std = F.relu(self.linear4_std(std))
        std = torch.cat([std,dim],dim=1)
        log_std = self.linear5_std(std)
        log_std = torch.clamp(log_std,LOG_SIG_MIN,LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state,dim):
        mean, log_std = self.forward(state,dim)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

##########################################################################################
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, init_w=3e-3, hidden_arr=[500,100,50,9]):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs,hidden_arr[0])
        self.linear2 = nn.Linear(hidden_arr[0],hidden_arr[1])
        self.linear3 = nn.Linear(hidden_arr[1],hidden_arr[2])
        self.linear4 = nn.Linear(hidden_arr[2],hidden_arr[3])
        self.mean = nn.Linear(hidden_arr[3]+3,num_actions)

        self.noise = torch.Tensor(num_actions)
        
        self.apply(weights_init_)

    def forward(self, state, dim):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = torch.tanh(self.mean(x))
        return mean


    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean