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
# Input:  State: [b,4,45,80]
#         Dim:   [b,3]
class QNetwork(nn.Module):
    '''
    Q(s,a)
    Input:  State: [b, 4, 45, 80]
            Dim:   [b, 4*3]
            Action:[b,2]
    Output: Q1(s,a) --> [b,2]
    '''
    def __init__(self,num_actions,init_w=3e-3,out_channels=[32,32,32],hidden_arr=[512,128,36],num_boxes=4):
        super(QNetwork, self).__init__()
        # bx4x45x80 --> bx32x20x38
        self.conv1  =  nn.Conv2d(in_channels=4, out_channels=out_channels[0],kernel_size=5,stride=2)
        # bx32x20x38 --> bx32x10x18
        self.conv2  =  nn.Conv2d(in_channels=out_channels[0],out_channels=out_channels[1],kernel_size=3,stride=2)
        # bx32x10x18 --> bx32x4x8
        self.conv3  =  nn.Conv2d(in_channels=out_channels[1],out_channels=out_channels[2],kernel_size=3,stride=2)

        # 1024 --> 512
        self.fc1    =  nn.Linear(32*4*8,hidden_arr[0])
        # 512  --> 128
        self.fc2    =  nn.Linear(hidden_arr[0],hidden_arr[1])
        # 128  --> 36
        self.fc3    =  nn.Linear(hidden_arr[1],hidden_arr[2])
        # 36 + 12 --> num_actions
        self.fc4    =  nn.Linear(hidden_arr[2]+3*num_boxes,num_actions)

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


class StochasticPolicyCNN(nn.Module):
    '''
    Pi(s)
    Input:  State --> [b,4,45,80]
            Dim   --> [b,4*3]
    Output: mean,log_std [b,2,2]
            action A(s)
    '''
    def __init__(self, num_actions=2, init_w=3e-3, out_channels=[32,32,32], hidden_arr=[500,100,50,9],num_boxes=4,batch_norm=False):
        super(StochasticPolicyCNN, self).__init__()
         # bx4x45x80 --> bx32x20x38
        self.conv1  =  nn.Conv2d(in_channels=4, out_channels=out_channels[0],kernel_size=5,stride=2)
        self.conv1_bn = nn.BatchNorm2d(out_channels[0])
        # bx32x20x38 --> bx32x10x18
        self.conv2  =  nn.Conv2d(in_channels=out_channels[0],out_channels=out_channels[1],kernel_size=3,stride=2)
        self.conv2_bn = nn.BatchNorm2d(out_channels[1])
        # bx32x10x18 --> bx32x4x8
        self.conv3  =  nn.Conv2d(in_channels=out_channels[1],out_channels=out_channels[2],kernel_size=3,stride=2)
        self.conv3_bn = nn.BatchNorm2d(out_channels[2])
        
        # mean
        # 1024 --> 512
        self.fc1_mean =  nn.Linear(32*4*8,hidden_arr[0])
        self.fc1_mean_bn = nn.BatchNorm1d(hidden_arr[0])
        # 512  --> 128
        self.fc2_mean =  nn.Linear(hidden_arr[0],hidden_arr[1])
        self.fc2_mean_bn = nn.BatchNorm1d(hidden_arr[1])
        # 128  --> 36
        self.fc3_mean =  nn.Linear(hidden_arr[1],hidden_arr[2])
        self.fc3_mean_bn = nn.BatchNorm1d(hidden_arr[2])
        # 36 + 12 --> num_actions
        self.fc4_mean =  nn.Linear(hidden_arr[2]+3*num_boxes,num_actions)
    
        # log-std
        # 1024 --> 512
        self.fc1_std =  nn.Linear(32*4*8,hidden_arr[0])
        self.fc1_std_bn = nn.BatchNorm1d(hidden_arr[0])
        # 512  --> 128
        self.fc2_std =  nn.Linear(hidden_arr[0],hidden_arr[1])
        self.fc2_std_bn = nn.BatchNorm1d(hidden_arr[1])
        # 128  --> 36
        self.fc3_std =  nn.Linear(hidden_arr[1],hidden_arr[2])
        self.fc3_std_bn = nn.BatchNorm1d(hidden_arr[2])
        # 36 + 12 --> num_actions
        self.fc4_std =  nn.Linear(hidden_arr[2]+3*num_boxes,num_actions)

        # init weights
        self.fc4_mean.weight.data.uniform_(-init_w,init_w)
        self.fc4_mean.bias.data.uniform_(-init_w,init_w)
        self.fc4_std.weight.data.uniform_(-init_w,init_w)
        self.fc4_std.bias.data.uniform_(-init_w,init_w)

        self.apply(weights_init_)

        self.batch_norm  = batch_norm

    def forward(self,state,dims):
        x = F.relu(self.conv1(state))
        if self.batch_norm:
            x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.conv2_bn(x)
        x = F.relu(self.conv3(x))
        if self.batch_norm:
            x = self.conv3_bn(x)
        x = x.view(-1,32*4*8)

        # mean
        mean = F.relu(self.fc1_mean(x))
        if self.batch_norm:
            mean = self.fc1_mean_bn(mean)
        mean = F.relu(self.fc2_mean(mean))
        if self.batch_norm:
            mean = self.fc2_mean_bn(mean)
        mean = F.relu(self.fc3_mean(mean))
        if self.batch_norm:
            mean = self.fc3_mean_bn(mean)

        z_mean = torch.cat([mean,dims],dim=1)
        z_mean = self.fc4_mean(z_mean)
        
        # std
        std = F.relu(self.fc1_mean(x))
        if self.batch_norm:
            std = self.fc1_std_bn(std)
        std = F.relu(self.fc2_std(std))
        if self.batch_norm:
            std = self.fc2_std_bn(std)
        std = F.relu(self.fc3_std(std))
        if self.batch_norm:
            std = self.fc3_std_bn(std)

        z_std = torch.cat([std,dims],dim=1)
        z_std = self.fc4_std(z_std)
        z_std = torch.clamp(z_std,LOG_SIG_MIN,LOG_SIG_MAX)
        
        return z_mean, z_std

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