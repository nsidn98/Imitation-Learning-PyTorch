import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class StochasticAgent(nn.Module):
    # if we want top view of box
    def __init__(self, h, w, outputs,hidden_arr=[1000,400,50,9]):
        super(Agent, self).__init__()
         # if we want raw dimensions of box  
        self.linear1 = nn.Linear(h*w,hidden_arr[0])
        self.linear2 = nn.Linear(hidden_arr[0],hidden_arr[1])
        self.linear3 = nn.Linear(hidden_arr[1],hidden_arr[2])
        self.linear4 = nn.Linear(hidden_arr[2],hidden_arr[3])
        self.linear5 = nn.Linear(hidden_arr[3]+3,outputs)

            
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, img,dim):
        # if we want raw dimensions of box  
        # print(img.shape,dim.shape)
        x = F.relu(self.linear1(img))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        z = torch.cat((x,dim),dim=1)
        print(x)
        out = self.linear5(z)
        return out




class QNetwork(nn.Module):
    '''
    Q(s,a)
    Input:  State [[b,45*80], [b,3]]
            Action [b,2]
    Output: Q1(s,a) --> [b,2]
    '''
    def __init__(self, input_size, num_actions, init_w=3e-3, hidden_arr=[500,100,50,9]):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(input_size,hidden_arr[0])
        self.linear2 = nn.Linear(hidden_arr[0],hidden_arr[1])
        self.linear3 = nn.Linear(hidden_arr[1],hidden_arr[2])
        self.linear4 = nn.Linear(hidden_arr[2],hidden_arr[3])
        self.linear5 = nn.Linear(hidden_arr[3]+3+num_actions,1)
        # 90x128x2
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=16,kernel_size=5,padding=0,stride=1)
        # 86x124x16
        self.pool1 = nn.MaxPool2d(2,2)
        # 43x62x16
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5,padding=0,stride=1)
        # 39x58x16
        self.pool2 = nn.MaxPool2d(2,2)
        # 19x29x16
        self.fc1   = nn.Linear(19*29*16,500)

        # Q1 Architecture
        self.fc2_q1 = nn.Linear(500,50)
        self.fc3_q1 = nn.Linear(50,8)
        self.fc4_q1 = nn.Linear(8+num_actions,1) 

        self.apply(weights_init_)

    def forward(self,state,action):
        x = self.pool1(F.relu(self.conv1(state)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1,19*29*16)
        x = F.relu(self.fc1(x))
        # q1 forward
        q1 = F.relu(self.fc2_q1(x))
        q1 = F.relu(self.fc3_q1(q1))
        q1 = torch.cat([q1, action], dim=1) # concatenate action_target
        q1 = self.fc4_q1(q1)
        # q2 forward
        q2 = F.relu(self.fc2_q2(x))
        q2 = F.relu(self.fc3_q2(q2))
        q2 = torch.cat([q2, action], dim=1) # concatenate action_target
        q2 = self.fc4_q2(q2)
        return q1, q2

class GaussianPolicyCNN(nn.Module):
    '''
    Pi(s)
    Input: State --> [b,2,90,128]
    Output: mean,log_std [b,2,2]
            action A(s)
    '''
    def __init__(self, num_actions, init_w=3e-3):
        super(GaussianPolicyCNN, self).__init__()
        
        # 90x128x2
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=16,kernel_size=5,padding=0,stride=1)
        # 86x124x16
        self.pool1 = nn.MaxPool2d(2,2)
        # 43x62x16
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5,padding=0,stride=1)
        # 39x58x16
        self.pool2 = nn.MaxPool2d(2,2)
        # 19x29x16
        self.fc1        = nn.Linear(19*29*16,500)
        self.fc2        = nn.Linear(500,50)

        # for mean
        self.mean_fc3   = nn.Linear(50,10)
        self.mean_fc4   = nn.Linear(10,num_actions) 
        self.mean_fc4.weight.data.uniform_(-init_w,init_w)
        self.mean_fc4.bias.data.uniform_(-init_w,init_w)

        # for log_std
        self.std_fc3   = nn.Linear(50,10)
        self.std_fc4   = nn.Linear(10,num_actions) 
        self.std_fc4.weight.data.uniform_(-init_w,init_w)
        self.std_fc4.bias.data.uniform_(-init_w,init_w)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.pool1(F.relu(self.conv1(state)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1,19*29*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = F.relu(self.mean_fc3(x))
        mean = self.mean_fc4(mean)

        log_std = F.relu(self.std_fc3(x))
        log_std = self.std_fc4(log_std)
        log_std = torch.clamp(log_std, LOG_SIG_MIN,LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean


    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean