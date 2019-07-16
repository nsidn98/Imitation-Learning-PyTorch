import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter

from make_data import BoxMaker
from model import StochasticPolicy
from config import args

# set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# configure CUDA availability
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class BehaviouralCloning():
    def __init__(self,args):
        self.ldc_len = 80
        self.ldc_wid = 45
        self.ldc_ht  = 45
        self.num_actions = 2
        self.input_size = self.ldc_len*self.ldc_wid
        self.data_maker = BoxMaker(self.ldc_ht,self.ldc_wid,self.ldc_len)
        self.policy = StochasticPolicy(self.input_size, self.num_actions)
        if args.tensorboard:
            print('Init tensorboardX')
            self.writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))


    def shift_action(self,action):
        x = action[:,0]
        y = action[:,1]
        y = y*self.ldc_len/2 +self.ldc_len/2
        x = x*self.ldc_wid/2 +self.ldc_wid/2
        return x,y

    def train(self):
        optimizer = optim.Adam(self.policy.parameters(),
                                     lr=args.lr)
        start_episode = 0

        if args.load_model:
            if not use_cuda:
                checkpoint = torch.load(args.save_path,map_location='cpu')
            else:
                checkpoint = torch.load(args.save_path)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            loss = checkpoint['loss']
            self.policy.train()

        for episodes in tqdm(range(args.episodes)):
            data = self.data_maker.get_data_dict()
            data = np.array(data)
            state = []
            dim = []
            action = []
            for i in range(len(data)):
                state.append(data[i,0])
                dim.append(data[i,1])
                action.append(data[i,2][:2])
            state   = torch.from_numpy(np.array(state))/self.ldc_ht
            dim     = torch.from_numpy(np.array(dim))/self.ldc_ht
            action  = torch.from_numpy(np.array(action))
            if use_cuda:
                state   = state.to(device)
                dim     = dim.to(device)
                action  = action.to(device)
            a,m,s   = self.policy.sample(state.float(),dim.float())
            x,y     = self.shift_action(a)
            pred = torch.cat([x.unsqueeze(1),y.unsqueeze(1)],dim=1)

            optimizer.zero_grad()
            loss = F.mse_loss(pred,action.float())
            loss.backward()
            # print(loss.item())
            if args.tensorboard:
                self.writer.add_scalar('Loss',loss.item(),episodes+start_episode)
            optimizer.step()

            if episodes % 100 == 0 and episodes !=0:
                # print('Saving model...')
                torch.save({
                            'episode': episodes,
                            'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, args.save_path)
            # print(loss)
        self.writer.close()

if __name__ == "__main__":
    if not os.path.exists('./Models'):
        os.makedirs('./Models')
    BC = BehaviouralCloning(args)
    BC.train()

        
