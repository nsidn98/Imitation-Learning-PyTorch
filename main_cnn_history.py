'''
CNN with history of 4 boxes as states
'''
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
from model import StochasticPolicyCNN
from config import args

# set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# configure CUDA availability
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer(object):
	def __init__(self, max_size=1e4):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		#[st_img,dim,action]
		x, y, u = [], [], []
		for i in ind:
			X, Y, U = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))

		return np.array(x), np.array(y), np.array(u)


class BehaviouralCloning():
    def __init__(self,args):
        self.ldc_len = 80
        self.ldc_wid = 45
        self.ldc_ht  = 45
        self.num_actions = 2
        self.input_size = self.ldc_len*self.ldc_wid
        self.data_maker = BoxMaker(self.ldc_ht,self.ldc_wid,self.ldc_len)
        self.policy = StochasticPolicyCNN()
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

        buff = ReplayBuffer(1e4)

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
            data = self.data_maker.get_data_dict(flatten=False)
            data = np.array(data)
            state = np.zeros((4,45,80))
            dim   = np.zeros((12))
            for i in range(len(data)):
                state = np.roll(state,axis=0,shift=1)
                state[0,:,:] = data[i][0]
                dim = np.roll(dim,shift=3)
                dim[:3] = data[i][1]
                action = data[i][2][:2]
                buff.add([state,dim,action])

            if len(buff.storage) >= args.batch_size:
                state_feed, dim_feed, action_feed = buff.sample(args.batch_size)
                state_feed   = torch.from_numpy(state_feed)/self.ldc_ht
                dim_feed     = torch.from_numpy(dim_feed)/self.ldc_ht
                action_feed  = torch.from_numpy(action_feed)
                if use_cuda:
                    state_feed   = state_feed.to(device)
                    dim_feed     = dim_feed.to(device)
                    action_feed  = action_feed.to(device)
                a,m,s   = self.policy.sample(state_feed.float(),dim_feed.float())
                x,y     = self.shift_action(a)
                pred = torch.cat([x.unsqueeze(1),y.unsqueeze(1)],dim=1)

                optimizer.zero_grad()
                loss = F.mse_loss(pred,action_feed.float())
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
        self.writer.close()

if __name__ == "__main__":
    if not os.path.exists('./Models'):
        os.makedirs('./Models')
    BC = BehaviouralCloning(args)
    BC.train()

        
