import argparse


parser = argparse.ArgumentParser(description='PyTorch Soft Actor Critic')

parser.add_argument('--episodes', type=int, default=1000000,
                    help='number of episodes to train the agent')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--save_path', type=str, default='./Models/policy.pt', 
                    help='file path to save the weights')
parser.add_argument('--load_model', type=int, default=0, 
                    help='bool to load model from pre-trained weights')
parser.add_argument('--tensorboard', type=int, default=1, 
                    help='Whether we want tensorboardX logging')




args = parser.parse_args()