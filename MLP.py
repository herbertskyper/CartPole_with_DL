#-*- coding: gbk-*-
from torch import nn
import torch
import torch.nn.functional as F

class CartPolePolicy(nn.Module):
    def __init__(self):
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self.drop=nn.Dropout(p=0.6)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x=F.relu(x)
        x=self.fc2(x)
        return F.softmax(x, dim=1) #作用于每一行，使得每一行的和为1
    
    #最小化策略损失即最大化奖励
    @staticmethod
    def compute_policy_loss(n,log_p):
        reward=[i for i in range(n,0,-1)] #越靠近结束的步数，奖励越小，因为导致了失败
        reward=torch.tensor(reward).float()
        reward=(reward-reward.mean())/reward.std()
        
        loss=0
        for pi,ri in zip(log_p,reward):
            loss+= -pi*ri
        
        return loss
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

        