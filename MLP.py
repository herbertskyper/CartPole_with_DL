from torch import nn
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
        return F.softmax(x, dim=1)