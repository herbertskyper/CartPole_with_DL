# -*- coding: gbk -*-
from torch import nn
from torch.distributions import Categorical
import torch
from MLP import CartPolePolicy
import gym
import pygame
import time

pygame.init() 
env = gym.make(id='CartPole-v1')
env.reset(seed=543)
torch.manual_seed(543)
    
policy=CartPolePolicy()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
policy.apply(CartPolePolicy.init_weights)

# ���ѵ��1000���غϣ�ÿ�غ�����ж�10000�Σ���������5000������Ϊ�ɹ�
max_episode = 1000
max_actions = 10000
max_steps = 5000


for episode in range(1,max_episode+1):
    state, _ =env.reset()
    step=0
    log_p=[]
    for step in range(1,max_actions+1):
        state=torch.from_numpy(state).float().unsqueeze(0)
        probs=policy(state)
        
        #��ֱ��ѡ���������Ǹ��ݸ���ѡ������������epsilon-greedy
        m=Categorical(probs)
        action=m.sample()
        
        state, _ , done, _, _= env.step(action.item())
        if done:
            break
        log_p.append(m.log_prob(action)) #����ö����Ķ�������
        
    if step>=max_steps:
        print(f"Episode {episode} failed at step {step}")
        break
    
    optimizer.zero_grad()
    loss=policy.compute_policy_loss(step,log_p)
    loss.backward()
    optimizer.step()
    
    if episode%10==0:
        print(f"Episode {episode} finished after {step} steps")

torch.save(policy.state_dict(), 'CartPolePolicy.pth')
