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

# 最多训练1000个回合，每回合最多行动10000次，步数超过5000次则视为成功
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
        
        #不直接选择动作，而是根据概率选择动作，类似于epsilon-greedy
        m=Categorical(probs)
        action=m.sample()
        
        state, _ , done, _, _= env.step(action.item())
        if done:
            break
        log_p.append(m.log_prob(action)) #计算该动作的对数概率
        
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
