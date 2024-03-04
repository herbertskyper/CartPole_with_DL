# -*encoding:gbk*-
import gym
import pygame
import time
import random
import numpy as np
import torch  

from MLP import CartPolePolicy

if __name__ == '__main__':
    print("请选择游玩模式：输入1表示手动模式，输入2表示自动模式")
    opt=input()
    pygame.init() 
    
    env = gym.make(id='CartPole-v1',render_mode='human')
    state, _ =env.reset()
    
    cart_position = state[0]
    cart_speed = state[1]
    pole_angle = state[2]
    pole_speed = state[3]
    
    print(f"Begin State: {state}")
    print(f"cart_position: {cart_position:.2f}")
    print(f"cart_speed: {cart_speed:.2f}")
    print(f"pole_angle: {pole_angle:.2f}")
    print(f"pole_speed: {pole_speed:.2f}")
    time.sleep(2)
    
    #实现游戏过程
    start_time = time.time()
    max_action = 1000   #最大执行次数
    
    step=0
    fail=False
    if opt=="1":
        for step in range(1,max_action+1):
            time.sleep(0.2)
            

            #以非阻塞的方式获取键盘输入
            keys=pygame.key.get_pressed()
            
            action=0
            
            #随机设置小车下一个动作
            if not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
                action=random.choice([0,1])
                
            if keys[pygame.K_LEFT]:
                action=0
                
            if keys[pygame.K_RIGHT]:
                action=1
                
            state, _ , done, _, _= env.step(action)
            
            if done:
                fail=True
                break
            
            print(f"Step {step}: {state} action: {action}",end='  ')
            print(f"angle={state[2]:.2f} position={state[0]:.2f}")
    else:
        policy=CartPolePolicy()
        # policy.load_state_dict(torch.load('CartPolePolicy.pth'))
        policy.eval()
        
        for step in range(1,max_action+1):
            time.sleep(0.2)
            
            
            state=torch.from_numpy(state).float().unsqueeze(0)
            action=policy(state).argmax(dim=1).item()
                
            state, _ , done, _, _= env.step(action)
            
            if done:
                fail=True
                break
            
            print(f"Step {step}: {state} action: {action}"
                f"angle={state[2]:.2f} position={state[0]:.2f}")
            
    end_time=time.time()
    game_time=end_time-start_time
        
    if fail:
        print(f"Failed at step {step} in {game_time:.2f} seconds")
    else:
        print(f"Succeeded in {step} steps in {game_time:.2f} seconds")
    env.close()
        
    