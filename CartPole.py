# -*encoding:gbk*-
import gym
import pygame
import time
import random
import numpy as np
import torch  

from MLP import CartPolePolicy

import tkinter as tk
from tkinter import messagebox,ttk

def start_manual_mode():
    # 在这里启动手动模式
    messagebox.showinfo("Mode", "手动模式已启动")
    pygame.init() 
    pygame.display.set_mode((800, 600))
    
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
    
    # 创建一个窗口
    screen = pygame.display.set_mode((800, 600))

    # 创建一个字体对象
    font = pygame.font.Font(None, 36)

    # 显示倒计时
    for i in range(3, 0, -1):
        # 创建一个包含倒计时的Surface对象
        text = font.render(f"Game starts in {i} seconds...", True, (255, 255, 255))

        # 将Surface对象绘制到屏幕上
        screen.fill((0, 0, 0))
        screen.blit(text, (200, 300))

        # 更新屏幕
        pygame.display.flip()

        # 等待1秒
        time.sleep(1)
    
    #实现游戏过程
    start_time = time.time()
    max_action = 1000   #最大执行次数
    
    step=0
    fail=False

    
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
        
        print(f"Step {step}: {state} action: {action}"
            f"angle={state[2]:.2f} position={state[0]:.2f}") 
    
    end_time=time.time()
    game_time=end_time-start_time
    if fail:
        print(f"Failed at step {step} in {game_time:.2f} seconds")
    else:
        print(f"Succeeded in {step} steps in {game_time:.2f} seconds")
    env.close()

def start_auto_mode():
    # 在这里启动自动模式
    messagebox.showinfo("Mode", "自动模式已启动")
    pygame.init() 
    pygame.display.set_mode((800, 600))
    
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
    
    # 创建一个窗口
    screen = pygame.display.set_mode((800, 600))

    # 创建一个字体对象
    font = pygame.font.Font(None, 36)

    # 显示倒计时
    for i in range(3, 0, -1):
        # 创建一个包含倒计时的Surface对象
        text = font.render(f"Game starts in {i} seconds...", True, (255, 255, 255))

        # 将Surface对象绘制到屏幕上
        screen.fill((0, 0, 0))
        screen.blit(text, (200, 300))

        # 更新屏幕
        pygame.display.flip()

        # 等待1秒
        time.sleep(1)
    
    #实现游戏过程
    start_time = time.time()
    max_action = 1000   #最大执行次数
    
    step=0
    fail=False

    
    policy=CartPolePolicy()
    policy.load_state_dict(torch.load('CartPolePolicy.pth'))
    policy.eval()
    
    for step in range(1,max_action+1):
        
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


if __name__ == '__main__':
    root = tk.Tk()
    root.title("CartPole游戏")

    # 加载背景图片
    bg_image = tk.PhotoImage(file="background.png")  # 请替换为你的图片文件路径

    # 创建一个用于显示背景图片的Label
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # 设置窗口大小
    root.geometry("300x200")

    # 使用ttk库来创建更现代的按钮
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 20), padding=10)

    manual_button = ttk.Button(root, text="手动模式", command=start_manual_mode)
    manual_button.pack(side="left", padx=20, pady=20)

    auto_button = ttk.Button(root, text="自动模式", command=start_auto_mode)
    auto_button.pack(side="right", padx=20, pady=20)

    root.mainloop()    