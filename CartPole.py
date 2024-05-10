# -*encoding:gbk*-
import gym
import pygame
import time
import random
import numpy as np
import torch  
import os

from MLP import CartPolePolicy

import tkinter as tk
from tkinter import messagebox,ttk

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QFont,QIcon
from PyQt5.QtCore import Qt,QUrl,QFileInfo
from Ui_window import Ui_MainWindow 

def start_manual_mode():
    pygame.init() 
    pygame.display.set_mode((600, 400))
    
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
    screen = pygame.display.set_mode((600, 400))

    # 创建一个字体对象
    font = pygame.font.Font(None, 36)

    # 显示倒计时
    for i in range(3, 0, -1):
        # 创建一个包含倒计时的Surface对象
        text = font.render(f"Game starts in {i} seconds...", True, (255, 255, 255))

        # 将Surface对象绘制到屏幕上
        screen.fill((0, 0, 0))
        screen.blit(text, (150, 200))

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
    # self.hide()
    if not os.path.exists("CartPolePolicy.pth"):
        print("模型不存在，请重新选择！")
        return 0
    # 在这里启动自动模式
    pygame.init() 
    pygame.display.set_mode((600, 400))
    
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
    screen = pygame.display.set_mode((600, 400))

    # 创建一个字体对象
    font = pygame.font.Font(None, 36)

    # 显示倒计时
    for i in range(3, 0, -1):
        # 创建一个包含倒计时的Surface对象
        text = font.render(f"Game starts in {i} seconds...", True, (255, 255, 255))

        # 将Surface对象绘制到屏幕上
        screen.fill((0, 0, 0))
        screen.blit(text, (150, 200))

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
    
    return 1
    
    # self.show()


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CartPole Game Menu")
        
        self.pushButton_auto.clicked.connect(self.start_auto_mode_entry)
        self.pushButton_man.clicked.connect(self.start_manual_mode_entry)
        
        self.player = QMediaPlayer()
        self.play_flag=1
        music_file_info = QFileInfo("resources/music.mp3")
        music_file_url = QUrl.fromLocalFile(music_file_info.absoluteFilePath())
        self.player.setMedia(QMediaContent(music_file_url))
        self.playMusic()

        self.pushButton_music.clicked.connect(self.playMusic)
        
        self.pushButton_back.clicked.connect(self.close)
        
    def close(self):
        sys.exit(app.exec_())
    
    def playMusic(self):
        if(self.play_flag==1):
            self.pushButton_music.setIcon(QIcon('./resources/music.gif'))
            self.player.play()
            self.play_flag=0
        else:
            self.pushButton_music.setIcon(QIcon('./resources/music_off.png'))
            self.player.pause()
            self.play_flag=1
        
    def start_auto_mode_entry(self):
        self.label_title.setText('自动模式已启动')
        self.label_title.setFont(QFont("Roman times", 30, QFont.Bold))
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.show()
        QApplication.processEvents()
        for i in range(100):
            self.setWindowOpacity(1-i/100) #设置透明度
            time.sleep(0.02)
            QApplication.processEvents()
        self.hide()
        if(not start_auto_mode()):
            self.label_title.setText('未发现模型，请先进行训练！')
            self.label_title.setFont(QFont("Roman times", 40, QFont.Bold))
            self.label_title.setAlignment(Qt.AlignCenter)
            self.label_title.show()
            QApplication.processEvents()
        self.setWindowOpacity(1)
        QApplication.processEvents()
        self.show()

    def start_manual_mode_entry(self):
        self.label_title.setText('手动模式已启动')
        self.label_title.setFont(QFont("Roman times", 30, QFont.Bold))
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.show()
        QApplication.processEvents()
        for i in range(100):
            self.setWindowOpacity(1-i/100) #设置透明度
            time.sleep(0.02)
            QApplication.processEvents()
        self.hide()
        start_manual_mode()
        self.setWindowOpacity(1)
        QApplication.processEvents()
        self.show()


if __name__ == '__main__': 
    app = QApplication(sys.argv) 
    myWindow = MyWindow() 
    myWindow.show()
    sys.exit(app.exec_())