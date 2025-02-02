# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 14:11:07 2021

@author: makot
"""


import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import time



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 11)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(11, 11)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(11, 11)
        self.last = nn.Linear(11, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.souftplus = nn.Softplus()
        
        self.features = nn.Sequential(
                self.l1,
                self.relu,
                self.l2,
                self.relu,
                self.l3,
                self.relu,
                self.last
                )
    
    def forward(self, x):
        x1 = self.features(x)
        return x1

class Net_u(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 11)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(11, 11)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(11, 11)
        self.last = nn.Linear(11, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.souftplus = nn.Softplus()

        self.features = nn.Sequential(
                self.l1,
                self.relu,
                self.l2,
                self.relu,
                self.l3,
                self.relu,
                self.last
                )
    
    def forward(self, x):
        x1 = self.features(x)
        return x1
    
class Environment:
    def __init__(self):
        self.Y0_list = []
        self.LOSS_list = []
        self.time_list = []
        self.Z_df = pd.DataFrame(index = range(1, 10000+1), columns = ['Z0', 'Z_X0', 'Z5', 'Z_X5', 'Z10', 'Z_X10', 'Z15', 'Z_X15', 'Z20', 'Z_X20', 'Z25', 'Z_X25'])

        self.result = 0

        
    def run_path(self):
        
        start = time.time()

        num_episode = 10000
        num_batch = 256
        
        x = 1.0
        x_ae_1 = 0
        T = 0.1
        eta = 5 * 0.001
        step = 25
        epsilon = 0.01

        net_u_list = []
        optimizer_u_list = []
        net_Z_list = []
        optimizer_Z_list = []

        for num in range(step+1):            
            net_u_list.append(Net_u())
            tmp_optimizer_u = optim.Adam(net_u_list[num].parameters(), lr= eta)
            optimizer_u_list.append(tmp_optimizer_u)

        for num in range(step):            
            net_Z_list.append(Net())
            tmp_optimizer_Z = optim.Adam(net_Z_list[num].parameters(), lr= eta)
            optimizer_Z_list.append(tmp_optimizer_Z)
        
        def b(t, x, y, z):
            return (-1)*epsilon/(epsilon)*(0.5*torch.sin(t+x)*torch.cos(t+x)*(y*y+z))
        
        def sigma(t, x, y, z):
            return 0.5*epsilon/(epsilon)*torch.cos(t+x)*(y*torch.sin(t+x)+z+1)

        def f(t, x, y, z):
            return epsilon/(epsilon)*y*z-torch.cos(t+x)
        
        def g(x):
            return torch.sin(T+x)


        dt = np.ones(num_batch) * (T / step)        
        X_pre_0 = np.ones(num_batch) * x
        X_AE_1_pre_0 = np.ones(num_batch) * x_ae_1
        
        dt = torch.tensor(np.array([dt]).reshape(-1, 1)).float()                
        X_pre_0 = torch.tensor(np.array([X_pre_0]).reshape(-1, 1)).float()                
        X_AE_1_pre_0 = torch.tensor(np.array([X_AE_1_pre_0]).reshape(-1, 1)).float()                

        for episode in range(0, num_episode):            
            
            for num in range(step):
                net_Z_list[num].train()
                optimizer_Z_list[num].zero_grad()

            for num in range(step+1):
                net_u_list[num].train()
                optimizer_u_list[num].zero_grad()

            L = 0

            X_pre = X_pre_0
            X_AE_1_pre = X_AE_1_pre_0
            
            input_u = torch.tensor(np.stack([np.squeeze(X_pre.detach().numpy())], axis=1)).float()            
            Y_pre = net_u_list[0](input_u)
            Y_AE_pre = np.sin(x) + epsilon*(np.cos(T+x)*(-1/8)*(np.power(np.sin(T+x), 4)-np.power(np.sin(x), 4))
                    +(1/8)*(np.cos(T+x)*np.power(np.sin(T+x), 4)-np.cos(x)*np.power(np.sin(x), 4))
                    +(1/6)*(np.power(np.cos(T+x), 3)*np.power(np.sin(T+x), 2)-np.power(np.cos(x), 3)*np.power(np.sin(x), 2))
                    -(1/15)*(np.power(np.cos(T+x), 5)-np.power(np.cos(x), 5))
                    -np.power(np.sin(x), 4)*(-np.cos(T+x)+np.cos(x)))
            Y_pre = Y_AE_pre + Y_pre

            Y_AE_1_pre = np.cos(T+x)*(-1/8)*(np.power(np.sin(T+x), 4)-np.power(np.sin(x), 4))\
                    +(1/8)*(np.cos(T+x)*np.power(np.sin(T+x), 4)-np.cos(x)*np.power(np.sin(x), 4))\
                    +(1/6)*(np.power(np.cos(T+x), 3)*np.power(np.sin(T+x), 2)-np.power(np.cos(x), 3)*np.power(np.sin(x), 2))\
                    -(1/15)*(np.power(np.cos(T+x), 5)-np.power(np.cos(x), 5))\
                    -np.power(np.sin(x), 4)*(-np.cos(T+x)+np.cos(x))
            
            for num in range(step):
                ww = np.random.randn(1, num_batch)
                ww = torch.tensor(np.array([ww]).reshape(num_batch, -1)).float()

                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                
                if num == 0:
                    u = Y_pre
                else:
                    input_u = torch.tensor(np.stack([np.squeeze(X_pre.detach().numpy())], axis=1)).float()            
                    u = net_u_list[num](input_u)
                    u = Y_AE_pre + u

                input_Z = torch.tensor(np.stack([np.squeeze(X_pre.detach().numpy())], axis=1)).float()                                
                Z = net_Z_list[num](input_Z)
                Z_AE = 0 + epsilon*(np.cos(T+x)*0.5*np.cos(t+x)*(np.sin(t+x)*np.sin(t+x)+1) + 0.5*np.cos(t+x)*(np.sin(t+x)*np.sin(t+x)+1)*(-1)*(np.cos(T+x)-np.cos(t+x)))
                Z = Z_AE + Z                

                Z_X = torch.pow(torch.cos(t+X_pre), 2)

                if num == 0:
                    self.Z_df.loc[[episode+1], ['Z0']] = Z.mean().item()
                    self.Z_df.loc[[episode+1], ['Z_X0']] = Z_X.mean().item()
                elif num == 4:
                    self.Z_df.loc[[episode+1], ['Z5']] = Z.mean().item()
                    self.Z_df.loc[[episode+1], ['Z_X5']] = Z_X.mean().item()
                elif num == 9:
                    self.Z_df.loc[[episode+1], ['Z10']] = Z.mean().item()
                    self.Z_df.loc[[episode+1], ['Z_X10']] = Z_X.mean().item()
                elif num == 14:
                    self.Z_df.loc[[episode+1], ['Z15']] = Z.mean().item()
                    self.Z_df.loc[[episode+1], ['Z_X15']] = Z_X.mean().item()
                elif num == 19:
                    self.Z_df.loc[[episode+1], ['Z20']] = Z.mean().item()
                    self.Z_df.loc[[episode+1], ['Z_X20']] = Z_X.mean().item()
                elif num == 24:
                    self.Z_df.loc[[episode+1], ['Z25']] = Z.mean().item()
                    self.Z_df.loc[[episode+1], ['Z_X25']] = Z_X.mean().item()

                X = X_pre + b(t, X_pre, u, Z) * dt + sigma(t, X_pre, u, Z) * ww * np.sqrt(dt)
                Y_AE_0 = np.sin(t+x)
                Z_AE_1 = np.cos(T+x)*0.5*np.cos(t+x)*(np.sin(t+x)*np.sin(t+x)+1) + 0.5*np.cos(t+x)*(np.sin(t+x)*np.sin(t+x)+1)*(-1)*(np.cos(T+x)-np.cos(t+x))
                X_AE_1 = X_AE_1_pre - 0.5*np.sin(t+x)*np.cos(t+x)*Y_AE_0*Y_AE_0*dt + 0.5*np.cos(t+x)*(Y_AE_0*np.sin(t+x)+1)*ww*np.sqrt(dt)
                Y_AE_1 = Y_AE_1_pre - np.sin(t+x)*X_AE_1_pre*dt + Z_AE_1*ww*np.sqrt(dt)
                
                Y_AE = Y_AE_0 + epsilon * Y_AE_1
                
                if num == 0:
                    self.Y0_list.append(Y_pre.mean().item())
                Y = Y_pre - f(t, X_pre, Y_pre, Z) * dt + Z * ww * np.sqrt(dt)

                input_u_n = torch.tensor(np.stack([np.squeeze(X.detach().numpy())], axis=1)).float()            
                u_n = Y_AE + net_u_list[num+1](input_u_n)                    
                L = L + (T/step) * torch.pow(Y - u_n, 2).mean()

                X_pre = X
                X_AE_1_pre = X_AE_1
                Y_pre = Y
                Y_AE_1_pre = Y_AE_1
                Y_AE_pre = Y_AE

            LOSS = torch.pow(Y - g(X), 2).mean() / 2 + L / 2
            
            self.LOSS_list.append(LOSS.item())

            LOSS.backward(retain_graph=True)

            for num in range(step):
                optimizer_u_list[num].step()
                optimizer_Z_list[num].step()

            self.time_list.append(time.time() - start)            


        
        
if __name__ == '__main__':
    
    for number in range(1, 11, 1):
        env = Environment()
        env.run_path()
        Y0_list = env.Y0_list
        LOSS_list = env.LOSS_list
        time_list = env.time_list
    
        Y0_df = pd.DataFrame(Y0_list, columns = ['Y0'])
        LOSS_df = pd.DataFrame(LOSS_list, columns = ['LOSS'])
        time_df = pd.DataFrame(time_list, columns = ['time'])

        result = pd.concat([Y0_df, LOSS_df], axis = 1)
        result = pd.concat([result, time_df], axis = 1)
        
        