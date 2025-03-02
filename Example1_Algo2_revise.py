# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 14:11:07 2021

@author: makot
"""


import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from torchinfo import summary
import torch.optim as optim
import math
import pandas as pd
from scipy.optimize import minimize as mmm
from matplotlib.font_manager import FontProperties
fp = FontProperties(fname=r'C:\WINDOWS\Fonts\meiryob.ttc')
import time



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 110)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(110, 110)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(110, 110)
        self.last = nn.Linear(110, 100)
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

        #x1 = self.l1(x)
        #x1 = self.l3(x1)
        return x1 #+ 1e-4

class Net_u(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 110)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(110, 110)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(110, 110)
        self.last = nn.Linear(110, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.souftplus = nn.Softplus()
        
        """
        nn.init.zeros_(self.l1.weight)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l3.weight)

        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)
        nn.init.zeros_(self.l3.bias)
        """
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

        #x1 = self.l1(x)
        #x1 = self.l3(x1)
        return x1 #+ 1e-4

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
        final_cnt = 0
        
        x = 1.0
        x_ae_1 = 0
        T = 0.1
        eta = 5 * 0.001
        step = 25
        d = 5
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
            return t/2*torch.pow(torch.cos(y+x), 2)
        
        def sigma(t, x, y, z):
            return t/2*torch.pow(torch.sin(y+x), 2)

        def f(t, x, y, z):
            sum_z = torch.sum(z, axis = 1).reshape(-1, 1)
            sum_x2 = 1/d*(1+t/2)*torch.sum(torch.pow(x, 2), axis = 1).reshape(-1, 1)

            sum_x = 0
            for i in range(0, d-1):
                sum_x += x[:, i].reshape(-1, 1) * (x[:, i+1].reshape(-1, 1) + t)            
            sum_x = t/d*(sum_x + x[:, d-1].reshape(-1, 1) * (x[:, 0].reshape(-1, 1) + t))

            sum_sin = 0
            for i in range(0, d-1):
                sum_sin += (x[:, i+1].reshape(-1, 1) + t)*torch.pow(torch.sin(y + x[:, i].reshape(-1, 1)), 4)
            sum_sin = 0.5*torch.pow(t/d, 2)*(sum_sin + (x[:, 0].reshape(-1, 1) + t)*torch.pow(torch.sin(y + x[:, d-1].reshape(-1, 1)), 4))

            return sum_z - sum_x2 - sum_x - sum_sin
        
        def g(x):
            sum_ = 0
            for i in range(0, d-1):
                sum_ += torch.pow(x[:, i].reshape(-1, 1), 2) * (x[:, i+1].reshape(-1, 1) + T)            
            sum_ = 1/d*(sum_ + torch.pow(x[:, d-1], 2).reshape(-1, 1) * (x[:, 0].reshape(-1, 1) + T))            
            return sum_

        def y_AE1_1(mu, x):
            return torch.sum(mu * x, axis = 1).reshape(-1, 1)

        def y_AE1_2(mu, x):
            sum_ = 0
            for i in range(0, d-1):
                sum_ += mu[:, i].reshape(-1, 1) * x[:, i+1].reshape(-1, 1)            
            sum_ = sum_ + mu[:, d-1].reshape(-1, 1) * x[:, 0].reshape(-1, 1)
            return sum_

        def y_AE1_3(mu):
            return torch.sum(mu, axis = 1).reshape(-1, 1)
        
        def y_AE0_f(t, x):            
            sum_x2 = 1/d*(1+t/2)*torch.sum(torch.pow(x, 2), axis = 1).reshape(-1, 1)

            sum_x = 0
            for i in range(0, d-1):
                sum_x += x[:, i].reshape(-1, 1) * (x[:, i+1].reshape(-1, 1) + t)            
            sum_x = t/d*(sum_x + x[:, d-1].reshape(-1, 1) * (x[:, 0].reshape(-1, 1) + t))

            sum_sin = 0
            for i in range(0, d-1):
                sum_sin += (x[:, i+1].reshape(-1, 1) + t)*torch.pow(torch.sin(x[:, i].reshape(-1, 1)), 4)
            sum_sin = 0.5*torch.pow(t/d, 2)*(sum_sin + (x[:, 0].reshape(-1, 1) + t)*torch.pow(torch.sin(x[:, d-1].reshape(-1, 1)), 4))

            return - sum_x2 - sum_x - sum_sin
        
        def AE0(x=1.0):
            
            X_AE0_k = [x for i in range(0, step+1, 1)]
            y_AE0_k = [0 for i in range(0, step+1, 1)]
            
            dt = np.ones(num_batch) * (T / step)        
            X_pre_0 = np.ones([d, num_batch]) * x
            
            dt = torch.tensor(np.array([dt]).reshape(-1, 1)).float()                
            X_AE0_pre = torch.tensor(np.array([X_pre_0]).reshape(-1, d)).float()                
                                    
            for num in range(step):
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                
                X_AE0 = X_AE0_pre + torch.pow(torch.cos(X_AE0_pre), 2) * dt                
                X_AE0_k[num] = X_AE0_pre
                X_AE0_pre = X_AE0
            X_AE0_k[step] = X_AE0
            
            Y_AE0_T = 0
            for i in range(0, d-1, 1):
                Y_AE0_T += torch.pow(X_AE0[:, i].reshape(-1, 1), 2) * (X_AE0[:, i+1].reshape(-1, 1) + T)
            Y_AE0_T = 1/d * (Y_AE0_T + torch.pow(X_AE0[:, d-1].reshape(-1, 1), 2) * (X_AE0[:, 0].reshape(-1, 1) + T))
            
            Y_AE0_t_pre = Y_AE0_T
            for num in range(0, step):
                num = step-num-1
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                Y_AE0_t = Y_AE0_t_pre + y_AE0_f(t, X_AE0_k[num]) * dt
                y_AE0_k[num] = Y_AE0_t
                Y_AE0_t_pre = Y_AE0_t
            y_AE0_k[step] = Y_AE0_T
            
            return X_AE0_k, y_AE0_k, Y_AE0_T


        def AE1(X_AE0_k, y_AE0_k):
            
            Z_AE1_k = [x for i in range(0, step+1, 1)]
            W_t_k = [1 for i in range(0, step+1, 1)]
            mu_bar_t_k = [1 for i in range(0, step+1, 1)]
            EX_AE1_k = [x for i in range(0, step+1, 1)]
            mu_hat_t_11_k = [1 for i in range(0, step+1, 1)]
            mu_hat_t_12_k = [1 for i in range(0, step+1, 1)]
            
            # W(t)
            dt = np.ones(num_batch) * (T / step)  
            W_t_pre = np.ones([d, num_batch]) * 1            
            dt = torch.tensor(np.array([dt]).reshape(-1, 1)).float()                
            W_t_pre = torch.tensor(np.array([W_t_pre]).reshape(-1, d)).float()                
            for num in range(step):
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                W_t = W_t_pre -t * torch.cos(X_AE0_k[num]) * torch.sin(X_AE0_k[num]) * W_t_pre * dt                
                W_t_k[num] = W_t_pre
                W_t_pre = W_t
            W_t_k[step] = W_t

            # mu_bar_T
            mu_bar_T = []
            for i in range(0, d-1, 1):
                mu_bar_T.append(1/d*2 * X_AE0_k[step][:, i].reshape(-1, 1) * (X_AE0_k[step][:, i+1].reshape(-1, 1) + T))
            mu_bar_T.append(1/d*2 * X_AE0_k[step][:, d-1].reshape(-1, 1) * (X_AE0_k[step][:, 0].reshape(-1, 1) + T))            
            mu_bar_T = torch.cat(mu_bar_T, dim = 1)            
            
            # mu_hat_t_1_0
            for num in range(step+1):            
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                mu_bar_t = []
                for i in range(0, d-1, 1):
                    mu_bar_t.append(-1/d*(1+t/2)*2* X_AE0_k[num][:, i].reshape(-1, 1)
                    - t/d * (X_AE0_k[num][:, i+1].reshape(-1, 1) + t)
                    -0.5*torch.pow(t/d, 2)*(X_AE0_k[num][:, i+1].reshape(-1, 1) + t)*4*torch.pow(torch.sin(X_AE0_k[num][:, i].reshape(-1, 1)), 3)*torch.cos(X_AE0_k[num][:, i].reshape(-1, 1)))
                mu_bar_t.append(-1/d*(1+t/2)*2* X_AE0_k[num][:, d-1].reshape(-1, 1)
                - t/d * (X_AE0_k[num][:, 0].reshape(-1, 1) + t)
                -0.5*torch.pow(t/d, 2)*(X_AE0_k[num][:, 0].reshape(-1, 1) + t)*4*torch.pow(torch.sin(X_AE0_k[num][:, d-1].reshape(-1, 1)), 3)*torch.cos(X_AE0_k[num][:, d-1].reshape(-1, 1)))
                mu_bar_t = torch.cat(mu_bar_t, dim = 1)
                
                mu_bar_t_k[num] = mu_bar_t
            
            # Z_AE1
            for num in range(step):    
                num = step-num-1
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                num_list = [i for i in range(num, step, 1)]
                num_list.reverse()
                mu_hat_int = 0
                for numnum in num_list:
                    mu_hat_int += mu_bar_t_k[numnum]*W_t_k[numnum]/W_t_k[num] * dt
                
                f_1_Y = mu_bar_T*W_t_k[step]/W_t_k[num] + mu_hat_int
                
                Z_AE1_k[num] = f_1_Y * t/2 * torch.pow(torch.sin(y_AE0_k[num] + X_AE0_k[num]), 2)
            
            Z_AE1_k[step] = mu_bar_T * t/2 * torch.pow(torch.sin(y_AE0_k[step] + X_AE0_k[step]), 2)

            # EX_AE1
            X_pre_0 = np.ones([d, num_batch]) * x
            EX_AE1_pre = torch.tensor(np.array([X_pre_0]).reshape(-1, d)).float()                
                                    
            for num in range(step):
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                
                EX_AE1 = EX_AE1_pre - t * torch.cos(X_AE0_k[num]) * torch.sin(X_AE0_k[num]) * (y_AE0_k[num] + EX_AE1_pre) * dt                
                EX_AE1_k[num] = EX_AE1_pre
                EX_AE1_pre = EX_AE1
            EX_AE1_k[step] = EX_AE1

            # mu_hat_t_1_1
            for num in range(step+1):            
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                mu_bar_t = []
                for i in range(0, d, 1):
                    mu_bar_t.append(-t/d*X_AE0_k[num][:, i].reshape(-1, 1)-0.5*torch.pow(t/d, 2)*torch.pow(torch.sin(X_AE0_k[num][:, i].reshape(-1, 1)), 4))
                mu_bar_t = torch.cat(mu_bar_t, dim = 1)
                
                mu_hat_t_11_k[num] = mu_bar_t   

            # mu_hat_t_1_2
            for num in range(step+1):            
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                mu_bar_t = []
                for i in range(0, d-1, 1):
                    mu_bar_t.append(-0.5*torch.pow(t/d, 2)*(X_AE0_k[num][:, i+1].reshape(-1, 1)+t)*4*torch.pow(torch.sin(X_AE0_k[num][:, i].reshape(-1, 1)), 3)*torch.cos(X_AE0_k[num][:, i].reshape(-1, 1))*y_AE0_k[num])
                mu_bar_t.append(-0.5*torch.pow(t/d, 2)*(X_AE0_k[num][:, 0].reshape(-1, 1)+t)*4*torch.pow(torch.sin(X_AE0_k[num][:, d-1].reshape(-1, 1)), 3)*torch.cos(X_AE0_k[num][:, d-1].reshape(-1, 1))*y_AE0_k[num])
                mu_bar_t = torch.cat(mu_bar_t, dim = 1)
                
                mu_hat_t_12_k[num] = mu_bar_t   

            # EY_AE1_T
            EY_AE1_T = 0
            for i in range(0, d-1, 1):
                EY_AE1_T += 2 * X_AE0_k[step][:, i].reshape(-1, 1) * EX_AE1_k[step][:, i].reshape(-1, 1) * (X_AE0_k[step][:, i+1].reshape(-1, 1) + T) + torch.pow(X_AE0_k[step][:, i].reshape(-1, 1), 2) * EX_AE1_k[step][:, i+1].reshape(-1, 1)
                
            EY_AE1_T = 1/d * (EY_AE1_T + 2 * X_AE0_k[step][:, d-1].reshape(-1, 1) * EX_AE1_k[step][:, d-1].reshape(-1, 1) * (X_AE0_k[step][:, 0].reshape(-1, 1) + T) + torch.pow(X_AE0_k[step][:, d-1].reshape(-1, 1), 2) * EX_AE1_k[step][:, 0].reshape(-1, 1))
            
            # Y_AE1_0
            Y_AE1_t_pre = EY_AE1_T
            for num in range(0, step):
                num = step-num-1
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                nakami = 0
                for i in range(0, d-1, 1):
                    nakami += mu_bar_t_k[num][:, i].reshape(-1, 1) * EX_AE1_k[num][:, i].reshape(-1, 1) + mu_hat_t_11_k[num][:, i].reshape(-1, 1) * EX_AE1_k[num][:, i+1].reshape(-1, 1) + mu_hat_t_12_k[num][:, i].reshape(-1, 1)
                nakami += mu_bar_t_k[num][:, d-1].reshape(-1, 1) * EX_AE1_k[num][:, d-1].reshape(-1, 1) + mu_hat_t_11_k[num][:, d-1].reshape(-1, 1) * EX_AE1_k[num][:, 0].reshape(-1, 1) + mu_hat_t_12_k[num][:, d-1].reshape(-1, 1)                
                Y_AE1_t = Y_AE1_t_pre + nakami * dt
                Y_AE1_t_pre = Y_AE1_t
            y_AE1_0 = Y_AE1_t
                
            return Z_AE1_k, y_AE1_0, mu_bar_t_k, mu_hat_t_11_k, mu_hat_t_12_k
        


        X_AE0_k, y_AE0_k, Y_AE0_T = AE0(x=1.0)
        Z_AE1_k, y_AE1_0, mu_bar_t_k, mu_hat_t_11_k, mu_hat_t_12_k = AE1(X_AE0_k, y_AE0_k)

        dt = np.ones(num_batch) * (T / step)        
        X_pre_0 = np.ones([d, num_batch]) * x
        X_AE_1_pre_0 = np.ones([d, num_batch]) * x_ae_1
        
        dt = torch.tensor(np.array([dt]).reshape(-1, 1)).float()                
        X_pre_0 = torch.tensor(np.array([X_pre_0]).reshape(-1, d)).float()
        X_AE_1_pre_0 = torch.tensor(np.array([X_AE_1_pre_0]).reshape(-1, d)).float()
              
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
            
            stack_list = []
            for sl in range(d):
                stack_list.append(np.squeeze([X_pre[:, sl].detach().numpy()], 1))
            input_u = torch.tensor(np.stack(stack_list, axis=1)).float()                
            Y_pre = net_u_list[0](input_u)
            Y_pre = y_AE0_k[0] + epsilon * y_AE1_0 + Y_pre

            Y_AE_1_pre = y_AE1_0
            Y_AE_pre = 0
            
            for num in range(step):
                ww = np.random.randn(d, num_batch)
                ww = torch.tensor(np.array([ww]).reshape(num_batch, d)).float()

                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()

                if num == 0:
                    u = Y_pre
                else:                
                    stack_list = []
                    for sl in range(d):
                        stack_list.append(np.squeeze([X_pre[:, sl].detach().numpy()], 1))
                    input_u = torch.tensor(np.stack(stack_list, axis=1)).float()  
                    u = Y_AE_pre + net_u_list[num](input_u)
                
                stack_list = []
                for sl in range(d):
                    stack_list.append(np.squeeze([X_pre[:, sl].detach().numpy()], 1))
                input_Z = torch.tensor(np.stack(stack_list, axis=1)).float()                
                Z = 0 + epsilon * Z_AE1_k[num] + net_Z_list[num](input_Z)

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
                Y_AE_0 = y_AE0_k[num]
                X_AE_1 = X_AE_1_pre - t * torch.cos(X_AE0_k[num]) * torch.sin(X_AE0_k[num]) * (y_AE0_k[num] + X_AE_1_pre) * dt + sigma(t, X_AE0_k[num], y_AE0_k[num], Z) * ww * np.sqrt(dt)
                Y_AE_1 = Y_AE_1_pre - (y_AE1_1(mu_bar_t_k[num], X_AE_1_pre) + y_AE1_2(mu_hat_t_11_k[num], X_AE_1_pre) + y_AE1_3(mu_hat_t_12_k[num])) * dt + torch.sum(Z_AE1_k[num] * ww, axis = 1).reshape(-1, 1) * np.sqrt(dt)
                
                Y_AE = Y_AE_0 + epsilon * Y_AE_1
                
                if num == 0:
                    if (episode + 1)%100 == 0:
                        print('Y0:', Y_pre.mean().item())
                    self.Y0_list.append(Y_pre.mean().item())

                Y = Y_pre - f(t, X_pre, Y_pre, Z) * dt + torch.sum(Z * ww, axis = 1).reshape(-1, 1) * np.sqrt(dt)
                
                stack_list = []
                for sl in range(d):
                    stack_list.append(np.squeeze([X[:, sl].detach().numpy()], 1))
                input_u_n = torch.tensor(np.stack(stack_list, axis=1)).float()                
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