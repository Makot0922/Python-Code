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
        self.l1 = nn.Linear(7, 17)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(17, 17)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(17, 17)
        self.last = nn.Linear(17, 5)
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

class Net_O(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(7, 17)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(17, 17)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(17, 17)
        self.last = nn.Linear(17, 1)
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
        self.l1 = nn.Linear(7, 17)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(17, 17)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(17, 17)
        self.last = nn.Linear(17, 1)
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
        self.expu_list = []
        self.result = 0

        
    def run_path(self):
        
        start = time.time()

        num_episode = 10000
        num_batch = 256
        final_cnt = 0
        num_batch_acc = 60000

        x = 1.0
        T = 0.5
        theta0 = 0.1
        s0 = 1.0
        sigma_2 = 0.21
        theta_bar_2 = 0.0871
        theta_h = 0.01
        k2 = 0.6950
        sigma_h = 0.1
        eta = 5 * 0.001
        gamma = 0.5
        step = 25
        d = 5
        d_O = 1
        h0 = 1
        epsilon = 0.01

        net_u_list = []
        optimizer_u_list = []
        net_Z_H_list = []
        optimizer_Z_H_list = []
        net_Z_O_list = []
        optimizer_Z_O_list = []

        for num in range(step+1):            
            net_u_list.append(Net_u())
            tmp_optimizer_u = optim.Adam(net_u_list[num].parameters(), lr= eta)
            optimizer_u_list.append(tmp_optimizer_u)

        for num in range(step):            
            net_Z_H_list.append(Net())
            tmp_optimizer_Z_H = optim.Adam(net_Z_H_list[num].parameters(), lr= eta)
            optimizer_Z_H_list.append(tmp_optimizer_Z_H)            

        for num in range(step):            
            net_Z_O_list.append(Net_O())
            tmp_optimizer_Z_O = optim.Adam(net_Z_O_list[num].parameters(), lr= eta)
            optimizer_Z_O_list.append(tmp_optimizer_Z_O)            

        
        def b(t, x, z, theta):
            return (1/(1-gamma))*x*torch.sum(theta*(z+theta), axis = 1).reshape(-1, 1)

        def b_AE0(t, x, theta):
            return (1/(1-gamma))*x*torch.sum(theta*theta, axis = 1).reshape(-1, 1)

        def b_AE1(t, x0, x1, theta0, theta1):
            return (1/(1-gamma))*x1*torch.sum(theta0*theta0, axis = 1).reshape(-1, 1) + (1/(1-gamma))*x0*torch.sum(2*theta0*theta1, axis = 1).reshape(-1, 1)

        
        def sigma(t, x, z, theta):
            return (1/(1-gamma))*x*(z+theta)

        def f(t, x, y, zh, zo, theta):
            return ((-1)*((gamma/(2*(gamma-1)))*(torch.pow(zh, 2).sum(axis = 1)+torch.pow(theta, 2).sum(axis = 1)) - 0.5*(torch.pow(zh, 2).sum(axis = 1)+torch.pow(zo, 2).sum(axis = 1)))).reshape(-1, 1)

        def f_AE0(t, theta):
            return ((-1)*((gamma/(2*(gamma-1)))*(torch.pow(theta, 2).sum(axis = 1)))).reshape(-1, 1)

        def f_EAE1(t, theta0, Etheta1):
            return ((-1)*((gamma/(gamma-1))*(theta0*Etheta1).sum(axis=1))).reshape(-1, 1)

        def f_ZAE1(t, theta0, theta_sigma):
            return ((-1)*((gamma/(2*(gamma-1)))*(theta0*theta_sigma).sum(axis=1))).reshape(-1, 1)
        
        def g(x, h):
            return (gamma-1)*torch.log(1 + x/h)

        def AE0(x, theta0, h0, num_batch):
            
            theta_AE0_k = [theta0 for i in range(0, step+1, 1)]
            X_AE0_k = [x for i in range(0, step+1, 1)]
            y_AE0_k = [0 for i in range(0, step+1, 1)]
            
            dt = np.ones(num_batch) * (T / step)        
            X_pre_0 = np.ones([1, num_batch]) * x
            theta_pre_0 = np.ones([d, num_batch]) * theta0
            
            dt = torch.tensor(np.array([dt]).reshape(-1, 1)).float()                
            X_AE0_pre = torch.tensor(np.array([X_pre_0]).reshape(-1, 1)).float()                
            theta_AE0_pre = torch.tensor(np.array([theta_pre_0]).reshape(-1, d)).float()
            H0_pre = h0

            for num in range(step):
                H0 = H0_pre + H0_pre * theta_h * dt
                H0_pre = H0

            for num in range(step):
                theta_AE0 = theta_AE0_pre + k2 * (theta_bar_2 - theta_AE0_pre) * dt
                theta_AE0_k[num] = theta_AE0_pre                
                theta_AE0_pre = theta_AE0
            theta_AE0_k[step] = theta_AE0
                    
            for num in range(step):
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                X_AE0 = X_AE0_pre + b_AE0(t, X_AE0_pre, theta_AE0_k[num]) * dt
                X_AE0_k[num] = X_AE0_pre
                X_AE0_pre = X_AE0
            X_AE0_k[step] = X_AE0
            
            Y_AE0_T = (gamma-1)*torch.log(X_AE0 + H0) - (gamma-1)*torch.log(X_AE0)
            
            Y_AE0_t_pre = Y_AE0_T
            for num in range(0, step):
                num = step-num-1
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                Y_AE0_t = Y_AE0_t_pre + f_AE0(t, theta_AE0_k[num]) * dt
                y_AE0_k[num] = Y_AE0_t
                Y_AE0_t_pre = Y_AE0_t
            y_AE0_k[step] = Y_AE0_T
            
            return X_AE0_k, y_AE0_k, Y_AE0_T, theta_AE0_k, H0

        def AE1(X_AE0_k, y_AE0_k, Y_AE0_T, theta_AE0_k, H0, x, theta0, num_batch):
            
            Etheta_AE1_k = [0 for i in range(0, step+1, 1)]
            EX_AE1_k = [0 for i in range(0, step+1, 1)]
            ZH_AE1_k = [0 for i in range(0, step+1, 1)]
            
            H1 = 0
            dt = np.ones(num_batch) * (T / step)        
            X_pre_0 = np.ones([1, num_batch]) * 0
            theta_pre_0 = np.ones([d, num_batch]) * 0
            
            dt = torch.tensor(np.array([dt]).reshape(-1, 1)).float()                
            X_AE1_pre = torch.tensor(np.array([X_pre_0]).reshape(-1, 1)).float()                
            theta_AE1_pre = torch.tensor(np.array([theta_pre_0]).reshape(-1, d)).float()                
            H1_pre = H1

            for num in range(step):
                H1 = H1_pre + H1_pre * theta_h * dt
                H1_pre = H1


            for num in range(step):
                theta_AE1 = theta_AE1_pre - k2 * theta_AE1_pre * dt
                Etheta_AE1_k[num] = theta_AE1_pre                
                theta_AE1_pre = theta_AE1
            Etheta_AE1_k[step] = theta_AE1
                    
            for num in range(step):
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                X_AE1 = X_AE1_pre + b_AE1(t, X_AE0_k[num], X_AE1_pre, theta_AE0_k[num], Etheta_AE1_k[num]) * dt
                EX_AE1_k[num] = X_AE1_pre
                X_AE1_pre = X_AE1
            EX_AE1_k[step] = X_AE1
            
            EY_AE1_T = (gamma-1) * (X_AE0_k[step]*H1-H0*X_AE1) / (X_AE0_k[step]*(X_AE0_k[step]+H0))
            
            y_AE1_0 = 0

            ZH_AE1_t_pre = (gamma-1) * (-H0*(1/(1-gamma))*X_AE0_k[step]*theta_AE0_k[step]) / (X_AE0_k[step]*(X_AE0_k[step]+H0))
            for num in range(0, step):
                num = step-num-1
                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                ZH_AE1_t = ZH_AE1_t_pre + f_ZAE1(t, theta_AE0_k[num], sigma_2) * dt
                ZH_AE1_k[num] = ZH_AE1_t
                ZH_AE1_t_pre = ZH_AE1_t
            ZH_AE1_k[step] = 0

            ZO_AE1_t = (gamma-1) * (X_AE0_k[step]*sigma_h*H0)
            
            return EX_AE1_k, y_AE1_0, EY_AE1_T, ZH_AE1_k, ZO_AE1_t
        
        X_AE0_k, y_AE0_k, Y_AE0_T, theta_AE0_k, H0 = AE0(x, theta0, h0, num_batch = num_batch)
        EX_AE1_k, y_AE1_0, EY_AE1_T, ZH_AE1_k, ZO_AE1_t = AE1(X_AE0_k, y_AE0_k, Y_AE0_T, theta_AE0_k, H0, x=1.0, theta0 = 0.1, num_batch = num_batch)

        X_AE0_k_acc, y_AE0_k_acc, Y_AE0_T_acc, theta_AE0_k_acc, H0 = AE0(x, theta0, h0, num_batch = num_batch_acc)
        EX_AE1_k_acc, y_AE1_0_acc, EY_AE1_T_acc, ZH_AE1_k_acc, ZO_AE1_t_acc = AE1(X_AE0_k_acc, y_AE0_k_acc, Y_AE0_T_acc, theta_AE0_k_acc, H0, x=1.0, theta0 = 0.1, num_batch = num_batch_acc)

        dt = np.ones(num_batch) * (T / step)        
        X_pre_0 = np.ones(num_batch) * x
        theta_pre_0 = np.ones([d, num_batch]) * theta0
        theta_AE1_pre_0 = np.ones([d, num_batch]) * 0
        h_pre_0 = np.ones([1, num_batch]) * h0
        s_pre_0 = np.ones([d, num_batch]) * s0
        
        dt = torch.tensor(np.array([dt]).reshape(-1, 1)).float()                
        X_pre_0 = torch.tensor(np.array([X_pre_0]).reshape(-1, 1)).float()  
        theta_pre_0 = torch.tensor(np.array([theta_pre_0]).reshape(-1, d)).float() 
        theta_AE1_pre_0 = torch.tensor(np.array([theta_AE1_pre_0]).reshape(-1, d)).float() 
        h_pre_0 = torch.tensor(np.array([h_pre_0]).reshape(-1, d_O)).float() 
        s_pre_0 = torch.tensor(np.array([s_pre_0]).reshape(-1, d)).float() 

        dt_acc = np.ones(num_batch_acc) * (T / step)        
        X_pre_0_acc = np.ones(num_batch_acc) * x
        theta_pre_0_acc = np.ones([d, num_batch_acc]) * theta0
        theta_AE1_pre_0_acc = np.ones([d, num_batch_acc]) * 0
        h_pre_0_acc = np.ones([1, num_batch_acc]) * h0
        s_pre_0_acc = np.ones([d, num_batch_acc]) * s0
        
        dt_acc = torch.tensor(np.array([dt_acc]).reshape(-1, 1)).float()                
        X_pre_0_acc = torch.tensor(np.array([X_pre_0_acc]).reshape(-1, 1)).float()  
        theta_pre_0_acc = torch.tensor(np.array([theta_pre_0_acc]).reshape(-1, d)).float() 
        theta_AE1_pre_0_acc = torch.tensor(np.array([theta_AE1_pre_0_acc]).reshape(-1, d)).float() 
        h_pre_0_acc = torch.tensor(np.array([h_pre_0_acc]).reshape(-1, d_O)).float() 
        s_pre_0_acc = torch.tensor(np.array([s_pre_0_acc]).reshape(-1, d)).float() 

              

        for episode in range(0, num_episode):            
            
            for num in range(step):
                net_Z_H_list[num].train()
                optimizer_Z_H_list[num].zero_grad()

            for num in range(step):
                net_Z_O_list[num].train()
                optimizer_Z_O_list[num].zero_grad()

            for num in range(step+1):
                net_u_list[num].train()
                optimizer_u_list[num].zero_grad()

            L = 0

            X_pre = X_pre_0
            theta_pre = theta_pre_0
            theta_AE1_pre = theta_AE1_pre_0
            h_pre = h_pre_0
            
            stack_list = []
            for sl in range(d):
                stack_list.append(np.squeeze([theta_pre[:, sl].detach().numpy()]))
            stack_list.append(np.squeeze([X_pre.detach().numpy()]))
            stack_list.append(np.squeeze([h_pre.detach().numpy()]))
            input_u = torch.tensor(np.stack(stack_list, axis=1)).float()                
            Y_pre = net_u_list[0](input_u)
            Y_pre = y_AE0_k[0] + epsilon * y_AE1_0 + Y_pre

            Y_AE_1_pre = y_AE1_0
            Y_AE_pre = 0
            
            for num in range(step):
                ww_H = np.random.randn(d, num_batch)
                ww_H = torch.tensor(np.array([ww_H]).reshape(num_batch, d)).float()
                ww_O = np.random.randn(d_O, num_batch)
                ww_O = torch.tensor(np.array([ww_O]).reshape(num_batch, d_O)).float()

                theta = theta_pre + k2 * (theta_bar_2 - theta_pre) * dt + sigma_2 * ww_H * np.sqrt(dt)
                theta_AE1 = theta_AE1_pre - k2 * theta_AE1_pre * dt + sigma_2 * ww_H * np.sqrt(dt)
                h = h_pre + h_pre * theta_h * dt + h_pre * sigma_h * ww_O * np.sqrt(dt)


                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()

                if num == 0:
                    u = Y_pre
                else:                
                    stack_list = []
                    for sl in range(d):
                        stack_list.append(np.squeeze([theta_pre[:, sl].detach().numpy()]))
                    stack_list.append(np.squeeze([X_pre.detach().numpy()]))
                    stack_list.append(np.squeeze([h_pre.detach().numpy()]))
                    input_u = torch.tensor(np.stack(stack_list, axis=1)).float()  
                    u = Y_AE_pre + net_u_list[num](input_u)
                
                stack_list = []
                for sl in range(d):
                    stack_list.append(np.squeeze([theta_pre[:, sl].detach().numpy()]))
                stack_list.append(np.squeeze([X_pre.detach().numpy()]))
                stack_list.append(np.squeeze([h_pre.detach().numpy()]))
                input_Z_H = torch.tensor(np.stack(stack_list, axis=1)).float()                
                Z_H = net_Z_H_list[num](input_Z_H)
                Z_H = 0 + epsilon*ZH_AE1_k[num] + Z_H

                stack_list = []
                for sl in range(d):
                    stack_list.append(np.squeeze([theta_pre[:, sl].detach().numpy()]))
                stack_list.append(np.squeeze([X_pre.detach().numpy()]))
                stack_list.append(np.squeeze([h_pre.detach().numpy()]))
                input_Z_O = torch.tensor(np.stack(stack_list, axis=1)).float()                
                Z_O = net_Z_O_list[num](input_Z_O)
                Z_O = 0 + epsilon*ZO_AE1_t + Z_O
                
                X = X_pre + b(t, X_pre, Z_H, theta_pre) * dt + torch.sum(sigma(t, X_pre, Z_H, theta_pre) * ww_H, axis = 1).reshape(-1, 1) * np.sqrt(dt)                
                
                Y_AE_0 = y_AE0_k[num]
                Y_AE_1 = Y_AE_1_pre - f_EAE1(t, theta_AE0_k[num], theta_AE1_pre) * dt + torch.sum(ZH_AE1_k[num] * ww_H, axis = 1).reshape(-1, 1) * np.sqrt(dt) + torch.sum(ZO_AE1_t * ww_O, axis = 1).reshape(-1, 1) * np.sqrt(dt)
                Y_AE = Y_AE_0 + epsilon * Y_AE_1


                if num == 0:
                    if ((episode + 1)%100 == 0)or(episode == 10):
                        print('Y0:', Y_pre.mean().item())
                    self.Y0_list.append(Y_pre.mean().item())

                Y = Y_pre - f(t, X_pre, Y_pre, Z_H, Z_O, theta_pre) * dt + torch.sum(Z_H * ww_H, axis = 1).reshape(-1, 1) * np.sqrt(dt) + torch.sum(Z_O * ww_O, axis = 1).reshape(-1, 1) * np.sqrt(dt)
                
                stack_list = []
                for sl in range(d):
                    stack_list.append(np.squeeze([theta[:, sl].detach().numpy()]))
                stack_list.append(np.squeeze([X.detach().numpy()]))
                stack_list.append(np.squeeze([h.detach().numpy()]))
                input_u_n = torch.tensor(np.stack(stack_list, axis=1)).float()                
                u_n = Y_AE + net_u_list[num+1](input_u_n)
                L = L + (T/step) * torch.pow(Y - u_n, 2).mean()

                X_pre = X
                Y_pre = Y
                theta_pre = theta
                theta_AE1_pre = theta_AE1
                h_pre = h
                Y_AE_1_pre = Y_AE_1
                Y_AE_pre = Y_AE

            LOSS = torch.pow(Y - g(X, h), 2).mean() / 2 + L / 2            
            self.LOSS_list.append(LOSS.item())

            LOSS.backward(retain_graph=True)

            for num in range(step):
                optimizer_u_list[num].step()
                optimizer_Z_H_list[num].step()
                optimizer_Z_O_list[num].step()


            self.time_list.append(time.time() - start)

            theta_pre = theta_pre_0_acc
            h_pre = h_pre_0_acc
            S_pre = s_pre_0_acc
            X_pre = X_pre_0_acc
            X_pre_acc = X_pre_0_acc
            
            for num in range(step):
                ww_H = np.random.randn(d, num_batch_acc)
                ww_H = torch.tensor(np.array([ww_H]).reshape(num_batch_acc, d)).float()
                ww_O = np.random.randn(d_O, num_batch_acc)
                ww_O = torch.tensor(np.array([ww_O]).reshape(num_batch_acc, d_O)).float()

                theta = theta_pre + k2 * (theta_bar_2 - theta_pre) * dt_acc + sigma_2 * ww_H * np.sqrt(dt_acc)
                h = h_pre + h_pre * theta_h * dt_acc + h_pre * sigma_h * ww_O * np.sqrt(dt_acc)
                S = S_pre + (theta_pre-0.5) * dt_acc + ww_H * np.sqrt(dt_acc)

                t = np.ones(num_batch) * (num * T / step)
                t = torch.tensor(np.array([t]).reshape(-1, 1)).float()
                
                stack_list = []
                for sl in range(d):
                    stack_list.append(np.squeeze([theta_pre[:, sl].detach().numpy()]))
                stack_list.append(np.squeeze([X_pre.detach().numpy()]))
                stack_list.append(np.squeeze([h_pre.detach().numpy()]))
                input_Z_H = torch.tensor(np.stack(stack_list, axis=1)).float()                
                Z_H =  0 + epsilon*ZH_AE1_k_acc[num] + net_Z_H_list[num](input_Z_H)

                pi = 1/(1-gamma)*(Z_H + theta_pre)
                X_acc = X_pre_acc + X_pre_acc * torch.sum(pi * (S - S_pre), axis = 1).reshape(-1, 1)
                X = X_pre + b(t, X_pre, Z_H, theta_pre) * dt_acc + torch.sum(sigma(t, X_pre, Z_H, theta_pre) * ww_H, axis = 1).reshape(-1, 1) * np.sqrt(dt_acc)                
                
                theta_pre = theta
                h_pre = h
                S_pre = S
                X_pre = X
                X_pre_acc = X_acc

            ExpU = 1/gamma * torch.pow(X_acc + h, gamma).mean()
            self.expu_list.append(ExpU.item())

            if ((episode + 1)%100 == 0)or(episode == 10):
                print('ExpU:', ExpU.item())
        
        
if __name__ == '__main__':
    for number in range(1, 11, 1):
        env = Environment()
        env.run_path()
        Y0_list = env.Y0_list
        LOSS_list = env.LOSS_list
        time_list = env.time_list
        expu_list = env.expu_list
    
        Y0_df = pd.DataFrame(Y0_list, columns = ['Y0'])
        LOSS_df = pd.DataFrame(LOSS_list, columns = ['LOSS'])
        time_df = pd.DataFrame(time_list, columns = ['time'])
        expu_df = pd.DataFrame(expu_list, columns = ['ExpU'])

        result = pd.concat([Y0_df, LOSS_df], axis = 1)
        result = pd.concat([result, time_df], axis = 1)
        result = pd.concat([result, expu_df], axis = 1)
