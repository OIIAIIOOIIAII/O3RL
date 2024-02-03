import json

import torch
from torch import nn
from torch.nn import functional as F
import utils
import torch.distributions as D
import pandas as pd
from math import  log, exp

def soft_clamp(x, low, high):
    x = torch.tanh(x)
    # x = 0 + 0.5 * (10) * (x + 1)
    x = 5 + 0.5*(200 - 5) * (x + 1)

    return x

def mu_soft_clamp(x):
    x = torch.relu(x)
    return x

class GaussMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth, dist_type):
        super().__init__()
        self.net = utils.MLP(input_shape=(state_dim), output_dim=2*action_dim,
                        width=width, depth=depth)
        self.log_std_bounds = (-5., 0.)
        self.mu_bounds = (-1., 1.)
        self.dist_type = dist_type
        self.relu = nn.ReLU(inplace=True)
        with open ('30new_guiyi_data.json','r') as f:
            self.data_dict = json.load(f)

    def norm(self, s):
        data_tuple = s.chunk(15, dim = -1)
        fixed_data = []
        for i in range(15):
            fixed_data.append ((data_tuple[i]-self.data_dict['mean'][i])/self.data_dict['std'][i])
        fixed_tuple = tuple(fixed_data)
        data = torch.cat(fixed_tuple,dim = -1)
        return data

    def forward(self, s, h, c):
        s = self.norm(s)
        identity = s
        # mu, log_std = self.net(s).chunk(2, dim=-1)
        out = self.net(s)
        # out += identity
        # out = self.relu(out)
        mu,log_std = out.chunk(2, dim = -1)
        mu = mu.abs()
        shape = (1,1)
        hidden_state = h
        cell_state = c
        mu = mu_soft_clamp(mu)
        print (mu)

        log_std = soft_clamp(log_std,1,1)
        # log_std = soft_clamp(log_std, *self.log_std_bounds)

        # std = log_std.exp()
        std = log_std

        # print('foward completed')
        # if self.dist_type == 'normal':
        #     dist = D.Normal(mu, std)
        # elif self.dist_type == 'trunc':
        #     dist = utils.TruncatedNormal(mu, std)
        # elif self.dist_type == 'squash':
        #     dist = utils.SquashedNormal(mu, std)
        # else:
        #     raise TypeError("Expected dist_type to be 'normal', 'trunc', or 'squash'")

        mu_l = mu.unsqueeze(dim=2)
        std_l = std.unsqueeze(dim=2)



        output = torch.cat((mu_l,std_l),dim=2)

        return output,hidden_state,cell_state


class Resnet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        input_dim = torch.tensor(input_shape).prod()
        self.Linear1 = nn.Linear(input_shape, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.Linear2 = nn.Linear(1024, 1024)
        self.Linear3 = nn.Linear(1024, 150)
        self.Linear4 = nn.Linear(150, output_shape)
        self.Linear5 = nn.Linear(150, 150)
        with open ('30new_guiyi_data.json','r') as f:
            self.data_dict = json.load(f)
        self.sigmoid = nn.Sigmoid()

    def log_out(self, out, max, min):
        log_out = out * (log(max) - log(min)) + log(min)
        out = torch.exp(log_out)
        return out

    def norm(self, s):
        data_tuple = s.chunk(30, dim = -1)
        fixed_data = []
        for i in range(30):
            fixed_data.append ((data_tuple[i]-self.data_dict['mean'][i])/self.data_dict['std'][i])
        fixed_tuple = tuple(fixed_data)
        data = torch.cat(fixed_tuple,dim = -1)
        return data

    def minmax(self, s):
        data_tuple = s.chunk(30, dim = -1)
        fix_data = []
        for i in range(30):
            fix_data.append((data_tuple[i] - self.data_dict['min'][i])/(self.data_dict['max'][i] - self.data_dict['min'][i]))
        fixed_tuple = tuple(fix_data)
        data = torch.cat(fixed_tuple, dim = -1)
        return data

    def forward(self, s, h, c):
        # with open('check.json','w')as f:
        #     json.dump(ss, f)
        # print(torch.isnan(s).all())
        s = self.norm(s)
        # s = self.minmax(s)
        # print (s.shape)
        identity = s
        out = self.Linear1(s)
        out = self.tanh(out)
        out = self.Linear2(out)
        out = self.tanh(out)
        # out = self.Linear2(out)
        # out = self.tanh(out)
        out = self.Linear3(out)
        out += identity
        out = self.tanh(out)
        out = self.Linear5(out)
        out  = self.relu(out)
        out = self.Linear4(out)
        # print (out)
        # out = self.relu(out)
        mu, std = out.chunk(2, dim = -1)
        # batch_norm = torch.nn.BatchNorm1d(1, device= 'cuda')
        # mu  = batch_norm(mu)
        # print(mu)
        mu = self.sigmoid(mu)
        mu = self.log_out(mu, 10e6, 10000)
        # print (mu)

        hidden_state = h
        cell_state = c

        # print(mu)

        std = soft_clamp(std, 1,1)
        # std = log_std.exp()
        # print('std:',std)

        mu_l = mu.unsqueeze(dim = 2)
        std_l  = std.unsqueeze(dim = 2)
        # print(std_l)
        output = torch.cat((mu_l, std_l), dim = 2)
        # print (output.shape)
        # output = output.squeeze(dim = 3)

        return output, hidden_state, cell_state


class MixedGaussMLP(nn.Module):
    def __init__(self, n_comp, state_dim, action_dim, width, depth):
        super().__init__()
        self.n_comp = n_comp
        self.action_dim = action_dim
        self.net = utils.MLP(input_shape=(state_dim), 
                        output_dim=(2*n_comp)*action_dim + n_comp,
                        width=width, depth=depth)
        self.log_std_bounds = (-5., 0.)
        self.mu_bounds = (-2., 2.)
    
    def forward(self, s):
        s = torch.flatten(s, start_dim=1)
        preds = self.net(s)
        cat = F.softmax(preds[:, :self.n_comp], dim=-1)
        mus = preds[:, self.n_comp: self.n_comp + self.n_comp*self.action_dim]
        log_stds = preds[:, self.n_comp + self.n_comp*self.action_dim:]
        
        # mus is  batch x n_comp x a_dim
        mus = soft_clamp(mus, *self.mu_bounds)
        log_stds = soft_clamp(log_stds, *self.log_std_bounds)
        stds = log_stds.exp().transpose(0,1)

        mus = mus.reshape(-1, self.n_comp, self.action_dim)
        stds = stds.reshape(-1, self.n_comp, self.action_dim)
        
        mix = D.Categorical(cat)
        #comp = D.Independent(utils.SquashedNormal(mus, stds), 1)
        #comp = D.Independent(utils.SquashedNormal(mus, stds), 1)
        comp = D.Independent(D.Normal(mus, stds), 1)
        dist = D.MixtureSameFamily(mix, comp)
        
        return dist

