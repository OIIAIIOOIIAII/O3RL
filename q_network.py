import torch
from torch import nn
from utils import MLP
import json
from math import  log, exp
class Resnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.Linear1 = nn.Linear(151, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.Linear2 = nn.Linear(1024, 1024)
        self.Linear3 = nn.Linear(1024, 151)
        self.Linear5 = nn.Linear(151, 151)
        self.Linear4 = nn.Linear(151, 1)
        with open('30new_guiyi_data.json', 'r') as f:
            self.data_dict = json.load(f)
        with open('action_guiyi_data.json', 'r') as f:
            self.action_data_dict = json.load(f)
        self.sigmoid = nn.Sigmoid()

    # def log_out(self, out, max, min):
    #     log_out = (torch.log(out) - log(min))/(log(max) - log(min))
    #     return log_out
    def norm(self, s, a):
        data_tuple = s.chunk(30, dim = -1)
        fixed_data = []
        for i in range(30):
            fixed_data.append ((data_tuple[i]-self.data_dict['mean'][i])/self.data_dict['std'][i])
        fixed_tuple = tuple(fixed_data)
        fixed_s = torch.cat(fixed_tuple,dim = -1)

        fixed_a = (a-self.action_data_dict['mean'])/self.action_data_dict['std']
        return fixed_s, fixed_a

    def minmax(self, s, a):
        data_tuple = s.chunk(30, dim = -1)
        fixed_data = []
        for i in range(30):
            fixed_data.append((data_tuple[i] - self.data_dict['min'][i])/(self.data_dict['max'][i]-self.data_dict['min'][i]))
        fixed_tuple = tuple(fixed_data)
        fixed_s = torch.cat(fixed_tuple, dim = -1)
        fixed_a = (a - self.action_data_dict['min'])/(self.action_data_dict['max'] - self.action_data_dict['min'])
        return fixed_s, fixed_a

    def log_out(self, out, max, min):
        log_out = out * (log(max) - log(min)) + log(min)
        out = torch.exp(log_out)
        return out

    def forward(self, s, a):
        # identity = torch.cat((s,a), dim = -1)
        # print (identity)
        # s, a = self.norm(s, a)
        s, a = self.norm(s, a)
        input = torch.cat((s, a), dim = -1)
        identity = input
        out = self.Linear1(input)
        print('out', torch.isnan(out).int().sum())
        out = self.tanh(out)
        out = self.Linear2(out)
        out = self.tanh(out)
        out = self.Linear3(out)
        # out += identity
        out = self.tanh(out)
        out = self.Linear5(out)
        out = self.relu(out)
        out = self.Linear4(out)
        out = self.sigmoid(out)
        out = self.log_out(out, 2000000, 300000)
        # out = self.tanh(out)
        # out = 300 + 0.5 * (1000 - 300) * (out + 1)
        # out = 600 * out + 400
        # out = torch.exp(out)
        # print (out)

        return  out


class QMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth):
        super().__init__()
        self.net = MLP(input_shape=(state_dim + action_dim), output_dim=1,
                        width=width, depth=depth)

    def forward(self, s, a):
        print ('QMLP used')
        x = torch.cat([
            torch.flatten(s, start_dim=1),
            torch.flatten(a, start_dim=1)
        ], axis=-1)
        # print('state+action shape:',x.shape)
        print(self.net(x))
        return self.net(x)

class DoubleQMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth):
        super().__init__()
        self.net1 = MLP(input_shape=(state_dim + action_dim), output_dim=1,
                        width=width, depth=depth)
        self.net2 = MLP(input_shape=(state_dim + action_dim), output_dim=1,
                        width=width, depth=depth)

    def forward(self, s, a):
        x = torch.cat([
            torch.flatten(s, start_dim=1),
            torch.flatten(a, start_dim=1)
        ], axis=-1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2
