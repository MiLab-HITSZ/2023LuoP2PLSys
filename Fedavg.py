import argparse
import copy
import argparse
import json
import os
import random
import numpy as np
import torch
import torchvision
from caffe2.perfkernels.hp_emblookup_codegen import args
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.transforms import ToTensor


transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
train_data = torchvision.datasets.MNIST("./data", train=True, transform=transform,
                                        download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=transform,
                                       download=True)
train_data_size = len(train_data)
test_dataset_size = len(test_data)
print("lenth of the train dataset:{}".format(train_data_size))
print("lenth of the test dataset:{}".format(test_dataset_size))



# class Mymodel(nn.Module):
#     def __init__(self):
#         super(Mymodel, self).__init__()
#         self.model1 = Sequential(
#             Conv2d(1, 16, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(16, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(576, 32),
#             Linear(32, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)
        # print("after conv1: {}".format(x.shape))
        x = F.relu6(x)
        x = self.conv2(x)
        # print("after conv2: {}".format(x.shape))
        x = F.relu6(x)
        x = F.max_pool2d(x, 2)
        # print("after max_pool2d: {}".format(x.shape))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print("after flatten: {}".format(x.shape))
        x = self.fc1(x)
        # print("after fc1: {}".format(x.shape))
        x = F.relu6(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # print("after fc2: {}".format(x.shape))
        output = F.log_softmax(x, dim=1)
        # print("after log_softmax: {}".format(output.shape))
        return output



class Server(object):
    def __init__(self, conf, eval_dataset):
        self.conf = conf
        self.global_model = Mymodel()
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.conf["batch_size"],
                                      shuffle=True)

    def model_aggrate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] / self.conf["no_models"]
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        lossf = nn.CrossEntropyLoss()
        self.global_model.eval()
        toatal_loss = 0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            output = self.global_model(data)
            toatal_loss += lossf(output, target).item()
            accuracy = (output.argmax(1) == target).sum()
            correct += accuracy
        acc =  (float(correct) / float(dataset_size))
        total_l = toatal_loss / dataset_size
        return acc, total_l


class Client(object):
    def __init__(self, conf, model, train_dataset, id=1):
        self.conf = conf
        # 配置文件
        self.local_model = copy.deepcopy(model)
        # 客户端本地模型
        self.client_id = id
        # 客户端ID
        self.train_dataset = train_dataset
        # 客户端本地数据集

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        indices = all_range[id * data_len: (id + 1) * data_len]
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def local_train(self, model):
        lossf = nn.CrossEntropyLoss()
        for name, param in model.state_dict().items():
            # 客户端首先用服务器端下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        # 定义最优化函数器，用于本地模型训练
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        # 本地模型训练
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = lossf(output, target)
                loss.backward()
                optimizer.step()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff

writer = SummaryWriter("./P2PSys-cifar10")

with open("settings.json", 'r') as f:
    conf = json.load(f)
result_acc = []
for glo_round in range(1):

    ind_print = 0
    clients = []
    server = Server(conf, test_data)
    acc, loss = server.model_eval()
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_data, c))
    maxacc=0
    for e in range (500):
        cand_ind = 1
        #采样k个客户端参与本轮联邦训练
        candidates = random.sample(clients, 10)
        drop_clients_num = 0
        trans_totensor = ToTensor()
        weight_accumulator = {}
        for name,params in server.global_model.state_dict ().items():
            weight_accumulator[name] = torch.zeros_like(params)
        for c in candidates:
            if cand_ind > drop_clients_num:
                diff = c.local_train(server.global_model)
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])
                writer.add_scalar('FedAvg-MNIST-100-10-3-v1-local3', acc, ind_print)
                ind_print = ind_print + 1
            cand_ind += 1
        server.model_aggrate(weight_accumulator)
        #模型聚合
        acc, loss = server.model_eval ()
        if acc > maxacc:
            maxacc = acc
        print ( "Global_Epoch: %d，acc: %f,1oss:%f\n" % (e, acc,loss))
        if ind_print > 5000:
            break
    print("maxAcc:{}".format(maxacc))
    result_acc.append(maxacc)
print(result_acc)