import argparse
import copy
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
from ResNet import ResNet18


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomGrayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
#
# transform = torchvision.transforms.Compose(
#     [torchvision.transforms.ToTensor(),
#      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform_train,download=True)
test_data=torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform_test,download=True)
train_len=len(train_data)
test_len=len(test_data)
print(train_len,test_len)


class Client(object):
    def __init__(self, conf, train_dataset, id=1):
        self.conf = conf
        # 配置文件
        self.local_model = ResNet18()
        self.local_model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.local_model.fc = torch.nn.Linear(512, 10)  # 将最后的全连接层改掉
        self.local_model = self.local_model.cuda()
        # 客户端本地模型
        self.client_id = id
        # 客户端ID
        self.train_dataset = train_dataset
        # 客户端本地数据集
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'], weight_decay=5e-4)
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma= 0.7)
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        indices = all_range[id * data_len: (id + 1) * data_len]
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=conf["batch_size"],
                                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def local_train(self):
        lossf = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            lossf = lossf.cuda()
        # 定义最优化函数器，用于本地模型训练
        # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

        # 本地模型训练
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                self.optimizer.zero_grad()
                output = self.local_model(data).cuda()
                loss = lossf(output, target)
                loss.backward()
                self.optimizer.step()
        # self.scheduler.step()
        return self.local_model

    def model_update(self, model):
        self.local_model.load_state_dict(model, strict= True)

    def model_eval(self, eval_loader):
        lossf = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            lossf = lossf.cuda()
        toatal_loss = 0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.local_model(data)
            toatal_loss += lossf(output, target).item()
            accuracy = (output.argmax(1) == target).sum()
            correct += accuracy
        acc =  (float(correct) / float(dataset_size))
        total_l = toatal_loss / dataset_size
        return acc, total_l


with open("settings.json", 'r') as f:
    conf = json.load(f)
clients = []
print(conf["model_name"])
need_update_iter = np.random.uniform(1, 10, conf["no_models"])
#need_update_iter = np.ones(conf["no_models"])
for c in range(conf["no_models"]):
    clients.append(Client(conf, train_data, c))
eval_loader = DataLoader(test_data, batch_size = conf["batch_size"],
                                      shuffle=True)
## record the staleness
already_update_iter = np.zeros_like(need_update_iter)

## record the staleness
staleness_iter = np.zeros_like(need_update_iter)

allclient_acc = np.zeros_like(need_update_iter)

NP = 5
writer = SummaryWriter("./P2PSys-cifar10")
for drop_rate in range(1):
    maxacc = 0
    wait_agg_list = []
    ind_gra = 0
    for ind in range(conf["no_models"]):
        acc, loss = clients[ind].model_eval(eval_loader)
        allclient_acc[ind] = acc

    for e in range (100000):
        print(e, "TIME*************************************")
        already_update_iter = already_update_iter + 1
        for cind in range(conf["no_models"]):
            if already_update_iter[cind] >= need_update_iter[cind] and (cind not in wait_agg_list):
                staleness_iter[cind] = staleness_iter[cind] + 1
                clients[cind].local_train()
                wait_agg_list.append(cind)
        while(len(wait_agg_list) >= NP):
            aggre_staleness = []
            aggre_model = []
            aggre_ind = []
            for i in range(NP):
                cind = wait_agg_list.pop(0)
                aggre_ind.append(cind)
                aggre_staleness.append(staleness_iter[cind])
                aggre_model.append(clients[cind].local_model)
            t_now = min(aggre_staleness)
            t_max = max(aggre_staleness)
            for i in range(NP):
                aggre_staleness[i] = (aggre_staleness[i] - t_now + 1)
            for i in range(NP):
                aggre_staleness[i] = np.exp(aggre_staleness[i])
            sum_aggre = sum(aggre_staleness)
            sum_model = None
            for i in range(NP):
                if sum_model is None:
                    sum_model = aggre_model[i].state_dict()
                    for var in sum_model:
                        sum_model[var] = sum_model[var] * aggre_staleness[i] / sum_aggre
                else:
                    for var in sum_model:
                        sum_model[var] = sum_model[var] + aggre_model[i].state_dict()[var] * aggre_staleness[i] / sum_aggre
            for i in range(NP):
                cind = aggre_ind[i]
                staleness_iter[cind] = t_max
                clients[cind].model_update(sum_model)
                acc, loss = clients[cind].model_eval(clients[cind].train_loader)
                print(cind, "train acc:%f , loss: %f *****" %(acc, loss))
                already_update_iter[cind] = 0
            acc, loss = clients[aggre_ind[0]].model_eval(eval_loader)
            allclient_acc[aggre_ind] = acc
            for i in range(NP):
                writer.add_scalars('100-cifar10-GraNum-2W-ExpAggregation-Resnet18', {'SumAcc': sum(allclient_acc) / conf["no_models"],
                                                                  'MaxAcc': max(allclient_acc),
                                                                  'MinAcc': min(allclient_acc)}, ind_gra)
                ind_gra = ind_gra + 1
            print(aggre_ind, ":acc: %f,1oss:%f " % (acc,loss))
            print("************:sum acc: %f , max acc : %f , min acc : %f  ****************\n" % (sum(allclient_acc) / conf["no_models"], max(allclient_acc), min(allclient_acc)))
        if ind_gra > 40000:
            break
        # writer.add_scalars('100-cifar10-ExpAggregation', {'SumAcc': sum(allclient_acc) / conf["no_models"],
        #                                     'MaxAcc': max(allclient_acc),
        #                                     'MinAcc': min(allclient_acc)}, e)
    print("************:sum acc: %f ****************" % (sum(allclient_acc) / conf["no_models"]))


    #     #模型聚合
    #     acc, loss = server.model_eval ()
    #     writer.add_scalar('GlobalLoss{}'.format(drop_rate), loss, e)
    #     if acc > maxacc:
    #         maxacc = acc
    #     print ( "Global_Epoch: %d，acc: %f,1oss:%f\n" % (e, acc,loss))
    # print("maxAcc:{}".format(maxacc))
    # torch.save(server.global_model, 'server_global_mode200CommNormSGD{}.pt'.format(drop_rate))