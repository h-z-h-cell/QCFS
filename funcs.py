import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from utils import *
from modules import LabelSmoothing
import torch.distributed as dist
import random
import os

def seed_all(seed=42):#设置随机数种子使得实验可重复
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True #使用确定性卷积算法


def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=5e-4, save=None, parallel=False, rank=0):
    model.cuda(device)#将所有模型参数和buffers转移到GPU上。
    para1, para2, para3 = regular_set(model)#
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': wd}, 
                                {'params': para2, 'weight_decay': wd}, 
                                {'params': para3, 'weight_decay': wd}
                                ],
                                lr=lr, 
                                momentum=0.9)#设置梯度下降的学习率和和权值衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)#使得学习率慢慢减小到0
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        length = 0
        model.train()#调用torch.nn.model的train函数设置训练模式
        for img, label in train_dataloader:#训练过程,按照batchsize分组进行
            img = img.cuda(device)#把图片数据迁移到CUDA上
            label = label.cuda(device)#把标签数据迁移到CUDA上
            optimizer.zero_grad()#清空参数，Pytorch的特性是张量的梯度不自动清零，因此每次反向传播后都需要清空梯度
            out = model(img)#计算当前输入在当前网络下的输出
            loss = loss_fn(out, label)#利用传入的交叉熵计算函数算出损失
            loss.backward()#反向传播,会得到一个梯度，利用它进行优化
            optimizer.step()#更新参数
            epoch_loss += loss.item()#计算该轮的总误差，后面是不是没用上?
            length += len(label)#计算目前总样本数
        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)#测试过程，计算准确度
        if parallel:#多进程的时候使用
            dist.all_reduce(tmp_acc)#减少所有机器上的张量数据，以使所有机器都得到最终结果。我的理解是让不同进程间通信现在的最高准确率
        print('Epoch {} -> Val_loss: {}, Acc: {}'.format(epoch, val_loss, tmp_acc), flush=True)
        if rank == 0 and save != None and tmp_acc >= best_acc:#如果要保存，且迭代后新的该模型比现在的优秀则保存；这里为什么要rank==0
            torch.save(model.state_dict(), './saved_models/' + save + '.pth')
        best_acc = max(tmp_acc, best_acc)
        print('best_acc: ', best_acc)
        scheduler.step()#更新学习率和权值衰减
    return best_acc, model

def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0#初始化总误差为0
    tot = torch.tensor(0.).cuda(device)#加载总的正确个数变量进gpu
    tot = torch.tensor(0.).cuda(device)#加载总的正确个数变量进gpu
    model.eval()#设置该模式为评估模式而不是训练模式
    model.cuda(device)#将所有模型参数和buffers转移到GPU上。
    length = 0#初始化已经处理的样本数为0
    with torch.no_grad():#表明当前计算不需要反向传播
        for img, label in test_dataloader:
            img = img.cuda(device)#把图片数据迁移到CUDA上
            label = label.cuda(device)#把标签数据迁移到CUDA上
            out = model(img)#计算当前输入在当前网络下的输出
            loss = loss_fn(out, label)#利用传入的交叉熵计算函数算出损失
            epoch_loss += loss.item()#计算该轮的总误差
            length += len(label)#计算目前总样本数
            tot += (label==out.max(1)[1]).sum().data#.max(axis)[index]axis也可以为1表示每行的最大值。index为1表示只返回最大值对应的索引
    return tot/length, epoch_loss/length#返回总正确率和平均误差

def eval_snn(test_dataloader, model, device, sim_len=8, rank=0):
    tot = torch.zeros(sim_len).cuda(device)#加载模拟时间长度个 记录总的正确个数的变量 进gpu
    length = 0#初始化总误差为0
    model = model.cuda(device)#将所有模型参数和buffers转移到GPU上。
    model.eval()#设置该模式为评估模式而不是训练模式
    # valuate
    with torch.no_grad():#表明当前计算不需要反向传播
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):#tqdm,可以在 Python长循环中添加一个进度提示信息
            spikes = 0#初始化脉冲数
            length += len(label)#计算目前总样本数
            img = img.cuda()#把图片数据迁移到CUDA上
            label = label.cuda()#把标签数据迁移到CUDA上
            for t in range(sim_len):#模拟sim_len个时间长度
                out = model(img)#计算当前输入在当前网络下的输出
                spikes += out#计算目前总脉冲数
                tot[t] += (label==spikes.max(1)[1]).sum()#计算正确个数，最后tot[sum_len]才是我们最终需要的
            reset_net(model)#更新模型,包括膜电位在内的参数
    return tot/length
    #样例输出 Accuracy:  tensor([0.1203, 0.6932, 0.7936, 0.8466, 0.8735, 0.8884, 0.9000, 0.9079],device='cuda:0')