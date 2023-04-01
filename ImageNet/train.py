import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torch import nn
import torch
from Models import modelpool
from Preprocess.getdataloader import GetImageNet
from funcs import train_ann, seed_all
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
import os

def main_worker(rank, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 主节点的地址和端口，就是用来供给init_method使用的。
    #init_method：指定如何初始化互相通讯的进程，可以选择tcp方式或者共享文档的方式；默认为 env：表示使用读取环境变量的方式进行初始化



    dist.init_process_group(backend='nccl', rank=rank, world_size=gpus)#nccl用于GPU，gloo用于CPU

    #设置当前使用的gpu
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    seed_all()

    #根据总的进程数决定每个进程要读入的数据集数量，GetImageNet会保证给每个进程返回不同的不重复的数据内容
    batchsize = int(args.batchsize / gpus)
    train, test = GetImageNet(batchsize)

    # model preparing
    model = modelpool(args.model)#建立模型
    model = replace_maxpool2d_by_avgpool2d(model)#使用最大池化
    model = replace_activation_by_floor(model, t=args.l)#使用QCFS函数

    criterion = nn.CrossEntropyLoss()#设置交叉熵函数

    model.cuda(device)#将所有模型参数和buffers转移到当前GPU上。
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])#使用DistributedDataParallel进行单机多卡或者多机多卡分布式训练

    train_ann(train, test, model, args.epochs, device, criterion, args.lr, args.wd, args.id, rank, True)#训练
    
    dist.destroy_process_group()