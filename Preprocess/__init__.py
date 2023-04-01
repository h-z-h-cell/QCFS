from .getdataloader import *

def datapool(DATANAME, batchsize):#在main.py中 调用
    if DATANAME.lower() == 'cifar10':#lower是将大写变小写，这里在判断使用什么数据集
        return GetCifar10(batchsize)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize)
    elif DATANAME.lower() == 'imagenet':
        return GetImageNet(batchsize)
    else:
        print("still not support this model")
        exit(0)