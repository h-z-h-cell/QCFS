from .ResNet import *
from .VGG import *

def modelpool(MODELNAME, DATANAME):
    if 'imagenet' in DATANAME.lower():
        num_classes = 1000#imagenet有100类
    elif '100' in DATANAME.lower():
        num_classes = 100#cifar100有100类
    else:
        num_classes = 10#cifar10有10类
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    else:
        print("still not support this model")
        exit(0)