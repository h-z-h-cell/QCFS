from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from Preprocess.augment import Cutout, CIFAR10Policy

# your own data dir
DIR = {'CIFAR10': 'F:\paperitem\SNN_conversion_QCFS-master\dataset', 'CIFAR100': 'F:\paperitem\SNN_conversion_QCFS-master\dataset', 'ImageNet': 'F:\paperitem\SNN_conversion_QCFS-master\dataset'}

# def GetCifar10(batchsize, attack=False):
#     if attack:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   Cutout(n_holes=1, length=16)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor()])
#     else:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                                   Cutout(n_holes=1, length=8)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#     train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
#     test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
#     train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
#     test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
#     return train_dataloader, test_dataloader

def GetCifar10(batchsize, attack=False):#获取cifar10数据集
    #串联多个图片变换的操作，猜测这些操作是为了使得图片更加随机
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),#在一个随机位置上对图像进行裁剪，32会被处理成(32,32)的大小
                                  transforms.RandomHorizontalFlip(),#随机水平翻转，默认概率0.5
                                  CIFAR10Policy(),#随机选择CIFAR10上最好的25个子策略之一。
                                  transforms.ToTensor(), #将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
                                  # 暂时还不明白这里的三组均值和方差的值是怎么定的
                                  Cutout(n_holes=1, length=16)#从图像中随机屏蔽一个或多个补丁。
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)#利用torchvision.datasets下载cifa10训练数据
    #参数为root: str,train: bool = True,transform: Optional[Callable] = None,target_transform: Optional[Callable] = None,download: bool = False,
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) #利用torchvision.datasets下载cifa10测试数据
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    #num_workers0的话表示数据导入在主进程中进行，其他大于0的数表示通过多个进程来导入数据，可以加快数据导入速度
    # 批训练，把数据变成一小批一小批数据进行训练。DataLoader就是用来包装所使用的数据，每次抛出一批数据
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader




def GetCifar100(batchsize):#获取cifar100数据集，详细注释同cifar10
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader

def GetImageNet(batchsize):#获取imagenet数据集
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    
    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    # 利用sampler确保多个进程里面dataloader只会load到整个数据集的一个特定子集
    # DistributedSampler为每一个子进程划分出一部分数据集，以避免不同进程之间数据重复
    train_dataloader =DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8, sampler=train_sampler, pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler) 
    return train_dataloader, test_dataloader
