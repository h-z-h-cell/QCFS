import torch.multiprocessing as mp
import argparse
from Models import modelpool
from Preprocess import datapool
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
from ImageNet.train import main_worker
import torch.nn as nn
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('action', default='train', type=str, help='Action: train or test.')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=128, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate') 
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=120, type=int, help='Training epochs') # better if set to 300 for CIFAR dataset
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--l', default=16, type=int, help='L')
    parser.add_argument('--t', default=16, type=int, help='T')
    parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='vgg16')
    args = parser.parse_args()

    #python main.py train --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --l=QUANTIZATION_STEP
    #python main.py test --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --mode={ann, snn} --t=SIMULATION_TIME
    #例如：python main.py test --bs=16 --model=vgg16 --data=cifar10 --id=model1 --mode=snn --t=8
    seed_all()#设置随机数种子方便重复实验

    # only ImageNet using multiprocessing,
    if args.gpus > 1:
        if args.data.lower() != 'imagenet':
            AssertionError('Only ImageNet using multiprocessing.')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
        #mainwork参数形式为(i, *args)形式,i是进程编号，nprocs是要生成的进程数,args为传递给main_worker的参数
    else:
        # preparing data
        train, test = datapool(args.data, args.bs)
        # preparing model
        model = modelpool(args.model, args.data)#建立模型
        model = replace_maxpool2d_by_avgpool2d(model)#使用最大池化
        model = replace_activation_by_floor(model, t=args.l)#使用QCFS函数
        criterion = nn.CrossEntropyLoss()#设置交叉熵函数
        if args.action == 'train':
            train_ann(train, test, model, args.epochs, args.device, criterion, args.lr, args.wd, args.id)#训练
        elif args.action == 'test' or args.action == 'evaluate':
            model.load_state_dict(torch.load('./saved_models/' + args.id + '.pth'))#载入已经训练好的模型
            if args.mode == 'snn':
                model = replace_activation_by_neuron(model)#替换脉冲神经元
                model.to(args.device)#将参数和buffers转换为指定的数据类型或转换到指定的设备上
                acc = eval_snn(test, model, args.device, args.t)#执行SNN
                print('Accuracy: ', acc)
            elif args.mode == 'ann':
                model.to(args.device)#将参数和buffers转换为指定的数据类型或转换到指定的设备上
                acc, _ = eval_ann(test, model, criterion, args.device)#执行ANN
                print('Accuracy: {:.4f}'.format(acc))
            else:
                AssertionError('Unrecognized mode')
