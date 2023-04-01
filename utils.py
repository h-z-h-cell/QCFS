import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from modules import TCL, MyFloor, ScaledNeuron, StraightThrough

def isActivation(name):#根据名字判断某一个模块是不是激活函数模块
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False

def replace_activation_by_module(model, m):#替换模块中的激活函数为指定的激活函数m
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):#将子模块中的模型的激活函数递归替换
            model._modules[name] = replace_activation_by_module(module, m)
        if isActivation(module.__class__.__name__.lower()):#如果是激活函数
            if hasattr(module, "up"):
                print(module.up.item())
                model._modules[name] = m(module.up.item())
            else:
                model._modules[name] = m()
    return model

def replace_activation_by_floor(model, t):#替换模块中的激活函数为QFCS中的离散函数，参数为t，t=0改用tcl，用于ANN的训练和测试
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):#这里为什么看up这个选项我还没有想明白，没查到相关属性，不过能够猜到这里是在做什么
                print(module.up.item())
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(module.up.item(), t)
            else:
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(8., t)
    return model

def replace_activation_by_neuron(model):#将模型中ann的激活函数为snn神经元的计算方式，用于SNN输出测试
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_neuron(module)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                model._modules[name] = ScaledNeuron(scale=module.up.item())#ScaledNeron就是SNN神经元
            else:
                model._modules[name] = ScaledNeuron(scale=1.)
    return model

def replace_maxpool2d_by_avgpool2d(model):#将模型中的平均池化层全部替换为最大池化层
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):#将子模型中递归替换
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

def reset_net(model):#将模型中的所以神经元初始化，用于SNN重新初始化
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model

def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()