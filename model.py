import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
from collections import OrderedDict
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict



__all__ = ['DPN', 'dpn92', 'dpn98', 'dpn131', 'dpn107', 'dpns', "SimpleNet",
           'se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           "densenet121", "densenet169", "densenet201", "densenet161"]


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Conv2d
        #     CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10, stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, stride=1)
        self.fc1 = nn.Linear(135424, 1000)
        self.fc2 = nn.Linear(1000, 2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return x

# Shake-shake implementation from https://github.com/owruby/shake-shake_pytorch/blob/master/models/shakeshake.py
class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):

        if training:
            alpha = torch.FloatTensor(x1.size(0)).uniform_().to("cuda:1")
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.FloatTensor(grad_output.size(0)).uniform_().to("cuda:1")
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        # beta = Variable(beta)

        return beta * grad_output, (1 - beta) * grad_output, None

# SENet
# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py

# from torchvision.models import ResNet

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, shake_shake=False, device="cuda:0"):
        super(SEBasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.reduction = reduction
        self.device=device

        self.shake_shake = shake_shake

        # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
        # https://arxiv.org/pdf/1610.02915.pdf
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout2d(p=0.3)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn3 = nn.BatchNorm2d(planes)

        if shake_shake:
            self.branch1 = self._make_branch(inplanes, planes, stride)
            self.branch2 = self._make_branch(inplanes, planes, stride)


    def _make_branch(self, inplanes, planes, stride=1):
        # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
        return nn.Sequential(
                nn.BatchNorm2d(inplanes),
                conv3x3(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=False),
                nn.Dropout2d(p=0.3),
                conv3x3(planes, planes, stride),
                nn.BatchNorm2d(planes),
                SELayer(planes, self.reduction))


    def forward(self, x):

        residual = x

        if not self.shake_shake:

            # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
            out = self.bn1(x)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.se(out)
            #######
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)

        elif self.shake_shake:
            h1 = self.branch1(x)
            h2 = self.branch2(x)
            out = ShakeShake.apply(h1, h2, self.training)
            assert h1.size() == out.size()
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, shake_shake=False):
        super(SEBottleneck, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

        # bn - 1*1conv - bn - relu - 3*3conv - bn - relu - 1*1conv - bn
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * 4)

    def forward(self, x):
        residual = x
        # bn - 1*1conv - bn - relu - 3*3conv - bn - relu - 1*1conv - bn
        # This architecture is proposed in Deep Pyramidal Residual Networks.

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn4(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes, if_mixup=False, if_shake_shake=False, first_conv_stride=2, first_pool=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, mixup_hidden=if_mixup, shake_shake=if_shake_shake,
                   first_conv_stride=first_conv_stride, first_pool=first_pool)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes, if_mixup=False, if_shake_shake=False, first_conv_stride=2, first_pool=True):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, mixup_hidden=if_mixup, shake_shake=if_shake_shake,
                   first_conv_stride=first_conv_stride, first_pool=first_pool)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes, if_mixup=False, if_shake_shake=False, first_conv_stride=2, first_pool=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, mixup_hidden=if_mixup, shake_shake=if_shake_shake,
                   first_conv_stride=first_conv_stride, first_pool=first_pool)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes, if_mixup=False, if_shake_shake=False, first_conv_stride=2, first_pool=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3],num_classes=num_classes, mixup_hidden=if_mixup, shake_shake=if_shake_shake,
                   first_conv_stride=first_conv_stride, first_pool=first_pool)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes, if_mixup=False, if_shake_shake=False, first_conv_stride=2, first_pool=True):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, mixup_hidden=if_mixup, shake_shake=if_shake_shake)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model






class ResNet(nn.Module):
    # This ResNet does Manifold-Mixup.
    # https://arxiv.org/pdf/1806.05236.pdf
    def __init__(self, block, layers, num_classes=2, zero_init_residual=True, mixup_hidden=True, shake_shake=False,
                 first_conv_stride=2, first_pool=True, device="cuda:0"):
        super(ResNet, self).__init__()
        self.mixup_hidden = mixup_hidden
        self.shake_shake = shake_shake
        self.inplanes = 64
        self.num_classes = num_classes
        self.first_pool = first_pool
        self.device=device

        widen_factor = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=first_conv_stride, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)


        self.layer1 = self._make_layer(block, 64*widen_factor, layers[0])
        self.layer2 = self._make_layer(block, 128*widen_factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256*widen_factor, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512*widen_factor, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * widen_factor, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Heの初期化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual and (not shake_shake):
            for m in self.modules():
                if isinstance(m, SEBottleneck):
                    nn.init.constant_(m.bn4.weight, 0)
                elif isinstance(m, SEBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if self.shake_shake:
                layers.append(block(self.inplanes, planes, shake_shake=True))
            else:
                layers.append(block(self.inplanes, planes, shake_shake=False))

        return nn.Sequential(*layers)

    def forward(self, x, lam=None, target=None, device=None):
        def mixup_process(out, target_reweighted, lam):
            # target_reweighted is one-hot vector
            # target is the taerget class.

            # shuffle indices of mini-batch
            indices = np.random.permutation(out.size(0))

            out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
            target_shuffled_onehot = target_reweighted[indices]
            target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))
            return out, target_reweighted

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
            return y_onehot.to(device)

        if self.mixup_hidden:
            layer_mix = np.random.randint(0,3)
        else:
            layer_mix = 0

        out = x

        if lam is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        if lam is not None and self.mixup_hidden and layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        if self.first_pool:
            out = self.maxpool(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        out = self.layer1(out)
        if lam is not None and self.mixup_hidden and layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.layer2(out)
        if lam is not None and self.mixup_hidden and layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)


        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if lam is None:
            return out
        else:
            return out, target_reweighted

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        # nn.sequentilaを継承, forwardする
        new_features = super(_DenseLayer, self).forward(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, if_selayer=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

        # if if_selayer:
        #     self.add_module("selayer", SELayer(num_input_features + growth_rate*num_layers, reduction=16))


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, if_selayer=False):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

        if if_selayer:
            # Squeeze-and-Excitation
            self.add_module("selayer", SELayer(num_output_features))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2,
                 mixup_hidden=False, if_selayer=False,
                 first_conv_stride=2, first_pool=True):

        super(DenseNet, self).__init__()

        self.mixup_hidden = mixup_hidden
        self.num_classes = num_classes
        self.se = if_selayer
        self.first_pool = first_pool

        # First convolution
        if first_pool:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=first_conv_stride, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
                ('norm1', nn.BatchNorm2d(num_init_features)),
                ('relu1', nn.ReLU(inplace=True)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, if_selayer=self.se)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, if_selayer=self.se)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Heの初期化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lam=None, target=None, device=None):
        def mixup_process(out, target_reweighted, lam):
            # target_reweighted is one-hot vector of an original target class
            # target is the taerget class.

            # shuffle indices of mini-batch
            indices = np.random.permutation(out.size(0))

            out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
            target_shuffled_onehot = target_reweighted[indices]
            target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))
            return out, target_reweighted

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
            return y_onehot.to(device)


        # features = self.features(x)

        if self.mixup_hidden:
            layer_mix = np.random.randint(0,4)
        else:
            layer_mix = 0

        if lam is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        out = x

        if lam is not None and self.mixup_hidden and layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        if self.first_pool:
            out = self.features.pool0(self.features.relu0(self.features.norm0(self.features.conv0(out))))
        else:
            out = self.features.relu0(self.features.norm0(self.features.conv0(out)))
            out = self.features.relu1(self.features.norm1(self.features.conv1(out)))

        out = self.features.denseblock1(out)
        out = self.features.transition1(out)
        if lam is not None and self.mixup_hidden and layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.features.denseblock2(out)
        out = self.features.transition2(out)
        if lam is not None and self.mixup_hidden and layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.features.denseblock3(out)
        out = self.features.transition3(out)
        if lam is not None and self.mixup_hidden and layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.features.denseblock4(out)

        out = self.features.norm5(out)

        out = F.relu(out, inplace=True)

        # out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.classifier(out)

        if lam is not None:
            return out, target_reweighted
        else:
            return out


def densenet121(pretrained=False, if_mixup=False, if_selayer=False, first_conv_stride=2, first_pool=True, drop_rate=0.2):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), mixup_hidden=if_mixup, if_selayer=if_selayer,
                     drop_rate=drop_rate, first_conv_stride=first_conv_stride, first_pool=first_pool)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model




def densenet169(pretrained=False, if_mixup=False, if_selayer=False, first_conv_stride=2, first_pool=True):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), mixup_hidden=if_mixup, if_selayer=if_selayer,
                      drop_rate=0.2, first_conv_stride=first_conv_stride, first_pool=first_pool)

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, if_mixup=False, if_selayer=False, first_conv_stride=2, first_pool=True):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), mixup_hidden=if_mixup,
                     if_selayer=if_selayer,  drop_rate=0.2, first_conv_stride=first_conv_stride, first_pool=first_pool)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, mixup_hidden=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), mixup_hidden=mixup_hidden,
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model






def dpn92(num_classes=2, if_selayer=False, if_mixup=False, first_conv_stride=2, first_pool=True):
    return DPN(num_init_features=64, k_R=96, G=32, k_sec=(3,4,20,3), inc_sec=(16,32,24,128), num_classes=num_classes,
               if_selayer=if_selayer, if_mixup=if_mixup, first_conv_stride=first_conv_stride, first_pool=first_pool)


def dpn98(num_classes=2, if_selayer=False, if_mixup=False, first_conv_stride=2, first_pool=True):
    return DPN(num_init_features=96, k_R=160, G=40, k_sec=(3,6,20,3), inc_sec=(16,32,32,128), num_classes=num_classes,
                if_selayer=if_selayer, if_mixup=if_mixup, first_conv_stride=first_conv_stride, first_pool=first_pool)


def dpn131(num_classes=2, if_selayer=False, if_mixup=False, first_conv_stride=2, first_pool=True):
    return DPN(num_init_features=128, k_R=160, G=40, k_sec=(4,8,28,3), inc_sec=(16,32,32,128), num_classes=num_classes,
                if_selayer=if_selayer, if_mixup=if_mixup, first_conv_stride=first_conv_stride, first_pool=first_pool)


def dpn107(num_classes=2, if_selayer=False, if_mixup=False, first_conv_stride=2, first_pool=True):
    return DPN(num_init_features=128, k_R=200, G=50, k_sec=(4,8,20,3), inc_sec=(20,64,64,128), num_classes=num_classes,
                if_selayer=if_selayer, if_mixup=if_mixup, first_conv_stride=first_conv_stride, first_pool=first_pool)

class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, increase, Groups, _type='normal', if_selayer=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.increase = increase

        if _type is 'proj':
            key_stride = 1
            self.has_proj = True
        if _type is 'down':
            key_stride = 2
            self.has_proj = True
        if _type is 'normal':
            key_stride = 1
            self.has_proj = False

        if self.has_proj:
            self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c+2*increase, kernel_size=1, stride=key_stride)

        if not if_selayer:
            self.layers = nn.Sequential(OrderedDict([
                ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
                ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=Groups)),
                ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+increase, kernel_size=1, stride=1))
            ]))
        else:
            self.layers = nn.Sequential(OrderedDict([
                ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
                ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=Groups)),
                ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+increase, kernel_size=1, stride=1)),
                ('se_layer', SELayer(num_1x1_c+increase))
            ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        return nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(in_chs)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)),
        ]))

    def forward(self, x):
        data_in = torch.cat(x, dim=1) if isinstance(x, list) else x

        if self.has_proj:
            data_o = self.c1x1_w(data_in)
            data_o1 = data_o[:, :self.num_1x1_c, :, :]
            data_o2 = data_o[:, self.num_1x1_c:, :, :]
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)

        summ = data_o1 + out[:, :self.num_1x1_c, :, :]
        dense = torch.cat([data_o2, out[:, self.num_1x1_c:, :, :]], dim=1)

        return [summ, dense]


class DPN(nn.Module):

    def __init__(self, num_init_features=64, k_R=96, G=32,
                 k_sec=(3, 4, 20, 3), inc_sec=(16,32,24,128) #DPN-92
                 , num_classes=2, if_selayer=False, if_mixup=False,
                  first_conv_stride=2, first_pool=True):

        super(DPN, self).__init__()

        self.mixup_hidden=if_mixup
        self.num_classes = num_classes
        self.first_pool = first_pool

        blocks = OrderedDict()

        # conv1
        if first_pool:
            blocks['conv1'] = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=first_conv_stride, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            blocks['conv1'] = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=first_conv_stride, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
            )

        # conv2
        bw = 256
        inc = inc_sec[0]
        R = int((k_R*bw)/256)
        blocks['conv2_1'] = DualPathBlock(num_init_features, R, R, bw, inc, G, 'proj', if_selayer=False)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0]+1):
            if i == k_sec[0]:
                blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            else:
                blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=False)
            in_chs += inc

        # conv3
        bw = 512
        inc = inc_sec[1]
        R = int((k_R*bw)/256)
        blocks['conv3_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down', if_selayer=False)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1]+1):
            if i == k_sec[1]:
                blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            else:
                blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=False)
            in_chs += inc


        # conv4
        bw = 1024
        inc = inc_sec[2]
        R = int((k_R*bw)/256)
        blocks['conv4_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down', if_selayer=False)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2]+1):
            if i == k_sec[2]:
                blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            else:
                blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=False)
            in_chs += inc

        # conv5
        bw = 2048
        inc = inc_sec[3]
        R = int((k_R*bw)/256)
        blocks['conv5_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down', if_selayer=False)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3]+1):
            if i == k_sec[3]:
                blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            else:
                blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=False)
            in_chs += inc

        self.conv2_block = nn.Sequential()
        for i in range(1, k_sec[0]+1):
            self.conv2_block.add_module("conv2_{}".format(i), blocks['conv2_{}'.format(i)])
        self.conv3_block = nn.Sequential()
        for i in range(1, k_sec[1]+1):
            self.conv3_block.add_module("conv3_{}".format(i), blocks['conv3_{}'.format(i)])
        self.conv4_block = nn.Sequential()
        for i in range(1, k_sec[2]+1):
            self.conv4_block.add_module("conv4_{}".format(i), blocks['conv4_{}'.format(i)])
        self.conv5_block = nn.Sequential()
        for i in range(1, k_sec[3]+1):
            self.conv5_block.add_module("conv5_{}".format(i), blocks['conv5_{}'.format(i)])


        self.features = nn.Sequential(blocks)
        self.classifier = nn.Linear(in_chs, num_classes)


    def forward(self, x, lam=None, target=None):
        def mixup_process(out, target_reweighted, lam):
            # target_reweighted is one-hot vector
            # target is the taerget class.
            if isinstance(out, list):
                threshold = out[0].size(1)
                out = torch.cat(out, dim=1)

            # shuffle indices of mini-batch
            indices = np.random.permutation(out.size(0))

            out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
            target_shuffled_onehot = target_reweighted[indices]
            target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))

            if isinstance(out, list):
                out = [out[:, :threshold, :, :], out[:, threshold:, :, :]]

            return out, target_reweighted

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
            return y_onehot.to("cuda:0")


        if lam is None:
            features = torch.cat(self.features(x), dim=1)
            out = F.avg_pool2d(features, kernel_size=7).view(features.size(0), -1)
            out = self.classifier(out)
            return out

        else:

            layer_mix = np.random.randint(0,4)

            if lam is not None:
                target_reweighted = to_one_hot(target, self.num_classes)

            out = x

            if lam is not None and layer_mix == 0:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)


            out = self.features.conv1(out)
            out = self.conv2_block(out)
            if lam is not None and layer_mix == 1:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

            out = self.conv3_block(out)
            if lam is not None and layer_mix == 2:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

            out = self.conv4_block(out)
            if lam is not None and layer_mix == 3:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

            out = self.conv5_block(out)
            features = torch.cat(out, dim=1)

            out = F.avg_pool2d(features, kernel_size=7).view(features.size(0), -1)
            out = self.classifier(out)
            return out, target_reweighted

## WideResNet
# https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
