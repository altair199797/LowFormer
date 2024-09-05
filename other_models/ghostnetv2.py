"""
Creates a GhostNetV2 Model as defined in:
GhostNetV2: Enhance Cheap Operation with Long-Range Attention By Yehui Tang, Kai Han, Jianyuan Guo, Chang Xu, Chao Xu, Yunhe Wang.
https://openreview.net/forum?id=vhKaBdOOobB
Modified from MindSpore implementation:
https://github.com/mindspore-ai/models/tree/master/research/cv/ghostnetv2
"""
import math
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ghostnetv2']


def _make_divisible(x, divisor=4):
    return int(np.ceil(x * 1. / divisor) * divisor)


class MyHSigmoid(nn.Module):
    """
    Hard Sigmoid definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> MyHSigmoid()
    """

    def __init__(self):
        super(MyHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        """ forward """
        return self.relu6(x + 3.) * 0.16666667


class Activation(nn.Module):
    """
    Activation definition.

    Args:
        act_func(string): activation name.

    Returns:
         Tensor, output tensor.
    """

    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = MyHSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.Hardswish()
        else:
            raise NotImplementedError

    def forward(self, x):
        """ forward """
        return self.act(x)


class GlobalAvgPooling(nn.Module):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """ forward """
        x = self.mean(x)
        return x


class SE(nn.Module):
    """
    SE warpper definition.

    Args:
        num_out (int): Output channel.
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> SE(4)
    """

    def __init__(self, num_out, ratio=4):
        super(SE, self).__init__()
        num_mid = _make_divisible(num_out // ratio)
        self.pool = GlobalAvgPooling()
        self.conv_reduce = nn.Conv2d(in_channels=num_out, out_channels=num_mid,
                                     kernel_size=1, bias=True, padding_mode='zeros')
        self.act1 = Activation('relu')
        self.conv_expand = nn.Conv2d(in_channels=num_mid, out_channels=num_out,
                                     kernel_size=1, bias=True, padding_mode='zeros')
        self.act2 = Activation('hsigmoid')

    def forward(self, x):
        """ forward of SE module """
        out = self.pool(x)
        out = self.conv_reduce(out)
        out = self.act1(out)
        out = self.conv_expand(out)
        out = self.act2(out)
        out = x * out
        return out


class ConvUnit(nn.Module):
    """
    ConvUnit warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (Union[int, tuple[int]]): Input kernel size.
        stride (int): Stride size.
        padding (Union[int, tuple[int]]): Padding number.
        num_groups (int): Output num group.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvUnit(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=num_groups,
                              bias=False,
                              padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = Activation(act_type) if use_act else nn.Identity()

    def forward(self, x):
        """ forward of conv unit """
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class GhostModule(nn.Module):
    """
    GhostModule warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        ratio (int): Reduction ratio.
        dw_size (int): kernel size of cheap operation.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostModule(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size//2, num_groups=1, use_act=use_act, act_type=act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                        padding=dw_size // 2, num_groups=init_channels,
                                        use_act=use_act, act_type=act_type)

    def forward(self, x):
        """ ghost module forward """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostModuleMul(nn.Module):
    """
    GhostModuleMul warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        ratio (int): Reduction ratio.
        dw_size (int): kernel size of cheap operation.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostModuleMul(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModuleMul, self).__init__()
        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gate_fn = Activation('sigmoid')
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size//2, num_groups=1, use_act=use_act, act_type=act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                        padding=dw_size // 2, num_groups=init_channels,
                                        use_act=use_act, act_type=act_type)
        self.short_conv = nn.Sequential(
            ConvUnit(num_in, num_out, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, num_groups=1, use_act=False),
            ConvUnit(num_out, num_out, kernel_size=(1, 5), stride=1,
                     padding=(0, 2), num_groups=num_out, use_act=False),
            ConvUnit(num_out, num_out, kernel_size=(5, 1), stride=1,
                     padding=(2, 0), num_groups=num_out, use_act=False),
            )

    def forward(self, x):
        """ ghost module forward """
        res = self.avgpool2d(x)
        res = self.short_conv(res)
        res = self.gate_fn(res)

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out * F.interpolate(res, size=out.shape[-2:], mode="bilinear", align_corners=True)
        return out


class GhostBottleneck(nn.Module):
    """
    GhostBottleneck warpper definition.

    Args:
        num_in (int): Input channel.
        num_mid (int): Middle channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        act_type (str): Activation type.
        use_se (bool): Use SE warpper or not.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostBottleneck(16, 3, 1, 1)
    """

    def __init__(self, num_in, num_mid, num_out, kernel_size, stride=1, act_type='relu', use_se=False, layer_id=None):
        super(GhostBottleneck, self).__init__()
        self.use_ori_module = layer_id <= 1
        if self.use_ori_module:
            self.ghost1 = GhostModule(num_in, num_mid, kernel_size=1,
                                      stride=1, padding=0, act_type=act_type)
        else:
            self.ghost1 = GhostModuleMul(num_in, num_mid, kernel_size=1,
                                         stride=1, padding=0, act_type=act_type)

        self.use_dw = stride > 1
        self.dw = nn.Identity()
        if self.use_dw:
            self.dw = ConvUnit(num_mid, num_mid, kernel_size=kernel_size, stride=stride,
                               padding=self._get_pad(kernel_size), act_type=act_type, num_groups=num_mid, use_act=False)

        self.use_se = use_se
        self.se = nn.Identity()
        if use_se:
            self.se = SE(num_mid)

        self.ghost2 = GhostModule(num_mid, num_out, kernel_size=1, stride=1,
                                  padding=0, act_type=act_type, use_act=False)

        self.down_sample = False
        if num_in != num_out or stride != 1:
            self.down_sample = True
        self.shortcut = nn.Identity()
        if self.down_sample:
            self.shortcut = nn.Sequential(
                ConvUnit(num_in, num_in, kernel_size=kernel_size, stride=stride,
                         padding=self._get_pad(kernel_size), num_groups=num_in, use_act=False),
                ConvUnit(num_in, num_out, kernel_size=1, stride=1,
                         padding=0, num_groups=1, use_act=False),
            )

    def forward(self, x):
        """ forward of ghostnet """
        shortcut = x
        out = self.ghost1(x)
        if self.use_dw:
            out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        if self.down_sample:
            shortcut = self.shortcut(shortcut)
        out = shortcut + out
        return out

    def _get_pad(self, kernel_size):
        """set the padding number"""
        pad = 0
        if kernel_size == 1:
            pad = 0
        elif kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        elif kernel_size == 7:
            pad = 3
        else:
            raise NotImplementedError
        return pad


class GhostNet(nn.Module):
    """
    GhostNet architecture.

    Args:
        model_cfgs (Cell): number of classes.
        num_classes (int): Output number classes.
        multiplier (int): Channels multiplier for round to 8/16 and others. Default is 1.
        final_drop (float): Dropout number.
        round_nearest (list): Channel round to . Default is 8.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostNet(num_classes=1000)
    """

    def __init__(self, model_cfgs, num_classes=1000, multiplier=1., final_drop=0.):
        super(GhostNet, self).__init__()
        self.cfgs = model_cfgs['cfg']
        self.inplanes = 16
        first_conv_in_channel = 3
        first_conv_out_channel = _make_divisible(multiplier * self.inplanes)

        self.conv_stem = nn.Conv2d(in_channels=first_conv_in_channel,
                                   out_channels=first_conv_out_channel,
                                   kernel_size=3, padding=1, stride=2,
                                   bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(first_conv_out_channel)
        self.act1 = Activation('relu')
        self.inplanes = first_conv_out_channel
        self.blocks = []
        layer_id = 0
        for layer_cfg in self.cfgs:
            # print(multiplier,_make_divisible(multiplier * layer_cfg[2]),multiplier * layer_cfg[2])
            self.blocks.append(self._make_layer(kernel_size=layer_cfg[0],
                                                exp_ch=_make_divisible(
                                                    multiplier * layer_cfg[1]),
                                                out_channel=_make_divisible(
                                                    multiplier * layer_cfg[2]),
                                                use_se=layer_cfg[3],
                                                act_func=layer_cfg[4],
                                                stride=layer_cfg[5],
                                                layer_id=layer_id))
            layer_id += 1
        output_channel = _make_divisible(
            multiplier * model_cfgs["cls_ch_squeeze"])
        self.blocks.append(ConvUnit(_make_divisible(multiplier * self.cfgs[-1][2]), output_channel,
                                    kernel_size=1, stride=1, padding=0, num_groups=1, use_act=True))
        self.blocks = nn.Sequential(*self.blocks)

        self.global_pool = GlobalAvgPooling()
        self.conv_head = nn.Conv2d(in_channels=output_channel,
                                   out_channels=model_cfgs['cls_ch_expand'],
                                   kernel_size=1, padding=0, stride=1,
                                   bias=True, padding_mode='zeros')
        self.act2 = Activation('relu')
        self.squeeze = nn.Flatten()
        self.final_drop = final_drop
        self.dropout = nn.Identity()
        if self.final_drop > 0:
            self.dropout = nn.Dropout(self.final_drop)

        self.classifier = nn.Linear(
            model_cfgs['cls_ch_expand'], num_classes, bias=True)

        self._initialize_weights()

    def forward(self, x):
        """ forward of GhostNet """
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.squeeze(x)
        if self.final_drop > 0:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, stride=1, layer_id=0):
        mid_planes = exp_ch
        out_planes = out_channel
        layer = GhostBottleneck(self.inplanes, mid_planes, out_planes,
                                kernel_size, stride=stride, act_type=act_func, use_se=use_se, layer_id=layer_id)
        self.inplanes = out_planes
        return layer

    def _initialize_weights(self):
        """
        Initialize weights.

        Args:

        Returns:
            None.

        Examples:
            >>> _initialize_weights()
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, std=np.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def ghostnetv2(model_name, **kwargs):
    """
    Constructs a GhostNet model
    """
    model_cfgs = {
        "1x": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                # stage1
                [3, 16, 16, False, 'relu', 1],
                # stage2
                [3, 48, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                # stage3
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                # stage4
                [3, 240, 80, False, 'relu', 2],
                [3, 200, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 480, 112, True, 'relu', 1],
                [3, 672, 112, True, 'relu', 1],
                # stage5
                [5, 672, 160, True, 'relu', 2],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, True, 'relu', 1],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, True, 'relu', 1]],
            "cls_ch_squeeze": 960,
            "cls_ch_expand": 1280,
        },

        "nose_1x": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                # stage1
                [3, 16, 16, False, 'relu', 1],
                # stage2
                [3, 48, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                # stage3
                [5, 72, 40, False, 'relu', 2],
                [5, 120, 40, False, 'relu', 1],
                # stage4
                [3, 240, 80, False, 'relu', 2],
                [3, 200, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 480, 112, False, 'relu', 1],
                [3, 672, 112, False, 'relu', 1],
                # stage5
                [5, 672, 160, False, 'relu', 2],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, False, 'relu', 1]],
            "cls_ch_squeeze": 960,
            "cls_ch_expand": 1280,
        }
    }

    return GhostNet(model_cfgs[model_name], **kwargs)


def get_ghostnet(version):
    model = ghostnetv2(model_name="1x",multiplier=float(version))
    # print(model)
    return model