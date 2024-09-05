# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
from torch import inf
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super(Conv2d_BN, self).__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False
        ))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                      groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(a))
        self.add_module('l', nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super(PatchMerging, self).__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5], **kwargs):
        super(CascadedGroupAttention, self).__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []

        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(torch.nn.ReLU(inplace=True), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))

        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        # for i, qkv in enumerate(self.qkvs):
        #     if i > 0:  # add the previous output to the input
        #         feat = feat + feats_in[i]
        #     feat = qkv(feat)
        #     q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
        #     q = self.dws[i](q)
        #     q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
        #     attn = (
        #             (q.transpose(-2, -1) @ k) * self.scale
        #             +
        #             (trainingab[i] if self.training else self.ab[i])
        #     )

        #     attn = attn.softmax(dim=-1)  # BNN
        #     feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
        #     feats_out.append(feat)
        
        for i, (qkv,dws) in enumerate(zip(self.qkvs, self.dws)):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
            q = dws(q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
            attn = (
                    (q.transpose(-2, -1) @ k) * self.scale
                    +
                    (trainingab[i] if self.training else self.ab[i])
            )

            attn = attn.softmax(dim=-1)  # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
            feats_out.append(feat)

        
        
        # feat = self.qkvs[0](feat)
        # q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
        # q = self.dws[0](q)
        # q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
        # attn = (
        #         (q.transpose(-2, -1) @ k) * self.scale
        #         +
        #         (trainingab[0] if self.training else self.ab[0])
        # )

        # attn = attn.softmax(dim=-1)  # BNN
        # feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
        # feats_out.append(feat)
        
        # i = 1
        # feat = feat + feats_in[1]
        # feat = self.qkvs[1](feat)
        # q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
        # q = self.dws[1](q)
        # q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
        # attn = (
        #         (q.transpose(-2, -1) @ k) * self.scale
        #         +
        #         (trainingab[1] if self.training else self.ab[1])
        # )

        # attn = attn.softmax(dim=-1)  # BNN
        # feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
        # feats_out.append(feat)
        # feat = feat + feats_in[1]
        
        
        # if self.num_heads>2:
        #     i = 2
        #     feat = feat + feats_in[2]
        #     feat = self.qkvs[2](feat)
        #     q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
        #     q = self.dws[2](q)
        #     q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
        #     attn = (
        #             (q.transpose(-2, -1) @ k) * self.scale
        #             +
        #             (trainingab[2] if self.training else self.ab[2])
        #     )

        #     attn = attn.softmax(dim=-1)  # BNN
        #     feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
        #     feats_out.append(feat)
        #     feat = feat + feats_in[2]
        
        # if self.num_heads>3:
        #     i = 3
        #     feat = feat + feats_in[3]
        #     feat = self.qkvs[3](feat)
        #     q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
        #     q = self.dws[3](q)
        #     q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
        #     attn = (
        #             (q.transpose(-2, -1) @ k) * self.scale
        #             +
        #             (trainingab[3] if self.training else self.ab[3])
        #     )

        #     attn = attn.softmax(dim=-1)  # BNN
        #     feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
        #     feats_out.append(feat)        
        #     feat = feat + feats_in[3]
            
        

        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                           attn_ratio=attn_ratio,
                                           resolution=window_resolution,
                                           kernels=kernels)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution

            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)

            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                                           C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)

        return x


class EfficientViTBlock(nn.Module):
    """ A basic EfficientViT building block.
    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], **kwargs):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar,
                                                       resolution=resolution,
                                                       window_resolution=window_resolution,
                                                       kernels=kernels))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        return x


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,
                 **kwargs):
        super().__init__()
        

        self.distillation = distillation
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        resolution = img_size
        # Patch embedding
        self.patch_embed = nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution),
                                         torch.nn.ReLU(),
                                         Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1,
                                                   resolution=resolution // 2), torch.nn.ReLU(),
                                         Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1,
                                                   resolution=resolution // 4), torch.nn.ReLU(),
                                         Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1,
                                                   resolution=resolution // 8))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(nn.Sequential(
                    Residual(
                        Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i],
                                  resolution=resolution)
                    ),
                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution))
                    )
                )

                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))

                resolution = resolution_

                blk.append(nn.Sequential(
                    Residual(
                        Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1],
                                  resolution=resolution)
                    ),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution))
                    )
                )


        self.blocks1 = nn.Sequential(*self.blocks1)
        self.blocks2 = nn.Sequential(*self.blocks2)
        self.blocks3 = nn.Sequential(*self.blocks3)


        # Classification head
        self.head = BN_Linear(embed_dim[-1], self.num_classes) if self.num_classes > 0 else nn.Identity()

        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], self.num_classes) if self.num_classes > 0 else nn.Identity()


    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}


    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = BN_Linear(self.embed_dim[-1], self.num_classes) if self.num_classes > 0 else nn.Identity()


    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        if self.distillation:
            assert False
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x
    
'''
Build the EfficientViT model family
'''
import torch
import torch.nn as nn
from timm.models import register_model

_checkpoint_url_format = 'https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/{}.pth'


def Replace_BatchNorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, nn.BatchNorm2d):
            setattr(net, child_name, nn.Identity())
        else:
            Replace_BatchNorm(child)


EfficientViT_m0 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [64, 128, 192],
    'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
    'window_size': [7, 7, 7],
    'kernels': [5, 5, 5, 5],
}

EfficientViT_m1 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 144, 192],
    'depth': [1, 2, 3],
    'num_heads': [2, 3, 3],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m2 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 192, 224],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 2],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m3 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 240, 320],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 4],
    'window_size': [7, 7, 7],
    'kernels': [5, 5, 5, 5],
}

EfficientViT_m4 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 256, 384],
    'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m5 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [192, 288, 384],
    'depth': [1, 3, 4],
    'num_heads': [3, 3, 4],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}


@register_model
def EfficientViT_M0(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m0):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m1):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m2):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m3, img_size=224):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m4, img_size=224):
    model_cfg["img_size"] = img_size
    model = EfficientViT(num_classes=num_classes, distillation=distillation,**model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M5(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m5, img_size=224):
    model_cfg["img_size"] = img_size
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


def get_model_effvit(version, image_size):
    command = "EfficientViT_"+ version.upper()+ "(img_size="+str(image_size)+")"
    print(command)
    model = eval(command)
    return model