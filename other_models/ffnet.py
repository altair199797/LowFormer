import torch
import torch.nn as nn
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg, _update_default_kwargs
import numpy as np
from pathlib import Path

def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 256, 256),
            'pool_size': None,
            'crop_pct': 0.95,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'ffnet_1': _cfg(url='',
                             crop_pct=0.9,
                             input_size=(3, 256, 256),
                             crop_mode='center'),
    'ffnet_2': _cfg(url='',
                             crop_pct=0.9,
                             input_size=(3, 256, 256),
                             crop_mode='center'),
    'ffnet_3': _cfg(url='',
                             crop_pct=0.95,
                             input_size=(3, 256, 256),
                             crop_mode='center'),
    'ffnet_4': _cfg(url='',
                             crop_pct=0.95,
                             input_size=(3, 256, 256),
                             crop_mode='center'),                                                           
}


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.in_channels = a
        self.groups = groups
        self.kernel_size = ks
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Conv2d(torch.nn.Sequential):   # only for 1x1 conv with bias = False
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.in_channels = a
        self.groups = groups
        self.kernel_size = ks
        self.add_module('bn', torch.nn.BatchNorm2d(a))
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, c = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[None, :, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        b = b @ c.weight.squeeze(-1).squeeze(-1).T
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepVGGDW(torch.nn.Module):
    def __init__(self, in_dim, out_dim, ks = 7, stride = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ks = ks
        self.stride = stride
        
        ks_ = ks // 2
        self.conv = Conv2d_BN(in_dim, out_dim, ks, stride, ks // 2, groups = in_dim)
        self.conv1 = Conv2d_BN(in_dim, out_dim, ks_, stride, ks_ // 2, groups = in_dim)
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [2,2,2,2])

        final_conv_w = conv_w + conv1_w 
        final_conv_b = conv_b + conv1_b
        
        fin_conv = nn.Conv2d(self.in_dim, self.out_dim, self.ks, stride = self.stride,
                             padding = self.ks // 2, groups = self.in_dim, bias = True)

        fin_conv.weight.data.copy_(final_conv_w)
        fin_conv.bias.data.copy_(final_conv_b)
        self.__delattr__('conv')
        self.__delattr__('conv1')
        return fin_conv


class ResidualBlock(torch.nn.Module):
    def __init__(self, main, dim, drop_path = 0., use_layer_scale = True, 
            layer_scale_init_value = 1e-5, post_act = None, pre_norm = None):
        super().__init__()
        self.main = main
        self.dim = dim
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad = True
            )
        self.layer_scale = torch.ones((dim, 1, 1))
        self.pre_norm = pre_norm
        self.post_act = post_act

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_main(self, x):
        if self.use_layer_scale:
            if self.pre_norm is None:
                return self.main(x) * self.layer_scale
            else:
                return self.main(self.pre_norm(x)) * self.layer_scale
        else:
            if self.pre_norm is None:
                return self.main(x)
            else:
                return self.main(self.pre_norm(x))

    def forward(self, x):
        if self.main is None:
            res = x
        else:
            res = self.drop_path(self.forward_main(x)) + x
            if not self.post_act is None:
                res = self.post_act(res)
        return res


class Token_Mixer(torch.nn.Module):
    def __init__(self, dim, ks1, ks2, use_gelu = True):
        super().__init__()
        
        self.fc = Conv2d_BN(dim, dim)
        if ks1 == 7:
            self.conv1 = RepVGGDW(dim, dim, ks1)
        else:
            self.conv1 = nn.Conv2d(dim, dim, ks1, 1, ks1 // 2, groups = dim)
        self.act = torch.nn.GELU() if use_gelu else torch.nn.ReLU()
        if ks2 == 7:
            self.conv2 = RepVGGDW(dim, dim, ks2)
        else:
            self.conv2 = nn.Conv2d(dim, dim, ks2, 1, ks2 // 2, groups = dim)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(self.fc(x))))


class Channel_Mixer(torch.nn.Module):
    def __init__(self, dim, ks, expansion_ratio, use_gelu = True):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        if ks == 7:
            self.conv = RepVGGDW(dim, dim, ks)
        else:
            self.conv = Conv2d_BN(dim, dim, ks, 1, ks // 2, groups = dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = torch.nn.GELU() if use_gelu else torch.nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.conv(x))))


class FFBlock(torch.nn.Module):
    def __init__(self, dim, kernel_size, use_se, use_gelu, expansion_ratio, ks1, ks2, 
                 drop_path = 0., use_layer_scale = True, layer_scale_init_value = 1e-5):
        super(FFBlock, self).__init__()

        self.token_mixer = torch.nn.Sequential(
            ResidualBlock(Token_Mixer(dim, ks1, ks2, use_gelu),
             dim, drop_path, use_layer_scale, layer_scale_init_value),
            SqueezeExcite(dim, 0.25) if use_se else nn.Identity(),
        ) 

        self.channel_mixer = ResidualBlock(Channel_Mixer(dim, kernel_size, expansion_ratio, use_gelu),
            dim, drop_path, use_layer_scale, layer_scale_init_value)

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class Conv2d_BN_skip(torch.nn.Sequential):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = BN_Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        return self.conv(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias

        identity = torch.eye(self.dim, device = conv_w.device)
        identity = identity.unsqueeze(-1).unsqueeze(-1)

        final_conv_w = conv_w + identity
        final_conv_b = conv_b 

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        layers = []
        
        layers.append(RepVGGDW(dim, out_dim, 7, 2))
        layers.append(Conv2d_BN_skip(out_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


class FFNet(torch.nn.Module):
    def __init__(self,
                 in_chans = 3,
                 num_classes = 1000,
                 embed_dim=[64, 128, 320, 512],
                 in_dim = 64,
                 depth=[1, 2, 14, 2],
                 in_channels = 3,
                 kernel_sizes = [7, 7, 7, 7],
                 kernel_sizes1 = [3, 3, 7, 7],
                 kernel_sizes2 = [3, 3, 7, 7],
                 use_gelus = [True, True, True, True],
                 use_ses = [False, False, False, False],
                 expansion_ratios = [3, 3, 3, 3],
                 down_ops=[[''], ['subsample', 2], ['subsample', 2], ['subsample', 2]],
                 use_layer_scale = False,
                 layer_scale_init_value = 1e-5,
                 drop_path_rate = 0.2, 
                 layer_norm_last = False,
                 distillation=False,
                 **kwargs):
        super().__init__()
        num_features = embed_dim[-1]
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        # Patch embedding
        self.patch_embed = nn.Sequential(
            Conv2d_BN(3, in_dim, 3, 2, 1), torch.nn.GELU(),
            Conv2d_BN(in_dim, embed_dim[0], 3, 2, 1)
        )

        # Build FFNet blocks
        self.stages = nn.ModuleList()
        cur = 0
        for i, (ed, dpth, ks, ks1, ks2, do, ug, use, er) in enumerate(
                zip(embed_dim, depth, kernel_sizes, kernel_sizes1, kernel_sizes2, down_ops, use_gelus, use_ses, expansion_ratios)):
            stage = []
            if do[0] == 'subsample':
                # Build FFNet downsample block
                stage.append(PatchMerging(*embed_dim[i - 1:i + 1]))
            for d in range(dpth):
                if d % 2 == 0:
                    stage.append(FFBlock(ed, ks, use, ug, er, ks1, ks2, dpr[cur + d], use_layer_scale, layer_scale_init_value))
                else:
                    stage.append(FFBlock(ed, ks, False, ug, er, ks1, ks2, dpr[cur + d], use_layer_scale, layer_scale_init_value))
            
            self.stages.append(nn.Sequential(*stage))
            cur += depth[i]
        
        self.norm = LayerNorm2d(num_features) if layer_norm_last else nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02) 
            if  m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return x
    
    def forward_head(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)



from timm.models import register_model


def ffnet_1(pretrained=False, **kwargs): 
    depth = kwargs.pop("depth", [2, 2, 8, 2])
    embed_dim = kwargs.pop("embed_dim", [80, 160, 320, 640])
    in_dim = kwargs.pop("in_dim", 64)
    kernel_sizes = kwargs.pop("kernel_sizes", [3, 3, 3, 3])
    kernel_sizes1 = kwargs.pop("kernel_sizes1", [3, 3, 7, 7])
    kernel_sizes2 = kwargs.pop("kernel_sizes2", [3, 3, 7, 7])
    expansion_ratios = kwargs.pop("expansion_ratios", [3, 3, 3, 3])
    drop_path_rate = kwargs.pop("drop_path_rate", 0.1)
    pretrained_cfg = resolve_pretrained_cfg('ffnet_1').to_dict()
    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = FFNet(depth=depth,
                      embed_dim=embed_dim,
                      in_dim=in_dim,
                      kernel_sizes = kernel_sizes,
                      kernel_sizes1 = kernel_sizes1,
                      kernel_sizes2 = kernel_sizes2,
                      expansion_ratios=expansion_ratios,
                      drop_path_rate=drop_path_rate,
                      **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


def ffnet_2(pretrained=False, **kwargs): 
    depth = kwargs.pop("depth", [3, 3, 15, 3])
    embed_dim = kwargs.pop("embed_dim", [88, 176, 352, 704])
    in_dim = kwargs.pop("in_dim", 64)
    kernel_sizes = kwargs.pop("kernel_sizes", [7, 7, 7, 7])
    kernel_sizes1 = kwargs.pop("kernel_sizes1", [3, 3, 7, 7])
    kernel_sizes2 = kwargs.pop("kernel_sizes2", [3, 3, 7, 7])
    expansion_ratios = kwargs.pop("expansion_ratios", [3, 3, 3, 3])
    drop_path_rate = kwargs.pop("drop_path_rate", 0.15)
    pretrained_cfg = resolve_pretrained_cfg('ffnet_2').to_dict()
    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = FFNet(depth=depth,
                      embed_dim=embed_dim,
                      in_dim=in_dim,
                      kernel_sizes = kernel_sizes,
                      kernel_sizes1 = kernel_sizes1,
                      kernel_sizes2 = kernel_sizes2,
                      expansion_ratios=expansion_ratios,
                      drop_path_rate=drop_path_rate,
                      **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


def ffnet_3(pretrained=False, **kwargs): 
    depth = kwargs.pop("depth", [4, 4, 22, 5])
    embed_dim = kwargs.pop("embed_dim", [96, 192, 384, 768])
    in_dim = kwargs.pop("in_dim", 64)
    kernel_sizes = kwargs.pop("kernel_sizes", [7, 7, 7, 7])
    kernel_sizes1 = kwargs.pop("kernel_sizes1", [3, 3, 7, 7])
    kernel_sizes2 = kwargs.pop("kernel_sizes2", [3, 3, 7, 7])
    expansion_ratios = kwargs.pop("expansion_ratios", [3, 3, 3, 3])
    use_layer_scale = kwargs.pop("use_layer_scale", True)
    layer_scale_init_value = kwargs.pop("layer_scale_init_value", 1e-6)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.35)
    pretrained_cfg = resolve_pretrained_cfg('ffnet_3').to_dict()
    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = FFNet(depth=depth,
                      embed_dim=embed_dim,
                      in_dim=in_dim,
                      kernel_sizes = kernel_sizes,
                      kernel_sizes1 = kernel_sizes1,
                      kernel_sizes2 = kernel_sizes2,
                      expansion_ratios=expansion_ratios,
                      use_layer_scale = use_layer_scale,
                      layer_scale_init_value = layer_scale_init_value,
                      drop_path_rate=drop_path_rate,
                      **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


def ffnet_4(pretrained=False, **kwargs): 
    depth = kwargs.pop("depth", [4, 4, 27, 3])
    embed_dim = kwargs.pop("embed_dim", [128, 256, 512, 1024])
    in_dim = kwargs.pop("in_dim", 64)
    kernel_sizes = kwargs.pop("kernel_sizes", [7, 7, 7, 7])
    kernel_sizes1 = kwargs.pop("kernel_sizes1", [3, 3, 7, 7])
    kernel_sizes2 = kwargs.pop("kernel_sizes2", [3, 3, 7, 7])
    expansion_ratios = kwargs.pop("expansion_ratios", [3, 3, 3, 3])
    use_layer_scale = kwargs.pop("use_layer_scale", True)
    layer_scale_init_value = kwargs.pop("layer_scale_init_value", 1e-6)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.4)
    pretrained_cfg = resolve_pretrained_cfg('ffnet_4').to_dict()
    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = FFNet(depth=depth,
                      embed_dim=embed_dim,
                      in_dim=in_dim,
                      kernel_sizes = kernel_sizes,
                      kernel_sizes1 = kernel_sizes1,
                      kernel_sizes2 = kernel_sizes2,
                      expansion_ratios=expansion_ratios,
                      use_layer_scale = use_layer_scale,
                      layer_scale_init_value = layer_scale_init_value,
                      drop_path_rate=drop_path_rate,
                      **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


def get_ffnet(version):
    model = eval("ffnet_"+str(version)+ "()")
    model.eval()
    return model