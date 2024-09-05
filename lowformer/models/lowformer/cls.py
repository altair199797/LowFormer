# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

import torch
import torch.ao.quantization
import torch.nn as nn

from lowformer.models.lowformer.backbone import LowFormerBackbone, EfficientViTLargeBackbone
from lowformer.models.nn import ConvLayer, LinearLayer, OpSequential
from lowformer.models.utils import build_kwargs_from_config

__all__ = [
    "LowFormerCls",
    ######################
    "efficientvit_cls_b0",
    "lowformer_cls_b1",
    "efficientvit_cls_b2",
    "efficientvit_cls_b3",
    ######################
    "efficientvit_cls_l1",
    "efficientvit_cls_l2",
    "efficientvit_cls_l3",
]


class ClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final",
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func, squeeze_it=True),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops)

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)


class ClsHeadTorchScript(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final",
    ):
        super().__init__()
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func, squeeze_it=True),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        self.opseq = OpSequential(ops)

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return self.opseq(x)
        # return OpSequential.forward(self, x)



class LowFormerCls(nn.Module):
    def __init__(self, backbone: LowFormerBackbone or EfficientViTLargeBackbone, head: ClsHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        feed_dict = self.backbone(x)
        output = self.head(feed_dict)
        
        self.dequant(output)
        return output



def lowformer_cls_b1(**kwargs) -> LowFormerCls:
    from lowformer.models.lowformer.backbone import lowformer_backbone_b1
    
    
    
    # print("clsb1:",kwargs)
    backbone, width_list = lowformer_backbone_b1(**kwargs)

    
    if "model_mult" in kwargs:
        model_mult = kwargs["model_mult"]
    else:
        model_mult = 1
    
    if "less_layers" in kwargs and kwargs["less_layers"]>0:
        return backbone

    
    widthlist = [1536, 1600]
    act_func = "hswish"
    if "bighead" in kwargs and kwargs["bighead"]:
        widthlist = [2304, 2560]
    if "biggerhead" in kwargs and kwargs["bighead"]:
        widthlist=[3072, 3200]
        act_func="gelu"
    

    if "torchscriptsave" in kwargs and kwargs["torchscriptsave"]:
        head = ClsHeadTorchScript(
                    in_channels=width_list[-1],
                    width_list=widthlist,
                    act_func=act_func,
                    **build_kwargs_from_config(kwargs, ClsHead),
                )
    else:
        head = ClsHead(
            in_channels=width_list[-1],
            width_list=widthlist,
            act_func=act_func,
            **build_kwargs_from_config(kwargs, ClsHead),
        )

    model = LowFormerCls(backbone, head)
    return model


def efficientvit_cls_b0(**kwargs) -> LowFormerCls:
    from lowformer.models.lowformer.backbone import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)

    head = ClsHead(
        in_channels=128,
        width_list=[1024, 1280],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = LowFormerCls(backbone, head)
    return model



# def efficientvit_cls_b1mine(**kwargs) -> LowFormerCls:
#     from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1
#     backbone = efficientvit_backbone_b1(**kwargs)

#     head = ClsHead(
#         in_channels=256,
#         width_list=[1536, 1600],
#         **build_kwargs_from_config(kwargs, ClsHead),
#     )
#     model = LowFormerCls(backbone, head)
#     return model



def efficientvit_cls_b2(**kwargs) -> LowFormerCls:
    from lowformer.models.lowformer.backbone import efficientvit_backbone_b2

    backbone = efficientvit_backbone_b2(**kwargs)

    head = ClsHead(
        in_channels=384,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = LowFormerCls(backbone, head)
    return model


def efficientvit_cls_b3(**kwargs) -> LowFormerCls:
    from lowformer.models.lowformer.backbone import efficientvit_backbone_b3

    backbone = efficientvit_backbone_b3(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = LowFormerCls(backbone, head)
    return model


def efficientvit_cls_l1(**kwargs) -> LowFormerCls:
    from lowformer.models.lowformer.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = LowFormerCls(backbone, head)
    return model


def efficientvit_cls_l2(**kwargs) -> LowFormerCls:
    from lowformer.models.lowformer.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = LowFormerCls(backbone, head)
    return model


def efficientvit_cls_l3(**kwargs) -> LowFormerCls:
    from lowformer.models.lowformer.backbone import efficientvit_backbone_l3

    backbone = efficientvit_backbone_l3(**kwargs)

    head = ClsHead(
        in_channels=1024,
        width_list=[6144, 6400],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = LowFormerCls(backbone, head)
    return model
