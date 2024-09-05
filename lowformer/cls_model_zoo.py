# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

from torch import inf
from lowformer.models.lowformer import (
    LowFormerCls,
    efficientvit_cls_b0,
    lowformer_cls_b1,
    efficientvit_cls_b2,
    efficientvit_cls_b3,
    efficientvit_cls_l1,
    efficientvit_cls_l2,
    efficientvit_cls_l3,
)
from lowformer.models.nn.norm import set_norm_eps
from lowformer.models.utils import load_state_dict_from_file

__all__ = ["create_cls_model"]



def create_cls_model(name: str, pretrained=True, weight_url: str or None = None,  **kwargs) -> LowFormerCls:
    model_dict = {
        "b0": efficientvit_cls_b0,
        "b1": lowformer_cls_b1,
        "b2": efficientvit_cls_b2,
        "b3": efficientvit_cls_b3,
        #########################
        "l1": efficientvit_cls_l1,
        "l2": efficientvit_cls_l2,
        "l3": efficientvit_cls_l3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)
    if model_id in ["l1", "l2", "l3"]:
        set_norm_eps(model, 1e-7)

    try:
        if pretrained:
            weight_url = weight_url #or REGISTERED_CLS_MODEL.get(name, None)
            if weight_url is None:
                raise ValueError(f"Do not find the pretrained weight of {name}.")
            else:
                weight = load_state_dict_from_file(weight_url)
                model.load_state_dict(weight)
    except Exception as e:
        print("Model weights could not be loaded!!!!!!!!!!!!!!!!!!!",e)
    return model
