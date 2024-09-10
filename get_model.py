
from lowformer.cls_model_zoo import create_cls_model
from lowformer.apps import setup

def get_lowformer(config_path="configs/cls/imagenet/b1.yaml", checkpoint_path=".exp/cls/imagenet/b1/checkpoint/evalmodel.pt"):
    
    config = setup.setup_exp_config(config_path, recursive=True, opt_args=None)
    
    model = create_cls_model(weight_url=checkpoint_path, pretrained=True, less_layers=0, torchscriptsave=False, **config["net_config"])
    
    return model

if __name__ == "__main__":
    get_lowformer()