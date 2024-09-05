# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

import argparse
import os, sys, torch, copy

from lowformer.apps import setup
from lowformer.apps.utils import dump_config, parse_unknown_args
from lowformer.cls_model_zoo import create_cls_model
from lowformer.clscore.data_provider import ImageNetDataProvider
from lowformer.clscore.trainer import ClsRunConfig, ClsTrainer
from lowformer.models.nn.drop import apply_drop_func

parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")
parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
parser.add_argument("--gpu", type=str, default=None)  # used in single machine experiments
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--timeit", action="store_true")
parser.add_argument("--evalnow", action="store_true")
parser.add_argument("--cpuspeed", action="store_true")
parser.add_argument("--bench", action="store_true")
parser.add_argument("--imgnetfinetune", action="store_true", default=False)
parser.add_argument("--onlymodel", action="store_true", default=False)


# initialization
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)

parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
parser.add_argument("--save_freq", type=int, default=1)


def main():
    
    # parse args
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # setup gpu and distributed training
    setup.setup_dist_env(args.gpu)
    print("Distributed env set up!")

    # setup path, update args, and save args to path
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # setup random seed
    setup.setup_seed(args.manual_seed, args.resume)

    # setup exp config
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    if not "bsizemult" in config["run_config"].keys():
        config["run_config"]["bsizemult"] = 1
    else:
        config["run_config"]["base_lr"] = config["run_config"]["base_lr"] * config["run_config"]["bsizemult"]
        
    # save exp config
    setup.save_exp_config(config, args.path)

    # setup data provider
    data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=True)

    # setup run config
    run_config = setup.setup_run_config(config, ClsRunConfig)

    # setup model
    model = create_cls_model( pretrained=False,**config["net_config"])#, dropout=config["net_config"]["dropout"]) config["net_config"]["name"]
    apply_drop_func(model.backbone.stages, config["backbone_drop"])

    # save model arch and print MACs
    from eval_cls_model import benchmark_it
    args.image_size = config["run_config"]["eval_image_size"][0]
    if  args.bench:
        try:
            benchmark_it(args, copy.deepcopy(model), outfile=os.path.join(args.path,"model_speed"+("_cpu" if args.cpuspeed else "")+".txt"), cpu=args.cpuspeed)
        except Exception as e:
            print("Benchmark didn't work:",str(e))
    with open(os.path.join(args.path,"model_arch.txt"), "w") as writefile:
        # macstring = mac_it(model,imgsize=config["run_config"]["eval_image_size"][0], timeit=args.timeit)
        writefile.write(str(model)+ "\n \n")
        
    if "no_eval" in config["run_config"]:
        no_eval = config["run_config"]["no_eval"]
    else:
        no_eval = False
        
    # setup trainer
    trainer = ClsTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
        auto_restart_thresh=args.auto_restart_thresh,
        bsizemult=config["run_config"]["bsizemult"],
        no_eval=no_eval,
    )
    # initialization
    setup.init_model(
        trainer.network,
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    # prep for training
    trainer.prep_for_training(run_config, config["ema_decay"], args.fp16)

    # resume
    if args.resume:
        trainer.load_model(model_fname="model_best.pt" if args.evalnow else None, imagenetfinetune=args.imgnetfinetune, only_model=args.onlymodel)
        trainer.data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=True)
    else:
        trainer.sync_model()

    # launch training
    trainer.train(save_freq=args.save_freq, evalnow=args.evalnow)

import time
def mac_it(model, imgsize, timeit=False):
    timstring=""
    if timeit and int(os.environ["LOCAL_RANK"]) == 0:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        bsize = 100
        device = "cuda:1"
        inp = torch.randn((bsize,3,imgsize,imgsize)).to(device)
        tempmodel = model.to(device)
        starter.record()
        for i in range(50):
            x = tempmodel(inp)

        takentime = endtim(starter,ender,bsize*50)
        timstring = "Time needed: "+ str(takentime)

    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, imgsize, imgsize), as_strings=True,
                                    print_per_layer_stat=False, verbose=False)
        printstring = '{:<30}  {:<8}'.format('Computational complexity: ', macs) +" | params: "+ str(params)+ " | imgsize: "+ str(imgsize) + " | "+timstring
        print(printstring)
        # from io import StringIO
        # printres = StringIO()
        # macs, params = get_model_complexity_info(model, (3, imgsize, imgsize), as_strings=True, ost=printres,
        #                             print_per_layer_stat=True, verbose=True)
        # printstring += "\n \n" + printres.getvalue()
        return printstring

def endtim(starter, ender, bsize):
    ender.record()
    torch.cuda.synchronize()
    timed = starter.elapsed_time(ender)/bsize
    return timed

if __name__ == "__main__":
    main()
