# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

import argparse, time
import math
import os
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import torch  

from lowformer.apps.utils import AverageMeter
from lowformer.cls_model_zoo import create_cls_model
from lowformer.apps import setup

from lowformer.apps.utils import export_onnx
# import multiprocessing
from termcolor import colored
from train_cls_model import mac_it
# from merge_conv_bn import fuse
from other_models.repvit import *


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

from torch.profiler import profile, record_function, ProfilerActivity
def profile_it(args, model):
    args.gpu = 0
    inputs = torch.randn(5, 3, 224, 224).to("cuda:%d" % args.gpu)
    model = model.to("cuda:%d" %args.gpu)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=False, with_stack=True) as prof: # , ProfilerActivity.CUDA
        with record_function("model_inference"):
            model(inputs)
    
    # print(vars(prof.key_averages(group_by_input_shape=True)))
    # print(dir(prof.key_averages(group_by_input_shape=True)))
    # print(repr(prof.key_averages(group_by_input_shape=True)))
    # return
    
    # res = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=50)
    res = prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=50)
    res = prof.key_averages(group_by_stack_n=5)
    
    print(res)
    
    with open("temp_data/profiler.txt", "w") as write_file:
        write_file.write(str(res))


def benchmark_it(args, model, outfile="", bsize=200, iterations=100, cpu=False):
    if args.bench:
        from deepspeed.profiling.flops_profiler import get_model_profile
        from deepspeed.accelerator import get_accelerator
    if cpu:
        model = model.to("cpu")
    else:
        model = model.to("cuda:0")
    
    if len(outfile)<1:
        outfile = "temp_data/"+args.model+"_bs"+str(bsize)+("_cpu" if cpu else "")+"_sz"+str(args.image_size)+".txt"
                
    if args.bench:
        with get_accelerator().device(0):
            print("num devices:",torch.cuda.device_count(), torch.cuda.get_device_name(0))

            
            flops, macs, params, flopmodel = get_model_profile(model=model, # model
                                        input_shape=(bsize, 3, args.image_size, args.image_size), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=True, # prints the model graph with the measured profile attached to each module
                                        detailed=True, # print the detailed profile
                                        iterations=iterations,
                                        module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=15, # the number of warm-ups before measuring the time of each module
                                        as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=outfile, # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling
            print(macs)
            print("Total Macs:","%f MMACS" % (macs/(iterations*bsize*1_000_000)))
            # print(flopmodel)
    if args.bench or args.testrun:
        tim_res = testrun_it_old(model=model, image_size=args.image_size, batch_size=bsize, cpu=cpu, iterations=iterations, optit=args.optit, args=args)
        with open(outfile,"a") as write_file:
            write_file.write("\n\n\n----------------------\n"+str(tim_res)+ " ms")


def other_macit(args, model):
    from deepspeed.profiling.flops_profiler import get_model_profile
    from deepspeed.accelerator import get_accelerator
    with get_accelerator().device(0):  
        flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(1, 3, args.image_size, args.image_size), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=False, # prints the model graph with the measured profile attached to each module
                                    detailed=False, # print the detailed profile
                                    # iterations=1,
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=1, # the number of warm-ups before measuring the time of each module
                                    as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling

        print("Total Macs:","%f MMACS" % (macs/(1_000_000)))

def testrun_it_old(model, image_size, proc=-1, iterations=100, batch_size=300, cpu=False, return_dict=None, optit=False, args=None):
    device = "cpu" if cpu else "cuda:0"
    inp = torch.randn(batch_size, 3, image_size, image_size).to(device)
    if optit:
        model.eval()
        model = torch.jit.script(model)#, example_inputs=[inp])    
        model = torch.jit.optimize_for_inference(model)
    if args.quantit:
        backend = "qnnpack" #"qnnpack"
        model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        model_static_quantized = torch.ao.quantization.prepare(model, inplace=False)
        model_static_quantized = torch.ao.quantization.convert(model_static_quantized, inplace=False)
        model = model_static_quantized
    model.eval()
    model.to(device)


    with torch.inference_mode():
        for i in range(5):
            out = model(inp)
        
        # print("all:",batch_size, iterations, args.image_size)
        timings = []
        for i in range(1):
            inp = torch.randn(batch_size,3,image_size,image_size).to(device)
            model.eval()
            model.to(device)
            for i in range(10):
                out = model(inp)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # print(inp.shape)
            for i in range(iterations):
                if cpu:
                    start_time = time.time()
                else:
                    starter.record()
                out = model(inp)
                if cpu:
                    timings.append(time.time() - start_time)
                else:
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender)/inp.shape[0])
                # print(timings[-1])
            model.to("cpu")
        
    timings = np.array(timings)
    print("backbonetime:", timings[-1],"| median:",np.median(timings), "mean:", np.mean(timings))
    
    if not return_dict is None:
        return_dict[proc] = {"median": np.median(timings), "mean": np.mean(timings)}
    else:
        return np.median(timings)


def testrun_it(proc, args, return_dict=None):
    device = "cpu" if args.cpu else "cuda:0"
    inp = torch.randn((args.batch_size, 3, args.image_size, args.image_size)).to(device).float()

    fac = args.facbs
    inp = torch.randn((args.batch_size*fac**2, 3, args.image_size//fac, args.image_size//fac)).to(device).float()
    
    # inp = torch.ones(args.batch_size, 3, args.image_size, args.image_size).to(device)
    
    # return_dict[proc] = inp.shape
    if args.ds and not args.onnxrun and not args.tensorrtrun:
        model = torch.jit.load(os.path.join("Wtotal_model",args.ds+".pt"))
        modelname = args.ds 
    else:
        model = get_model(args.model, args.other,image_size=args.image_size, args=args).to(device)
        modelname = args.model if len(args.other)<1 else args.other
        
    if args.optit:
        model.eval()
        model = torch.jit.script(model)#, example_inputs=[inp])    
        model = torch.jit.optimize_for_inference(model)
        
    if args.tensorrt or args.tensorrtrun:
        import torch_tensorrt
        modelpath = os.path.join("Wtensorrt",modelname +".ts")
        if args.tensorrtrun:
            # model = torch.export.load(modelpath).module()
            if args.ds:
                modelpath = os.path.join("Wdsrt",args.ds+".ts")
                
            model = torch.jit.load(modelpath).cuda()
        else:
            if proc==0:
                exinps = [torch_tensorrt.Input((1, 3, args.image_size, args.image_size), dtype=torch.float32)]
                otherinps = [torch.randn((1, 3, args.image_size, args.image_size), dtype=torch.float32).cuda()]
                model = torch_tensorrt.compile(model, inputs = otherinps,
                enabled_precisions = {torch.float32}, )
                
                # torch_tensorrt.save(model, modelpath, output_format="torchscript", inputs=otherinps)
                torch_tensorrt.save(model, modelpath, inputs=[torch.randn((1, 3, args.image_size, args.image_size), dtype=torch.float32).cuda()])
            else:
                return
        # torch.jit.save(model,os.path.join("Wtensorrt",modelname +".ts"))
    if args.ds:
        args.image_size = 512

    if args.onnx:
        if proc == 0:
            if not os.path.exists(os.path.join("Wonnxmodels",modelname+".onnx")):
                torch.onnx.export(model, inp, os.path.join("Wonnxmodels",modelname+".onnx"), do_constant_folding=True,opset_version=16, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
                print("ONNX MODEL CREATED SUCCESSFULLY!")
            else:
                print("ALREADY EXISTS!")
            return
        else:
            return
    
    if args.onnxrun  and args.cpu:
        import onnx 
        onnx_model = onnx.load(os.path.join("Wonnxmodels",modelname+"_"+str(args.image_size)+".onnx"))
        
        if False:
            if False: # static quantization!
                from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
                quantize_static(os.path.join("Wonnxmodels",modelname+".onnx"), os.path.join("Wonnxmodels",modelname+"_quant.onnx"), None, quant_format=QuantFormat.QDQ, per_channel=False, weight_type=QuantType.QInt8)
                onnx_model = onnx.load(os.path.join("Wonnxmodels",modelname+"_quant.onnx"))
            else: # dynamic quantization
                if os.path.exists(os.path.join("Wonnxmodels",modelname+"_quant.onnx")):
                    onnx.load(os.path.join("Wonnxmodels",modelname+"_quant.onnx"))
                else:
                    os.system("python -m onnxruntime.quantization.preprocess --input "+ os.path.join("Wonnxmodels",modelname+".onnx") +" --output " + os.path.join("Wonnxmodels",modelname+"_quant.onnx"))
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    onnx_model = onnx.load(os.path.join("Wonnxmodels",modelname+"_quant.onnx")) 
                    onnx_model = quantize_dynamic(onnx_model, os.path.join("Wonnxmodels",modelname+"_quant.onnx"))
                    onnx_model =  onnx.load(os.path.join("Wonnxmodels",modelname+"_quant.onnx"))
        
        onnx.checker.check_model(onnx_model)
        import onnxruntime as ort
        if args.ds: # os.path.join("Wdsonnx",args.ds+".onnx")
            ort_session = ort.InferenceSession(os.path.join("Wdsonnx",args.ds+".onnx"))
        else:
            ort_session = ort.InferenceSession(os.path.join("Wonnxmodels",modelname+"_"+str(args.image_size)+".onnx"))
        inp =  np.random.randn(args.batch_size, 3, args.image_size, args.image_size).astype(np.float32)

        for i in range(10):
            outputs = ort_session.run(None,{"input": inp},)
        
        timings = []
        for i in range(args.iterations):
            start_time = time.time()
            outputs = ort_session.run(None,{"input": inp},)
            timings.append(1000*(time.time()-start_time)/inp.shape[0])
            # print(outputs)
            # print(len(outputs))
            # for l in outputs:
                # print(l.shape)
            # assert False
        timings = np.array(timings)
        med = np.median(timings)
        mean = np.mean(timings)
        return_dict[proc] = {"median": med, "mean": mean}
        return
    elif args.onnxrun :
        import onnx 
        import onnxruntime, psutil
        sess_options = onnxruntime.SessionOptions()
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL


        # maybe remove
        sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
        inp =  np.random.randn(args.batch_size, 3, args.image_size, args.image_size).astype(np.float32)
        if args.ds:
            size_fac = 2 if "shvit" in args.ds or "fat" in args.ds else 1
            out0 = np.random.randn(args.batch_size, 720,64//size_fac, 64//size_fac).astype(np.float32) 
            out1 = np.random.randn(args.batch_size, 720,32//size_fac, 32//size_fac).astype(np.float32)
            out2 = np.random.randn(args.batch_size, 720,16//size_fac, 16//size_fac).astype(np.float32)
            out3 = np.random.randn(args.batch_size, 720,8//size_fac, 8//size_fac).astype(np.float32)
            out4 = np.random.randn(args.batch_size, 720,4//size_fac, 4//size_fac).astype(np.float32)
            out5 = np.random.randn(args.batch_size, 36,64//size_fac, 64//size_fac).astype(np.float32)
            out6 = np.random.randn(args.batch_size, 36,32//size_fac, 32//size_fac).astype(np.float32)
            out7 = np.random.randn(args.batch_size, 36,16//size_fac, 16//size_fac).astype(np.float32)
            out8 = np.random.randn(args.batch_size, 36,8//size_fac, 8//size_fac).astype(np.float32)
            out9 = np.random.randn(args.batch_size, 36, 4//size_fac, 4//size_fac).astype(np.float32)
            if "fat" in args.ds:
                size_fac = 1
                out2 = np.random.randn(args.batch_size, 720,16//size_fac, 16//size_fac).astype(np.float32)
                out3 = np.random.randn(args.batch_size, 720,8//size_fac, 8//size_fac).astype(np.float32)
                out4 = np.random.randn(args.batch_size, 720,4//size_fac, 4//size_fac).astype(np.float32)
                out7 = np.random.randn(args.batch_size, 36,16//size_fac, 16//size_fac).astype(np.float32)
                out8 = np.random.randn(args.batch_size, 36,8//size_fac, 8//size_fac).astype(np.float32)
                out9 = np.random.randn(args.batch_size, 36, 4//size_fac, 4//size_fac).astype(np.float32)
                
            
        else:
            out = np.random.randn(args.batch_size, 1000).astype(np.float32)


        # retinanet output: torch.Size([1, 720, 64, 64]) torch.Size([1, 720, 32, 32]) torch.Size([1, 36, 64, 64]) torch.Size([1,36,32,32])
        if args.ds:
            session = onnxruntime.InferenceSession(os.path.join("Wdsonnx",args.ds+".onnx"), sess_options, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": True})], verbose=True)
        else:
            session = onnxruntime.InferenceSession(os.path.join("Wonnxmodels",modelname+"_"+str(args.image_size)+".onnx"), sess_options, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": True})], verbose=True)
        # options = session.get_provider_options()
        # cuda_options = options[("CUDAExecutionProvider", {"enable_cuda_graph": True})]
        # cuda_options['cudnn_conv_use_max_workspace'] = '1'
        # session.set_providers([("CUDAExecutionProvider", {"enable_cuda_graph": True})], [cuda_options])
        ro = onnxruntime.RunOptions()
        ro.add_run_config_entry("gpu_graph_id", "1")
        io_binding = session.io_binding()

        x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(inp, 'cuda', 0)
        if args.ds:
            total_outs = []
            for i in range(10):
                total_outs.append(onnxruntime.OrtValue.ortvalue_from_numpy(eval("out"+str(i)), 'cuda', 0))
                io_binding.bind_ortvalue_output('output'+str(i), total_outs[-1])
        else:
            y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(out, 'cuda', 0)
            io_binding.bind_ortvalue_output('output', y_ortvalue)
        
        io_binding.bind_ortvalue_input('input', x_ortvalue)
        # io_binding.bind_cpu_input("input", inp)
        # io_binding.bind_output("output", device)


        for i in range(10):
            if False:
                out = session.run(None,{"input":inp})
            else:
                session.run_with_iobinding(io_binding, ro)
                # out = y_ortvalue.numpy()


        timings = []
        for i in range(args.iterations):
            if False:
                start_time = time.time()
                out = session.run(None,{"input":inp})
                timings.append(1000*(time.time()-start_time))
            else:
                start_time = time.time()
                session.run_with_iobinding(io_binding, ro)
                timings.append(1000*(time.time()-start_time)/args.batch_size)
                if args.ds:  
                    ort_outs = total_outs[0].numpy()
                    # print(ort_outs)
                else:
                    ort_outs = y_ortvalue.numpy()
        timings = np.array(timings)
        med = np.median(timings)
        mean = np.mean(timings)
        return_dict[proc] = {"median": med, "mean": mean}
        
        return
    # ort_session = onnxruntime.InferenceSession(os.path.join("Wonnxmodels",args.model), providers=["CUDAExecutionProvider"])
    # # ort_inputs =  {ort_session.get_inputs()[0].name: inp.detach().cpu().numpy()}
    # # ort_outs = ort_session.run(None,ort_inputs)
    
    
    ############################################
    model.eval()
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for i in range(10):
            _ = model(inp)
        
        # print(inp.shape)
        # print(inp.get_device())
        timings = []
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        if True:
            for i in range(args.iterations):
                if args.cpu:
                    start_time = time.time()
                else:
                    starter.record()
                
                for i in range(args.multexec):
                    out = model(inp)
                    # print([out[key].shape for key in out])
                if args.cpu:
                    timings.append(1000*(time.time()-start_time)/(args.multexec*args.batch_size))
                else:
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender)/(args.multexec * args.batch_size))
                # print(timings[-1])
                
            timings = np.array(timings)
            med = np.median(timings)
            mean = np.mean(timings)
            return_dict[proc] = {"median": med, "mean": mean}
        
        else:
            starter.record()
            for i in range(args.iterations):
                out = model(inp)
                
            ender.record()
            torch.cuda.synchronize()
            med = starter.elapsed_time(ender)/(inp.shape[0]*args.iterations)
            mean = med
            return_dict[proc] = {"median": med, "mean": mean}


def get_model(modelname, other="", weight_url="", image_size=224, less_layers=0, args=None):
    config = setup.setup_exp_config("configs/cls/imagenet/"+modelname+".yaml", recursive=True, opt_args=None)
    
    if len(other) > 0:
        from other_models.shvit_temp import shvit_s1, shvit_s2, shvit_s3, shvit_s4
        from timm.models import create_model
        from other_models.efficientModulation import efficientmod
        from other_models.mobilevig import get_mobile_vig
        
        if "fastvit" in other:
            model = create_model(other)
        elif "efficientmod" in other:
            model = efficientmod(other.replace("efficientmod","")) 
        elif "mobilevig" in other:
            model = get_mobile_vig(other.replace("mobilevig",""))
        elif "iformer" in other:
            from other_models.inceptionFormer import get_iformer
            model = get_iformer(other.replace("iformer",""))
        elif "mobileone" in other: 
            from other_models.mobileone_repr import get_mobileone
            model = get_mobileone(other.replace("mobileone",""))
        elif "ffnet" in other:
            from other_models.ffnet import get_ffnet
            model = get_ffnet(other.replace("ffnet",""))
        elif "ghostnet" in other:
            from other_models.ghostnetv2 import get_ghostnet
            model = get_ghostnet(other.replace("ghostnet",""))
        elif "efficientvit" in other:
            from other_models.efficientvitmemory import get_model_effvit
            model = get_model_effvit(other.replace("efficientvit",""), image_size=image_size)
        elif "edgevit" in other:
            from other_models.edgevit import edegvit_model
            model = edegvit_model(other.replace("edgevit",""), image_size)
        elif "pvtv2" in other:
            from other_models.pvtv2 import pvtv2_model
            model = pvtv2_model(other.replace("pvtv2",""))
        elif "FAT" in other:
            from other_models.FAT import get_fat_model
            model = get_fat_model(other.replace("FAT",""))
        elif "mobilenetv3" in other:
            from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
            model = eval("mobilenet_v3_"+other.replace("mobilenetv3","")+"()")
        elif "coatnet" in other:
            from other_models.coatnet import get_coatnet
            model = get_coatnet(other.replace("coatnet",""))
        elif "efficientformerv2" in other:
            from other_models.efficientformerv2 import get_efficientformer
            model = get_efficientformer(other.replace("efficientformerv2",""), image_size)
        elif "resnet" in other:
            from torchvision.models import resnet50, resnet18, resnet101
            model = eval(other+ "()")
        else:
            model = eval(other+"()")
        # model = other
        # print(model)
    else:
        if weight_url == "":
            model = create_cls_model(weight_url="", pretrained=False, less_layers=less_layers, torchscriptsave=args.latency or args.testrun,**config["net_config"])
        else:    
            model = create_cls_model(weight_url=weight_url, pretrained=True, less_layers=less_layers, torchscriptsave=args.latency or args.testrun, **config["net_config"])
    return model 

def run_total(model, device, args, proc_num=0, return_dict=None):
    
    if args.optit:
        model.eval()
        model = model.to(device)
        model = torch.jit.script(model)#, example_inputs=[inp])    
        model = torch.jit.optimize_for_inference(model)
    model.eval()
    
    
    inp = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
    with torch.inference_mode():
        for i in range(10):
            out = model(inp)
        
        if not args.cpu:
            torch.cuda.synchronize()
        start_time = time.time()

        for i in range(args.iterations):
            for i in range(args.multexec):
                out = model(inp)  #[N,1000]
            # torch.cuda.synchronize()    
            inp = inp * out[0,0]
            
        if not args.cpu:
            torch.cuda.synchronize()
        passed_time = (time.time() - start_time) * (1000/(args.iterations * args.batch_size * args.multexec))
    
    if not return_dict is None:
        return_dict[proc_num] = passed_time 
    
    return passed_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("savedir", metavar="FILE", help="config file")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--testrun", action="store_true", default=False)
    parser.add_argument("--latency", action="store_true", default=False)
    parser.add_argument("--optit", action="store_true", default=False)
    parser.add_argument("--ds", type=str, default="")
    parser.add_argument("--onnx", action="store_true", default=False)
    parser.add_argument("--onnxrun", action="store_true", default=False)
    parser.add_argument("--tensorrt", action="store_true", default=False)
    parser.add_argument("--tensorrtrun", action="store_true", default=False)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--other", type=str, default="")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=200)
    parser.add_argument("--path", type=str, default="../../datasets/imagenetsmall/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val")
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--prof", action="store_true", default=False)
    parser.add_argument("--bench", action="store_true", default=False)
    parser.add_argument("--fusebn", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--maconly", action="store_true", default=False)
    parser.add_argument("--quantit", action="store_true", default=False)
    parser.add_argument("--nomac", action="store_true", default=False)
    parser.add_argument("--total_exec", action="store_true", default=False)
    parser.add_argument("--multexec", type=int, default=1)
    parser.add_argument("--facbs", type=int, default=1)
    
    
    args = parser.parse_args()
    # torch.backends.cudnn.benchmark=True
    # if args.gpu == "all":
    #     device_list = range(torch.cuda.device_count())
    #     args.gpu = ",".join(str(_) for _ in device_list)
    # else:
    #     device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.batch_size = args.batch_size #* max(len(device_list), 1)

    args.model = args.savedir
    args.savedir = ".exp/cls/imagenet/" + args.savedir
    if os.path.exists(args.savedir+"/checkpoint/evalmodel.pt"):
        args.weight_url = args.savedir+"/checkpoint/evalmodel.pt"#model_best.pt" # b1-r224.pt"#
    else:
        args.weight_url = args.savedir+"/checkpoint/model_best.pt"
    
    # args.weight_url = args.savedir+"/checkpoint/model_best.pt" # TODO REMOVE
    
    
    # try:
    #     config = setup.setup_exp_config(args.savedir+"/config.yaml", recursive=True, opt_args=None)
    # except:
    
    if not args.testrun:
        data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                args.path,
                transforms.Compose(
                    [
                        transforms.Resize(
                            int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC
                        ),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

    model = get_model(args.model, args.other, args.weight_url, args.image_size, args=args)

    # print(model)
    device = "cpu" if args.cpu else "cuda:0"
    model.eval()
    if args.fusebn:
        model = fuse(model)
    with open("temp_data/modelprint.txt", "w") as write_file:
        write_file.write(str(model))

    if not args.nomac:
        mac_it(model=model,imgsize=args.image_size)
        other_macit(args, model)
    if args.maconly:
        return
    
    if args.total_exec:
        if args.jobs > 1:
            if True: # GOOD WAY
                import torch.multiprocessing as mp
                mp.set_start_method("spawn",force=True)
                manager = mp.Manager()
                return_dict = manager.dict()
                jobs = []
                for i in range(args.jobs): # model, args, proc=-1, iterations=100, batch_size=300, cpu=False
                    p = mp.Process(target=run_total, args=(model, device, args, i,return_dict))#, args=(model, args.image_size, i, args.iterations, args.batch_size, args.cpu))
                    jobs.append(p)
                    p.start()
                    # print("Started",p.name)
                    
                for proc in jobs:
                    proc.join()
                meantime = np.mean([return_dict[key] for key in return_dict])
                print(colored("Mean time needed:","light_green"), meantime) # colored(args.model,"light_green")+" on ImageNet")
                return
            
            else:
                import multiprocessing as mp
                mp.set_start_method("spawn",force=True)
                manager = mp.Manager()
                return_dict = manager.dict()
                jobs = []
                for i in range(args.jobs): # model, args, proc=-1, iterations=100, batch_size=300, cpu=False
                    p = mp.Process(target=run_total, args=(model, device, args, i,return_dict))#, args=(model, args.image_size, i, args.iterations, args.batch_size, args.cpu))
                    jobs.append(p)
                    p.start()
                    # print("Started",p.name)
                    
                for proc in jobs:
                    proc.join()
                meantime = np.mean([return_dict[key] for key in return_dict])
                print(colored("Mean time needed:","light_green"), meantime) # colored(args.model,"light_green")+" on ImageNet")
                return
            
        passed_time = run_total(model, device, args)
        print("Time needed:", passed_time, "bs:", args.batch_size )
        return
    if args.onnxrun:
        modelname = args.model if len(args.other)<1 else args.other
        # model.to(device)
        inp = torch.randn((args.batch_size, 3, args.image_size, args.image_size)).float()#.to(device).float()
        if not os.path.exists( os.path.join("Wonnxmodels",modelname+"_"+str(args.image_size)+".onnx")):
            torch.onnx.export(model, inp, os.path.join("Wonnxmodels",modelname+"_"+str(args.image_size)+".onnx"), do_constant_folding=True,opset_version=16, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    if args.latency: # model, args, iterations=100, batch_size=300, cpu=False

        if args.jobs > 1:
            import multiprocessing as mp
            mp.set_start_method("spawn",force=True)
            
            # args.batch_size = 1
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []  
            for i in range(args.jobs ): # model, args, proc=-1, iterations=100, batch_size=300, cpu=False
                p = mp.Process(target=testrun_it, args=(i, args, return_dict))#, args=(model, args.image_size, i, args.iterations, args.batch_size, args.cpu))
                jobs.append(p)
                p.start()
                # print("Started",p.name)
                
            for proc in jobs:
                proc.join()
            # print("return dict:",return_dict)
            meanmed = np.mean([return_dict[key]["median"] for key in return_dict])
            meanmean = np.mean([return_dict[key]["mean"] for key in return_dict])
            print("Meanmed:", meanmed, "Meanmean:", meanmean, [return_dict[key]["median"] for key in return_dict])
        else:
            ret_dict = {}
            testrun_it(0,args,ret_dict)
            print("Med:",{key:float(ret_dict[0][key]) for key in ret_dict[0]})
        return
        # testrun_it(procnum=procnum, model=model, args=args, iterations=args.iterations, batch_size=args.batch_size, cpu=args.cpu)
    if args.prof:
        profile_it(args, model)
        return 
    if args.bench or args.testrun:
        benchmark_it(args, model, bsize=args.batch_size, cpu=args.cpu, iterations=args.iterations)
        return
    
    ## reset BN TODO
    if False:
        from lowformer.models.nn.norm import reset_bn
        for i in data_loader:
            print(i[0].shape,i[1].shape)
            print(i.shape)
        ###CONTINUE!!!!
        reset_bn(model,data_loader)

    
    if False:
        dummy_input = torch.rand((10, 3, args.image_size,args.image_size))
        export_onnx(model, args.savedir+"/model.onnx", dummy_input, simplify=True, opset=11)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    dummy_input = torch.rand((10, 3, args.image_size,args.image_size))
    dumout = model(dummy_input)


    # Timing
    total_time_list = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc="Eval "+ colored(args.model,"light_green")+" on ImageNet") as t: # light_grey
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()

                starter.record()

                # compute output
                output = model(images)

                bsize = images.shape[0]
                ender.record()
                torch.cuda.synchronize()
                total_time_list.append(starter.elapsed_time(ender)/images.shape[0])

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix(
                    {
                        "top1": top1.avg,
                        "medtim:": np.median(np.array(total_time_list)),
                        "top5": top5.avg,
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)
    print("len total timelist:",len(total_time_list),"bsize:",bsize)
    print("median time needed:",np.median(np.array(total_time_list))," mean:",np.mean(np.array(total_time_list)))
    print(f"Top1 Acc={top1.avg:.3f}, Top5 Acc={top5.avg:.3f}")


if __name__ == "__main__":
    main()

# python eval_cls_model.py b1_testing7_other --gpu 4 --batch_size 1 --image_size 224  --testrun --iterations 4000  --latency


#  python eval_cls_model.py b1_testing10 --gpu 0  --batch_size 20 --image_size 224 --testrun --other custom1 --cpu

