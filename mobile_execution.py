# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

import os, sys
from torch.jit import mobile
import torch
from eval_cls_model import get_model
from torch.utils.mobile_optimizer import optimize_for_mobile
import onnx

def main():
    jitit = False
    other = ""
    
    # modelname = "b1_testing2_hugerun2"
    # modelname = "b1_testing9"
    modelname = "b1_testing10"
    # modelname = "b1_testing7_other"
    modelname = "b1_testing8"
    # modelname = "b1"
    
    # other = "shvit_s4"              ##
    # other = "repvit_m0_9"
    # other = "repvit_m1_1"           ##
    # other = "repvit_m1_5"
    # other = "repvit_m2_3"
    # other = "fastvit_t8"
    # other = "fastvit_sa12"
    # other = "fastvit_sa24"
    # other = "fastvit_ma36"
    
    
    # other = "efficientmods"         
    # other = "efficientvitm4"
    # other = "pvtv2b0"
    # other = "mobilenetv3large"
    # other = "resnet50"
    # other = "pvtv2b1"
    # other = "FATb0"
    # other = "shvit_s4"
    # other = "efficientformerv2s0"  ##
    # other = "efficientformerv2s1"
    # other = "efficientformerv2l"
    
    # other = "edgevitxxs"
    # other = "repvit_m1_5"
    # other = "repvit_m2_3"
    # other = "mobilevigti"
    # other = "fastvit_sa24"
    # other = "ghostnet1.6"
    # other = "mobileones4"
    img_size = 224
    # img_size = 256
    img_size = 512
    # img_size = 384
    # img_size = 320
    
    
    mobile_opt = False
    model = get_model(modelname=modelname,other=other, image_size=img_size)    
    if len(other) > 0: 
        modelname = other
    model.eval()
    inp = torch.randn(1,3,img_size,img_size)
    out = model(inp)
    print(out.shape)
    if jitit:
        jit_model = torch.jit.script(model, example_inputs=[inp])    
        if mobile_opt:
            jit_model = optimize_for_mobile(jit_model)

        out2 = jit_model(inp)    

        print("diff:", (out-out2).abs().max().detach().numpy())
        print("Model:","Wjitmodels/"+modelname+("_opt" if mobile_opt else "")+".ptl")
        jit_model._save_for_lite_interpreter("Wjitmodels/"+modelname+("_opt" if mobile_opt else "")+".ptl")
    else:
        from tinynn.converter import TFLiteConverter
        dummy_input = inp = torch.randn(1,3,img_size,img_size)
        with torch.no_grad():
            converter = TFLiteConverter(model, dummy_input, tflite_path=os.path.join("Wonnxmodels",modelname+"_"+str(img_size)+".tflite"))
            converter.convert()
        
        # torch.onnx.export(model, inp, os.path.join("Wonnxmodels","intermed",modelname+".onnx"), do_constant_folding=True,opset_version=12, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
        # import tensorflow as tf
        # onnx_model = onnx.load( os.path.join("Wonnxmodels","intermed",modelname+".onnx"))
        # from onnx_tf.backend import prepare 
        # tf_rep = prepare(onnx_model)
        # tf_model_path = os.path.join("Wonnxmodels","intermed",modelname+".pb")
        # tf_rep.export_graph(tf_model_path )
        
        # converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        # converter.target_spec.supported_ops = [
        #     tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        #     tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        #     ]

        # tflite_model = converter.convert()
        # with open( os.path.join("Wonnxmodels",modelname+".tflite"), 'wb') as f:
        #     f.write(tflite_model)

if __name__ == "__main__":
    main()
    
    
# # adb push <speedbenchmark-torch> /data/local/tmp
# adb push <path-to-scripted-model> /data/local/tmp

# adb shell "/data/local/tmp/speed_benchmark_torch --model=/data/local/tmp/model.pt" --input_dims="1,3,224,224" --input_type="float" --warmup=10 --iter 100 --report_pep false
