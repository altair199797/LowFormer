# LowFormer
Accepted at [WACV2025](https://wacv2025.thecvf.com/).


## Install

## Training


### Run Training Single-GPU
To run on one GPU, specify the GPU-id with `CUDA_VISIBLE_DEVICES` and execute the following command:

`CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 
--nproc_per_node=1  --rdzv-endpoint localhost:29411 \
train_cls_model.py configs/cls/imagenet/b1_alternative.yaml  \
    --data_provider.image_size "[128,160,192,224]"   \
    --run_config.eval_image_size "[224]" \
    --path .exp/cls/imagenet/b1_alternative/`


### Run Training Multi-GPU
To run on 8 GPUs, just run the following command:

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 
--nproc_per_node=8  --rdzv-endpoint localhost:29411 \
train_cls_model.py configs/cls/imagenet/b1_alternative.yaml  \
    --data_provider.image_size "[128,160,192,224]"   \
    --run_config.eval_image_size "[224]" \
    --path .exp/cls/imagenet/b1_alternative/`

>Caveat: The total batch size in the config file is multiplied with the GPU instances and as well is the learning rate in the config file!

### Gradient Accumulation
In order to simulate a bigger batch size, there is a parameter in the configs called `bsizemult`, which is normally set to 1. The learning rate is also multiplied with it, as `bsizemult` increases the effective batch size.


## Testing
For testing and speed analysis `eval_cls_model.py` can be used. 
We also feature a vast library of popular backbone architectures. We adapted their code such that they can be converted to torchscript and onnx for speed measurement. 

### Evaluation on ImageNet
To evaluate a model given in `configs/cls/imagenet`, just run the following command:

`python eval_cls_model.py b1 --image_size 224 --batch_size 100  --gpu 6`


### Throughput Measurement
The following command runs the model LowFormer-B1 (from configs/cls/imagenet) for 400 iterations, with a batch size of 200, it uses torchscript optimization (optit) and has an input resolution of 224x224 (throughput measurement):

`python eval_cls_model.py b1 --image_size 224 --batch_size 200 --testrun --iterations 400 --gpu 6 --optit`

### Latency Measurement
You can also convert LowFormer-B1 to onnx and benchmark its latency:

`python eval_cls_model.py b1 --image_size 224 --batch_size 1 --testrun --latency --onnxrun --iterations 4000 --gpu 6 --optit --jobs 1`


### Speed Measurement of popular Backbone Models
When you append the argument `--other` followed by a string, you can run a lot of other backbones. Most of these backbones do not load their weights, so this functionality is purely for speed measurement (but could be extended for that purpose). The following command benchmarks MobileOne-S1 [1]:

`python eval_cls_model.py b1 --image_size 224 --batch_size 1 --testrun --latency --onnxrun --iterations 4000 --gpu 6 --optit --jobs 1 --other mobileones1`


## Acknowledgements

Please take a look at "DOCUMENT" for all the citations (TODO)


