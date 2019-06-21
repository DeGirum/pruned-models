# Pruning Neural Networks
## Introduction
Pruning weights is one of the techniques to reduce the number of parameters and compute operations in neural networks. A variety of approaches have been studied for pruning, from fine-grained element-wise pruning to coarse-grained methods in which entire filters are pruned. See the [Documentation Page](https://nervanasystems.github.io/distiller/index.html) of the [Distiller](https://github.com/NervanaSystems/distiller) repo from [Nervana Systems](https://github.com/NervanaSystems) for a very good introduction and overview of various pruning techniques.

Recently, pruning API has been added to [TensorFlow ](https://www.tensorflow.org/). See the [announcement](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a). While the current gains are mainly from model compression, latency improvements have been mentioned as [future work](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/g3doc/guide/pruning/index.md).

At DeGirum, we are working on HW solutions that leverage sparsity in neural networks to provide significant latency improvements. We will provide links to the computation gains in the near future.

## Current State
Pruning has so far been seen as a method to reduce the number of parameters, thereby reducing the model size. This is attractive for edge applications which are resource constrained. There have been some efforts to leverage pruning to provide latency improvements, notably on FPGA platforms by [Xilinx](https://www.xilinx.com/applications/megatrends/machine-learning.html). Special libraries to handle sparsity have also shown some gains. See Baidu's [benchmark results](https://github.com/baidu-research/DeepBench) for different cases. It is worth noting that the gains are mainly reported for fully connected layers and for the sparsity of 90% or more. While pruning fully connected layers reduces the number of parameters significantly, it does not result in significant computation reduction. Consequently, the gains achieved can mainly be due to DRAM bandwidth savings. 

Some network architectures such as the recurrent neural networks used in natural language processing (NLP) applications and speech applications (speech-to-text and text-to-speech) have a lot of parameters. While these networks are also computationally intensive, the parameter reuse is very low thereby making their performance memory bound. Moreover, from a system cost point of view, compute is cheaper than memory bandwidth. Hence, it is no wonder that some architectures exploit sparsity only for reducing memory bandwidth and not computation. These architectures typically store compressed weights in system memory, decompress the weights to their full size on chip and compute on the full matrix ([TensorFlow Lite](https://www.tensorflow.org/lite) can be thought of as one example where model compression is used as an optimization technique).

Convolutional neural networks (CNNs), on the other hand, have excellent reuse of filter weights and are much more computationally intensive. Networks used for image classification easily have the number of multiply-accumulate (MAC) operations in the order of billions whereas networks for object detection run into tens of billions of MACs. Image segmentation networks can have hundreds of billions of operations. See the [convnet-burden](https://github.com/albanie/convnet-burden) for an overview of computation burden in different networks. While the network sizes are large in these cases, the computation load can be so high as to make these networks compute bound. Hence, for convolutional neural networks, properly designed hardware can exploit sparsity to reduce computation and provide latency improvements.

Researchers working on pruning methods have found that carefully pruned models with much less number of parameters and operations do not provide any performance improvement (in terms of frames per second), even on custom HW. Such observations have even led to questions about the usage of pruning as an optimization technique. See the paper [Pruning neural networks: is it time to nip it in the bud?](https://openreview.net/forum?id=r1lbgwFj5m) for some interesting comments and conclusions.


**NOTE:** Effort has been made to provide a general overview of the current state regarding pruning: (a) references to research (by pointing to documentation page of Distiller) (b) tools used (TensorFlow, Distiller), (c) state of HW (Baidu benchmark numbers, Xilinx FPGA link), (d) some views on advantages of sparsity and (e) counterpoint views. The above overview is by no means exhaustive. If you are aware of any other work that can add to this overview, please let us know by opening an issue. Also, if any of our understanding of the work of others is incorrect, please let us know.

## Relation Between Parameter Reduction and Compute Reduction
For a pruned network, parameter reduction is defined as the ratio of number of parameters in the original network to the number of parameters in the pruned network. For a single layer, the relationship between parameter reduction and computation speedup is straightforward. However, for a network composed of multiple layers, filter coefficients in different layers have different reuse. Pruning a coefficient in the initial layer saves more computation than pruning a coefficient in later layers. Similarly, for the same percentage of pruning, later layers reduce the absolute number of parameters much more than earlier layers.

Consider a simple network made of the following two layers:

Layer   | Input Shape (h, w, c) | Filter Shape (h, w, c) | Number of Filters | Output Shape (h, w, c) | Num. Filter Params | Num. MACs
--------|-----------------------|------------------------|-------------------|------------------------|--------------------|-----------
Layer0  | (56, 56, 64)          | (3, 3, 64)             | 256               | (28, 28, 256)          | 147456             | 115.6M
Layer1  | (28, 28, 256)         | (3, 3, 256)            | 256               | (14, 14, 256)          | 589824             | 115.6M 

Pruning Layer0 by 50% reduces the number of parameters by 73728 while cutting the MACs by 57.8Million. Pruning Layer1 by 50% reduces the number of parameters by 294912 while cutting the MACs by the same number 57.8M MACs. Unless all the layers are pruned by the same ratio, the overall reduction in the number of parameters cannot always give an accurate estimation of computation reduction. Hence, in the results, both the parameter reduction and computation reduction are reported. Parameter reduction is useful for saving DRAM bandwidth and computation reduction is useful for saving the number of MAC units in the HW.

## Some Results
This section provides the results on the performance of pruned networks for different models. The pruned models have been obtained using the [Distiller Tool](https://github.com/NervanaSystems/distiller). The Distiller tool also has options to evaluate the performance of [quantized models](https://github.com/NervanaSystems/distiller/tree/master/examples/quantization/post_train_quant). Results are reported for post-training quantization in which the weights and activations are quantized using 8-bits. The quantization scheme used was asymmetric and the activations were not clipped (option 6 in this [table](https://github.com/NervanaSystems/distiller/blob/master/examples/quantization/post_train_quant/command_line.md#sample-invocations)). 

Currently, the results are reported for ResNet50 trained on ImageNet dataset. Results for other image classification models (e.g. Inception-v3, DenseNet161, MobileNet, etc) and object detection models (e.g. Yolo-v3) will be added in the near future. 

Various research studies show that language models, as well as models used in speech recognition, can be pruned by as much as 90% without sacrificing accuracy. Moreover, sparse models have been found to outperform dense models with the same number of parameters. See [1](https://arxiv.org/pdf/1704.05119.pdf) and [2](https://arxiv.org/pdf/1710.01878.pdf) for some very interesting results. Reporting results for these models on is also on the roadmap.

### Image Classification
Name of Network     | Number of GMACS (Compute Reduction) | Number of Parameters (Parameter Reduction) | Top1/Top5 Accuracy (fp32) | Top1/Top5 Accuracy (INT8) 
--------------------|-----------------|----------------------|--------------------|-----------------
ResNet50            | 4.089 (1.00x)   | 25.5M (1.00x)        | 76.130/92.862      | 75.702/92.680   
ResNet50_Pruned_70  | 1.846 (2.21x)   | 7.48M (3.41x)        | 75.944/92.960      | 75.504/92.662   
ResNet50_Pruned_83  | 1.143 (3.58x)   | 4.24M (6.01x)        | 75.766/92.920      | 75.194/92.634   
ResNet50_Pruned_85  | 0.714 (5.73x)   | 3.93M (6.48x)        | 75.516/92.718      | 74.874/92.376   

## Reproducing Above Results
In order to enable the research community as well as product developers to replicate the above results, links are provided to download the training checkpoints as well as the mode state dictionaries for our pruned models. The pruning schedule yaml files used to generate the pruned models (on Distiller) are also provided. The yaml file contains other important metadata such as the accuracy performance, the command lines to run training, evaluation and quantization, the sparsity profile, the best epoch number and the ideal number of MACS needed for the pruned network.

Name of Network     | Model Checkpoint  | Model State Dict  | Pruning Scheduler 
--------------------|-------------------|-------------------|------------------
ResNet50_Pruned_70  | [ckpt](https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_70_best.pth.tar) | [state dict](https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_70_state_dict.pth)                  | [resnet50_pruned_70_schedule.yaml](pruning_schedulers/resnet50_pruned_70_schedule.yaml)
ResNet50_Pruned_83  | [ckpt](https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_83_best.pth.tar) | [state dict](https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_83_state_dict.pth)            | [resnet50_pruned_83_schedule.yaml](pruning_schedulers/resnet50_pruned_83_schedule.yaml)
ResNet50_Pruned_85  | [ckpt](https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_85_best.pth.tar) | [state dict](https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_85_state_dict.pth)                 | [resnet50_pruned_85_schedule.yaml](pruning_schedulers/resnet50_pruned_85_schedule.yaml)

### Using Pytorch
The performance of the pruned models can be replicated using this [jupyter notebook](EvalPrunedModelsStateDict.ipynb) which uses state dictionary and this [jupyter notebook](EvalPrunedModelsStateDict.ipynb) which uses the checkpoint. The notebooks are self-contained and only require pytorch to be installed (tested with Pytorch 1.1.0). 

### Using Distiller
With Distiller installed, results on quantization can be replicated. Also, models can be trained from scratch to obtain the pruned models with desired sparsity profile. 

## Call for Collaboration
Researchers are welcome to share their pruned models, especially in speech recognition applications where bandwidth requirements are high thereby making them more difficult to bring to the edge.

## License
All the model weight checkpoints, state dictionaries, scheduler yaml files, jupyter notebooks to replicate the results and any other software are licensed under the Apache License 2.0. 

## Acknowledgements
1. Thanks to the group maintaining Distiller repo. All the pruned models have been obtained by using their package.
2. The pytorch community
