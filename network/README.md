# Cloth net

This repository contains code to train and test neural networks to perform cloth simulations.
The loss is based on the SNUG paper (https://arxiv.org/abs/2204.02219).
It shares some similarity with the HOOD paper (https://dolorousrtur.github.io/hood/), however we are using a grid-based U-Net and not a GNN and take cloth parameters (stretching / shearing / bending) as additional parameters.

## Training
To train a net, you can call for example:

``` bash
python train.py --clip_grad_value=1 --max_stretching=5000 --min_stretching=100 --max_shearing=20 --min_shearing=0.05 --max_bending=1 --min_bending=0.001 --lr=0.001 --dataset_size=5000 --batch_size=300 --g=0.125 --net=SMP_param_a_gated3
```

Meaning of parameters:  
- --clip_grad_value: gradient clipping  
- --max_stretching: maximum value of stretching range during training  
- --min_stretching: minimum value of stretching range during training  
- --max_shearing: maximum value of shearing range during training  
- --min_shearing: minimum value of shearing range during training  
- --max_bending: maximum value of bending range during training  
- --min_bending: minimum value of bending range during training  
- --net: different network architectures (preffered at the moment: SMP_param_a_gated)  
- --lr: learning rate of Adam optimizer  
- --dataset_size: number of randomized domains during training  
- --batch_size: batch size  

More Infos:  
--help  
(or look into get_param.py)
