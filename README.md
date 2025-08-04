# HKT-ResNet CIFAR-10

The purpose of this repo is to provide a valid pytorch implementation of Hereditary Knowledge Transfer HKT applied between ResNet-100 (P) and ResNet-20 (C) for CIFAR10 as described in the original paper. The following models are provided:

| Name      | # layers | # params| Test err(paper) |
|-----------|---------:|--------:|:-----------------:|
|[ResNet20](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th)   |    20    | 0.27M   | 8.75%|
|[HKT-ResNet20](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32-d509ac18.th)  |    20    | 0.27M   | 7.60%| 
|[ResNet110](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th)  |   110    |  1.7M   | 6.43%|

This implementation matches description of the original paper, with comparable or better test error.

## How to run?
```bash
git clone https://github.com/akamaster/pytorch_resnet_cifar10
cd pytorch_resnet_cifar10
chmod +x run.sh && ./run.sh
```

## Details of training
Our implementation follows the paper in straightforward manner with some caveats: **First**, the training in the paper uses 45k/5k train/validation split on the train data, and selects the best performing model based on the performance on the validation set. We *do not perform* validation testing; if you need to compare your results on ResNet head-to-head to the orginal paper keep this in mind. **Second**, if you want to train ResNet1202 keep in mind that you need 16GB memory on GPU.

## Pretrained models for download
1. [ResNet20, 8.27% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20.th)
2. [HKT-ResNet20, 7.60% err](https://drive.google.com/file/d/1NrxhQ29Ptns6-ohu8UNoQxWMNt9Tl8Ev/view?usp=sharing)
3. [ResNet1202, 6.18% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202.th)



