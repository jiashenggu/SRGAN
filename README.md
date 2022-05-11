# SRGAN
A PyTorch implementation of SRGAN based on CVPR 2017 paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv pandas matplotlib
```

## Datasets

### Train、Val Dataset
The train and val datasets are sampled from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
Train dataset has 600 images and Val dataset has 200 images.


### Test Image Dataset
The test image dataset are sampled from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
Test dataset has 100 images.

## MDFloss
### MDFloss.py
MDFloss implementation.

### SinGAN
MDFloss is trained based n SinGAN architecture.

### Ds_SISR.pth
Weight of MDFloss for SISR.

## testing_results

Comparison between bicubic, original and super-resolution images.

## data_utils.py

Implementaion of dataset, dataloader

### PIL.ipynb
Test data_utils.

## metrics.ipynb
Plot metrics, PSNR, SSIM

## architecture_visualization.ipynb
Visualize model architecture by netron in onnx format.

## Usage

### Train
```
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 88]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 4, 8])
--num_epochs                  train epoch number [default value is 100]
```
The output val super resolution images are on `training_results` directory.

### Test Benchmark Datasets
```
python test_benchmark.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution images are on `benchmark_results` directory.

### Test Single Image
```
python test_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution image are on the same directory.

### Test Single Video
```
python test_video.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--video_name                  test low resolution video name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution video and compared video are on the same directory.



