# Single infrared image optical noise removal using a deep convolutional neural network

Torch implementation for learning a mapping from input images to output images

<img src="imgs/img1.png" width="900px"/>


## Setup

### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN

### Getting Started
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph` and `display`
```bash
luarocks install nngraph
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
- Download this repo

- Download the network that has been trained and put it under checkpoints/OP/unet_L1+cGAN/ folder
```bash
[[latest_net_G.t7]](https://drive.google.com/file/d/0B3pG20Tbq8Nec09LV3lSMDJSWDA/view)
```

## Test
```bash
DATA_ROOT=/datasets/OP/ name=unet_L1+cGAN which_direction=AtoB phase=val_0.002 th test.lua
```
