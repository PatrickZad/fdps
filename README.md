
# Code for Fully Decoupled End-to-End Person Search: An Approach without Conflicting Objectives

## Fully checked version is coming soon

## Prerequest

1. Main dependecies

    ```shell
        torch==1.8.2
        torchvision==0.9.2
        cuda==11.1
        detectron2==0.5
        numpy=1.20.2
        opencv-python==4.8.0
    ```

2. Preparing data

    * Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](https://github.com/liangzheng06/PRW-baseline) for person search.

    * Create folder ```Data/``` and place the data as

        ```shell
                Data
                ├── cuhk_sysu
                │   ├── annotation
                │   ├── Image
                │   └── README.txt
                └── PRW
                ├── annotations
                ├── frames
                └── ......
        ```

3. Preparing model weight

    * Download pre-trained [ResNet-50](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl) weight as ```tvR-50.pkl``` into ```Data/model_zoo/```
  
## Training

* Train any model with the config

    ```shell
    python tools/train_tips_p.py --config-file ${file_path}  --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60888
    ```

    Make sure the detection side-network is ready before training the two-stage full TIPS model.

* We recommend to use [Tensorboard](https://www.tensorflow.org/tensorboard) to monitor the training process:

    ```shell
    tensorboard --logdir outputs/${model_output}$
    ```

## Inference

* Test any model with the config

    ```shell
    python tools/train_tips_p.py --config-file ${file_path}  --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60888 --eval-only
    ```

    Note that we employ a fast parallel evaluation mechanism that is incompatible with Pytorch Distributed Data Parallel.

## Code structure

* The overall structure of the code is based on [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html). This repository is organized as

    ```shell
    fdps
    ├── configs # Where model configuration files saved
    │   ├── det # detection side-network configuration files
    │   ├── ps  # full TIPS model configuration files
    │   └── frcnn_base.yaml
    ├── Data # Where data and model weights saved 
    ├── LICENSE
    ├── outputs # Where training logs and checkpoints saved
    ├── fdps # The main code
    │   ├── checkpoint # Automatic checkpoint tool
    │   ├── config # Definition of the basic configuration for all models
    │   ├── data # Reading and augmenting data
    │   ├── engine # The main training loop
    │   ├── evaluation # Evaluators for testing the model 
    │   ├── __init__.py
    │   ├── layers # Widely-used neural network modules
    │   ├── modeling # Definition of models
    │   │   ├── backbone # Vision backbone models
    │   │   ├── meta_arch
    │   │   │   ├── person_search # Person search models
    │   │   │   └── ......
    │   │   ├── proposal_generator # RPN and its variants
    │   │   ├── roi_heads # Prediction modules
    │   │   ├── transformer # Transformer models
    │   │   └── ......
    │   ├── model_zoo
    │   ├── solver # Optimizers and lr schedulers
    │   ├── structures # Commonly used data structures
    │   └── utils # Commonly used tools, e.g. logging, visualization and distributed communication.
    ├── README.md # This file
    └── tools # Train/test interface and preprocess tools
        └── train_tips_p.py

    ```

## Acknowledgement

This code is greatly inspired by [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html).
