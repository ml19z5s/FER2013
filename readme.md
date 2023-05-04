# Facial Expression Recognition using Deep Learning Models

This repository contains a Python implementation of deep learning models for facial
expression recognition (FER).
The models provided in this repository include VGG16, VGG19, ResNet50, ResNet101, ResNet152, ViT,
HybridViTCNN, CNNBeforeViT, and ViTBeforeCNN.

## Install

```
pip install -r requirements.txt
```

## Usage

To train and test a model, run the train.py script with the desired arguments. For example:
```
python train.py -d FER2013 -m ViT
```
If you want to train a hybrid model, you can use the following command:
```
python train.py -d FER2013 -m HybridViTCNN -cnn VGG16
```
