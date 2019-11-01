# U-Net Cat Image Segmentation
[![python](https://img.shields.io/badge/Python-3.x-ff69b4.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)
[![tensorflow](https://img.shields.io/badge/Tensorflow-1.1x%7C2.0-brightgreen.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)
[![OpenCV](https://img.shields.io/badge/OpenCV-3.x%7C4.x-orange.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)

U-Net implementation for performing image segmentation on cat images.

## Models
The project supports these semantic segmentation models as follows:

>1. UNet - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

## Dataset

The training dataset consists of **60 cat images**, along with their **masks**.

To increase the size of our dataset, we perform augmentations like **flipping, rotations, shears, and translations** on each of the images. 

We also have the option for applying a tranfer learning approach, for which we use a training dataset consisting of **308 horse images**, along with their **masks**.

## Results
