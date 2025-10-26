

# UQ-Medical-Image-Segmentation

This repository provides a Monte Carlo UNet model for uncertainty quantification in the task of medical image segmentation.
This model has been coded using Pytorch. This code is a modified version of the original the original implementation of UNet model in Pytorch by milesial (https://github.com/milesial/Pytorch-UNet).
It consists of adding a dropout layer at the end of each convolution layer, which is used both during training and testing times.

## Dataset (2016 ISIC Challenge for Skin Lesion Segmentation)

The training dataset contains **900 dermoscopic lesion images** in JPEG format, along with the corresponding segmentation mask.

The ISIC Lesion segmentation dataset is originally introduced in the [ISIC Challenge webpage](https://challenge.isic-archive.com/), both training and testing dataset can be downloaded at [ISIC Challenge data](https://challenge.isic-archive.com/data/). 

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.

--

## Pretrained model
A [pretrained model](https://github.com/xingcheg/UQ-MedSeg/checkpoints/checkpoint_epoch20.pth) is available for the ISIC dataset. 


## Results

- Mean Dice score on the test set: 0.889
- Median Dice score on the test set: 0.926

![Segmentation and UQ results](diagram/result.png)

--

## Quick start
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. **[training]:** Download the training data (put them in `data/imgs` and `data/masks` folders respectively) and run training:
```bash
python train_bayesian.py --amp
```
5. **[predicting]:** Download the testing data (put them in `data/imgs_test` and `data/masks_test` folders respectively) and run predict:
```bash
python predict.py --input-dir ./data/imgs_test --output-dir ./data/masks_test_pred -m ./checkpoints/checkpoint_epoch20.pth
```
6. **[UQ]:** For each testing image:
```bash
python predict_bayesian.py -i ./data/imgs_test/ISIC_0000023.jpg -o ./data/test_uq/ISIC_0000023/ISIC_0000023.png --mc 20 --save-prob -m ./checkpoints/checkpoint_epoch20.pth
```
