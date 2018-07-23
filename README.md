# abdominal-multi-organ-segmentation
abdominal multi-organ segmentation using pytorch

the data come from an online challenge called **Multi-atlas labeling Beyond the Cranial Vault**, for the detail, you can check this link:**https://www.synapse.org/#!Synapse:syn3193805/wiki/217752**. in this challenge, the task is to segement 13 different kind of organ as follow:

![img1](https://github.com/assassint2017/abdominal-multi-organ-segmentation/blob/master/img/abdomen_overview_small.png)

## data management
i use the trainging set given bt the competition organizer. The training set include 30 CT data.I randomly divided it into 25 for training and 5 for evaluation. and organize them as follow:

.
├── train
│   ├── CT
│   │   ├── img0001.nii
│   │   ├── img0002.nii
│   │   ├── img0003.nii
│   │   ├── img0004.nii
│   │   ├── img0005.nii
│   │   ├── img0006.nii
│   │   ├── img0008.nii
│   │   ├── img0009.nii
│   │   ├── img0010.nii
│   │   ├── img0021.nii
│   │   ├── img0022.nii
│   │   ├── img0023.nii
│   │   ├── img0025.nii
│   │   ├── img0026.nii
│   │   ├── img0027.nii
│   │   ├── img0028.nii
│   │   ├── img0029.nii
│   │   ├── img0031.nii
│   │   ├── img0032.nii
│   │   ├── img0033.nii
│   │   ├── img0034.nii
│   │   ├── img0035.nii
│   │   ├── img0036.nii
│   │   ├── img0038.nii
│   │   └── img0040.nii
│   └── GT
│       ├── label0001.nii
│       ├── label0002.nii
│       ├── label0003.nii
│       ├── label0004.nii
│       ├── label0005.nii
│       ├── label0006.nii
│       ├── label0008.nii
│       ├── label0009.nii
│       ├── label0010.nii
│       ├── label0021.nii
│       ├── label0022.nii
│       ├── label0023.nii
│       ├── label0025.nii
│       ├── label0026.nii
│       ├── label0027.nii
│       ├── label0028.nii
│       ├── label0029.nii
│       ├── label0031.nii
│       ├── label0032.nii
│       ├── label0033.nii
│       ├── label0034.nii
│       ├── label0035.nii
│       ├── label0036.nii
│       ├── label0038.nii
│       └── label0040.nii
└── val
    ├── CT
    │   ├── img0007.nii
    │   ├── img0024.nii
    │   ├── img0030.nii
    │   ├── img0037.nii
    │   └── img0039.nii
    └── GT
        ├── label0007.nii
        ├── label0024.nii
        ├── label0030.nii
        ├── label0037.nii
        └── label0039.nii

## data process


## network architecture


## implementation detail


## result
|spleen|right kidney|left kidney|gallbladder|esophagus|liver|stomach|aorta|inferior vena cava|portal vein and splenic vein|pancreas|right adrenal gland|left adrenal gland|
|-|-|-|-|-|-|-|-|-|-|-|-|-|

## references
1. Roth H R, Shen C, Oda H, et al. A multi-scale pyramid of 3D fully convolutional networks for abdominal multi-organ segmentation[J]. arXiv preprint arXiv:1806.02237, 2018.

2. Milletari F, Navab N, Ahmadi S A. V-net: Fully convolutional neural networks for volumetric medical image segmentation[C]//3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016: 565-571.

3. Fidon L, Li W, Garcia-Peraza-Herrera L C, et al. Generalised wasserstein dice score for imbalanced multi-class segmentation using holistic convolutional networks[C]//International MICCAI Brainlesion Workshop. Springer, Cham, 2017: 64-76.

4. Sudre C H, Li W, Vercauteren T, et al. Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations[M]//Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. Springer, Cham, 2017: 240-248.
