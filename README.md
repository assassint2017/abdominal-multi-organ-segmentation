# abdominal-multi-organ-segmentation
abdominal multi-organ segmentation using pytorch

the data come from an online challenge called **Multi-atlas labeling Beyond the Cranial Vault**, for the detail, you can check this link:**https://www.synapse.org/#!Synapse:syn3193805/wiki/217752**. in this challenge, the task is to segement 13 different kind of organ as follow:

<div align=center><img src="https://github.com/assassint2017/abdominal-multi-organ-segmentation/blob/master/img/abdomen_overview_small.png" alt="各器官说明图"/></div>

## data management
i use the trainging set given by the competition organizer. The training set include 30 CT data.I randomly divided it into 25 for training and 5 for evaluation. and organize them as follow:

<div align=center><img src="https://github.com/assassint2017/abdominal-multi-organ-segmentation/blob/master/img/data_management.png"alt="数据管理示意图"/></div>

## data process
i normalized the axial spacing to 3mm. and truncated the hu value to a certain range. only the slice contain organ are used to train the network.

## network architecture
i use two u-shape like 3D FCN, and add residual connection at a certain group of convlayers. In order to increase the receptive field，i add some hybrid dilated convlayer to the last two stage of the encoder.most idea come form [1].

## implementation detail
i use adam optim and set the initial learning rate to 1e-4, train on three GTX 1080TI with batch size equal to three.the whole trainging process take about 13 hours.

## result
i use mean dice coefficient as metrics.

|strategy|spleen|right kidney|left kidney|gallbladder|esophagus|liver|stomach|aorta|inferior vena cava|portal vein and splenic vein|pancreas|right adrenal gland|left adrenal gland|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|ava_dice_loss|0.830|0.745|0.712|0.143|0.000|0.880|0.654|0.686|0.605|0.500|0.429|0.089|0.111|
|ava_dice_loss_with_bg|0.000|0.793|0.753|0.202|0.268|0.865|0.586|0.474|0.344|0.001|0.466|0.126|0.196|
|genernalised_dice_loss|---|---|---|---|---|---|---|---|---|---|---|---|---|
|genernalised_dice_loss_with_weight|---|---|---|---|---|---|---|---|---|---|---|---|---|

Here is the best of the above results:
<div align=center><img src="https://github.com/assassint2017/abdominal-multi-organ-segmentation/blob/master/img/bset.png"alt="最好结果三维展示图"/></div>

you can copy the value in bset_result.xlsx to show.xlsx to get the above picture

## TODO:
- [X] other loss function
- [ ] data augmentation

## references
1. Roth H R, Shen C, Oda H, et al. A multi-scale pyramid of 3D fully convolutional networks for abdominal multi-organ segmentation[J]. arXiv preprint arXiv:1806.02237, 2018.

2. Milletari F, Navab N, Ahmadi S A. V-net: Fully convolutional neural networks for volumetric medical image segmentation[C]//3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016: 565-571.

3. Fidon L, Li W, Garcia-Peraza-Herrera L C, et al. Generalised wasserstein dice score for imbalanced multi-class segmentation using holistic convolutional networks[C]//International MICCAI Brainlesion Workshop. Springer, Cham, 2017: 64-76.

4. Sudre C H, Li W, Vercauteren T, et al. Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations[M]//Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. Springer, Cham, 2017: 240-248.
