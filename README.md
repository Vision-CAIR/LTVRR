# LTVRR Challenge

This is a starter code for the [LTVRR challenge](https://ltvrr.github.io/challenge/) associated with the ICCV 2021 paper [Exploring Long Tail Visual Relationship Recognition with Large Vocabulary](https://arxiv.org/abs/2004.00436).

This code is for the GQA-LT and VG8K-LT datasets. Below you can find instructions to run the code to train your own models and produce an output in the format that is required for the submission.

## Requirements
* Python 3
* Python packages
  * pytorch 1.7.1
  * cython
  * matplotlib
  * numpy
  * scipy
  * opencv
  * pyyaml
  * packaging
  * [pycocotools](https://github.com/cocodataset/cocoapi)
  * tensorboardX
  * tqdm
  * pillow
  * scikit-image
  * gensim
* An NVIDIA GPU and CUDA 10.2. Make sure you install the pytorch version compatible with your CUDA version. We only give instructions here to install pytorch 1.7 with CUDA 10.2. 

### Installation

```
conda create -n ltvrd python=3.8
conda activate ltvrd
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

## Install the dependencies:
```
pip install -r requirements.txt
```

## Compilation
Compile the CUDA code in the Detectron submodule and in the repo:
```
cd $ROOT/lib
python setup.py build develop
```

## Annotations

Create a data folder at the top-level directory of the repository:
```
# ROOT=path/to/cloned/repository
cd $ROOT
mkdir data
```

### GQA
Download it [here](https://drive.google.com/file/d/1ypmMOq2TkZyLNVuU9agHS7_QcsfTtBmn/view?usp=sharing). Unzip it under the data folder. You should see a `gvqa` folder unzipped there. It contains seed folder called `seed0` that contains .json annotations that suit the dataloader used in this repo.

### Visual Genome
Download it [here](https://drive.google.com/file/d/1S8WNnK0zt8SDAGntkCiRDfJ8rZOR3Pgx/view?usp=sharing). Unzip it under the data folder. You should see a `vg8k` folder unzipped there. It contains seed folder called `seed3` that contains .json annotations that suit the dataloader used in this repo.


### Word2Vec Vocabulary
Create a folder named `word2vec_model` under `data`. Download the Google word2vec vocabulary from [here](https://code.google.com/archive/p/word2vec/). Unzip it under the `word2vec_model` folder and you should see `GoogleNews-vectors-negative300.bin` there.

## Images

### GQA
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/gvqa
mkdir images
```
Download GQA images from the [here](https://cs.stanford.edu/people/dorarad/gqa/download.html)

### Visual Genome
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/vg8k
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images (part 1 and part 2) into `VG_100K/`. There should be a total of 108249 files.

## Pre-trained Object Detection Models
Download pre-trained object detection models [here](https://drive.google.com/open?id=16JVQkkKGfiGt7AUt789pUPX3o84Cl2hL). Unzip it under the root directory and you should see a `detection_models` folder there.

## Our Trained Relationship Detection Models
Download our trained models [here](https://drive.google.com/open?id=12zvgkUjgxAGEE99o0l6rcCCeE4QP4cYZ). Unzip it under the root folder and you should see a `trained_models` folder there.

## Directory Structure
The final directories for data and detection models should look like:
```
|-- data
|   |-- vg
|   |   |-- VG_100K    <-- (contains Visual Genome images)
|   |   |-- seed3    <-- (contains annotations)
|   |   |   |-- rel_annotations_train.json
|   |   |   |-- rel_annotations_val.json
|   |   |   |-- ...
|   |-- gvqa
|   |   |-- images    <-- (contains GQA training images)
|   |   |-- seed0    <-- (contains annotations)
|   |       |-- annotations_train.json
|   |       |-- annotations_val.json
|   |       |-- ...
|   |-- word2vec_model
|   |   |-- GoogleNews-vectors-negative300.bin
|-- trained_models
|   |-- e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only
|   |   |-- gvqa
|   |       |-- Mar02-02-16-02_gpu214-10_step_with_prd_cls_v3
|   |           |-- ckpt
|   |               |-- best.pth
|   |-- ...
```
## Evaluating Pre-trained Relationship Detection models

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to test with less or more than 8 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use. Remove the
`--multi-gpu-test` for single-gpu inference.

### GQA
**NOTE:** May require at least 64GB RAM to evaluate on the GQA test set

```
python tools/test_net_rel.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness.yaml --do_val --load_ckpt Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness/gvqa/Mar11-07-01-07_gpu210-18_step_with_prd_cls_v3/ckpt/best.pth  --use_gt_boxes --use_gt_labels --seed 0
```

### Visual Genome
**NOTE:** May require at least 64GB RAM to evaluate on the Visual Genome test set

```
python tools/test_net_rel.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_y_loss_only_hubness.yaml --do_val --load_ckpt Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness/vg8k/Mar11-07-01-07_gpu210-18_step_with_prd_cls_v3/ckpt/best.pth  --use_gt_boxes --use_gt_labels --seed 0
```

## Training Relationship Detection Models

The section provides the command-line arguments to train our relationship detection models given the pre-trained object detection models described above.

CHANGE variable `NUM_GPUS` to control the number of GPUs you want to train with (4 or 8) in the provided config files(configs/xx/xxxx.yaml).You can also use the environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs to use.

With the following command lines, the training results (models and logs) should be in `$ROOT/Outputs/xxx/` where `xxx` is the .yaml file name used in the command without the ".yaml" extension. If you want to test with your trained models, simply run the test commands described above by setting `--load_ckpt` as the path of your trained models.

<b> These are the Base case config files that you can use to kickstart your submissions </b>


### GQA
To train our relationship network using a VGG16 backbone with the ViL-Hubless loss, run
```
python tools/train_net_step_rel.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness100k.yaml --nw 8 --use_tfboard --seed 0
```

To train our relationship network using a VGG16 backbone without the ViL-Hubless loss, run
```
python tools/train_net_step_rel.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_baseline.yaml --nw 8 --use_tfboard --seed 0
```

To train our relationship network using a VGG16 backbone with the RelMix augmentation, run
```
python tools/train_net_step_rel.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_baseline_relmix.yaml --nw 8 --use_tfboard --seed 0
```

To run models with different ViL-Hubless scales create a new config file under `configs/gvqa/` (by copying the file `configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness.yaml`) and change the variable `TRAIN.HUBNESS_SCALE` to the desired value.
Also confirm the ViL-Hubless loss is activated by making sure the variable `TRAIN.HUBNESS` is set to `True`

### Visual Genome
To train our relationship network using a VGG16 backbone with the ViL-Hubless loss, run
```
python tools/train_net_step_rel.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_y_loss_only_hubness100k.yaml --nw 8 --use_tfboard --seed 3
```

To train our relationship network using a VGG16 backbone without the ViL-Hubless loss, run
```
python tools/train_net_step_rel.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_y_loss_only_baseline.yaml --nw 8 --use_tfboard --seed 3
```

To train our relationship network using a VGG16 backbone with the RelMix augmentation, run
```
python tools/train_net_step_rel.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_y_loss_only_baseline_relmix.yaml --nw 8 --use_tfboard --seed 3
```

To run models with different ViL-Hubless scales create a new config file under `configs/vg8k/` (by copying the file `configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_y_loss_only_hubness.yaml`) and change the variable `TRAIN.HUBNESS_SCALE` to the desired value.
Also confirm the ViL-Hubless loss is activated by making sure the variable `TRAIN.HUBNESS` is set to `True`
