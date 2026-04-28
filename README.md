# SSA-Pose: Spatial Shape Awareness for RGB-Based Category-Level Object Pose Estimation

## Environment Setup

To install the required dependencies, use the following commands:

```bash
conda env create -f Lapose.yaml
```

## Data Preparation
* Download the data from [NOCS](https://github.com/hughw19/NOCS_CVPR2019).
* Download the segmentation predictions on CAMERA25 and REAL275 from [DualPose-Net](https://github.com/Gorilla-Lab-SCUT/DualPoseNet)

Run the following scripts to prepare training instances:

```bash
python prepare_data/pose_data.py
python prepare_data/shape_data.py
```

## Train

* Train on the CAMERA+Real dataset.

```bash
python engine/train.py --model_save="./output/model_save"
```

* Train on the CAMERA dataset.

```bash
python engine/train.py  --model_save="./output/model_save_C" --dataset=CAMERA
```

* Train scale net.

```bash
python engine/train_scale_net.py --model_save="./output_scale_net/model_save"
```

## Evaluate

* Evaluate on the Real dataset.

```bash
python evaluation/evaluate.py --resume_model="./output/model_save/model.pth" --dataset=Real --use_scale_net --sn_path='./output_scale_net/model_save/model.pth'
```

* Evaluate on the CAMERA dataset.

```bash
python evaluation/evaluate.py --resume_model="./output/model_save_C/model.pth" --dataset=CAMERA --use_scale_net --sn_path='./output_scale_net/model_save/model.pth'
```

## Citation

If you find our work useful, please cite:

## Acknowledgement
