# Semantic-Segmentation-for-Natural-Terrains

### DeepLabv3-ResNet50 fine-tuned on RUGD for off-road terrain segmentation

This repository contains the code used to fine-tune and evaluate a DeepLabv3-ResNet50 semantic segmentation model for off-road terrain classification, along with a script for running inference on new images.

## Overview

- **Base architecture:** DeepLabv3 with a ResNet50 backbone (`COCO_WITH_VOC_LABELS_V1` pretrained checkpoint)
- **Dataset:** [RUGD](http://rugd.vision/) (RGB Unstructured Ground Driving) dataset, used for fine-tuning on terrain/off-road classes
- **Training environment:** Google Colab, using an A100 GPU hardware accelerator
- **Final model:** referred to as **Model 3** in this repo — the best-performing checkpoint from training/experimentation, and the one used for downstream inference

## Repository Contents

| File | Description |
|---|---|
| `Code.py` | Full model pipeline: training/fine-tuning setup and benchmarking (metrics such as mIoU) used to evaluate model performance |
| `model_finetuning.py` | Standalone model code — defines and fine-tunes the DeepLabv3-ResNet50 architecture on RUGD, without the benchmarking steps |
| `model_prediction.py` | Inference script — loads and combines the saved model weight files, then runs the model on an input image to produce a semantic segmentation prediction |

## Usage

1. **Training/fine-tuning:** Run `model_finetuning.py` (or `Code.py` for training + benchmarking together) in an environment with GPU access — this was originally run on Google Colab with an A100 GPU.
2. **Evaluation:** `Code.py` includes benchmarking code to assess segmentation performance (e.g., mIoU) against the RUGD validation/test set.
3. **Inference:** Use `model_prediction.py` to load the trained model weights and run semantic segmentation on a new image. This script handles combining the weight files needed to reconstruct the final model before prediction.

## Notes

- Model 3 is the final selected model after experimentation with earlier versions/checkpoints.
- Training was performed on Colab's A100 GPU due to the compute requirements of fine-tuning on the full RUGD dataset.

## Dataset Citation

RUGD: RGB Unstructured Ground Driving Dataset — used under its respective license/terms for academic/research purposes.
