# Pneumonia Detection using Swin Transformer

This repository contains a fine-tuned implementation of **Swin Transformer (Tiny)** for classifying Chest X-Ray images (Normal vs. Pneumonia).

This project was developed for the **Computer Vision** course to demonstrate the effectiveness of Hierarchical Vision Transformers in medical imaging tasks, comparing their complexity and accuracy against traditional methods.

## Key Features & Modifications

Unlike the original Microsoft repository, this version includes:
* **Custom Data Splitter**: A script to re-balance the dataset into 80% Train, 10% Val, 10% Test (`dataset_splitter.py`).
* **Modern PyTorch Support**: Patched `utils.py` to fix compatibility issues with PyTorch 2.6+ (`weights_only=False` fix).
* **Local Inference Script**: Optimized `test_swin.py` for running inference on local machines, with support for **Apple Metal (MPS)** acceleration on Mac.
* **Visualization Tools**: Notebooks to generate Confusion Matrices and Error Analysis plots.

## Dataset

The model is trained on the **Chest X-Ray Images (Pneumonia)** dataset.
* **Classes**: NORMAL, PNEUMONIA
* **Structure**: The code expects the dataset to be organized in `dataset/chest_xray_new/` with `train`, `val`, and `test` subfolders.

*(Note: The dataset is not included in this repo due to size limits. You can download it from Kaggle at https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).*

## Usage

### 1. Installation
```bash
git clone [https://github.com/TuoNomeUtente/Swin-Transformer-ChestXray.git](https://github.com/TuoNomeUtente/Swin-Transformer-ChestXray.git)
cd Swin-Transformer-ChestXray
pip install -r requirements.txt
```

### Training (Fine_tuning)
```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
--cfg configs/swin/chest_xray_finetune.yaml \
--data-path /path/to/dataset \
--batch-size 32 \
--tag split_80_10_10
```

### Inference and Evaluation
```bash
python test_swin.py
```

### Results

    Best Accuracy: ~91.5% (on Test set)

    Sensitivity (Recall on Pneumonia): High sensitivity achieved to minimize False Negatives in a medical context.
