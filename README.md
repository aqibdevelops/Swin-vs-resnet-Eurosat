# Swin Transformer vs ResNet50 for EuroSAT Classification

This project implements and compares **Swin Transformer (Tiny)** and **ResNet50** models for land use and land cover classification using the [EuroSAT dataset](https://github.com/phelber/eurosat).

## Evaluation Results

We evaluated both models on the EuroSAT dataset. The Swin Transformer (Tiny) demonstrates competitive performance compared to the ResNet50 baseline. The Swin transformer outperforms the ResNet50 model in terms of accuracy, with a final accuracy of 98.76% compared to 88.37% for the ResNet50 model.

### Swin Transformer (Tiny)
![Swin Transformer Evaluation](Swin%20eval.png)

### ResNet50
![ResNet50 Evaluation](RESNET50%20eval.png)

## Dataset
The project uses the **EuroSAT** dataset, which consists of 27,000 labeled and georeferenced satellite images.
- **Download**: [EuroSAT Dataset (RGB)](http://madm.dfki.de/files/sentinel/EuroSAT.zip)
- **Structure**: Ensure the dataset is extracted into `data/eurosat/2750/`.

## Setup

1.  **Environment**: It is recommended to use a Conda environment.
    ```bash
    conda create -n torchmps python=3.10
    conda activate torchmps
    ```

2.  **Dependencies**: Install the required packages.
    ```bash
    pip install torch torchvision timm tqdm wandb pillow
    ```
    

3.  **Weights & Biases (WandB)**:
    This project uses WandB for experiment tracking and model artifact storage.
    ```bash
    wandb login
    ```

## Usage

### 1. Training
You can train either the Swin Transformer or ResNet50 model.

**Train Swin Transformer:**
```bash
python train_swin.py --data_dir data/eurosat/2750 --batch_size 32 --finetune_epochs 40
```

**Train ResNet50:**
```bash
python train_resnet.py --data_dir data/eurosat/2750 --batch_size 32 --finetune_epochs 40
```

**Arguments:**
- `--data_dir`: Path to the dataset directory (default: `data/eurosat/2750`).
- `--batch_size`: Batch size for training (default: 32).
- `--warmup_epochs`: Number of epochs for head-only warmup (default: 5).
- `--finetune_epochs`: Number of epochs for full fine-tuning (default: 40).
- `--lr`: Learning rate (default: 0.001).
- `--dry-run`: Run a single batch for debugging.

### 2. Inference
The `inference.py` script allows you to classify new images. **Crucially, it automatically downloads the trained model weights from WandB if they are not found locally.**

**Run Inference:**
```bash
python inference.py <path_to_image> --model_type <swin|resnet>
```

**Example:**
```bash
python inference.py data/eurosat/2750/Forest/Forest_1.jpg --model_type swin
```

**Arguments:**
- `image_path`: Path to the input image.
- `--model_type`: Model architecture to use (`swin` or `resnet`). Default: `swin`.
- `--model_path`: Path to the model checkpoint. If not found, it downloads from WandB. Default: `best_swin_eurosat.pth`.
- `--device`: Device to use (`cpu`, `cuda`, `mps`). Auto-detected if not specified.

### 3. Model Management
To upload your trained local models to WandB for remote storage and versioning:
```bash
python upload_models.py
```
This script uploads `best_resnet50_eurosat.pth` and `best_swin_eurosat.pth` to the `eurosat-classification` project on WandB.

## Project Structure
- `train_swin.py`: Training script for Swin Transformer.
- `train_resnet.py`: Training script for ResNet50.
- `inference.py`: Inference script with auto-download capabilities.
- `upload_models.py`: Helper script to upload models to WandB.
- `utils.py`: Utility functions for data loading and metrics.
- `data/`: Directory containing the EuroSAT dataset.
