# GCSANet

GCSANet is a PyTorch-based land-use and land-cover segmentation pipeline for vegetation-focused remote sensing experiments. This repository packages the workflow from dataset setup through model training, evaluation, and ablation studies.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for training
- Local dataset directories matching the absolute paths configured in `01_setup_data.py`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Update the dataset paths in `01_setup_data.py` if your training and test data live somewhere else.

If you want the optional U-Net efficiency comparison in `03_training_eval.py`, install `segmentation-models-pytorch` separately.

## Usage

Run the full pipeline:

```bash
python main.py --stage all
```

Run a single stage:

```bash
python main.py --stage train
```

Available stages:

1. `setup` - data paths, preprocessing, dataloaders, and class weights
2. `model` - architecture, losses, and training utilities
3. `train` - sanity check, training, evaluation, and analyses
4. `ablation` - ablation studies and summary plots

## Project layout

- `01_setup_data.py` - dataset paths, preprocessing, dataloaders, and class weights
- `02_model_losses.py` - GCSANet architecture, losses, and train/validation helpers
- `03_training_eval.py` - training, test-set evaluation, metrics, and figures
- `04_ablation_studies.py` - ablation experiments and summary plots
- `figure_export_utils.py` - safe Matplotlib export helpers
- `main.py` - command-line entry point

## License

Released under the MIT License. See `LICENSE`.

Outputs are written under `results\`.
