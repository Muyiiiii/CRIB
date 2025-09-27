# CRIB

![CRIB](./pic/model.png)

# Key Features

- **CRIB Model**: Consistency-Regularized Information Bottleneck model for Multivariate Time Series Forecasting with Missing Values
- **Multiple Time Series Models**: Supports DLinear, SegRNN, Transformer, iTransformer, PatchTST, TSMixer, WPMixer, PAttn, and other models
- **Missing Value Handling**: Supports various missing patterns, including point missing, block missing, and column missing
- **Multi-dataset Support**: Compatible with ETTh1, PEMS, Metr, Elec
- **Flexible Training Configuration**: Supports various loss function combinations and training parameter settings

# Project Structure

```
CRIB/
├── train.py                 # Main training script
├── train.ipynb             # Jupyter notebook version
├── requirements.txt        # Dependency package list
├── bash/                   # Batch training scripts
│   └── bash_train.sh      # Automated training script
├── data/                   # Dataset directory
│   ├── ETT/               # ETT dataset
│   ├── PEMS/              # PEMS dataset
│   ├── metr_la/           # METR-LA dataset
│   ├── Electricity/       # Electricity dataset
├── TSL_models/            # Time series model implementations
│   ├── CRIB.py            # Main CRIB model file
│   ├── CRIB_module.py     # CRIB module components
│   ├── CRIB_embedding.py  # CRIB embedding layers
│   ├── DLinear.py         # DLinear model
│   ├── PatchTST.py        # PatchTST model
│   └── ...                # Other model files
├── layers/                # Base network layers
│   ├── Embed.py           # Embedding layers
│   ├── SelfAttention_Family.py # Self-attention mechanisms
│   └── Transformer_EncDec.py   # Transformer encoder-decoder
├── utils/                 # Utility functions
│   ├── utils.py           # Core utility functions
│   ├── dataClass.py       # Data class definitions
│   ├── masking.py         # Missing value mask processing
│   └── metrics.py         # Evaluation metrics
├── data_provider/         # Data providers
│   ├── data_factory.py    # Data factory
│   └── data_loader.py     # Data loader
├── revin/                 # RevIN normalization
├── log/                   # Training logs
└── result/                # Experimental results
```

# Requirements

## Python Version
- Python 3.9+

## Main Dependencies
```
torch==2.0.0
numpy==1.24.3
pandas==1.5.3
scikit-learn==1.2.2
tqdm==4.66.5
Muyi==0.0.8
```

# Installation

1. **Clone the repository**
```bash
git clone git@github.com:Muyiiiii/CRIB.git
cd CRIB
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare datasets**
    Place datasets in the `data/` directory. [Google Drive for 4 datasets](https://drive.google.com/file/d/1pb0qz6k3a0PJ96TS69F7NUFjHLjTSDlZ/view?usp=sharing). 

  Supported datasets include:

  - ETTh1: ETT dataset

  - PEMS: PEMS-Bay traffic dataset

  - Metr: METR-LA traffic dataset

  - Elec: Electricity consumption dataset

# Usage

## 1. Single Training Run

```bash
python train.py \
    --dataset ETTh1 \
    --model CRIB \
    --batch_size 32 \
    --missing_pattern point \
    --missing_rate 0.2 \
    --seq_len 24 \
    --pred_len 24 \
    --train_epochs 10 \
    --model_dim 32 \
    --seed 123
```

## 2. Batch Training

Use the provided bash script for batch training:

```bash
bash bash/bash_train.sh
```

## 3. Main Parameters

### Model Parameters
- `--model`: Model name, options include CRIB, DLinear, SegRNN, Transformer, iTransformer, PatchTST, TSMixer, WPMixer, PAttn
- `--dataset`: Dataset name, supports ETTh1, PEMS, Metr, Elec, BeijingAir, etc.
- `--model_dim`: Model dimension, default 32
- `--heads_num`: Number of attention heads, default 4
- `--enc_num`: Number of encoder layers, default 3

### Training Parameters
- `--train_epochs`: Number of training epochs, default 10
- `--learning_rate`: Learning rate, default 0.001
- `--batch_size`: Batch size, default 32
- `--seq_len`: Input sequence length, default 24
- `--pred_len`: Prediction sequence length, default 24

### Missing Value Parameters
- `--missing_rate`: Missing rate, default 0.7
- `--missing_pattern`: Missing pattern, options: point, block, col
- `--missing_block_width`: Block missing width (for block pattern)
- `--missing_block_height`: Block missing height (for block pattern)

### Loss Function Parameters
- `--loss_type`: Loss type for CRIB, options: 1, 2, 3, 12, 13, 23, 123
- `--IB_weight`: Information bottleneck weight, default 1.0
- `--KL_weight`: KL divergence weight, default 1e-6
- `--Consis_weight`: Consistency weight, default 1.0

# CRIB Model Details

The CRIB model is based on the information bottleneck principle and is implemented through the following components:

1. **Encoder**: Uses Transformer architecture to extract time series features
2. **Information Bottleneck**: Controls information flow through prediction loss and KL divergence regularization
3. **Consistency Constraint**: Ensures consistency of feature representations across different paths
4. **Prediction Head**: Performs time series prediction based on learned representations

## Loss Functions
The CRIB model supports three types of loss function combinations:
- **Loss 1 & 3**: Prediction loss and KL divergence loss ( based on information bottleneck)
- **Loss 2**: Behavioral consistency loss

# Experimental Results

Training results are saved to CSV files, including the following metrics:
- Test_MAE: Test set Mean Absolute Error
- Test_MSE: Test set Mean Squared Error
- Test_MAPE: Test set Mean Absolute Percentage Error

# Supported Models

| Model Name | Description | Use Case |
|------------|-------------|----------|
| CRIB | Consistency-Regularized Information Bottleneck model | Time series forecasting with missing values |
| DLinear | Linear decomposition model | Multivariate time series forecasting |
| PatchTST | Patch-based Transformer | Multivariate time series forecasting |
| TSMixer | Time series mixer | Multivariate time series forecasting |
| iTransformer | Inverted Transformer | Multivariate time series forecasting |
| ... | ... | ... |

# Dataset Information

| Dataset | Variables | Description | Application Domain |
|---------|-----------|-------------|-------------------|
| ETTh1 | 7 | Electric transformer temperature data | Power systems |
| PEMS | 325 | Traffic flow data | Traffic forecasting |
| Metr | 207 | Traffic speed data | Traffic forecasting |
| Elec | 321 | Electricity consumption data | Energy forecasting |

# Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests to improve the project.

# Acknowledgement
We appreciate the following github repo very much for the valuable code base and datasets:

[https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)