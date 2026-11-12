# CAPL: Context-Aware Periodic Learning for Time Series Forecasting

This repository contains the official implementation of CAPL: Context-Aware Periodic Learning for Time Series Forecasting. CAPL combines shared daily representations with window-specific forecasting context. Run the commands below from this directory.

## 1. Set up the environment

Use Python 3.11 and PyTorch 2.6.0. For GPU experiments, install a CUDA-compatible PyTorch build.

```bash
conda create -n capl python=3.11 -y
conda activate capl
python -m pip install -r requirements.txt
```

## 2. Download and preprocess data

Download and prepare the datasets, then verify the outputs:

```bash
python -B prepare_data.py --download
python -B prepare_data.py --check
```

Source CSVs are cached in `raw_data/`; prepared CSVs are written to `data/`. Add `--dataset ETTh1` to prepare a single dataset. ETT, Weather, and Electricity retain the published benchmark CSVs; Appliances removes the artificial columns `rv1` and `rv2`.

For manual downloads, use these links:

| Dataset name (`--dataset`) | Download |
|---|---|
| ETTh1, ETTh2, ETTm1, ETTm2 | [ETTh1](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv), [ETTh2](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh2.csv), [ETTm1](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTm1.csv), [ETTm2](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTm2.csv) |
| Weather | [weather.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/weather/weather.csv) |
| Electricity (ECL in the paper) | [electricity.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/electricity/electricity.csv) |
| Appliances | [UCI ZIP](https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip); extract `energydata_complete.csv` |

Place the downloaded CSVs in one directory and prepare them offline:

```bash
python -B prepare_data.py --source_dir /path/to/downloads
```

Keep source files unchanged: the script checks their SHA-256 hashes. Use `--data_dir /path/to/data` in both `prepare_data.py` and `run.py` to store prepared data elsewhere.

## 3. Run experiments

Train and evaluate one configuration on CPU:

```bash
python -B run.py --dataset ETTh1 --pred_len 96 --seed 2021 --run
```

Add `--device cuda:0` for GPU execution. Omitting `--run` prints the configuration without training.

Run the configured forecast horizons for a dataset using Bash (or WSL on Windows):

```bash
bash scrips/ETTh1.sh --seed 2021 --device cuda:0
```

To run CAPL across datasets and random seeds, use:

```bash
for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Appliances; do
  for seed in 2021 2022 2023; do
    bash "scrips/${dataset}.sh" --seed "$seed" --device cuda:0 || exit 1
  done
done
```

Replace `cuda:0` with `cpu` if needed. Each run tests the checkpoint with the lowest validation MSE and prints MSE/MAE to the terminal. Add `> run.log` to a command to save its output. Temporary checkpoints are deleted after the run.
