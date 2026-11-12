"""Preview a CAPL configuration; add --run to train and evaluate it."""
import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATASETS = {
    'ETTh1': ('ETTh1', 'ETT-small/ETTh1.csv', 7, 'h'),
    'ETTh2': ('ETTh2', 'ETT-small/ETTh2.csv', 7, 'h'),
    'ETTm1': ('ETTm1', 'ETT-small/ETTm1.csv', 7, 't'),
    'ETTm2': ('ETTm2', 'ETT-small/ETTm2.csv', 7, 't'),
    'Weather': ('custom', 'weather/weather.csv', 21, 't'),
    'Electricity': ('custom', 'electricity/electricity.csv', 321, 'h'),
    'Appliances': ('custom', 'appliances_energy/appliances_energy.csv', 26, 't'),
}


def sampling_interval(path):
    dates = pd.to_datetime(pd.read_csv(path, usecols=['date'], nrows=97)['date'])
    intervals = dates.diff().dropna().dt.total_seconds().unique()
    if len(intervals) != 1 or intervals[0] <= 0 or 43200 % intervals[0] != 0:
        raise ValueError('Sampling must be regular and divide 12 hours')
    return int(intervals[0])


def build_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='ETTh1')
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or cuda:N')
    parser.add_argument('--cpu_threads', type=int, default=4)
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if args.pred_len < 16 or args.pred_len % 16:
        parser.error('--pred_len must be a positive multiple of 16')
    if min(args.batch_size, args.cpu_threads) < 1:
        parser.error('Batch size and CPU threads must be positive')
    try:
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Only cpu and cuda devices are supported')
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    args.device = str(device)
    args.data, relative_path, args.channels, args.freq = DATASETS[args.dataset]
    path = (args.data_dir / relative_path).resolve()
    if not path.is_file():
        parser.error(
            f'Dataset not found: {path}. Prepare it first with: '
            f'python "{ROOT / "prepare_data.py"}" --download --dataset {args.dataset} '
            f'--data_dir "{args.data_dir.resolve()}". See README.md for offline preparation.'
        )
    args.root_path, args.data_path = str(path.parent), path.name
    args.step_seconds = sampling_interval(path)
    args.features = 'MS' if args.dataset == 'Appliances' else 'M'
    args.target = 'Appliances' if args.dataset == 'Appliances' else 'OT'
    args.seq_len = 96
    args.train_epochs = 10
    args.learning_rate = 1e-4
    args.lradj = 'cosine'
    args.checkpoints = args.checkpoints.resolve()
    del args.data_dir
    return args


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def cleanup_checkpoint(args, setting):
    (args.checkpoints / setting / 'checkpoint.pth').unlink(missing_ok=True)


def main(argv=None):
    args = build_args(argv)
    print(json.dumps(vars(args), indent=2, default=str))
    if not args.run:
        print('Preview only. Add --run to train for 10 epochs and test the best Val-MSE checkpoint.')
        return
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select --device cpu or a working CUDA environment')
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    seed_everything(args.seed)
    os.chdir(ROOT)
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    experiment = Exp_Long_Term_Forecast(args)
    setting = f'CAPL_{args.dataset}_L{args.seq_len}_H{args.pred_len}_seed{args.seed}'
    try:
        experiment.train(setting)
        experiment.test(setting)
    finally:
        cleanup_checkpoint(args, setting)


if __name__ == '__main__':
    main()
