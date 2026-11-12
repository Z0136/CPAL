from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom


DATASETS = {
    'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    dataset = DATASETS[args.data](
        root_path=args.root_path, data_path=args.data_path,
        seq_len=args.seq_len, pred_len=args.pred_len,
        features=args.features, target=args.target, freq=args.freq, flag=flag,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag != 'test', num_workers=0, pin_memory=False, drop_last=False,
    )
    print(flag, len(dataset))
    return dataset, loader
