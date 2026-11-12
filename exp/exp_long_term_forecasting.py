"""FP32 forecasting with MSE training and MSE/MAE evaluation."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from data_provider.data_factory import data_provider
from models.CAPL import CAPL
from utils.tools import adjust_learning_rate


class Exp_Long_Term_Forecast:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = CAPL(
            channels=args.channels, pred_len=args.pred_len, seq_len=args.seq_len,
            step_seconds=args.step_seconds, freq=args.freq, target_only=args.features == 'MS',
        ).float().to(self.device)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _forward_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            value.float().to(self.device) for value in batch)
        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
        target = batch_y
        if self.args.features == 'MS':
            outputs, target = outputs[..., -1:], target[..., -1:]
        return outputs, target

    def _evaluate(self, loader):
        totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        count = 0
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    outputs, target = self._forward_batch(batch)
                    batch_size = target.shape[0]
                    totals[0] += F.mse_loss(outputs, target).double() * batch_size
                    totals[1] += F.l1_loss(outputs, target).double() * batch_size
                    count += batch_size
            if count == 0:
                raise ValueError('Evaluation loader yielded no forecast windows')
            mse, mae = (totals / count).tolist()
            return {'mse': mse, 'mae': mae}
        finally:
            self.model.train(was_training)

    def vali(self, loader):
        return self._evaluate(loader)

    def train(self, setting):
        _, train_loader = self._get_data('train')
        _, vali_loader = self._get_data('val')
        path = Path(self.args.checkpoints) / setting
        path.mkdir(parents=True, exist_ok=True)
        best_mse = math.inf
        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            totals = torch.zeros(2, dtype=torch.float64, device=self.device)
            count = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                outputs, target = self._forward_batch(batch)
                loss = criterion(outputs, target)
                batch_size = target.shape[0]
                totals[0] += loss.detach().double() * batch_size
                totals[1] += F.l1_loss(outputs.detach(), target).double() * batch_size
                count += batch_size
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print(f'Iteration {i + 1}, epoch {epoch + 1} | MSE: {loss.item():.7f}')
            if count == 0:
                raise ValueError('Training loader yielded no forecast windows')
            train_mse, train_mae = (totals / count).tolist()
            validation = self.vali(vali_loader)
            print(f'Epoch {epoch + 1} | Train MSE: {train_mse:.7f} Train MAE: {train_mae:.7f} '
                  f'Val MSE: {validation["mse"]:.7f} Val MAE: {validation["mae"]:.7f}')
            if not all(math.isfinite(value) for value in validation.values()):
                raise RuntimeError('Validation metrics are non-finite; refusing to select a checkpoint')
            if validation['mse'] <= best_mse:
                best_mse = validation['mse']
                torch.save(self.model.state_dict(), path / 'checkpoint.pth')
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path / 'checkpoint.pth',
                                             map_location=self.device, weights_only=True))
        return self.model

    def test(self, setting, test=0):
        if test:
            path = Path(self.args.checkpoints) / setting / 'checkpoint.pth'
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        _, loader = self._get_data('test')
        metrics = self._evaluate(loader)
        record = json.dumps(metrics, allow_nan=False)
        print(record)
        return metrics
