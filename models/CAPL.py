"""CAPL: Context-Aware Periodic Learning for Time Series Forecasting."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _Calendar:
    def __init__(self, freq):
        if freq not in ('h', 't'):
            raise ValueError("Calendar frequency must be 'h' or 't'")
        self.hour_index = 0 if freq == 'h' else 1
        self.minute_index = None if freq == 'h' else 0

    def seconds(self, marks):
        hour = ((marks[..., self.hour_index] + .5) * 23).round().long()
        minute = (torch.zeros_like(hour) if self.minute_index is None else
                  ((marks[..., self.minute_index] + .5) * 59).round().long())
        return (hour * 3600 + minute * 60).remainder(86400)

    def vectors(self, marks):
        phase = (self.seconds(marks).to(marks.dtype) / 86400)[..., None]
        angles = phase * (2 * math.pi)
        return torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)


class _LatentTemplate(nn.Module):
    nodes, width = 72, 128

    def __init__(self, targets, step_seconds):
        super().__init__()
        self.step_seconds = step_seconds
        self.E = nn.Parameter(torch.zeros(targets, self.nodes, self.width))
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(self.E, std=.02)

    def forward(self, patch_seconds):
        table = self.E - self.E.mean(dim=1, keepdim=True)
        # Doubled timestamps represent the half-step center of each 16-point patch.
        center = (2 * patch_seconds + 15 * self.step_seconds).remainder(172800)
        scaled = center * self.nodes
        coordinate = (scaled.div(172800, rounding_mode='floor')
                      + scaled.remainder(172800).float() / 172800).remainder(self.nodes)
        left = coordinate.floor().long()
        fraction = (coordinate - left)[..., None, None]
        target = torch.arange(table.shape[0], device=table.device)[None, None]
        features = ((1 - fraction) * table[target, left[..., None].remainder(self.nodes)]
                    + fraction * table[target, (left[..., None] + 1).remainder(self.nodes)])
        return features.permute(0, 2, 1, 3)


class _FeatureTime(nn.Module):
    def __init__(self, shared):
        super().__init__()
        initial_weight = shared.weight.detach().repeat(128, 1, 1)
        initial_bias = shared.bias.detach()
        weight = initial_weight.new_empty(128, initial_weight.shape[1] * 2 - 1, initial_weight.shape[2])
        bias = initial_bias.new_empty(initial_bias.numel() * 2 - 1)
        weight[:, ::2] = initial_weight
        weight[:, 1::2] = (initial_weight[:, :-1] + initial_weight[:, 1:]) / 2
        bias[::2] = initial_bias
        bias[1::2] = (initial_bias[:-1] + initial_bias[1:]) / 2
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return torch.einsum('bcdn,dqn->bcdq', x, self.weight) + self.bias


class _RefinementBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_attention = nn.LayerNorm(128)
        initial_projection = nn.Linear(128, 3 * 128)
        self.output = nn.Linear(128, 128)
        self.norm_ffn = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(256, 128))
        self.dropout = nn.Dropout(.1)
        for layer in (self.output, self.ffn[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Keep value-projection allocation independent of the next block's RNG.
        with torch.random.fork_rng(devices=[]):
            self.value = nn.Linear(128, 128)
        with torch.no_grad():
            self.value.weight.copy_(initial_projection.weight[256:])
            self.value.bias.copy_(initial_projection.bias[256:])

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1])
        x = x + self.dropout(self.output(self.value(self.norm_attention(x))))
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x.reshape(shape)


class _Backbone(nn.Module):
    def __init__(self, seq_len, pred_len, channels, target_only, dropout):
        super().__init__()
        starts = torch.arange(0, seq_len - 15, 8)
        if starts[-1] != seq_len - 16:
            starts = torch.cat((starts, torch.tensor([seq_len - 16])))
        self.register_buffer('history_index', starts[:, None] + torch.arange(16))
        self.register_buffer('future_position',
                             ((torch.arange(0, pred_len - 15, 8) + 1) / seq_len)[:, None])
        targets = torch.arange(channels)
        self.register_buffer('targets', targets[-1:] if target_only else targets)
        history_patches = len(starts)

        self.encoder = nn.Sequential(nn.Linear(16, 128), nn.Identity(), nn.LayerNorm(128),
                                     nn.Dropout(dropout))
        self.history_position = nn.Parameter(torch.empty(history_patches, 128))
        self.variable_role = nn.Parameter(torch.empty(channels, 128))
        nn.init.normal_(self.history_position, std=.02)
        nn.init.normal_(self.variable_role, std=.02)
        self.future_embedding = nn.Linear(1, 128)
        self.ordered_readout = nn.Linear(history_patches, pred_len // 16)
        self.self_decoder = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.self_head = nn.Linear(128, 16)
        self.direct = nn.Linear(seq_len, pred_len, bias=False)
        self.level_response = nn.Parameter(torch.zeros(pred_len))
        nn.init.zeros_(self.self_head.weight)
        nn.init.zeros_(self.self_head.bias)
        nn.init.zeros_(self.direct.weight)
        self.ordered_readout = _FeatureTime(self.ordered_readout)
        with torch.random.fork_rng(devices=[]):
            self.memory_blocks = nn.Sequential(_RefinementBlock(), _RefinementBlock())


class _OverlapMean(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        starts, points = torch.arange(0, horizon - 15, 8), torch.arange(horizon)
        covered = (points[None] >= starts[:, None]) & (points[None] < starts[:, None] + 16)
        self.register_buffer('count', covered.sum(0), persistent=False)

    def forward(self, patches):
        batch_size, channels = patches.shape[:2]
        folded = F.fold(patches.flatten(0, 1).transpose(1, 2), (1, self.horizon),
                        kernel_size=(1, 16), stride=(1, 8))
        return folded.reshape(batch_size, channels, self.horizon) / self.count


def _zero_projection(input_width):
    with torch.random.fork_rng(devices=[]):
        layer = nn.Linear(input_width, 128, bias=False)
        nn.init.zeros_(layer.weight)
    return layer


def _recent_state(history, step_seconds):
    tail = history[..., -(21600 // step_seconds + 1):]
    mean = tail.mean(-1)
    variation = tail.diff(dim=-1).abs().mean(-1) / (step_seconds / 3600)
    return torch.stack((mean, variation), -1).tanh()


class CAPL(nn.Module):
    def __init__(self, channels, pred_len, step_seconds, freq, *, seq_len=96,
                 target_only=False, dropout=.1):
        super().__init__()
        if seq_len < 16 or pred_len < 16 or pred_len % 16:
            raise ValueError('CAPL requires seq_len >= 16 and pred_len divisible by 16')
        if step_seconds <= 0 or 43200 % step_seconds:
            raise ValueError('Sampling interval must divide 12 hours')
        self.calendar = _Calendar(freq)
        self.step_seconds = step_seconds
        self.base = _Backbone(seq_len, pred_len, channels, target_only, dropout)
        with torch.random.fork_rng(devices=[]):
            self.phase_projection = nn.Linear(16, 128, bias=False)
        self.time_projection = _zero_projection(2)
        self.readout = _OverlapMean(pred_len)
        history_patches = len(self.base.history_index)
        self.response_weight = nn.Parameter(torch.zeros(2, 128, history_patches))
        self.halfday_response = nn.Parameter(torch.zeros(2, 128, history_patches))
        self.shape_projection = _zero_projection(16)
        self.scale_projection = _zero_projection(1)
        self.day_coefficients = nn.Parameter(torch.zeros(len(self.base.targets), 2, 2))
        self.latent_template = _LatentTemplate(len(self.base.targets), step_seconds)
        self.register_buffer('future_patch_starts', torch.arange(0, pred_len - 15, 8), persistent=False)

    def _encode_history(self, history, marks):
        memory = self.base.encoder(history[..., self.base.history_index]) + self.base.history_position
        calendar = self.calendar.vectors(marks)[:, self.base.history_index].mean(-2)
        memory = self.base.memory_blocks(memory + self.time_projection(calendar)[:, None])
        return memory[:, self.base.targets]

    def _temporal_state(self, memory, future_marks, future_calendar):
        state = self.base.ordered_readout(memory.transpose(-1, -2)).transpose(-1, -2)
        phase = future_calendar[:, 0]
        condition = phase[:, None].expand(-1, len(self.base.targets), -1)
        daily = torch.einsum('bcnd,jdn,bcj->bcd', memory, self.response_weight, condition)
        state = state + daily[:, :, None]

        day_fraction = self.calendar.seconds(future_marks)[:, :1].float() / 86400
        angle = (2 * day_fraction) * (2 * torch.pi)
        halfday_phase = torch.stack((angle.sin(), angle.cos()), -1)[:, 0]
        halfdaily = torch.einsum('bcnd,jdn,bj->bcd', memory, self.halfday_response, halfday_phase)
        return state + halfdaily[:, :, None]

    def _daily_response(self, history, future_calendar):
        recent = _recent_state(history, self.step_seconds)
        coefficients = torch.einsum('bcj,cjf->bcf', recent, self.day_coefficients)
        return torch.einsum('bcf,bhf->bch', coefficients, future_calendar)

    def forward(self, history, history_marks, future_marks):
        history = history.transpose(1, 2)
        variance, mean = torch.var_mean(history, dim=-1, unbiased=False, keepdim=True)
        scale = (variance + 1e-5).sqrt()
        normalized = (history - mean) / scale
        target_history = normalized[:, self.base.targets]
        target_scale = scale[:, self.base.targets]
        shape = self.base.direct(target_history)
        direct = mean[:, self.base.targets] * (1 + self.base.level_response) + target_scale * shape

        memory = self._encode_history(normalized, history_marks)
        future_calendar = self.calendar.vectors(future_marks)
        temporal = self._temporal_state(memory, future_marks, future_calendar)
        daily = self._daily_response(target_history, future_calendar)
        position = (self.base.variable_role[self.base.targets, None, :]
                    + self.base.future_embedding(self.base.future_position)[None])
        state = temporal + position + self.phase_projection(daily.unfold(-1, 16, 8))

        patch_seconds = self.calendar.seconds(future_marks[:, self.future_patch_starts])
        state = state + self.latent_template(patch_seconds)
        state = state + self.shape_projection(shape.unfold(-1, 16, 8))
        state = state + self.scale_projection(target_scale)[:, :, None]
        patches = self.base.self_head(self.base.self_decoder(state))
        residual = self.readout(patches)
        return (direct + residual + daily).transpose(1, 2)
