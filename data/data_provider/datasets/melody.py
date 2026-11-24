"""Numerical MELODY encounters: 36 h input, next 3 observed rows.

The audited exporter owns harmonisation, patient splits and train-input-only
standardisation. Loading never fits preprocessing or aliases validation as test.
Exact relative times are stored in int64 nanoseconds; model marks use a fixed
48-hour scale. See CHORD_MELODY/docs for provenance and clinical policies.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn as _standard_collate,
    collate_fn_patch as _patch_collate,
)


class Data(Dataset):
    def __init__(self, configs, flag='train', **kwargs):
        if flag not in ('train', 'val', 'test', 'test_all'):
            raise ValueError('melody supports train, val, test, and explicit test_all only.')
        if configs.task_name not in ('short_term_forecast', 'long_term_forecast'):
            raise ValueError('melody currently supports numerical forecasting only.')
        if configs.seq_len != 2160 or configs.pred_len != 3:
            raise ValueError('This melody export is fixed at --seq_len 2160 --pred_len 3 (minutes of input, future observed rows). Re-export before changing the task.')
        if getattr(configs, 'missing_rate', 0) != 0:
            raise ValueError('The audited melody task requires --missing_rate 0.')
        self.configs = configs
        self.flag = flag
        self.root = Path(configs.dataset_root_path).expanduser().resolve()
        meta_path = self.root / 'metadata.json'
        if not meta_path.is_file():
            raise FileNotFoundError(f'No completed melody export at {meta_path}. The clinical audit/export must finish before training.')
        self.metadata = json.loads(meta_path.read_text())
        if self.metadata.get('format') != 'melody-imts-v1' or self.metadata.get('status') != 'complete':
            raise ValueError('Unsupported or incomplete melody export.')
        if self.metadata['context_ns'] != 36 * 3600 * 10**9 or self.metadata['target_rows'] != 3:
            raise ValueError('Dataset task does not match the 36 h / 3-row loader contract.')
        self.L_TOTAL = 2880
        self.scale_ns = int(self.metadata['time_scale_ns'])
        if self.scale_ns != 48 * 3600 * 10**9:
            raise ValueError('melody expects fixed 48-hour time scale, not fitted time normalisation.')
        d = int(self.metadata['n_features'])
        for key in ('enc_in', 'dec_in', 'c_out'):
            if getattr(configs, key) != d:
                raise ValueError(f'--{key} must equal metadata n_features={d}. Use scripts/globals.sh get_dataset_info melody.')
        self._values = np.load(self.root / 'values.npy', mmap_mode='r', allow_pickle=False)
        self._times = np.load(self.root / 'relative_time_ns.npy', mmap_mode='r', allow_pickle=False)
        if self._values.dtype != np.float32 or self._values.ndim != 2 or self._values.shape[1] != d:
            raise ValueError('Invalid melody values shape or dtype.')
        if self._times.dtype != np.int64 or self._times.shape != (len(self._values),):
            raise ValueError('Invalid exact time array.')
        with (self.root / 'samples.csv').open(newline='', encoding='utf-8') as stream:
            self._all = list(csv.DictReader(stream))
        for row in self._all:
            for key in ('sample_id', 'offset', 'input_rows', 'target_rows'):
                row[key] = int(row[key])
            if row['split'] not in ('train','val','test') or row['target_rows'] != 3 or row['input_rows'] < 1:
                raise ValueError('Invalid split or sample lengths in melody index.')
            end = row['offset'] + row['input_rows'] + 3
            if row['offset'] < 0 or end > len(self._values):
                raise ValueError('Sample index exceeds array bounds.')
        patients = {split: {r['patient_id'] for r in self._all if r['split'] == split} for split in ('train','val','test')}
        if any(patients[a] & patients[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('Patient leakage between melody train, validation, or test.')
        if len({r['sample_id'] for r in self._all}) != len(self._all):
            raise ValueError('Duplicate melody sample identifiers.')
        self.rows = [r for r in self._all if flag == 'test_all' or r['split'] == flag]
        if not self.rows:
            raise ValueError(f'No {flag} samples in this melody export; regenerate the requested split.')
        self.sample_index = torch.tensor([r['sample_id'] for r in self.rows], dtype=torch.int64)
        self._configure_padding()

    def _configure_padding(self):
        """Inspect only time-array/index geometry; never validation/test feature values.

        A shared fixed shape is necessary for several PyOmniTS regular models.
        Geometry across all three splits is structural metadata, not fitted scaling.
        """
        c = self.configs
        mode = c.collate_fn or 'collate_fn'
        if mode not in ('collate_fn', 'collate_fn_patch', 'collate_fn_fractal'):
            raise ValueError(f'Unsupported melody collate: {mode}')
        c.seq_len_max_irr = max(r['input_rows'] for r in self._all)
        c.pred_len_max_irr = 3
        if mode == 'collate_fn':
            return
        patch_minutes = c.patch_len_list[-1] if mode == 'collate_fn_fractal' else c.patch_len
        if patch_minutes <= 0:
            raise ValueError('Patch length must be positive.')
        width = int(patch_minutes * 60 * 10**9)
        cutoff = self.metadata['context_ns']
        capacity, residual = 3, 0
        n_full = c.seq_len // patch_minutes
        for row in self._all:
            lo, n = row['offset'], row['input_rows']
            t = self._times[lo:lo+n]
            if len(t):
                _, counts = np.unique(t // width, return_counts=True)
                capacity = max(capacity, int(counts.max()))
                residual = max(residual, int(np.count_nonzero(t >= n_full * width)))
        if mode == 'collate_fn_patch':
            c.patch_len_max_irr = capacity
            c.seq_len_max_irr = math.ceil(c.seq_len / patch_minutes) * capacity
            c.pred_len_max_irr = capacity  # all 3 future rows live in the final target patch
        else:
            # Same layout as ReIMTS: complete input patches, a padded residual,
            # then target rows. Last patch retains all targets even if late.
            c.patch_len_max_irr = max(capacity, residual + 3)
            n_all = math.ceil((c.seq_len + c.pred_len) / patch_minutes)
            c.seq_len_max_irr = n_full * c.patch_len_max_irr + residual
            c.pred_len_max_irr = n_all * c.patch_len_max_irr - c.seq_len_max_irr
            if len(c.patch_len_list) > 1:
                # Require nested, integer-ratio scales; pad future-only tail so
                # the true input/target boundary cannot move across samples.
                for larger, smaller in zip(c.patch_len_list, c.patch_len_list[1:]):
                    if larger % smaller:
                        raise ValueError('Fractal patch scales must be nested integer multiples.')
                multiple = c.patch_len_list[0] // patch_minutes
                extra = (-n_all) % multiple
                c.pred_len_max_irr += extra * c.patch_len_max_irr
            c._melody_fractal_residual = residual

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start, nx = row['offset'], row['input_rows']
        times = self._times[start:start+nx+3]
        if (np.diff(times) < 0).any() or (times[:nx] >= self.metadata['context_ns']).any() or (times[nx:] < self.metadata['context_ns']).any():
            raise ValueError('Nonmonotone or incorrectly split sample times.')
        if times[0] < 0 or times[-1] >= self.scale_ns:
            raise ValueError('Sample lies outside the declared 48-hour horizon.')
        values = np.array(self._values[start:start+nx+3], copy=True)
        if np.isinf(values).any():
            raise ValueError('Nonfinite model values must be NaN (missing), never infinite.')
        t = torch.from_numpy(times.astype(np.float64) / self.scale_ns)
        v = torch.from_numpy(values)
        return dict(x=v[:nx], y=v[nx:], x_mark=t[:nx], y_mark=t[nx:],
                    x_mask=v[:nx].isfinite(), y_mask=v[nx:].isfinite(),
                    sample_ID=row['sample_id'], _configs=self.configs, _L_TOTAL=self.L_TOTAL)


def collate_fn(batch):
    # Shared padding helper constructs float32 synthetic time marks. Convert
    # only at the model boundary; the artifact and pre-collate times stay exact.
    prepared = [dict(s, x_mark=s["x_mark"].float(), y_mark=s["y_mark"].float()) for s in batch]
    result = _standard_collate(prepared)
    result["sample_ID"] = torch.tensor([s["sample_ID"] for s in batch], dtype=torch.int64)
    return result


def collate_fn_patch(batch):
    # The local shared implementation includes all remaining target rows in
    # the final forecast patch; pred_len=3 is not a three-minute cutoff.
    result = _patch_collate(batch)
    result["sample_ID"] = torch.tensor([s["sample_ID"] for s in batch], dtype=torch.int64)
    return result


def collate_fn_fractal(batch):
    """ReIMTS-compatible layout without truncating targets at seq_len+3 minutes.

    Original observed marks are never rounded or recomputed. Added positions
    carry mask=0. The last patch may extend in time to any of the three targets.
    """
    c = batch[0]['_configs']
    d = batch[0]['x'].shape[-1]
    capacity = c.patch_len_max_irr
    width = c.patch_len_list[-1] / batch[0]['_L_TOTAL']
    full = c.seq_len // c.patch_len_list[-1]
    residual = c._melody_fractal_residual
    total = c.seq_len_max_irr + c.pred_len_max_irr
    values = torch.zeros((len(batch), total, d), dtype=torch.float32)
    masks = torch.zeros_like(values)
    marks = torch.zeros((len(batch), total, 1), dtype=torch.float32)
    for i, sample in enumerate(batch):
        for patch in range(full):
            use = (sample['x_mark'] >= patch * width) & (sample['x_mark'] < (patch+1) * width)
            n = int(use.sum()); lo = patch * capacity
            if n > capacity:
                raise ValueError('Input patch exceeds declared padding capacity.')
            values[i,lo:lo+n] = torch.nan_to_num(sample['x'][use])
            masks[i,lo:lo+n] = sample['x_mask'][use]
            marks[i,lo:lo+n,0] = sample['x_mark'][use].float()
        use = sample['x_mark'] >= full * width
        n = int(use.sum()); lo = full * capacity
        if n > residual:
            raise ValueError('Residual exceeds declared padding capacity.')
        values[i,lo:lo+n] = torch.nan_to_num(sample['x'][use])
        masks[i,lo:lo+n] = sample['x_mask'][use]
        marks[i,lo:lo+n,0] = sample['x_mark'][use].float()
        lo = c.seq_len_max_irr; n = len(sample['y'])
        values[i,lo:lo+n] = torch.nan_to_num(sample['y'])
        masks[i,lo:lo+n] = sample['y_mask']
        marks[i,lo:lo+n,0] = sample['y_mark'].float()
    cut=c.seq_len_max_irr
    return dict(x=values[:,:cut], y=values[:,cut:], x_mask=masks[:,:cut], y_mask=masks[:,cut:],
                x_mark=marks[:,:cut], y_mark=marks[:,cut:],
                sample_ID=torch.tensor([s['sample_ID'] for s in batch],dtype=torch.int64))
