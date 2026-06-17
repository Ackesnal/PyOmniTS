# Code from: https://github.com/Ladbaby/PyOmniTS
import warnings

import torch
from sklearn.model_selection import train_test_split

from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (  # collate_fns must be imported here for PyOmniTS's --collate_fn argument to work
    collate_fn,
    collate_fn_fractal,
    collate_fn_patch,
    collate_fn_tpatch,
    tsdmDataset,
)
from data.dependencies.tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
from utils.ExpConfigs import ExpConfigs

warnings.filterwarnings('ignore')

class Data(tsdmDataset):
    '''
    wrapper for MIMIC IV Bilos2021 dataset implemented in tsdm
    tsdm: https://openreview.net/forum?id=a-bD9-0ycs0

    - title: "MIMIC-IV, a freely accessible electronic health record dataset"
    - paper link: https://www.nature.com/articles/s41597-022-01899-x
    - tasks: forecasting
    - sampling rate (rounded): 1 minute
    - max time length (padded): 971 (48 hours)
    - seq_len -> pred_len:
        - 2160 -> 3
        - 2160 -> 720
    - number of variables: 96
        - Current local export keeps `Value_label_0` to `Value_label_95` in `~/.tsdm/rawdata/MIMIC_IV_Bilos2021/full_dataset.csv`.
        - The exact `label_code -> label` mapping is generated into `processed/label_dict.csv` by `data/dependencies/MIMIC_IV/preprocess/6_DataMerging.py`.
        - That `label_dict.csv` is not present in the remaining local artifacts on this machine, so the previous bundled 100-variable name list has been removed instead of leaving a stale mapping.
    - number of samples: 22018 (17834 + 1982 + 2202)
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        super(Data, self).__init__(configs=configs, flag=flag)
        self.L_TOTAL = 2880 # overwrite None in parent class

        self._check_lengths()
        self._preprocess()
        self._get_sample_index() # overwrite self.sample_index=None in parent class

    def __getitem__(self, index): # redundant, just for clarity
        return super().__getitem__(index)

    def __len__(self): # redundant, just for clarity
        return super().__len__()

    def _check_lengths(self): # redundant, just for clarity
        return super()._check_lengths()

    def _preprocess_base(self, task): # redundant, just for clarity
        return super()._preprocess_base(task)

    def _preprocess(self):
        if self.configs.task_name == "imputation":
            backbone_pred_len = 0
        elif self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            backbone_pred_len = self.pred_len
        else:
            raise NotImplementedError()

        task = MIMIC_IV_Bilos2021(
            seq_len=self.seq_len - 0.5,
            pred_len=backbone_pred_len
        )
        self._preprocess_base(task) # implemented in parent class

    def _get_sample_index(self):
        N_SAMPLES = 22018
        sample_index_all = torch.arange(N_SAMPLES)
        sample_index_train_val, sample_index_test = train_test_split(sample_index_all, test_size=0.1, shuffle = False)
        sample_index_train, sample_index_val = train_test_split(sample_index_train_val, test_size=0.1, shuffle = False)
        if self.flag == "train":
            self.sample_index = sample_index_train
        elif self.flag == "val":
            self.sample_index = sample_index_val
        elif self.flag == "test":
            self.sample_index = sample_index_test
        elif self.flag == "test_all":
            self.sample_index = sample_index_all
        else:
            raise NotImplementedError(f"Unknown {self.flag=}")