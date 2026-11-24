# Code from: https://github.com/Ladbaby/PyOmniTS
import unittest

import torch

from exp.exp_main import Exp_Main
from utils.configs import get_configs


class TestExpMain(unittest.TestCase):

    def test_exp_main(self):
        configs = get_configs(args=[])
        exp = Exp_Main(configs)
        self.assertIsNotNone(exp.configs)
        self.assertIsNotNone(exp.device)

    def test_optimizer_defaults_to_adam(self):
        configs = get_configs(args=[])
        optimizer = Exp_Main(configs)._select_optimizer(torch.nn.Linear(1, 1))

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertNotIsInstance(optimizer, torch.optim.AdamW)

    def test_adamw_optimizer_can_be_selected(self):
        configs = get_configs(args=["--optimizer", "AdamW"])
        optimizer = Exp_Main(configs)._select_optimizer(torch.nn.Linear(1, 1))

        self.assertIsInstance(optimizer, torch.optim.AdamW)
