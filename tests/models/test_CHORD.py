import unittest

import torch

from models.CHORD import IMTS_SubModel, Model
from utils.configs import get_configs


class TestCHORD(unittest.TestCase):

    def test_model(self):
        configs = get_configs(
            args=[
                "--model_name",
                "CHORD",
                "--model_id",
                "CHORD",
                "--batch_size",
                "2",
                "--seq_len",
                "6",
                "--pred_len",
                "3",
                "--enc_in",
                "5",
                "--dec_in",
                "5",
                "--c_out",
                "5",
                "--d_model",
                "16",
                "--d_ff",
                "32",
                "--n_heads",
                "4",
                "--n_layers",
                "2",
            ]
        )
        model = Model(configs)

        x = torch.randn(configs.batch_size, configs.seq_len, configs.enc_in)
        x_mark = torch.rand(configs.batch_size, configs.seq_len, 1)
        x_mark, _ = torch.sort(x_mark, dim=1)
        x_mask = (torch.rand(configs.batch_size, configs.seq_len, configs.enc_in) > 0.35).float()
        x_mask[0, 0, 0] = 1.0
        x_mask[1, 0, 1] = 1.0

        result_dict = model(**{"x": x, "x_mark": x_mark, "x_mask": x_mask})

        self.assertEqual(
            result_dict["pred"].shape,
            torch.Size((configs.batch_size, configs.pred_len, configs.c_out)),
        )
        self.assertEqual(
            result_dict["true"].shape,
            torch.Size((configs.batch_size, configs.pred_len, configs.c_out)),
        )
        self.assertTrue(torch.isfinite(result_dict["pred"]).all().item())

    def test_y_query_embed_only_keeps_true_mask_variates(self):
        configs = get_configs(
            args=[
                "--model_name",
                "CHORD",
                "--model_id",
                "CHORD",
                "--enc_in",
                "3",
                "--dec_in",
                "3",
                "--c_out",
                "3",
                "--d_model",
                "8",
                "--d_ff",
                "16",
                "--n_heads",
                "2",
                "--n_layers",
                "1",
            ]
        )
        model = IMTS_SubModel(configs)

        y_mask = torch.tensor(
            [[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        y_query_embed = model._build_y_query_embed(y_mask)

        expected_query = model.query.view(1, 1, 3, configs.d_model).expand(1, 2, -1, -1)
        self.assertTrue(torch.allclose(y_query_embed[0, 0, 0], expected_query[0, 0, 0]))
        self.assertTrue(torch.allclose(y_query_embed[0, 0, 2], expected_query[0, 0, 2]))
        self.assertTrue(torch.equal(y_query_embed[0, 0, 1], torch.zeros_like(y_query_embed[0, 0, 1])))
        self.assertTrue(torch.allclose(y_query_embed[0, 1, 1], expected_query[0, 1, 1]))