import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name

        self.model = IMTS_SubModel(configs)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None,
        y_mask: Tensor | None = None,
        **kwargs,
    ) -> dict:
        batch_size, seq_len, enc_in = x.shape
        y_len = self.configs.pred_len if self.configs.pred_len != 0 else seq_len
        if x_mark is None:
            x_mark = repeat(
                torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device)
                / max(x.shape[1], 1),
                "L -> B L 1",
                B=x.shape[0],
            )
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            logger.warning(
                "y is missing for the model input. This is only reasonable when the model is testing flops!"
            )
            y = torch.ones((batch_size, y_len, enc_in), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(
                torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device)
                / max(y.shape[1], 1),
                "L -> B L 1",
                B=y.shape[0],
            )
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        
        predictions = self.model(x, x_mark, x_mask, y_mark, y_mask)

        if self.configs.task_name in ["long_term_forecast", "short_term_forecast"]:
            f_dim = -1 if self.configs.features == "MS" else 0
            return {
                "pred": predictions[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        raise NotImplementedError


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True, mask=None):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    If mask ([B, N] bool) is provided, only tokens where mask=True are eligible for dropping;
    tokens where mask=False are always kept.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    if mask is None:
        # Original behavior: drop entire sample uniformly
        random_tensor = x.new_empty(x.shape[:-1]).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
    else:
        # Per-token drop: mask=True tokens are eligible, mask=False tokens are always kept
        # mask: [B, N], x: [B, N, C]
        random_tensor = x.new_empty(x.shape[:-1]).bernoulli_(keep_prob)  # [B, N]
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        # Where mask is False, force factor to 1.0 (no drop)
        drop_factor = torch.where(mask, random_tensor, torch.ones_like(random_tensor))  # [B, N]
        return x * drop_factor.unsqueeze(-1)  # broadcast over C


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    Optionally accepts a boolean mask [B, N] to restrict dropping to eligible tokens only.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep, mask=mask)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def ensure_non_empty_rows(mask: Tensor) -> Tensor:
    row_has_key = mask.any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(mask)
    fallback[..., 0] = True
    return torch.where(row_has_key, mask, fallback)


class StageOneEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(StageOneEncoder, self).__init__()
        self.window_predictor = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.attn_norm = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.n_heads = n_heads
        self.out = nn.Linear(d_model, d_model)
        self.droppath1 = DropPath(drop_prob=dropout)
        
        self.ffn_norm = nn.RMSNorm(d_model)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.droppath2 = DropPath(drop_prob=dropout)

    def forward(self, x, x_mark, x_mask, tau_time=0.1, eps=1e-6):
        """
        Forward pass for the stage one encoder.

        Args:
            x: Input tensor of shape (B, V, L, D).
            x_mark: Time mark tensor of shape (B, V, L).
            x_mask: Mask tensor of shape (B, V, L).

        Returns:
            Encoded tensor of shape (B, V, L, D).
        """
        B, V, L, D = x.shape
        
        # 1. Predict the attention window for each event.
        time_pred = torch.sigmoid(self.window_predictor(x))  # B, V, L, 1
        # print(time_pred.max(), time_pred.min(), time_pred.mean(), time_pred.std(), time_pred.median())

        # 2. Query time and key time
        query_time = x_mark.reshape(B, V, L, 1)  # B, V, L, 1
        key_time = x_mark.reshape(B, V, 1, L)    # B, V, 1, L

        # 3. Calculate the time boundary for each event based on the predicted window.
        left_boundary = query_time - time_pred
        left_boundary = left_boundary.clamp(min=0.0, max=1.0)  # B, V, L, 1

        right_boundary = query_time.clamp(min=0.0, max=1.0)  # B, V, L, 1

        # 4. Calculate the time soft gate as additive log-bias.
        # Equivalent to:
        #   log(sigmoid((key_time - left_boundary) / tau_time))
        # + log(sigmoid((right_boundary - key_time) / tau_time))
        #
        # This is more numerically stable than:
        #   torch.log(left_gate * right_gate)
        tau = max(tau_time, 1e-6)

        left_time_bias = F.logsigmoid(
            (key_time - left_boundary) / tau
        )  # B, V, L, L

        right_time_bias = F.logsigmoid(
            (right_boundary - key_time) / tau
        )  # B, V, L, L

        attention_bias = left_time_bias + right_time_bias  # B, V, L, L
        
        # 5. Calculate the valid query-key mask.
        query_mask = x_mask.reshape(B, V, L, 1)  # B, V, L, 1
        key_mask = x_mask.reshape(B, V, 1, L)    # B, V, 1, L

        pair_mask = query_mask & key_mask        # B, V, L, L
        hard_mask = ensure_non_empty_rows(pair_mask)

        # 6. Apply hard mask.
        attention_bias = attention_bias.masked_fill(~hard_mask, float("-inf"))

        # 7. Expand to heads.
        attention_bias = attention_bias.unsqueeze(2).expand(
            -1, -1, self.n_heads, -1, -1
        )  # B, V, H, L, L
        
        # 5. Self-attention
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1) # B, V, L, D each
        q = q.reshape(B, V, L, self.n_heads, D // self.n_heads).permute(0, 1, 3, 2, 4) # B, V, H, L, D_head
        k = k.reshape(B, V, L, self.n_heads, D // self.n_heads).permute(0, 1, 3, 2, 4) # B, V, H, L, D_head
        v = v.reshape(B, V, L, self.n_heads, D // self.n_heads).permute(0, 1, 3, 2, 4) # B, V, H, L, D_head
        x = self.droppath1(self.out(torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_bias).permute(0, 1, 3, 2, 4).reshape(B, V, L, D)), mask=x_mask) + x
        
        # 5. Feedforward Network
        x = self.droppath2(self.down(self.act(self.up(self.ffn_norm(x)))), mask=x_mask) + x # B, V, L, D
        
        return x
    

class StageTwoEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_variates: int,
        dropout: float,
    ):
        super(StageTwoEncoder, self).__init__()
        self.window_predictor = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.variate_predictor = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_variates),
        )
        self.n_variates = n_variates
        self.attn_norm = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.n_heads = n_heads
        self.out = nn.Linear(d_model, d_model)
        self.droppath1 = DropPath(drop_prob=dropout)

        self.ffn_norm = nn.RMSNorm(d_model)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.droppath2 = DropPath(drop_prob=dropout)

    def forward(self, x, x_mark, x_mask, tau_time=0.1, eps=1e-6, tau_variate=0.1):
        """
        Forward pass for the stage two encoder.

        Args:
            x: Input tensor of shape (B, N, D).
            x_mark: Packed time mark tensor of shape (B, N), encoded as variate_index + timestamp.
            x_mask: Mask tensor of shape (B, N).

        Returns:
            Encoded tensor of shape (B, N, D).
        """
        B, N, D = x.shape
        
        # 1. Predict time window and variate logits
        time_pred = torch.sigmoid(self.window_predictor(x)).squeeze(-1)  # B, N
        variate_logits = self.variate_predictor(x)                       # B, N, V

        # 2. Decode packed marks into per-token variate ids and timestamps
        source_variate = torch.floor(x_mark).clamp(
            min=0,
            max=self.n_variates - 1
        ).long()  # B, N

        event_time = x_mark - source_variate.to(x_mark.dtype)  # B, N

        # 3. Calculate time boundaries
        left_boundary = (event_time - time_pred).clamp(
            min=0.0,
            max=1.0
        ).unsqueeze(-1)  # B, N, 1

        right_boundary = event_time.clamp(min=0.0, max=1.0).unsqueeze(-1)  # B, N, 1

        # 4. Time soft gate, written as additive log-bias
        key_time = event_time.unsqueeze(1)  # B, 1, N

        left_time_bias = F.logsigmoid(
            (key_time - left_boundary) / max(tau_time, eps)
        )  # B, N, N

        right_time_bias = F.logsigmoid(
            (right_boundary - key_time) / max(tau_time, eps)
        )  # B, N, N

        time_bias = left_time_bias + right_time_bias  # B, N, N

        # 5. Variate soft gate
        # For each query i and key j, select query i's logit for key j's variate id.
        key_variate_index = source_variate.unsqueeze(1).expand(
            -1, N, -1
        )  # B, N, N

        selected_variate_logits = variate_logits.gather(
            dim=-1,
            index=key_variate_index
        )  # B, N, N

        variate_bias = F.logsigmoid(
            selected_variate_logits / max(tau_variate, eps)
        )  # B, N, N

        # 6. Padding mask
        pair_mask = x_mask.unsqueeze(-1) & x_mask.unsqueeze(1)  # B, N, N
        hard_mask = ensure_non_empty_rows(pair_mask)

        # 7. Final additive attention bias
        attention_bias = time_bias + variate_bias
        attention_bias = attention_bias.masked_fill(~hard_mask, float("-inf"))

        # 8. Expand to heads
        attention_bias = attention_bias.unsqueeze(1).expand(
            -1, self.n_heads, -1, -1
        )  # B, H, N, N
        
        # 9. Self-attention
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1) # B, N, D each
        q = q.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
        k = k.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
        v = v.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
        x = self.droppath1(self.out(torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_bias).permute(0, 2, 1, 3).reshape(B, N, D)), mask=x_mask) + x

        # 10. Feedforward network
        x = self.droppath2(self.down(self.act(self.up(self.ffn_norm(x)))), mask=x_mask) + x # B, N, D

        return x



class IMTS_SubModel(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(IMTS_SubModel, self).__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.n_variates = configs.enc_in

        self.value_embedding = nn.Parameter(torch.empty(self.n_variates, self.d_model))
        self.time_embedding = nn.Linear(1, self.d_model)
        self.variate_embedding = nn.Parameter(torch.empty(self.n_variates, self.d_model))
        self.event_norm = nn.RMSNorm(self.d_model)

        self.query = nn.Parameter(torch.rand(1, 1, 1, self.d_model))
        self.query_time_embedding = nn.Linear(1, self.d_model)
        self.query_norm = nn.RMSNorm(self.d_model)

        self.stage1_encoder = nn.ModuleList([
            StageOneEncoder(
                d_model=self.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
            )
            for _ in range(configs.n_layers)
        ])
        self.stage2_encoder = nn.ModuleList([
            StageTwoEncoder(
                d_model=self.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                n_variates=self.n_variates,
                dropout=configs.dropout,
            )
            for _ in range(configs.n_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        model_std = self.d_model ** -0.5
        nn.init.normal_(self.value_embedding, mean=0.0, std=model_std)
        nn.init.normal_(self.variate_embedding, mean=0.0, std=model_std)
        nn.init.normal_(self.query, mean=0.0, std=model_std)

    def _build_event_embedding(self, x, x_mark, x_mask):
        batch_size, _, n_variates = x.shape
        value_embed = torch.einsum("bnv,vd->bnvd", x, self.value_embedding)
        time_embed = self.time_embedding(x_mark).unsqueeze(2)
        variate_embed = self.variate_embedding.view(1, 1, n_variates, self.d_model).expand(batch_size, -1, -1, -1)
        event_embed = self.event_norm(value_embed + time_embed + variate_embed)
        return event_embed

    def _build_query_embedding(self, y_mark, y_mask):
        batch_size, _, n_variates = y_mask.shape
        query_embed = self.query 
        query_time_embed = self.time_embedding(y_mark).unsqueeze(2)
        query_variate_embed = self.variate_embedding.view(1, 1, n_variates, self.d_model).expand(batch_size, -1, -1, -1)
        query_embed = self.query_norm(query_embed + query_time_embed + query_variate_embed)
        return query_embed 
    
    def _relocate_events_stage1(self, x, x_mark, x_mask, x_len):
        """
        x: (B, L, V, D)
        x_mark: (B, L, 1)
        x_mask: (B, L, V)
        x_len: int, length of the original x portion (positions >= x_len belong to y)
        """
        B, L, V, D = x.shape
        
        # 1. Reshape x, x_mark, x_mask to (B, V, L, D), (B, 1, L, 1), (B, V, L)
        x = x.permute(0, 2, 1, 3) # B, V, L, D
        x_mark = x_mark.reshape(B, 1, L) # B, 1, L
        x_mask = x_mask.permute(0, 2, 1) # B, V, L
        
        # 2. Get the maximum event length for each variate
        x_mask = (x_mask == 1) # B, V, L
        event_len_max = int(x_mask.sum(dim=2).max().item())
        
        # 3. New allocation for the input events, with shape (B, V, event_len_max, D)
        x_new = x.new_zeros((B, V, event_len_max, D)) # B, V, event_len_max, D
        x_mask_new = torch.zeros((B, V, event_len_max), dtype=torch.bool, device=x.device) # B, V, event_len_max
        x_mark_new = x_mark.new_zeros((B, V, event_len_max)) # B, V, event_len_max
        
        # 4. For each True position in mask_m, compute its new position after compaction
        #    Example: [False, True, True, False, True, True]
        #    ranks:   [-1,    0,    1,    1,     2,    3]
        ranks = x_mask.long().cumsum(dim=2) - 1
        
        # 5. Get indices of selected tokens
        b_idx, v_idx, old_pos_idx = x_mask.nonzero(as_tuple=True)
        new_pos_idx = ranks[b_idx, v_idx, old_pos_idx] # (num_selected_tokens,)
        
        # 6. Copy selected tokens from x to x_new and re-generate the mask
        x_new[b_idx, v_idx, new_pos_idx] = x[b_idx, v_idx, old_pos_idx]
        x_mask_new[b_idx, v_idx, new_pos_idx] = True
        x_mark_new[b_idx, v_idx, new_pos_idx] = x_mark[b_idx, 0, old_pos_idx]
        
        # 7. Build y-position map: mark which compacted positions came from y (old_pos >= x_len)
        #    and record the original l-index within y for each y-token
        y_mask_new = torch.zeros((B, V, event_len_max), dtype=torch.bool, device=x.device)
        y_orig_l_new = torch.full((B, V, event_len_max), -1, dtype=torch.long, device=x.device)
        is_y = old_pos_idx >= x_len
        y_mask_new[b_idx[is_y], v_idx[is_y], new_pos_idx[is_y]] = True
        y_orig_l_new[b_idx[is_y], v_idx[is_y], new_pos_idx[is_y]] = (old_pos_idx[is_y] - x_len).long()
        
        return x_new, x_mark_new, x_mask_new, y_mask_new, y_orig_l_new
    
    def _relocate_events_stage2(self, x, x_mark, x_mask, y_mask, y_orig_l):
        """
        x: (B, V, L, D)
        x_mark: (B, V, L)
        x_mask: (B, V, L)
        y_mask: (B, V, L), bool mask indicating which positions in stage1 output came from y
        y_orig_l: (B, V, L), original l-index in y for each y-token (-1 for non-y tokens)
        """
        B, V, L, D = x.shape
        
        # 1. Reshape x_mark to add variate information
        x = x.reshape(B, V*L, D) # B, V*L, D
        x_mask = x_mask.reshape(B, V*L) # B, V*L
        y_mask = y_mask.reshape(B, V*L) # B, V*L
        y_orig_l = y_orig_l.reshape(B, V*L) # B, V*L
        # v_flat[i] = variate index for flat position i in V*L
        v_flat = torch.arange(V, device=x.device).unsqueeze(1).expand(V, L).reshape(V * L) # V*L
        x_mark = x_mark + torch.arange(V, device=x_mark.device).unsqueeze(0).unsqueeze(2) # B, V, L
        x_mark = x_mark.reshape(B, V*L) # B, V*L
        
        # 2. Get the maximum length for each batch
        batch_len_max = int(x_mask.sum(dim=1).max().item())
        
        # 3. New allocation for the input events, with shape (B, V, event_len_max, D)
        x_new = x.new_zeros((B, batch_len_max , D)) # B, V, event_len_max, D
        x_mask_new = torch.zeros((B, batch_len_max ), dtype=torch.bool, device=x.device) # B, V, event_len_max
        x_mark_new = x_mark.new_zeros((B, batch_len_max )) # B, V, event_len_max
        
        # 4. For each True position in mask_m, compute its new position after compaction
        #    Example: [False, True, True, False, True, True]
        #    ranks:   [-1,    0,    1,    1,     2,    3]
        ranks = x_mask.long().cumsum(dim=1) - 1
        
        # 5. Get indices of selected tokens
        b_idx, old_pos_idx = x_mask.nonzero(as_tuple=True)
        new_pos_idx = ranks[b_idx, old_pos_idx] # (num_selected_tokens,)
        
        # 6. Copy selected tokens from x to x_new and re-generate the mask
        x_new[b_idx, new_pos_idx] = x[b_idx, old_pos_idx]
        x_mask_new[b_idx, new_pos_idx] = True
        x_mark_new[b_idx, new_pos_idx] = x_mark[b_idx, old_pos_idx]
        
        # 7. Build y-position map: propagate y tracking and original (l, v) indices through stage2 compaction
        y_mask_new = torch.zeros((B, batch_len_max), dtype=torch.bool, device=x.device)
        y_orig_l_new = torch.full((B, batch_len_max), -1, dtype=torch.long, device=x.device)
        y_orig_v_new = torch.full((B, batch_len_max), -1, dtype=torch.long, device=x.device)
        is_y = y_mask[b_idx, old_pos_idx]
        y_mask_new[b_idx[is_y], new_pos_idx[is_y]] = True
        y_orig_l_new[b_idx[is_y], new_pos_idx[is_y]] = y_orig_l[b_idx[is_y], old_pos_idx[is_y]]
        y_orig_v_new[b_idx[is_y], new_pos_idx[is_y]] = v_flat[old_pos_idx[is_y]]
        
        return x_new, x_mark_new, x_mask_new, y_mask_new, y_orig_l_new, y_orig_v_new

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        y_mark: Tensor,
        y_mask: Tensor,
    ) -> Tensor:
        
        n_variates = x.shape[2]
        x_len = x_mark.shape[1]
        
        # 1. Embedding inputs and queries
        x = self._build_event_embedding(x, x_mark, x_mask)
        y = self._build_query_embedding(y_mark, y_mask)
        
        # 2. Concatenate x and y along the sequence dimension to get the input for encoder
        new_x = torch.cat([x, y], dim=1) # B, L+L', V, D
        new_x_mark = torch.cat([x_mark, y_mark], dim=1) # B, L+L', 1
        new_x_mask = torch.cat([x_mask, y_mask], dim=1) # B, L+L', V
        
        # 2. Stage 1
        x, x_mark, x_mask, y_mask_s1, y_orig_l_s1 = self._relocate_events_stage1(new_x, new_x_mark, new_x_mask, x_len)
        # x: B, V, L, D
        # x_mark: B, V, L
        # x_mask: B, V, L
        # y_mask_s1: B, V, L (True where position came from y)
        # y_orig_l_s1: B, V, L (original l-index in y for y-tokens)
        for blk in self.stage1_encoder:
            x = blk(x, x_mark, x_mask) # B, V, L, D
        
        # 3. Stage 2
        x, x_mark, x_mask, y_mask_s2, y_orig_l_s2, y_orig_v_s2 = self._relocate_events_stage2(x, x_mark, x_mask, y_mask_s1, y_orig_l_s1)
        # x: B, N, D
        # x_mark: B, N
        # x_mask: B, N
        # y_mask_s2: B, N (True where position came from y)
        # y_orig_l_s2: B, N (original l-index in y for y-tokens)
        # y_orig_v_s2: B, N (original v-index in y for y-tokens)
        for blk in self.stage2_encoder:
            x = blk(x, x_mark, x_mask) # B, N, D

        # 4. Scatter y-tokens from x back to (B, L', V, D) using tracked original positions
        B = x.shape[0]
        L_prime = y_mark.shape[1]
        decoded = x.new_zeros((B, L_prime, n_variates, self.d_model))
        b_idx_y, n_idx_y = y_mask_s2.nonzero(as_tuple=True)
        if b_idx_y.numel() > 0:
            l_idx_y = y_orig_l_s2[b_idx_y, n_idx_y]
            v_idx_y = y_orig_v_s2[b_idx_y, n_idx_y]
            decoded[b_idx_y, l_idx_y, v_idx_y] = x[b_idx_y, n_idx_y]

        outputs = self.output_projection(decoded).squeeze(-1)
        return outputs #* y_mask.to(outputs.dtype)