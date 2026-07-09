import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

# CHORD: Hirar"CH"ical Missingness-aware Transf"OR"mer with "D"ynamic 
# Attention Scope for Irregular Multivariate Time Series Forecasting

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


class StageEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_variates: int,
        dropout: float,
        stage: int,
    ):
        super(StageEncoder, self).__init__()
        
        self.stage = stage
        self.n_variates = n_variates
        
        if self.stage in [1, 2]:
            self.window_predictor = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            
        if self.stage in [2]:
            self.variate_predictor = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_variates),
            )
        
        if self.stage in [1, 2, 3]:
            self.attn_norm = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, d_model * 3)
            self.n_heads = n_heads
            self.out = nn.Linear(d_model, d_model)
            self.droppath1 = DropPath(drop_prob=dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.droppath2 = DropPath(drop_prob=dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [
            self.window_predictor if hasattr(self, "window_predictor") else None,
            self.variate_predictor if hasattr(self, "variate_predictor") else None,
            self.attn_norm if hasattr(self, "attn_norm") else None,
            self.qkv if hasattr(self, "qkv") else None,
            self.out if hasattr(self, "out") else None,
            self.ffn_norm if hasattr(self, "ffn_norm") else None,
            self.up if hasattr(self, "up") else None,
            self.down if hasattr(self, "down") else None,
        ]:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x, x_mark, x_mask, tau_time=0.1, eps=1e-6, tau_variate=0.1):
        """
        Forward pass for a stage encoder.

        Args:
            x: Input tensor of shape (B, N, D).
            x_mark: Packed time mark tensor of shape (B, N) or (B, N, 1),
                encoded as variate_index + timestamp.
            x_mask: Mask tensor of shape (B, N) or (B, N, 1).

        Returns:
            Encoded tensor of shape (B, N, D).
        """
        B, N, D = x.shape
        if x_mark.dim() == 3:
            x_mark = x_mark.squeeze(-1)
        if x_mask.dim() == 3:
            x_mask = x_mask.squeeze(-1)

        if self.stage != 0:
            if self.stage in [1, 2]:
                # First get the original event time by subtracting the variate index
                source_variate = torch.floor(x_mark).clamp(
                    min=0,
                    max=self.n_variates - 1
                ).long()  # B, N
                event_time = x_mark - source_variate.to(x_mark.dtype)  # B, N
                
                # Second, compute the observation time window for each event
                time_pred = torch.sigmoid(self.window_predictor(x)).squeeze(-1)  # B, N
                
                # Third, compute the left and right boundaries for each event
                left_boundary = (event_time - time_pred).unsqueeze(-1)  # B, N, 1
                right_boundary = event_time.unsqueeze(-1)  # B, N, 1
                key_time = event_time.unsqueeze(1)  # B, 1, N
                
                # Finally, compute the soft time gate (attention bias)
                left_time_bias = F.logsigmoid(
                    (key_time - left_boundary) / max(tau_time, eps)
                )  # B, N, N
                right_time_bias = F.logsigmoid(
                    (right_boundary - key_time) / max(tau_time, eps)
                )  # B, N, N
                attention_bias = left_time_bias + right_time_bias  # B, N, N
            else:
                attention_bias = torch.zeros((B, N, N), dtype=x.dtype, device=x.device)

            if self.stage == 1:
                # For stage 1, we only allow attention between events of the same variate
                pair_mask = (
                    x_mask.unsqueeze(-1)
                    & x_mask.unsqueeze(1)
                    & (source_variate.unsqueeze(-1) == source_variate.unsqueeze(1))
                )  # B, N, N
            elif self.stage == 2:
                # For stage 2, we allow attention between events of different variates, 
                # but we add a soft variate gate (variate-based bias)
                variate_logits = self.variate_predictor(x)  # B, N, V
                key_variate_index = source_variate.unsqueeze(1).expand(-1, N, -1)  # B, N, N
                selected_variate_logits = variate_logits.gather(
                    dim=-1,
                    index=key_variate_index
                )  # B, N, N
                attention_bias = attention_bias + F.logsigmoid(
                    selected_variate_logits / max(tau_variate, eps)
                )  # B, N, N
                pair_mask = x_mask.unsqueeze(-1) & x_mask.unsqueeze(1)  # B, N, N
            elif self.stage == 3:
                # For stage 3, we allow attention between all events, regardless of variate
                pair_mask = x_mask.unsqueeze(-1) & x_mask.unsqueeze(1)  # B, N, N
            else:
                raise ValueError(f"Unsupported stage: {self.stage}")

            # Ensure that each row has at least one True value to avoid empty rows in attention
            hard_mask = ensure_non_empty_rows(pair_mask)
            attention_bias = attention_bias.masked_fill(~hard_mask, float("-inf"))
            attention_bias = attention_bias.unsqueeze(1).expand(
                -1, self.n_heads, -1, -1
            )  # B, H, N, N

            # Self-attention with scaled dot-product attention
            q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1) # B, N, D each
            q = q.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
            k = k.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
            v = v.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
            x = self.droppath1(
                self.out(
                    torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_bias)
                    .permute(0, 2, 1, 3)
                    .reshape(B, N, D)
                ),
                mask=x_mask,
            ) + x

        x = self.droppath2(self.down(self.act(self.up(self.ffn_norm(x)))), mask=x_mask) + x # B, N, D
        
        return x



class IMTS_SubModel(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(IMTS_SubModel, self).__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.n_variates = configs.enc_in

        self.value_embedding = nn.Parameter(torch.empty(self.n_variates, self.d_model))
        self.value_embedding_act = nn.GELU()
        self.time_embedding = nn.Linear(1, self.d_model)
        self.time_embedding_act = nn.GELU()
        self.variate_embedding = nn.Parameter(torch.empty(self.n_variates, self.d_model))
        self.event_missing_embedding = nn.Embedding(2, self.d_model)
        self.temporal_missing_embedding = nn.Linear(1, self.d_model)
        self.missing_embedding_act = nn.GELU()
        self.query = nn.Parameter(torch.rand(1, 1, 1, self.d_model))
        self.event_norm = nn.LayerNorm(self.d_model)

        self.stage0_encoder = nn.ModuleList([
            StageEncoder(
                d_model=self.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                n_variates=self.n_variates,
                dropout=configs.dropout,
                stage=0,
            )
            for _ in range(configs.n_layers)
        ])
        self.stage1_encoder = nn.ModuleList([
            StageEncoder(
                d_model=self.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                n_variates=self.n_variates,
                dropout=configs.dropout,
                stage=1,
            )
            for _ in range(configs.n_layers)
        ])
        self.stage2_encoder = nn.ModuleList([
            StageEncoder(
                d_model=self.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                n_variates=self.n_variates,
                dropout=configs.dropout,
                stage=2,
            )
            for _ in range(configs.n_layers)
        ])
        self.stage3_encoder = nn.ModuleList([
            StageEncoder(
                d_model=self.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                n_variates=self.n_variates,
                dropout=configs.dropout,
                stage=3,
            )
            for _ in range(configs.n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        model_std = self.d_model ** -0.5
        nn.init.normal_(self.value_embedding, mean=0.0, std=model_std)
        nn.init.normal_(self.variate_embedding, mean=0.0, std=model_std)
        nn.init.normal_(self.event_missing_embedding.weight, mean=0.0, std=model_std)
        nn.init.normal_(self.query, mean=0.0, std=model_std)
        self.time_embedding.reset_parameters()
        self.temporal_missing_embedding.reset_parameters()
        for module in self.output_projection:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.event_norm.reset_parameters()
        for block in self.stage0_encoder:
            block.reset_parameters()
        for block in self.stage1_encoder:
            block.reset_parameters()
        for block in self.stage2_encoder:
            block.reset_parameters()
        for block in self.stage3_encoder:
            block.reset_parameters()
        self.output_norm.reset_parameters()
        for layer in self.output_projection:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _build_missingness_embedding(self, input_mark, input_mask):
        input_mask = input_mask > 0
        event_missing_embed = self.event_missing_embedding(input_mask.long())

        time_axis = input_mark.squeeze(-1).unsqueeze(-1).expand_as(input_mask)
        observed_time = torch.where(input_mask, time_axis, torch.zeros_like(time_axis))
        last_observed_time = torch.cummax(observed_time, dim=1).values
        has_history = input_mask.cumsum(dim=1) > 0
        temporal_gap = torch.where(has_history, time_axis - last_observed_time, time_axis)
        temporal_gap = temporal_gap.unsqueeze(-1)
        temporal_missing_embed = self.temporal_missing_embedding(temporal_gap)
        
        missingness_embed = self.missing_embedding_act(event_missing_embed + temporal_missing_embed)
        return missingness_embed

    def _build_event_embedding(self, x, x_mark, x_mask, y_mark, y_mask):
        batch_size, _, n_variates = x.shape
        
        # Embedding original values and query token
        value_embed = self.value_embedding_act(torch.einsum("bnv,vd->bnvd", x, self.value_embedding))
        value_embed  = torch.cat([value_embed, self.query.expand(batch_size, y_mark.shape[1], n_variates, -1)], dim=1) # B, L+L', V, D
        
        # Embedding time marks
        input_mark = torch.cat([x_mark, y_mark], dim=1) # B, L+L', 1
        time_embed = self.time_embedding_act(self.time_embedding(input_mark)).unsqueeze(2)
        
        # Embedding missingness
        input_mask = torch.cat([x_mask, y_mask], dim=1) # B, L+L', V
        missingness_embed = self._build_missingness_embedding(input_mark, input_mask)
        
        # Embedding variate
        variate_embed = self.variate_embedding.view(1, 1, n_variates, self.d_model)
        
        # Combine all embeddings and normalize
        event_embed = self.event_norm(value_embed + time_embed + variate_embed + missingness_embed)
        return event_embed
    
    def _relocate_events(self, events, events_mark, events_mask, x_len):
        """
        events: (B, L, V, D)
        events_mark: (B, L, 1)
        events_mask: (B, L, V)
        x_len: int, length of the original x portion (positions >= x_len belong to y)
        """
        B, total_len, V, D = events.shape

        # 1. Pack events into a single per-sample sequence and encode variate ids into the mark.
        events = events.permute(0, 2, 1, 3).reshape(B, V * total_len, D) # B, V*L, D
        events_mask = (events_mask.permute(0, 2, 1).reshape(B, V * total_len) == 1) # B, V*L
        variate_offset = torch.arange(V, device=events_mark.device, dtype=events_mark.dtype).view(1, V, 1)
        events_mark = (
            events_mark.reshape(B, 1, total_len).expand(-1, V, -1) + variate_offset
        ).reshape(B, V * total_len, 1) # B, V*L, 1

        # 2. Get the maximum packed length for each batch.
        batch_len_max = int(events_mask.sum(dim=1).max().item())

        # 3. Compact the valid events into a dense batch-major sequence.
        events_new = events.new_zeros((B, batch_len_max, D)) # B, L_M, D
        events_mask_new = torch.zeros((B, batch_len_max, 1), dtype=torch.bool, device=events.device) # B, L_M, 1
        events_mark_new = events_mark.new_zeros((B, batch_len_max, 1)) # B, L_M, 1

        # 4. For each True position in mask_m, compute its new position after compaction.
        ranks = events_mask.long().cumsum(dim=1) - 1

        # 5. Get indices of selected tokens.
        b_idx, old_pos_idx = events_mask.nonzero(as_tuple=True)
        new_pos_idx = ranks[b_idx, old_pos_idx] # (num_selected_tokens,)

        # 6. Copy selected tokens into the compacted tensors.
        events_new[b_idx, new_pos_idx] = events[b_idx, old_pos_idx]
        events_mask_new[b_idx, new_pos_idx, 0] = True
        events_mark_new[b_idx, new_pos_idx, 0] = events_mark[b_idx, old_pos_idx, 0]

        # 7. Keep y-token positions and their original (l, v) indices for decoding.
        y_mask_new = torch.zeros((B, batch_len_max), dtype=torch.bool, device=events.device)
        y_orig_l_new = torch.full((B, batch_len_max), -1, dtype=torch.long, device=events.device)
        y_orig_v_new = torch.full((B, batch_len_max), -1, dtype=torch.long, device=events.device)
        l_flat = torch.arange(total_len, device=events.device).repeat(V) # V*L
        v_flat = torch.arange(V, device=events.device).unsqueeze(1).expand(V, total_len).reshape(V * total_len) # V*L
        is_y = l_flat[old_pos_idx] >= x_len
        y_mask_new[b_idx[is_y], new_pos_idx[is_y]] = True
        y_orig_l_new[b_idx[is_y], new_pos_idx[is_y]] = (l_flat[old_pos_idx[is_y]] - x_len).long()
        y_orig_v_new[b_idx[is_y], new_pos_idx[is_y]] = v_flat[old_pos_idx[is_y]]
        
        return events_new, events_mark_new, events_mask_new, y_mask_new, y_orig_l_new, y_orig_v_new

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
        events = self._build_event_embedding(x, x_mark, x_mask, y_mark, y_mask)
        events_mark = torch.cat([x_mark, y_mark], dim=1) # B, L+L', 1
        events_mask = torch.cat([x_mask, y_mask], dim=1) # B, L+L', V
        
        # 2. Relocate events once after embedding.
        events, events_mark, events_mask, y_mask, y_orig_l, y_orig_v = self._relocate_events(
            events, events_mark, events_mask, x_len
        )
        # events: B, N, D
        # events_mark: B, N, 1
        # events_mask: B, N, 1
        # y_mask: B, N (True where position came from y)
        # y_orig_l: B, N (original l-index in y for y-tokens)
        # y_orig_v: B, N (original v-index in y for y-tokens)
        for blk in self.stage0_encoder:
            events = blk(events, events_mark, events_mask) # B, N, D
        for blk in self.stage1_encoder:
            events = blk(events, events_mark, events_mask, tau_time=1e-6) # B, N, D
        for blk in self.stage2_encoder:
            events = blk(events, events_mark, events_mask, tau_time=1e-6) # B, N, D
        for blk in self.stage3_encoder:
            events = blk(events, events_mark, events_mask) # B, N, D

        # 3. Scatter y-tokens from x back to (B, L', V, D) using tracked original positions
        B = events.shape[0]
        L_prime = y_mark.shape[1]
        decoded = events.new_zeros((B, L_prime, n_variates, self.d_model))
        b_idx_y, n_idx_y = y_mask.nonzero(as_tuple=True)
        if b_idx_y.numel() > 0:
            l_idx_y = y_orig_l[b_idx_y, n_idx_y]
            v_idx_y = y_orig_v[b_idx_y, n_idx_y]
            decoded[b_idx_y, l_idx_y, v_idx_y] = events[b_idx_y, n_idx_y]

        decoded = self.output_norm(decoded)
        outputs = self.output_projection(decoded).squeeze(-1)
        return outputs #* y_mask.to(outputs.dtype)