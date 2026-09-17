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
        
        predictions = self.model(
            x,
            x_mark,
            x_mask,
            y_mark,
            y_mask,
            current_epoch=kwargs.get("current_epoch"),
        )

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


def blend_log_gate(log_gate: Tensor, predicted_weight: float) -> Tensor:
    """Blend a predicted gate with an always-open global gate in probability space."""
    if predicted_weight <= 0.0:
        return torch.zeros_like(log_gate)
    if predicted_weight >= 1.0:
        return log_gate
    return torch.logaddexp(
        log_gate + math.log(predicted_weight),
        torch.full_like(log_gate, math.log1p(-predicted_weight)),
    )


class PerVariateLinear(nn.Module):
    def __init__(self, n_variates: int, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(n_variates, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(n_variates, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.in_features ** -0.5
        nn.init.normal_(self.weight, mean = 0.0, std = std)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...vi,voi->...vo", x, self.weight) + self.bias


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
                
        for module in [
            self.window_predictor if hasattr(self, "window_predictor") else None,
            self.variate_predictor if hasattr(self, "variate_predictor") else None,
        ]:
            if module is None:
                continue
            for i, m in enumerate(module):
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
            
    def forward(
        self,
        x,
        x_mark,
        x_mask,
        tau_time=0.1,
        eps=1e-6,
        tau_variate=0.1,
        query_mask=None,
        predicted_gate_weight=1.0,
    ):
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
        x_mask = x_mask.bool()
        if query_mask is None:
            query_mask = torch.zeros_like(x_mask)
        elif query_mask.dim() == 3:
            query_mask = query_mask.squeeze(-1)
        query_mask = query_mask.bool() & x_mask

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
                # valid_time_pred = time_pred[x_mask]
                # print(
                #     valid_time_pred.shape,
                #     valid_time_pred.max(),
                #     valid_time_pred.min(),
                #     valid_time_pred.median(),
                # )
                
                # Third, compute the left and right boundaries for each event
                left_boundary = (event_time - time_pred).unsqueeze(-1)  # B, N, 1
                right_boundary = event_time.unsqueeze(-1)  # B, N, 1
                key_time = event_time.unsqueeze(1)  # B, 1, N
                
                # Compute the soft time gate.
                left_time_bias = F.logsigmoid(
                    (key_time - left_boundary) / max(tau_time, eps)
                )  # B, N, N
                right_time_bias = F.logsigmoid(
                    (right_boundary - key_time) / max(tau_time, eps)
                )  # B, N, N
                attention_bias = blend_log_gate(
                    left_time_bias + right_time_bias,
                    predicted_gate_weight,
                )  # B, N, N
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
                variate_logits = self.variate_predictor(x) # self.variate_predictor_norm(x))  # B, N, V
                key_variate_index = source_variate.unsqueeze(1).expand(-1, N, -1)  # B, N, N

                selected_variate_logits = variate_logits.gather(
                    dim=-1,
                    index=key_variate_index
                )  # B, N, N
                variate_bias = F.logsigmoid(
                    selected_variate_logits / max(tau_variate, eps)
                )
                attention_bias = attention_bias + blend_log_gate(
                    variate_bias,
                    predicted_gate_weight,
                )  # B, N, N
                pair_mask = x_mask.unsqueeze(-1) & x_mask.unsqueeze(1)  # B, N, N
            elif self.stage == 3:
                # For stage 3, we allow attention between all events, regardless of variate
                pair_mask = x_mask.unsqueeze(-1) & x_mask.unsqueeze(1)  # B, N, N
            else:
                raise ValueError(f"Unsupported stage: {self.stage}")

            # Query tokens act only as queries; all attention keys are real events.
            pair_mask = pair_mask & ~query_mask.unsqueeze(1)
            row_has_key = pair_mask.any(dim=-1)

            # Ensure that each row has at least one True value to avoid empty rows in attention
            hard_mask = ensure_non_empty_rows(pair_mask)
            attention_bias = attention_bias.masked_fill(~hard_mask, float("-10000"))
            # print(attention_bias)
            attention_bias = attention_bias.unsqueeze(1).expand(
                -1, self.n_heads, -1, -1
            )  # B, H, N, N

            # Self-attention with scaled dot-product attention
            q, k, v = self.qkv(x).chunk(3, dim=-1) # B, N, D each
            q = q.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
            k = k.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
            v = v.reshape(B, N, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3) # B, H, N, D_head
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attention_bias,
            ).permute(0, 2, 1, 3).reshape(B, N, D)
            attention_output = self.out(attention_output) * row_has_key.unsqueeze(-1)
            x = self.attn_norm(x + self.droppath1(attention_output, mask=x_mask)) # B, N, D

        ffn_output = self.down(self.act(self.up(x)))
        x = self.ffn_norm(x + self.droppath2(ffn_output, mask=x_mask)) # B, N, D
        
        return x



class IMTS_SubModel(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(IMTS_SubModel, self).__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.n_variates = configs.enc_in
        self.gate_warmup_epochs = configs.gate_warmup_epochs

        self.value_encoder = nn.Sequential(
            PerVariateLinear(self.n_variates, 1, self.d_model),
            nn.GELU(),
            PerVariateLinear(self.n_variates, self.d_model, self.d_model),
        )
        self.time_embedding = nn.Linear(self.d_model, self.d_model)
        self.register_buffer(
            "time_divisor",
            torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / self.d_model)
            ),
        )
        self.time_gap_embedding = nn.Linear(self.d_model, self.d_model)
        self.frequency_embedding = nn.Linear(self.d_model, self.d_model)
        self.variate_embedding = nn.Parameter(torch.empty(self.n_variates, self.d_model))
        self.query = nn.Parameter(torch.rand(1, 1, self.n_variates, self.d_model))
        self.value_norm = nn.LayerNorm(self.d_model)
        self.time_norm = nn.LayerNorm(self.d_model)
        self.missingness_norm = nn.LayerNorm(self.d_model)
        self.variate_norm = nn.LayerNorm(self.d_model)
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
        
        # self.output_norm_0 = nn.LayerNorm(self.d_model)
        # self.output_norm_1 = nn.LayerNorm(self.d_model)
        # self.output_norm_2 = nn.LayerNorm(self.d_model)
        # self.output_norm_3 = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Sequential(
            PerVariateLinear(self.n_variates, self.d_model*4, self.d_model),
            nn.GELU(),
            PerVariateLinear(self.n_variates, self.d_model, 1),
            # nn.Linear(self.d_model*4, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        model_std = self.d_model ** -0.5
        nn.init.normal_(self.variate_embedding, mean=0.0, std=model_std)
        nn.init.normal_(self.query, mean=0.0, std=model_std)
        for module in self.value_encoder:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.time_gap_embedding.reset_parameters()
        self.frequency_embedding.reset_parameters()
        self.time_embedding.reset_parameters()
        self.value_norm.reset_parameters()
        self.time_norm.reset_parameters()
        self.variate_norm.reset_parameters()
        self.missingness_norm.reset_parameters()
        self.event_norm.reset_parameters()
        for block in self.stage0_encoder:
            block.reset_parameters()
        for block in self.stage1_encoder:
            block.reset_parameters()
        for block in self.stage2_encoder:
            block.reset_parameters()
        # self.output_norm_0.reset_parameters()
        # self.output_norm_1.reset_parameters()
        # self.output_norm_2.reset_parameters()
        # self.output_norm_3.reset_parameters()
        for layer in self.output_projection:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _get_observation_history(self, observation_mask: Tensor) -> tuple[Tensor, Tensor]:
        sequence_length = observation_mask.shape[1]
        positions = torch.arange(
            sequence_length,
            device=observation_mask.device,
            dtype=torch.long,
        ).view(1, sequence_length, 1)
        # Shift cumulative observation indices by one step to exclude the current value.
        last_observation_positions = torch.where(
            observation_mask,
            positions,
            -1,
        ).cummax(dim=1).values
        previous_observation_positions = torch.cat(
            [
                torch.full_like(last_observation_positions[:, :1], -1),
                last_observation_positions[:, :-1],
            ],
            dim=1,
        )
        return positions, previous_observation_positions

    def _build_missingness_embedding(self, input_mark, observation_mask, observation_history=None):
        """Embed previous-observation gaps and unit-duration/count per variable."""
        observation_mask = observation_mask > 0
        time_axis = input_mark.expand_as(observation_mask)
        _, previous_observation_positions = (
            observation_history
            if observation_history is not None
            else self._get_observation_history(observation_mask)
        )

        has_previous_observation = previous_observation_positions >= 0
        previous_observation_time = time_axis.gather(
            dim=1,
            index=previous_observation_positions.clamp_min(0),
        )
        time_since_previous = torch.where(
            has_previous_observation,
            (time_axis - previous_observation_time).clamp_min(0),
            torch.ones_like(time_axis),
        )
        time_gap_embed = self.time_gap_embedding(
            self._build_sinusoidal_embedding(time_since_previous.unsqueeze(-1))
        )
        observation_count = observation_mask.sum(dim=1, keepdim=True)
        frequency = observation_count.clamp_min(1).to(time_axis.dtype).reciprocal()
        frequency_embed = self.frequency_embedding(
            self._build_sinusoidal_embedding(frequency.unsqueeze(-1))
        )
        
        missingness_embedding = (time_gap_embed + frequency_embed) / 2
        return missingness_embedding

    def _build_sinusoidal_embedding(self, time_marks: Tensor) -> Tensor:
        time_angles = time_marks * (2 * math.pi) * self.time_divisor
        return torch.stack(
            [torch.sin(time_angles), torch.cos(time_angles)],
            dim=-1,
        ).flatten(start_dim=-2)[..., :self.d_model]

    def _build_time_embedding(self, time_marks: Tensor) -> Tensor:
        return self.time_embedding(self._build_sinusoidal_embedding(time_marks))

    def _build_value_embedding(self, x, y_mask):
        # Encode the input values
        x_value_embed = self.value_encoder(x.unsqueeze(-1))
        # Embed the pseudo values
        y_pseudo_value_embed = self.query * y_mask.unsqueeze(-1)
        return torch.cat([x_value_embed, y_pseudo_value_embed], dim=1)

    def _build_event_embedding(self, x, x_mark, x_mask, y_mark, y_mask):
        input_mark = torch.cat([x_mark, y_mark], dim=1) # B, L+L', 1

        # 1. Generate the value embedding
        value_embed = self._build_value_embedding(x, y_mask)
        
        # 2. Generate the time embedding
        time_embed = self._build_time_embedding(input_mark).unsqueeze(2)
        
        # 3. Generate the missingness embedding
        observation_mask = torch.cat([x_mask > 0, torch.zeros_like(y_mask, dtype=torch.bool)], dim=1)
        observation_history = self._get_observation_history(observation_mask)
        missingness_embed = self._build_missingness_embedding(
            input_mark,
            observation_mask,
            observation_history,
        )
        
        # 4. Generate the variate embedding
        variate_embed = self.variate_embedding.view(1, 1, self.n_variates, self.d_model)

        # 5. Combine all embeddings to form the final event embedding.
        event_embed = self.value_norm(value_embed) + \
                      self.time_norm(time_embed) + \
                      self.variate_norm(variate_embed) + \
                      self.missingness_norm(missingness_embed)
        return self.event_norm(event_embed)
    
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

    def _decode_query_tokens(
        self,
        events: Tensor,
        query_mask: Tensor,
        y_orig_l: Tensor,
        y_orig_v: Tensor,
        pred_len: int,
    ) -> Tensor:
        """Scatter packed query tokens back to prediction-time and variable axes."""
        decoded = events.new_zeros(
            (events.shape[0], pred_len, self.n_variates, self.d_model)
        )
        batch_index, event_index = query_mask.nonzero(as_tuple=True)
        if batch_index.numel() > 0:
            decoded[
                batch_index,
                y_orig_l[batch_index, event_index],
                y_orig_v[batch_index, event_index],
            ] = events[batch_index, event_index]
        return decoded

    def _aggregate_batch_context(
        self,
        events: Tensor,
        events_mask: Tensor,
        query_mask: Tensor,
    ) -> Tensor:
        """Mean-pool all valid encoded x tokens within each batch sample."""
        historical_mask = events_mask.squeeze(-1).bool() & ~query_mask
        weights = historical_mask.to(events.dtype)
        return (events * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)

    def _get_predicted_gate_weight(self, current_epoch: int | None) -> float:
        if current_epoch is None or self.gate_warmup_epochs <= 0:
            return 1.0
        return min(max((current_epoch + 1) / self.gate_warmup_epochs, 0.0), 1.0)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        y_mark: Tensor,
        y_mask: Tensor,
        current_epoch: int | None = None,
    ) -> Tensor:
        x_len = x_mark.shape[1]
        original_y_mask = y_mask
        predicted_gate_weight = self._get_predicted_gate_weight(current_epoch)
        
        # 1. Embedding inputs and queries
        events = self._build_event_embedding(x, x_mark, x_mask, y_mark, y_mask)
        events_mark = torch.cat([x_mark, y_mark], dim=1) # B, L+L', 1
        events_mask = torch.cat([x_mask, y_mask], dim=1) # B, L+L', V
        
        # 2. Relocate events once after embedding.
        events, events_mark, events_mask, query_mask, y_orig_l, y_orig_v = self._relocate_events(
            events, events_mark, events_mask, x_len
        )
        # events: B, N, D
        # events_mark: B, N, 1
        # events_mask: B, N, 1
        # query_mask: B, N (True where position came from y)
        # y_orig_l: B, N (original l-index in y for y-tokens)
        # y_orig_v: B, N (original v-index in y for y-tokens)

        # 3. Capture query representations after each hierarchical stage.
        multiscale_queries = []
        
        for blk in self.stage0_encoder:
            events = blk(
                events, 
                events_mark, 
                events_mask, 
                query_mask=query_mask
            ) # B, N, D
        multiscale_queries.append(
            self._decode_query_tokens(
                events, 
                query_mask, 
                y_orig_l, 
                y_orig_v, 
                y_mark.shape[1]
            )
        )

        for blk in self.stage1_encoder:
            events = blk(
                events,
                events_mark,
                events_mask,
                tau_time=0.1,
                query_mask=query_mask,
                predicted_gate_weight=predicted_gate_weight,
            ) # B, N, D
        multiscale_queries.append(
            self._decode_query_tokens(
                events, 
                query_mask, 
                y_orig_l, 
                y_orig_v, 
                y_mark.shape[1]
            )
        )

        for blk in self.stage2_encoder:
            events = blk(
                events,
                events_mark,
                events_mask,
                tau_time=0.1,
                tau_variate=0.1,
                query_mask=query_mask,
                predicted_gate_weight=predicted_gate_weight,
            ) # B, N, D
        multiscale_queries.append(
            self._decode_query_tokens(
                events, 
                query_mask, 
                y_orig_l, 
                y_orig_v, 
                y_mark.shape[1]
            )
        )

        # 4. Append a global context pooled from every valid encoded x token.
        batch_context = self._aggregate_batch_context(
            events,
            events_mask,
            query_mask,
        ).view(events.shape[0], 1, 1, self.d_model)
        
        multiscale_queries.append(
            batch_context.expand(-1, y_mark.shape[1], self.n_variates, -1)
        )

        # 5. Concatenate stage-wise query features and global context for decoding.
        decoded = torch.cat(multiscale_queries, dim=-1)
        outputs = self.output_projection(decoded).squeeze(-1)
        return outputs * original_y_mask