import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import math


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        return F.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.eps)


class RepHead(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int, use_ln: bool = True, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.ln = RMSNorm(hidden_dim) if use_ln else nn.Identity()

        self.down = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(proj_dim, hidden_dim, bias=False)

        if self.use_gate:
            self.dynamic_gate = nn.Sequential(
                nn.Linear(hidden_dim, 64, bias=False),
                RMSNorm(64),
                nn.SiLU(),
                nn.Linear(64, 1)
            )

            last_linear = self.dynamic_gate[-1]
            if last_linear.bias is not None:
                nn.init.constant_(last_linear.bias, 0.0)
            nn.init.xavier_normal_(last_linear.weight)

        nn.init.zeros_(self.up.weight)
        nn.init.kaiming_normal_(self.down.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        norm_x = self.ln(x)
        delta = self.up(self.act(self.down(norm_x)))

        if self.use_gate:
            sig_score = torch.sigmoid(self.dynamic_gate(x))
            return delta, sig_score
        else:
            return delta, None


class LayerRepAdapter(nn.Module):

    def __init__(self, dim: int, proj_dim: int, use_ln: bool = True, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.head = RepHead(dim, proj_dim, use_ln, use_gate=use_gate)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        delta, sig_score = self.head(x)

        if self.use_gate:
            mask = (sig_score >= 0.5).float()

            if self.training:
                gate_score = mask - sig_score.detach() + sig_score
            else:
                gate_score = mask

            return delta * gate_score * self.scale, gate_score, sig_score
        else:
            return delta * self.scale, None, None


class ExternalRepHeadModel(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            layers: List[int],
            proj_dim: int = 512,
            use_ln: bool = True,
            use_rep_heads: bool = True,
            use_gate: bool = True,
            device: str = 'cuda',
            intervention_strategy: str = "all",
            reft_p: int = 2,
            reft_s: int = 6
    ):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.device = device
        self.use_rep_heads = use_rep_heads
        self.use_gate = use_gate

        hidden_dim = backbone.config.hidden_size

        self.rep_adapters = nn.ModuleDict({
            str(l): LayerRepAdapter(hidden_dim, proj_dim, use_ln, use_gate=use_gate)
            for l in layers
        })

        self.to(self.device)
        self._register_hooks()

        self.intervention_strategy = intervention_strategy
        self.reft_p = reft_p
        self.reft_s = reft_s
        self.current_intervention_boundary = None

    def set_intervention_boundary(self, boundary: Union[int, List[int], torch.Tensor, None]):
        self.current_intervention_boundary = boundary

    def _make_hook(self, layer_id: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if not self.use_rep_heads:
                module._last_ffn_out = hidden_states
                return output

            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device

            if self.current_intervention_boundary is not None:
                if isinstance(self.current_intervention_boundary, int):
                    bounds = torch.full((batch_size,), self.current_intervention_boundary, device=device,
                                        dtype=torch.long)
                elif isinstance(self.current_intervention_boundary, (list, tuple)):
                    bounds = torch.tensor(self.current_intervention_boundary, device=device, dtype=torch.long)
                elif isinstance(self.current_intervention_boundary, torch.Tensor):
                    bounds = self.current_intervention_boundary.to(device)
                else:
                    bounds = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
                bounds = torch.clamp(bounds, max=seq_len)
            else:
                bounds = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)

            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            bounds_col = bounds.unsqueeze(1)

            intervention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

            if self.intervention_strategy == "reft_p_s":
                p = self.reft_p
                s = self.reft_s
                if p > 0:
                    prefix_mask = positions < p
                    intervention_mask = intervention_mask | prefix_mask
                if s > 0:
                    start_s = (bounds_col - s).clamp(min=0)
                    suffix_mask = (positions >= start_s) & (positions < bounds_col)
                    intervention_mask = intervention_mask | suffix_mask

            elif self.intervention_strategy == "all":
                intervention_mask = positions < bounds_col

            module._last_basis_weights = None
            module._last_sig_scores = None

            if self.training:
                modified_hidden_states = hidden_states.clone()
            else:
                modified_hidden_states = hidden_states

            adapter = self.rep_adapters[layer_id]
            selected_states = modified_hidden_states[intervention_mask]

            if selected_states.shape[0] > 0:
                delta, weights, raw_sig_scores = adapter(selected_states)

                origin_dtype = modified_hidden_states.dtype
                modified_hidden_states_fp32 = modified_hidden_states.float()
                modified_hidden_states_fp32[intervention_mask] = (selected_states + delta).float()
                modified_hidden_states = modified_hidden_states_fp32.to(origin_dtype)

                if self.use_gate and weights is not None:
                    rank = weights.shape[-1]
                    full_weights = torch.zeros(
                        (batch_size, seq_len, rank),
                        dtype=weights.dtype,
                        device=device
                    )
                    full_weights[intervention_mask] = weights
                    module._last_basis_weights = full_weights

                    full_sig_scores = torch.zeros(
                        (batch_size, seq_len, 1),
                        dtype=raw_sig_scores.dtype,
                        device=device
                    )
                    full_sig_scores[intervention_mask] = raw_sig_scores
                    module._last_sig_scores = full_sig_scores

            module._last_ffn_out = modified_hidden_states

            if isinstance(output, tuple):
                return (modified_hidden_states,) + output[1:]
            else:
                return modified_hidden_states

        return hook

    def _register_hooks(self):
        if hasattr(self.backbone, "model"):
            model_layers = self.backbone.model.layers
        elif hasattr(self.backbone, "transformer"):
            model_layers = self.backbone.transformer.h
        else:
            raise ValueError("Unsupported backbone architecture")

        for layer_idx in self.layers:
            layer_module = model_layers[layer_idx]
            hook_fn = self._make_hook(str(layer_idx))
            layer_module.register_forward_hook(hook_fn)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    def train(self, mode=True):
        self.rep_adapters.train(mode)
        return super().train(mode)

    @property
    def config(self):
        return self.backbone.config

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)


@contextlib.contextmanager
def _rep_heads(model: ExternalRepHeadModel, enabled: bool):
    old = getattr(model, "use_rep_heads", True)
    setattr(model, "use_rep_heads", enabled)
    try:
        yield model
    finally:
        setattr(model, "use_rep_heads", old)