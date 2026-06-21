from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams

@dataclass
class OREHyperParams(HyperParams):
    model_name: str
    unfreeze_layers: List[int]
    epochs: int
    batch_size: int
    lr: float
    lr_min: float

    kl_coef: float
    ce_coef: float
    orth_coef: float

    unrel_rank: int
    proj_dim: int

    orth_method: str