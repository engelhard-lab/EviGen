# config.py
from dataclasses import dataclass
from typing import Optional
import random
import numpy as np
import torch
import yaml


@dataclass
class Config:
    # Embedding / query / expert parameters
    emb_dim: int

    # Total number of queries (0 is shared note query)
    num_queries: int

    # The first num_note_queries are note queries (0 is shared)
    # The remaining num_code_queries are code queries
    # Must satisfy: num_note_queries + num_code_queries == num_queries
    num_note_queries: int
    num_code_queries: int

    k_retrieval: int
    attn_temperature: float

    # Retrieval sampling
    use_sampling_retrieval: bool
    init_temperature: float
    final_temperature: float
    temp_anneal_epochs: int  # Annealing happens per-batch over this many epochs worth of steps

    # Patient summary aggregator
    aggregator: str          # "mean" or "transformer"
    transformer_layers: int
    transformer_heads: int
    transformer_dim_ff: int
    dropout: float

    # Expert MLP
    expert_hidden_dim: int

    # Optimization
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    aux_loss_weight: float
    simplicity_loss_weight: float  # Weight for simplicity term in aux loss (diversity + alpha*simplicity)
    max_grad_norm: Optional[float]

    # Dataset
    note_parquet: str
    code_parquet: str

    # LanceDB (for visualization only)
    db_uri: str
    note_table: str
    code_table: str

    train_set_ratio: float
    random_state: int

    # Misc
    device: str
    seed: int

    # Static-IRIS ablation: if False, skip dynamic gating so all queries are
    # always active (mask_q = 1). Leaves threshold params unused but harmless.
    enable_dynamic_gating: bool = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config_from_yaml(path: str) -> Config:
    """
    Load configuration from a YAML file into a Config dataclass.

    Example:
        cfg = load_config_from_yaml("config.yaml")

    Notes:
    - You must ensure num_note_queries + num_code_queries == num_queries.
      For a notes-only setting, for example:
          num_queries: 8
          num_note_queries: 8
          num_code_queries: 0
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Construct Config strictly from YAML; missing fields will raise a TypeError
    cfg = Config(**data)

    # Device check
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[load_config_from_yaml] CUDA not available, fallback to CPU.")
        cfg.device = "cpu"

    # Query split checks
    if cfg.num_note_queries < 1:
        raise ValueError("Config.num_note_queries must be >= 1 (query 0 is shared note query).")

    if cfg.num_note_queries + cfg.num_code_queries != cfg.num_queries:
        raise ValueError(
            f"num_note_queries + num_code_queries must equal num_queries "
            f"({cfg.num_note_queries} + {cfg.num_code_queries} != {cfg.num_queries}). "
            f"For note-only, e.g., set num_note_queries={cfg.num_queries}, num_code_queries=0."
        )

    return cfg
