from typing import Optional
from dataclasses import dataclass, field

@dataclass
class VocabConfig:
    vocab_size: int = 10000
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    vocab_path: str = "model/ts_vocab.json"
    merges_path: str = "model/ts_merges.txt"

@dataclass
class DataConfig:
    train_data_path: str = "data/TinyStoriesV2-GPT4-train.txt"
    train_bin_path: str = "data/TinyStoriesV2-GPT4-train.bin"
    valid_data_path: str = "data/TinyStoriesV2-GPT4-valid.txt"
    valid_bin_path: str = "data/TinyStoriesV2-GPT4-valid.bin"
    eval_batches: int = 10
    context_length: int = 256
    np_dtype: str = "uint16"

@dataclass
class ModelConfig:
    d_model: int = 512
    d_ff: int = 1344
    rope_theta: int = 10000
    num_layer: int = 4
    num_heads: int = 16

@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-2
    weight_decay: float = 0.01
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 1e-8
    max_learning_rate: float = 1e-2
    min_learning_rate: float = 1e-3
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 8000
    max_l2_norm: float = 1.0

@dataclass
class TrainingConfig:
    max_step: int = 10000
    batch_size: int = 16
    runs_dir: str = "runs/ts"
    train_log_step: int = 20
    eval_log_step: int = 20
    seed: int = 42

@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "cs336-a1"
    name: str = "train-ts"

@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def get_default_config():
    config = TrainConfig()
    return config
