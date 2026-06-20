import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # data
    data_path: str = "data/stranger.txt"

    # model
    vocab_size: int = 93       # character-level: 65 unique chars in Shakespeare (93 in stranger)
    block_size: int = 256      # max sequence length (context window)
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding dimension

    # training
    max_steps: int = 200 # 5000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20

    # clearml
    use_clearml: bool = False

    @classmethod
    def from_args(cls) -> "Config":
        defaults = cls()
        parser = argparse.ArgumentParser(description="Train a GPT model from scratch")
        parser.add_argument("data_path", nargs="?", default=defaults.data_path, help="Path to training data file")
        parser.add_argument("--max_steps", type=int, default=defaults.max_steps, help="Number of training steps")
        parser.add_argument("--batch_size", type=int, default=defaults.batch_size, help="Batch size for training")
        parser.add_argument("--n_layer", type=int, default=defaults.n_layer, help="Number of transformer layers")
        parser.add_argument("--n_head", type=int, default=defaults.n_head, help="Number of attention heads")
        parser.add_argument("--n_embd", type=int, default=defaults.n_embd, help="Embedding dimension")
        parser.add_argument("--block_size", type=int, default=defaults.block_size, help="Context window size")
        parser.add_argument("--clearml", action="store_true", help="Offload training to ClearML")
        args = parser.parse_args()

        return cls(
            data_path=args.data_path,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            use_clearml=args.clearml,
        )
