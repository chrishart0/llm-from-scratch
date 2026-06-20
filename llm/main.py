from dataclasses import dataclass

from rich import print

from llm.model import GPT
from llm.tokenizer import Tokenizer
from llm.trainer import Trainer


@dataclass
class GPTConfig:
    vocab_size: int = 93       # character-level: 65 unique chars in Shakespeare (93 in stranger)
    block_size: int = 256      # max sequence length (context window)
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding dimension
    max_steps=5000,
    batch_size=64,
    use_clearml=False


def main(args):
  with open(args.data_path).read() if args.data_path else open("data/stranger.txt").read() as f:
    text = f.read()

  tok = Tokenizer(text)
  config = GPTConfig()
  model = GPT(config)
  trainer = Trainer(model, text, config)


  print(f"\nLLM from scratch. We have {len(tok)} tokens.")
  # hello_vec = tok.char_encode("Hello")
  # print("\nChar enc/dec")
  # print(hello_vec)
  # print(tok.char_decode(hello_vec))
  # print("\nTok enc/dec")
  # print(tok.tok_encode("Hello"))
  # print(tok.tok_decode(tok.tok_encode("Hello")))

  n_params = sum(p.numel() for p in model.parameters())
  print(f"\nModel has {n_params/1e6:.2f}M params")
  print("\nModel:")
  print(model)

  trainer.train()


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Train a GPT model from scratch")
  parser.add_argument(
    "data_path",
    nargs="?",
    default="data/shakespeare.txt",
    help="Path to training data file"
  )
  parser.add_argument("--max_steps", type=int, default=5000, help="Number of training steps")
  parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
  parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
  parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
  parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
  parser.add_argument("--block_size", type=int, default=256, help="Context window size")
  parser.add_argument("--local", action="store_true", help="Run locally without ClearML")
  args = parser.parse_args()

  main(args)
