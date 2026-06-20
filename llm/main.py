from dataclasses import dataclass

from rich import print

from llm.model import GPT
from llm.tokenizer import Tokenizer


@dataclass
class GPTConfig:
    vocab_size: int = 93       # character-level: 65 unique chars in Shakespeare (93 in stranger)
    block_size: int = 256      # max sequence length (context window)
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding dimension


def main():
  text = open("data/stranger.txt").read()

  tok = Tokenizer(text)

  print(f"\nLLM from scratch. We have {len(tok)} tokens.")
  hello_vec = tok.char_encode("Hello")

  print("\nChar enc/dec")
  print(hello_vec)
  print(tok.char_decode(hello_vec))
  print("\nTok enc/dec")
  print(tok.tok_encode("Hello"))
  print(tok.tok_decode(tok.tok_encode("Hello")))

  config = GPTConfig()
  model = GPT(config)

  n_params = sum(p.numel() for p in model.parameters())
  print(f"\nModel has {n_params/1e6:.2f}M params")
  print("\nModel:")
  print(model)


if __name__ == "__main__":
  main()
