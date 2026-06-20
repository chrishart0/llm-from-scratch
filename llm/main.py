import os

import torch
from rich import print

from llm.config import Config
from llm.model import GPT
from llm.tokenizer import Tokenizer
from llm.trainer import Trainer


def _resolve_checkpoint(config):
  base = config.checkpoint
  if base is None:
    runs = sorted(d for d in os.listdir("trials") if os.path.isdir(os.path.join("trials", d))) if os.path.isdir("trials") else []
    if not runs:
      raise SystemExit("No runs found in trials/. Train a model first with `uv run train`.")
    base = runs[-1]

  run_dir = os.path.join("trials", base)
  for name in ("checkpoint_lowest_loss.pt", "checkpoint_final.pt"):
    path = os.path.join(run_dir, name)
    if os.path.exists(path):
      return path
  raise SystemExit(f"No checkpoint found in {run_dir}/")


def train():
  config = Config.from_args()
  text = open(config.data_path).read()
  tokenizer = Tokenizer(text, encoder=config.encoder)
  config.vocab_size = tokenizer.vocab_size
  model = GPT(config)
  trainer = Trainer(model, tokenizer, text, config)

  trainer.train()


def main():
  config = Config.from_args()

  ckpt_path = _resolve_checkpoint(config)
  checkpoint = torch.load(ckpt_path, weights_only=False)

  ckpt_config = checkpoint["config"]
  model = GPT(ckpt_config)
  model.load_state_dict(checkpoint["model_state_dict"])

  if ckpt_config.encoder == "char":
    tokenizer = Tokenizer.from_maps(checkpoint["stoi"], checkpoint["itos"])
  else:
    tokenizer = Tokenizer(encoder="tiktoken")

  print(f"Loaded {ckpt_path}")
  output = model.generate(
    config.prompt,
    tokenizer,
    temperature=config.temperature,
    top_k=config.top_k,
    max_new_tokens=config.max_new_tokens,
  )
  print(output)


if __name__ == "__main__":
  main()
