from rich import print

from llm.config import Config
from llm.model import GPT
from llm.tokenizer import Tokenizer
from llm.trainer import Trainer


def main():
  config = Config.from_args()

  text = open(config.data_path).read()
  tok = Tokenizer(text)
  model = GPT(config)
  trainer = Trainer(model, text, config)

  print(f"\nLLM from scratch. We have {len(tok)} tokens.")

  n_params = sum(p.numel() for p in model.parameters())
  print(f"\nModel has {n_params/1e6:.2f}M params")
  print("\nModel:")
  print(model)

  trainer.train()


if __name__ == "__main__":
  main()
