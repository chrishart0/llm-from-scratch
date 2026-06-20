import json
import math
import os
from datetime import datetime

import torch
from clearml import Task


class Trainer:
  def __init__(self, model, tokenizer, data, config):
    self.tokenizer = tokenizer
    self.data = data
    self.config = config
    self.device = self.get_device()
    self.model = model.to(self.device)


  def get_device(self):
    if torch.backends.mps.is_available():
        return torch.device("mps")     # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")    # NVIDIA GPU
    return torch.device("cpu")


  def get_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


  def setup_clearml(self, enabled=False):
      """Load .env file and initialize ClearML task."""
      if not enabled:
          return None
      try:
          # hosted environments pass env vars directly, so .env is optional
          try:
              from dotenv import load_dotenv
              load_dotenv()
          except ImportError:
              pass

          task = Task.init(
              project_name="Sandbox",
              task_name="gpt-training",
              auto_connect_frameworks={"pytorch": True},
          )
          return task
      except Exception as e:
          print(f"Warning: Could not initialize ClearML: {e}")
          return None


  def load_data(self, block_size, batch_size, device):
      tokens = torch.tensor(self.tokenizer.encode(self.data), dtype=torch.long)
      print(f"Dataset: {len(tokens):,} tokens, vocab size: {self.tokenizer.vocab_size}")

      def get_batch(split_tokens):
          ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
          x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
          y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
          return x, y

      n = int(0.9 * len(tokens))

      def get_train():
          return get_batch(tokens[:n])

      def get_val():
          return get_batch(tokens[n:])

      return get_train, get_val


  def train(self):
      # Initialize ClearML if available
      task = self.setup_clearml(enabled=self.config.use_clearml)
      logger = task.get_logger() if task else None
      if task:
          task.connect({
            "max_steps": self.config.max_steps,
            "batch_size": self.config.batch_size,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_embd": self.config.n_embd,
            "block_size": self.config.block_size
          })
          task.execute_remotely(queue_name="lattice_cuda")

      device = self.device
      print(f"Using device: {device}")

      run_dir = os.path.join("trials", datetime.now().strftime("%Y%m%d_%H%M%S"))
      os.makedirs(run_dir, exist_ok=True)
      print(f"Saving artifacts to {run_dir}/")

      get_train_batch, get_val_batch = self.load_data(
          self.config.block_size,
          self.config.batch_size,
          self.device
      )
      stoi = self.tokenizer.stoi
      itos = self.tokenizer.itos

      print(f"Model: {self.config.n_layer}L/{self.config.n_head}H/{self.config.n_embd}D, "
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params")

      optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

      max_lr = self.config.learning_rate
      min_lr = max_lr * 0.1
      warmup_steps = self.config.warmup_steps

      loss_log = {"steps": [], "train": [], "val": []}
      # sample_steps = set(torch.linspace(1, self.config.max_steps, min(50, self.config.max_steps), dtype=torch.int64).tolist())  # Part 4
      best_loss = float('inf')
      best_step = -1

      for step in range(self.config.max_steps):
          # --- validation loss ---
          if step % self.config.eval_interval == 0:
              self.model.eval()
              with torch.no_grad():
                  val_losses = []
                  for _ in range(self.config.eval_iters):
                      x, y = get_val_batch()
                      _, loss = self.model(x, y)
                      val_losses.append(loss.item())
                  val_loss = sum(val_losses) / len(val_losses)
                  print(f"Step {step:5d} | val loss: {val_loss:.4f}")
                  if logger:
                      logger.report_scalar("Loss", "val", value=val_loss, iteration=step)
              self.model.train()

          # --- update learning rate ---
          lr = self.get_lr(step, warmup_steps, self.config.max_steps, max_lr, min_lr)
          for param_group in optimizer.param_groups:
              param_group["lr"] = lr

          # --- training step ---
          x, y = get_train_batch()
          _, loss = self.model(x, y)
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
          optimizer.step()

          if logger:
              logger.report_scalar("Loss", "train", value=loss.item(), iteration=step)

          # --- track absolute lowest loss ---
          current_loss = loss.item()
          if current_loss < best_loss:
              best_loss = current_loss
              best_step = step
              if step > 0 and step % 1000 != 0:
                  torch.save({
                      "step": step,
                      "model_state_dict": self.model.state_dict(),
                      "config": self.config,
                      "stoi": stoi,
                      "itos": itos,
                  }, os.path.join(run_dir, "checkpoint_lowest_loss.pt"))

          # --- log loss ---
          loss_log["steps"].append(step)
          loss_log["train"].append(loss.item())
          if step % self.config.eval_interval == 0:
              loss_log["val"].append(val_loss)

          # --- Part 4: generate sample ---
          # if step in sample_steps:
          #     self.model.eval()
          #     sample = self.generate(self.model, "To be or not", stoi, itos,
          #                     max_new_tokens=100, temperature=0.8)
          #     print(f"\n--- Step {step} sample ---\n{sample}\n---\n")
          #     if logger:
          #         logger.report_text(title="Generated Samples", series="shakespeare",
          #                           iteration=step, msg=sample)
          #     self.model.train()

          # --- save checkpoint ---
          if step > 0 and step % 1000 == 0:
              torch.save({
                  "step": step,
                  "model_state_dict": self.model.state_dict(),
                  "config": self.config,
                  "stoi": stoi,
                  "itos": itos,
              }, os.path.join(run_dir, f"checkpoint_{step}.pt"))

      # --- emit best-loss info at the end ---
      if best_step >= 0:
          print(f"\n=== Best loss {best_loss:.4f} at step {best_step} ===")
          # Part 4: reload the best checkpoint and generate a sample (needs `import os` and `from llm.model import GPT`)
          # ckpt_path = "checkpoint_lowest_loss.pt" if os.path.exists("checkpoint_lowest_loss.pt") else "checkpoint_final.pt"
          # ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
          # best_model = GPT(ckpt["config"]).to(device)
          # best_model.load_state_dict(ckpt["model_state_dict"])
          # best_model.eval()
          # best_sample = self.generate(best_model, "To be or not", stoi, itos,
          #                       max_new_tokens=100, temperature=0.8)
          # print(f"\n--- Best sample ---\n{best_sample}\n---\n")
          # if logger:
          #     logger.report_text(title="Best Loss Sample", series="best",
          #                       iteration=best_step, msg=best_sample)

      # --- save final checkpoint and loss log ---
      torch.save({
          "step": self.config.max_steps,
          "model_state_dict": self.model.state_dict(),
          "config": self.config,
          "stoi": stoi,
          "itos": itos,
      }, os.path.join(run_dir, "checkpoint_final.pt"))

      with open(os.path.join(run_dir, "loss_log.json"), "w") as f:
          json.dump(loss_log, f)

      if best_step >= 0:
          final_dir = f"{run_dir}_loss{best_loss:.3f}"
          os.rename(run_dir, final_dir)
          print(f"\nRun saved to {final_dir}/")

      return self.model, stoi, itos
