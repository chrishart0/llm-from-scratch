import tiktoken


class Tokenizer:
  def __init__(self, text):
    self.gpt_enc = tiktoken.get_encoding("gpt2")
    self.chars = sorted(set(text))
    self.vocab_size = len(self.chars)  # 65 for Shakespeare

    self.stoi = {c: i for i, c in enumerate(self.chars)}  # string to int
    self.itos = {i: c for c, i in self.stoi.items()}  # int to string

  def char_encode(self, s):
      return [self.stoi[c] for c in s]

  def tok_encode(self, s):
      return self.gpt_enc.encode(s)

  def char_decode(self, ids):
      return "".join([self.itos[i] for i in ids])

  def tok_decode(self, ids):
      return self.gpt_enc.decode(ids)

  def __print__(self):
    print(f"vocab size: {self.vocab_size}")

  def __len__(self):
    return self.vocab_size

