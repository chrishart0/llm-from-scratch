import tiktoken


class Tokenizer:
  def __init__(self, text=None, encoder="char"):
    self.encoder = encoder
    self.gpt_enc = tiktoken.get_encoding("gpt2")
    self.chars = []
    self.stoi = {}  # string to int
    self.itos = {}  # int to string
    if encoder == "char" and text is not None:
      self.chars = sorted(set(text))
      self.stoi = {c: i for i, c in enumerate(self.chars)}
      self.itos = {i: c for c, i in self.stoi.items()}
    self.vocab_size = len(self.stoi) if encoder == "char" else self.gpt_enc.n_vocab

  @classmethod
  def from_maps(cls, stoi, itos):
    tok = cls(encoder="char")
    tok.stoi = stoi
    tok.itos = itos
    tok.chars = list(stoi)
    tok.vocab_size = len(stoi)
    return tok

  def char_encode(self, s):
      return [self.stoi[c] for c in s if c in self.stoi]

  def tok_encode(self, s):
      return self.gpt_enc.encode(s)

  def char_decode(self, ids):
      return "".join([self.itos[i] for i in ids])

  def tok_decode(self, ids):
      return self.gpt_enc.decode(ids)

  def encode(self, s):
      return self.char_encode(s) if self.encoder == "char" else self.tok_encode(s)

  def decode(self, ids):
      return self.char_decode(ids) if self.encoder == "char" else self.tok_decode(ids)

  def __print__(self):
    print(f"vocab size: {self.vocab_size}")

  def __len__(self):
    return self.vocab_size
