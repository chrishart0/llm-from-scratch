# Glossary

This glossary is ordered foundationally, not alphabetically. Terms that explain the basic representations come first, then model architecture, training, generation, and workshop operations.

## Dataset

**Core idea:** A dataset is the body of text used to train and evaluate the model.

**More Details:** The included dataset is `data/shakespeare.txt`, about one million characters. The model learns statistical patterns from this text. Dataset size and quality determine how much the model can learn before it starts memorizing. For the competition, better poetry data can matter more than simply having more text.

**Related:** Training Data, Validation Data, Data Curation, Overfitting

## Data Curation

**Core idea:** Data curation is choosing and cleaning the text that the model trains on.

**More Details:** Good curation removes irrelevant, noisy, duplicated, or low-quality text. A small, focused poetry dataset can beat a much larger messy dataset because the model spends its capacity learning the desired style instead of learning junk patterns.

**Related:** Dataset, Training Data, Poetry Foundation, Project Gutenberg

## Token

**Core idea:** A token is the unit of text the model operates on.

**More Details:** In this workshop, each character is one token. In larger language models, a token is often a word piece produced by BPE. The model receives token IDs as input and predicts the next token ID as output. Tokens are the bridge between human text and numerical computation.

**Related:** Token ID, Tokenization, BPE

## Token Type

**Core idea:** A token type is one distinct possible token in the vocabulary.

**More Details:** If the character `e` appears thousands of times, those appearances are many token instances of one token type. The vocabulary contains token types, not every occurrence. Shakespeare has about 65 character token types in this workshop.

**Related:** Vocabulary, Token, Token ID

## Token ID

**Core idea:** A token ID is the integer assigned to a token.

**More Details:** Neural networks operate on numbers, not strings. The tokenizer maps each token to an integer, such as mapping `H` to `20`. These integers are then used to index into embedding tables.

**Related:** Token, Integer Sequence, Embedding Table, `stoi`, `itos`

## Integer Sequence

**Core idea:** An integer sequence is text after tokenization.

**More Details:** A string like `Hello` becomes a list of token IDs. The model sees this list, usually arranged into batches of fixed-length chunks. During generation, the model produces more integers, which are decoded back into text.

**Related:** Encode, Decode, Token ID, Batch

## Tokenization

**Core idea:** Tokenization converts text into tokens and token IDs.

**More Details:** Tokenization defines the model's basic alphabet. A character-level tokenizer uses individual characters. BPE uses common character sequences or word fragments. The tokenizer choice affects sequence length, vocabulary size, model size, data sparsity, and training difficulty.

**Related:** Character-Level Tokenization, BPE, Vocabulary, Encode, Decode

## Character-Level Tokenization

**Core idea:** Character-level tokenization treats each unique character as its own token.

**More Details:** This is simple and works well for small datasets because the vocabulary is tiny and every token pattern appears many times. The downside is that sequences are longer and the model must learn spelling, words, and punctuation patterns from scratch.

**Related:** Vocabulary Size, Bigrams, BPE

## Byte-Pair Encoding (BPE)

**Core idea:** BPE is a tokenizer that groups common character sequences into single tokens.

**More Details:** BPE reduces sequence length by representing frequent chunks like word pieces as one token. GPT-2 uses a BPE tokenizer with 50,257 tokens. On tiny datasets, that large vocabulary can make training sparse because most token pairs appear rarely.

**Related:** Tokenization, Vocabulary Size, Sparse Statistics, `tiktoken`

## Word-Level Tokenization

**Core idea:** Word-level tokenization treats whole words as tokens.

**More Details:** A word-level model predicts words instead of characters or word pieces. This can make outputs more word-aware, but it requires enough data to learn many distinct words. Rare words are a problem because each needs enough examples.

**Related:** Token, BPE, Vocabulary, Training Data

## `tiktoken`

**Core idea:** `tiktoken` is a library that provides OpenAI-style tokenizers such as GPT-2 BPE.

**More Details:** The docs mention it as the easy way to use GPT-2's tokenizer. It is useful for larger datasets, but for the workshop's Shakespeare setup, character-level tokenization is intentionally better and simpler.

**Related:** BPE, GPT-2, Tokenization

## Vocabulary

**Core idea:** The vocabulary is the set of all token types the model can represent.

**More Details:** With character-level Shakespeare, the vocabulary is all unique characters in the dataset. The vocabulary determines the size of the token embedding table and the output layer. A larger vocabulary gives more expressive token units but makes the model spread probability across more possible next tokens.

**Related:** Vocabulary Size, Token Type, Embedding Table, Output Layer

## Vocabulary Size

**Core idea:** Vocabulary size is the number of distinct token types.

**More Details:** In the workshop, `vocab_size` is about 65. GPT-2's BPE vocabulary is 50,257. Vocabulary size directly affects parameter count and training difficulty because the model must choose among that many possible next tokens at every position.

**Related:** Vocabulary, Embedding Table, Logits, Cross-Entropy

## `stoi`

**Core idea:** `stoi` means string-to-integer mapping.

**More Details:** It is a dictionary that maps each character to its token ID. It is needed for encoding prompts and training text. Checkpoints save `stoi` so generation can reproduce the same mapping used during training.

**Related:** `itos`, Encode, Token ID, Checkpoint

## `itos`

**Core idea:** `itos` means integer-to-string mapping.

**More Details:** It is the reverse of `stoi`. It maps generated token IDs back to characters so model output can be displayed as text. If `itos` does not match the training tokenizer, decoded output will be wrong.

**Related:** `stoi`, Decode, Token ID, Checkpoint

## Encode

**Core idea:** Encoding converts text into token IDs.

**More Details:** In the character-level tokenizer, encoding walks through a string and looks up each character in `stoi`. The output is a list of integers suitable for turning into a PyTorch tensor.

**Related:** Decode, `stoi`, Token ID, Integer Sequence

## Decode

**Core idea:** Decoding converts token IDs back into text.

**More Details:** Decoding walks through generated IDs, maps each through `itos`, and joins the resulting characters. It is the last step of text generation.

**Related:** Encode, `itos`, Generation, Token ID

## Bigrams

**Core idea:** A bigram is a pair of adjacent tokens.

**More Details:** Character-level Shakespeare has only about `65²` possible character bigrams, so each pair appears many times. This gives dense learning signal. With a huge BPE vocabulary on small data, most token bigrams are rare, making sequential patterns harder to learn.

**Related:** Sparse Statistics, Character-Level Tokenization, Training Data

## Sparse Statistics

**Core idea:** Sparse statistics means the model sees too few examples of many patterns.

**More Details:** If the vocabulary is too large for the dataset, many token combinations appear once or not at all. The model then cannot reliably learn what follows what. This is why GPT-2 BPE is a poor fit for tiny Shakespeare training in the workshop.

**Related:** BPE, Vocabulary Size, Bigrams, Training Loss

## Model

**Core idea:** A model is the neural network that maps input token IDs to predictions for the next token.

**More Details:** The workshop model is a small GPT-style transformer. It has learned parameters, accepts batches of token sequences, and returns logits plus an optional loss when targets are provided.

**Related:** GPT, Parameters, Forward Pass, Logits

## Language Model

**Core idea:** A language model predicts likely text.

**More Details:** More precisely, it estimates a probability distribution over the next token given previous tokens. If repeated one token at a time, this prediction process becomes text generation.

**Related:** Autoregressive Language Model, Next-Token Prediction, Probability Distribution

## Autoregressive Language Model

**Core idea:** An autoregressive language model predicts each token using only earlier tokens.

**More Details:** GPT is autoregressive because it models text left-to-right. During training, causal masking prevents the model from seeing future tokens. During generation, the model predicts one token, appends it, then uses the expanded sequence to predict the next token.

**Related:** Language Model, Autoregressive, Causal Masking, Generation

## GPT

**Core idea:** GPT is a transformer language model that predicts the next token from previous tokens.

**More Details:** GPT stands for Generative Pretrained Transformer. In this workshop, GPT refers to the architecture style, not a pretrained commercial model. The workshop model is trained from scratch.

**Related:** Transformer, Autoregressive Language Model, Causal Masking

## GPT-2

**Core idea:** GPT-2 is a well-known GPT model whose design choices influence this workshop.

**More Details:** The docs reference GPT-2's BPE tokenizer, GELU activation, and general architecture style. The workshop does not load GPT-2 weights. It builds a much smaller GPT-like model from scratch.

**Related:** GPT, BPE, GELU, Pretrained Weights

## Transformer

**Core idea:** A transformer is a neural network architecture built around attention and repeated blocks.

**More Details:** The workshop transformer is a stack of identical blocks. Each block uses self-attention to gather information across positions and an MLP to process each position. Residual connections and layer normalization make the stack trainable.

**Related:** Transformer Block, Self-Attention, MLP, LayerNorm

## Transformer Block

**Core idea:** A transformer block is one repeated layer of the GPT architecture.

**More Details:** Each workshop block follows the pattern: LayerNorm, self-attention, residual connection, LayerNorm, MLP, residual connection. Stacking blocks lets the model build increasingly abstract representations of the context.

**Related:** `n_layer`, Self-Attention, MLP, Residual Connection

## Parameter

**Core idea:** A parameter is a learned number in the model.

**More Details:** Parameters include embedding values, linear layer weights, and normalization weights. Training changes parameters to reduce loss. More parameters give more capacity but also increase compute and overfitting risk.

**Related:** Parameter Count, Model Capacity, Optimizer, Weight

## Parameter Count

**Core idea:** Parameter count is the total number of learned numbers in the model.

**More Details:** The default workshop model has about 10.8 million parameters. Most live in transformer blocks, especially attention and MLP linear layers. Parameter count grows with `n_layer`, `n_head`, `n_embd`, and vocabulary size.

**Related:** Parameters, Model Size, Model Capacity, Overfitting

## Model Capacity

**Core idea:** Model capacity is how much complexity the model can represent.

**More Details:** A larger model can learn richer patterns, but on small data it can memorize instead of generalizing. The workshop demonstrates this with a 10M parameter model trained on about 1M characters.

**Related:** Parameter Count, Overfitting, Data Size, Model Size

## Weight

**Core idea:** A weight is a learned parameter used in a mathematical operation.

**More Details:** Linear layers, embeddings, and output projections all contain weights. During backpropagation, gradients tell the optimizer how to adjust these weights to reduce loss.

**Related:** Parameter, Gradient, Optimizer, Linear Layer

## Weight Tying

**Core idea:** Weight tying reuses the token embedding matrix as the output projection matrix.

**More Details:** The same learned token representation is used when reading tokens and when scoring possible output tokens. This reduces parameters and encourages consistency between input and output token meanings.

**Related:** Token Embedding, Output Projection, `lm_head`, Parameters

## Tensor

**Core idea:** A tensor is a multidimensional array of numbers.

**More Details:** PyTorch represents token batches, embeddings, logits, losses, gradients, and parameters as tensors. Shapes matter because they describe what each dimension means, such as batch, time, channels, heads, or vocabulary.

**Related:** Shape, Batch, PyTorch, Matrix

## Shape

**Core idea:** A shape describes the dimensions of a tensor.

**More Details:** The docs use shapes like `(B, T, C)`, where `B` is batch size, `T` is sequence length, and `C` is embedding dimension. Correct shapes are essential because attention, linear layers, and loss calculations expect specific layouts.

**Related:** Tensor, Batch Size, Block Size, Embedding Dimension

## Batch

**Core idea:** A batch is a group of training examples processed together.

**More Details:** In the training loop, a batch contains multiple random chunks of token IDs. Batching makes training more efficient and produces a more stable gradient estimate than using one sequence at a time.

**Related:** Batch Size, Training Step, Tensor, Gradient

## Batch Size

**Core idea:** Batch size is the number of examples processed in one training step.

**More Details:** Larger batches use more memory but can make gradient estimates smoother. The default batch size is 64. If context length or model size increases, batch size may need to decrease to fit in memory.

**Related:** Batch, Memory, Training Step, Block Size

## Sequence Length

**Core idea:** Sequence length is the number of tokens in one input example.

**More Details:** In the docs, sequence length is often represented by `T`. The model processes a fixed-length window of tokens during training. Longer sequences let the model use more context but require more compute and memory.

**Related:** Block Size, Context Window, Token, Tensor Shape

## Block Size

**Core idea:** `block_size` is the maximum sequence length the model can process.

**More Details:** It defines the context window and the size of the position embedding table. During generation, if the sequence gets longer than `block_size`, only the most recent tokens are passed to the model.

**Related:** Context Window, Sequence Length, Position Embedding, `idx_cond`

## Context Window

**Core idea:** The context window is the span of previous tokens the model can see.

**More Details:** A longer context helps the model learn broader structure like stanzas, speaker turns, and rhyme patterns. It also increases memory use, especially in attention, because attention compares positions to other positions.

**Related:** Block Size, Attention, Long Context, Memory

## Embedding

**Core idea:** An embedding is a learned vector representation of a discrete item.

**More Details:** Token IDs are arbitrary integers, so the model first maps them to dense vectors. These vectors are learned during training. Embeddings let the model operate in continuous space where similarity and composition can be represented numerically.

**Related:** Token Embedding, Position Embedding, Embedding Table, Vector

## Vector

**Core idea:** A vector is a list of numbers representing one item or position.

**More Details:** Each token position in the model is represented by a vector of size `n_embd`. The values are not hand-written features; they are learned internal coordinates useful for predicting the next token.

**Related:** Embedding, Hidden State, Embedding Dimension

## Embedding Dimension

**Core idea:** `n_embd` is the width of each token representation vector.

**More Details:** In the default model, each position is represented by 384 numbers. Larger embedding dimensions give the model more room to represent information but increase parameters and compute.

**Related:** Vector, Hidden State, Model Size, Head Dimension

## Embedding Table

**Core idea:** An embedding table maps IDs to learned vectors.

**More Details:** Token embeddings use a table of shape `[vocab_size, n_embd]`. Position embeddings use a table of shape `[block_size, n_embd]`. Looking up an ID returns the corresponding learned row.

**Related:** Token Embedding, Position Embedding, Vocabulary Size

## Token Embedding

**Core idea:** A token embedding is the learned vector for a token ID.

**More Details:** The workshop names this table `wte`. It converts token IDs into dense vectors before the transformer blocks. With weight tying, this table is also reused by the output layer.

**Related:** `wte`, Embedding Table, Weight Tying, Token ID

## Position Embedding

**Core idea:** A position embedding tells the model where each token occurs in the sequence.

**More Details:** Without position information, the model would see the same set of token embeddings for different word orders. The workshop adds token embeddings and position embeddings so each position carries both content and location.

**Related:** `wpe`, Block Size, Sequence Length, Token Embedding

## Broadcasting

**Core idea:** Broadcasting lets tensors with compatible shapes be combined automatically.

**More Details:** In the forward pass, position embeddings shaped `(T, n_embd)` are added to token embeddings shaped `(B, T, n_embd)`. PyTorch broadcasts the position embeddings across the batch dimension.

**Related:** Tensor Shape, Token Embedding, Position Embedding, PyTorch

## Hidden State

**Core idea:** A hidden state is the model's internal vector representation at a position.

**More Details:** After embeddings are added, each transformer block updates the hidden states. By the final layer, each position's vector should contain information useful for predicting the next token at that position.

**Related:** Vector, Transformer Block, Logits, Embedding Dimension

## Forward Pass

**Core idea:** A forward pass is running inputs through the model to produce outputs.

**More Details:** The GPT forward pass embeds tokens and positions, applies transformer blocks, normalizes the result, projects to logits, and optionally computes loss if targets are provided.

**Related:** Logits, Loss, Model, Inference

## Linear Layer

**Core idea:** A linear layer applies a learned matrix transformation to vectors.

**More Details:** Linear layers are used for QKV projections, attention output projection, MLP expansion and contraction, and the final output layer. They are where many of the model's parameters live.

**Related:** Matrix Multiplication, Projection, Weight, Bias

## Projection

**Core idea:** A projection maps vectors from one representation space to another.

**More Details:** In the docs, projections create queries, keys, values, attention outputs, MLP hidden vectors, and final logits. A projection is usually implemented as a linear layer.

**Related:** Linear Layer, QKV Projections, Output Projection

## Bias

**Core idea:** A bias is an added learned offset in a linear layer.

**More Details:** The final `lm_head` in the docs uses `bias=False`, but many linear layers include biases by default. A bias lets a layer shift outputs independently of the input values.

**Related:** Linear Layer, Weight, `lm_head`

## Logits

**Core idea:** Logits are raw, unnormalized scores for each possible next token.

**More Details:** The model outputs logits of shape `(B, T, vocab_size)`. Higher logits mean the model prefers those tokens more, but logits are not probabilities until softmax is applied.

**Related:** Softmax, Probability Distribution, Output Layer, Temperature

## Output Layer

**Core idea:** The output layer maps hidden states to logits over the vocabulary.

**More Details:** In the workshop, `lm_head` is a linear layer from `n_embd` to `vocab_size`. At every position, it scores every possible next token.

**Related:** `lm_head`, Logits, Vocabulary Size, Weight Tying

## `lm_head`

**Core idea:** `lm_head` is the language-modeling output layer.

**More Details:** It turns final hidden states into logits. Its output dimension equals the vocabulary size because the model must score every possible next token.

**Related:** Output Layer, Logits, Weight Tying

## Attention

**Core idea:** Attention lets one token position use information from other token positions.

**More Details:** Instead of processing each token in isolation, attention computes which previous positions are relevant. It then forms a weighted combination of their value vectors. This is the main mechanism that lets GPT use context.

**Related:** Self-Attention, Query, Key, Value, Context Window

## Self-Attention

**Core idea:** Self-attention is attention within the same sequence.

**More Details:** Each token position creates a query, key, and value from the same input sequence. A position compares its query to keys from other positions, then uses the resulting weights to combine values.

**Related:** Attention, Causal Self-Attention, QKV Projections

## Causal Self-Attention

**Core idea:** Causal self-attention prevents a token from attending to future tokens.

**More Details:** Position `i` may attend only to positions `0..i`. This matches next-token prediction: the model should predict the future from the past, not peek at the answer.

**Related:** Causal Masking, Autoregressive, Self-Attention

## Query

**Core idea:** A query is the vector that asks, “What information am I looking for?”

**More Details:** In attention, each position has a query vector. The query is compared with key vectors to produce attention scores. A high query-key match means that key's position is relevant.

**Related:** Key, Value, Attention Score, QKV Projections

## Key

**Core idea:** A key is the vector that advertises what information a position contains.

**More Details:** Queries compare against keys using dot products. If a query and key align strongly, the model gives that key's position more attention weight.

**Related:** Query, Value, Attention Score, Dot Product

## Value

**Core idea:** A value is the vector containing the information attention will copy or mix.

**More Details:** Attention scores decide how much of each value vector contributes to the output. Keys are used for matching; values are used for the resulting content.

**Related:** Query, Key, Attention Weight, Weighted Combination

## QKV Projections

**Core idea:** QKV projections create query, key, and value vectors from the input.

**More Details:** The workshop uses one linear layer, `c_attn`, to produce three `n_embd`-sized outputs, then splits them into Q, K, and V. This is efficient and common in GPT implementations.

**Related:** Query, Key, Value, Linear Layer, `c_attn`

## Dot Product

**Core idea:** A dot product measures alignment between two vectors.

**More Details:** Attention uses dot products between queries and keys. Larger dot products mean stronger similarity or compatibility. These scores are scaled and passed through softmax.

**Related:** Attention Score, Query, Key, Matrix Multiplication

## Attention Score

**Core idea:** An attention score is a raw relevance score between a query position and a key position.

**More Details:** Scores come from `Q @ K^T / sqrt(head_dim)`. Future positions are masked out in causal attention. The remaining scores become attention weights after softmax.

**Related:** Scaled Dot-Product Attention, Causal Masking, Softmax

## Scaled Dot-Product Attention

**Core idea:** Scaled dot-product attention computes weighted context using `softmax(QK^T / sqrt(head_dim)) @ V`.

**More Details:** Scaling by the square root of the head dimension keeps dot products from becoming too large as vector size grows. Softmax turns scores into weights. Multiplying by values produces the attended output.

**Related:** Query, Key, Value, Softmax, Head Dimension

## Causal Masking

**Core idea:** Causal masking hides future tokens from the attention calculation.

**More Details:** During training, the full sequence is available in memory, including targets. The mask enforces the rule that prediction at a position can only depend on current and previous tokens.

**Related:** Causal Self-Attention, Autoregressive, Future Tokens

## Softmax

**Core idea:** Softmax converts scores into a probability distribution.

**More Details:** It exponentiates each score and divides by the sum of exponentiated scores. In attention, softmax converts attention scores into attention weights. In generation, softmax converts logits into token probabilities.

**Related:** Logits, Probability Distribution, Attention Weight, Temperature

## Attention Weight

**Core idea:** An attention weight says how much one position should use another position's value.

**More Details:** Attention weights are nonnegative and sum to one across the allowed positions. A high weight means the model is relying heavily on that position's value vector.

**Related:** Softmax, Weighted Combination, Value, Attention Score

## Weighted Combination

**Core idea:** A weighted combination mixes vectors according to weights.

**More Details:** In attention, the output at each position is a weighted sum of value vectors. This lets the model gather information from multiple previous positions in different proportions.

**Related:** Attention Weight, Value, Self-Attention

## Attention Head

**Core idea:** An attention head is one independent attention operation over part of the embedding dimension.

**More Details:** Each head has its own query, key, and value subspace. A head can specialize in a type of relationship, such as recent characters, line breaks, names, or punctuation patterns.

**Related:** Multi-Head Attention, `n_head`, Head Dimension

## Multi-Head Attention

**Core idea:** Multi-head attention runs several attention heads in parallel.

**More Details:** Instead of one large attention operation, the model splits the embedding dimension across heads. The head outputs are concatenated and projected back to the model width. This lets the model track multiple relationships at once.

**Related:** Attention Head, Head Dimension, Concatenate, Output Projection

## Head Dimension

**Core idea:** Head dimension is the vector width assigned to each attention head.

**More Details:** It is computed as `n_embd / n_head`. With `n_embd=384` and `n_head=6`, each head has dimension 64. `n_embd` must divide evenly by `n_head`.

**Related:** Attention Head, Multi-Head Attention, Embedding Dimension

## Concatenate

**Core idea:** Concatenate means join tensors along a dimension.

**More Details:** Multi-head attention concatenates the outputs of all heads to recover the full embedding dimension. Generation also concatenates each sampled next token onto the running sequence.

**Related:** Multi-Head Attention, Tensor Shape, Generation

## Output Projection

**Core idea:** An output projection maps a sub-layer's output back to the model width.

**More Details:** Attention concatenates heads, then uses `c_proj` to return to `n_embd`. The MLP also projects expanded vectors back down to `n_embd`. This keeps residual additions shape-compatible.

**Related:** Projection, `c_proj`, Residual Connection, Embedding Dimension

## MLP

**Core idea:** The MLP is the feed-forward network inside each transformer block.

**More Details:** It processes each position independently. In the workshop, it expands from `n_embd` to `4 * n_embd`, applies GELU, then projects back down. Attention gathers context; the MLP transforms the gathered representation.

**Related:** Feed-Forward Network, GELU, `c_fc`, `c_proj`

## Feed-Forward Network

**Core idea:** A feed-forward network applies learned transformations without looking across positions.

**More Details:** In a transformer block, the feed-forward network is the MLP. It is the same operation applied independently to each token position, unlike attention, which mixes information across positions.

**Related:** MLP, Linear Layer, GELU, Position-Wise

## Non-Linearity

**Core idea:** A non-linearity lets the model represent patterns beyond simple linear transformations.

**More Details:** Without non-linear functions, stacking linear layers would collapse into one linear transformation. GELU is the non-linearity used in the workshop MLP.

**Related:** GELU, MLP, ReLU

## GELU

**Core idea:** GELU is a smooth activation function used as the MLP non-linearity.

**More Details:** GELU stands for Gaussian Error Linear Unit. It is smoother than ReLU because it does not have a hard cutoff at zero. GPT-2 uses a fast tanh approximation of GELU.

**Related:** Non-Linearity, ReLU, MLP

## ReLU

**Core idea:** ReLU is an activation function that outputs zero for negative values and the input for positive values.

**More Details:** The docs mention ReLU only to contrast it with GELU. ReLU is simple and common, but GPT-style models typically use GELU because its smoother behavior helps optimization.

**Related:** GELU, Non-Linearity, MLP

## Layer Normalization

**Core idea:** Layer normalization stabilizes the scale of activations.

**More Details:** LayerNorm normalizes each position's vector before attention or MLP in the workshop's pre-norm transformer. Stable activation scale helps training avoid exploding, vanishing, or erratic updates.

**Related:** Pre-Norm, Transformer Block, Training Stability

## Pre-Norm

**Core idea:** Pre-norm means applying LayerNorm before each sub-layer.

**More Details:** The workshop block uses `LayerNorm → attention` and `LayerNorm → MLP`. This is now common in GPT-style models because it improves training stability, especially in deeper networks.

**Related:** Layer Normalization, Post-Norm, Transformer Block

## Post-Norm

**Core idea:** Post-norm means applying LayerNorm after a sub-layer and residual addition.

**More Details:** The original transformer paper used post-norm, but the workshop notes that pre-norm is now standard for GPT-like models because it tends to train more stably.

**Related:** Pre-Norm, Layer Normalization, Transformer

## Residual Connection

**Core idea:** A residual connection adds a sub-layer's input back to its output.

**More Details:** The pattern is `x = x + sublayer(x)`. This gives gradients a direct path backward through the network and makes deep models much easier to train.

**Related:** Gradient, Backpropagation, Transformer Block, Training Stability

## Gradient

**Core idea:** A gradient tells how a parameter should change to reduce loss.

**More Details:** After the forward pass computes loss, backpropagation computes gradients for parameters. The optimizer uses those gradients to update weights.

**Related:** Backpropagation, Loss, Optimizer, Gradient Clipping

## Backpropagation

**Core idea:** Backpropagation computes gradients by moving backward through the model's operations.

**More Details:** It applies the chain rule from the loss back to every parameter that contributed to the prediction. In PyTorch, `loss.backward()` performs this computation.

**Related:** Gradient, Loss, Training Step, Optimizer

## Training

**Core idea:** Training adjusts model parameters so predictions become better.

**More Details:** Each training step samples a batch, runs a forward pass, computes loss, backpropagates gradients, clips gradients, updates parameters, and logs progress. Over many steps, the model learns patterns in the data.

**Related:** Training Loop, Training Step, Optimizer, Loss

## Training Data

**Core idea:** Training data is the portion of the dataset used to update model weights.

**More Details:** The docs split the tokenized dataset roughly 90/10. The first portion is used for training batches. Loss on training data usually decreases as the model learns, even if generalization gets worse.

**Related:** Validation Data, Train Loss, Overfitting, Dataset Split

## Validation Data

**Core idea:** Validation data is held-out data used to estimate generalization.

**More Details:** The model does not update weights on validation batches. Validation loss helps detect overfitting: if train loss keeps dropping while validation loss rises, the model is memorizing instead of generalizing.

**Related:** Val Loss, Overfitting, Dataset Split, Evaluation

## Dataset Split

**Core idea:** A dataset split divides data into separate training and validation portions.

**More Details:** The workshop uses about 90% for training and 10% for validation. Keeping validation separate makes it possible to estimate whether the model learned reusable patterns or merely memorized the training text.

**Related:** Training Data, Validation Data, Overfitting

## Target

**Core idea:** A target is the correct answer the model is trained to predict.

**More Details:** In next-token prediction, the target sequence is the input sequence shifted one token to the left. If `x` is positions `i` through `i + block_size`, `y` is positions `i + 1` through `i + block_size + 1`.

**Related:** Input, Next-Token Prediction, Cross-Entropy

## Label

**Core idea:** A label is the supervised answer for a training example.

**More Details:** This workshop is self-supervised because labels come from the text itself. The next character in the dataset acts as the label for each input position.

**Related:** Target, Self-Supervised Learning, Next-Token Prediction

## Next-Token Prediction

**Core idea:** Next-token prediction trains the model to predict the following token from previous tokens.

**More Details:** Given `[t0, t1, ..., tn]`, the model predicts `[t1, t2, ..., tn+1]`. This objective is enough to train a language model because every position in ordinary text supplies its own target.

**Related:** Autoregressive Language Model, Target, Cross-Entropy, Self-Supervised Learning

## Self-Supervised Learning

**Core idea:** Self-supervised learning creates labels automatically from the data.

**More Details:** No human labels are needed. The text itself provides training examples because the next token after each position is known. This is why raw text can train a language model.

**Related:** Next-Token Prediction, Label, Training Data

## Loss

**Core idea:** Loss is a number measuring how wrong the model is.

**More Details:** Lower loss means the model assigned more probability to the correct next token. Training tries to minimize loss. Validation loss is especially important because it measures performance on held-out data.

**Related:** Cross-Entropy, Train Loss, Val Loss, Optimization

## Cross-Entropy

**Core idea:** Cross-entropy is the loss function for choosing the correct token from many possibilities.

**More Details:** It compares the model's predicted probability distribution against the actual next token. If the model assigns high probability to the correct token, cross-entropy is low. If it spreads probability incorrectly, cross-entropy is high.

**Related:** Loss, Probability Distribution, Logits, Vocabulary Size

## Train Loss

**Core idea:** Train loss is loss measured on training data.

**More Details:** Train loss usually decreases as the optimizer fits the data. Very low train loss is not automatically good, because it may mean memorization if validation loss is rising.

**Related:** Loss, Validation Loss, Overfitting, Memorization

## Validation Loss

**Core idea:** Validation loss is loss measured on held-out validation data.

**More Details:** Validation loss is the workshop's main signal for model quality. The best checkpoint is usually near the lowest validation loss, not the lowest training loss.

**Related:** Val Loss, Train Loss, Overfitting, Early Stopping

## Val Loss

**Core idea:** Val loss is shorthand for validation loss.

**More Details:** The docs print validation loss every 100 steps. For character-level Shakespeare, values around 1.5–2.0 indicate recognizable text, while rising val loss after improvement indicates overfitting.

**Related:** Validation Loss, Loss, Evaluation, Checkpoint

## Optimization

**Core idea:** Optimization is the process of changing parameters to reduce loss.

**More Details:** The optimizer uses gradients, learning rate, and update rules to modify weights. Good optimization is stable: loss generally decreases without diverging or spiking wildly.

**Related:** Optimizer, AdamW, Learning Rate, Gradient

## Optimizer

**Core idea:** An optimizer updates model parameters using gradients.

**More Details:** The workshop uses AdamW. After `loss.backward()` computes gradients, `optimizer.step()` applies parameter updates. `optimizer.zero_grad()` clears old gradients before the next step.

**Related:** AdamW, Gradient, Learning Rate, Weight Decay

## AdamW

**Core idea:** AdamW is an optimizer commonly used for transformer training.

**More Details:** AdamW adapts update sizes using moving estimates of gradient statistics and applies decoupled weight decay. The workshop uses a simple AdamW setup because the small character-level model does not need the full GPT-2 optimization recipe.

**Related:** Optimizer, Learning Rate, Weight Decay, Betas

## Learning Rate

**Core idea:** The learning rate controls how large each optimizer update is.

**More Details:** Too low can make training slow or stuck. Too high can cause instability or divergence. The workshop uses a schedule that warms up then decays.

**Related:** Max LR, Min LR, Warmup, Cosine Decay

## Max LR

**Core idea:** Max LR is the peak learning rate reached after warmup.

**More Details:** In the docs, `max_lr` defaults to `1e-3`. It is the largest update scale used during training before cosine decay reduces it.

**Related:** Learning Rate, Warmup, Cosine Decay

## Min LR

**Core idea:** Min LR is the floor learning rate reached near the end of training.

**More Details:** The docs set `min_lr` to one tenth of `max_lr`. This keeps updates small during refinement instead of stopping learning abruptly.

**Related:** Learning Rate, Cosine Decay, Training Step

## Learning Rate Schedule

**Core idea:** A learning rate schedule changes the learning rate over training.

**More Details:** The workshop schedule uses warmup first, then cosine decay. This gives the optimizer gentle early steps and smaller late steps as the model refines what it has learned.

**Related:** Warmup, Cosine Decay, Learning Rate

## Warmup

**Core idea:** Warmup gradually increases the learning rate at the start of training.

**More Details:** Early in training, optimizer statistics are poorly calibrated and weights are random. Warmup avoids making large updates before the optimizer has useful moment estimates.

**Related:** Learning Rate Schedule, AdamW, Training Stability

## Cosine Decay

**Core idea:** Cosine decay smoothly lowers the learning rate over time.

**More Details:** The schedule starts with larger updates for exploration and ends with smaller updates for refinement. The cosine shape avoids abrupt changes in update size.

**Related:** Learning Rate Schedule, Max LR, Min LR

## Weight Decay

**Core idea:** Weight decay gently discourages large weights.

**More Details:** It acts as a form of regularization. The workshop uses light weight decay with AdamW. It can help generalization but does not by itself solve severe overfitting on tiny data.

**Related:** AdamW, Regularization, Overfitting

## Betas

**Core idea:** Betas are Adam-style optimizer settings that control moving averages.

**More Details:** The docs mention GPT-2's larger-scale recipe using custom betas. For the workshop, default AdamW settings are sufficient because the setup is smaller and simpler.

**Related:** AdamW, Optimizer, Gradient Statistics

## Gradient Clipping

**Core idea:** Gradient clipping limits gradient size before the optimizer update.

**More Details:** The workshop caps total gradient norm at 1.0. This prevents occasional large gradients from causing unstable jumps in parameter values.

**Related:** Gradient, Training Stability, Loss Spikes

## Training Step

**Core idea:** A training step is one parameter update.

**More Details:** Each step gets a batch, computes loss, clears old gradients, runs backpropagation, clips gradients, updates parameters, logs loss, and sometimes validates, generates, or checkpoints.

**Related:** Training Loop, Batch, Optimizer, Max Steps

## Max Steps

**Core idea:** Max steps is the total number of training updates to run.

**More Details:** The default is 5000 steps. More steps are not always better: after validation loss starts rising, additional steps can make the model worse at generating novel text.

**Related:** Training Step, Overfitting, Early Stopping

## Training Loop

**Core idea:** The training loop repeatedly performs training steps.

**More Details:** It coordinates batching, learning rate updates, forward and backward passes, validation, sample generation, checkpointing, and loss logging. It is where the static model architecture becomes a learned model.

**Related:** Training, Training Step, Validation, Checkpointing

## Evaluation

**Core idea:** Evaluation measures model performance without updating weights.

**More Details:** During validation and generation, the model is put in eval mode and gradients are disabled when appropriate. This saves memory and avoids changing model parameters.

**Related:** Validation Loss, `model.eval()`, `torch.no_grad()`

## `model.train()`

**Core idea:** `model.train()` puts a PyTorch model in training mode.

**More Details:** Training mode matters for modules such as dropout. The workshop switches back to train mode after validation or sample generation so subsequent steps behave correctly.

**Related:** `model.eval()`, Dropout, Training Loop

## `model.eval()`

**Core idea:** `model.eval()` puts a PyTorch model in evaluation mode.

**More Details:** Eval mode disables training-specific behavior such as dropout. The workshop uses eval mode during validation and sample generation.

**Related:** `model.train()`, Evaluation, Dropout

## `torch.no_grad()`

**Core idea:** `torch.no_grad()` disables gradient tracking.

**More Details:** Gradients are unnecessary for validation and generation because parameters are not being updated. Disabling gradients saves memory and computation.

**Related:** Evaluation, Inference, Generation, Backpropagation

## CUDA

**Core idea:** CUDA is NVIDIA's GPU computing platform.

**More Details:** If `torch.cuda.is_available()` is true, PyTorch can run tensors and model operations on an NVIDIA GPU. CUDA usually speeds up training compared with CPU.

**Related:** PyTorch

## MPS

**Core idea:** MPS is Apple's GPU acceleration backend for PyTorch on Apple Silicon.

**More Details:** The docs use MPS when available on MacBooks. It can give a 2–3x speedup over CPU for this workshop.

**Related:** PyTorch

## Training Stability

**Core idea:** Training stability means loss decreases without exploding, diverging, or behaving erratically.

**More Details:** LayerNorm, residual connections, warmup, gradient clipping, and reasonable learning rates all improve stability. Instability can show up as loss spikes or nonsense output.

**Related:** Gradient Clipping, Learning Rate, LayerNorm, Loss Spikes

## Loss Spike

**Core idea:** A loss spike is a sudden sharp increase in loss.

**More Details:** Spikes can come from too large a learning rate, unstable gradients, or bugs. The docs suggest reducing learning rate or checking gradient clipping when spikes appear.

**Related:** Training Stability, Gradient Clipping, Learning Rate

## Loss Plateau

**Core idea:** A loss plateau is when loss stops improving.

**More Details:** A plateau can mean the model has learned what it can from the available data and capacity. Escaping it may require more data, a bigger model, better optimization, or fixing a bug.

**Related:** Loss, Model Capacity, Data Size, Training Loop

## Overfitting

**Core idea:** Overfitting happens when the model memorizes training data instead of learning general patterns.

**More Details:** In the workshop, a 10M parameter model has enough capacity to memorize about 1M characters of Shakespeare. The signal is train loss decreasing while validation loss increases. Generated text may become less novel even as train loss improves.

**Related:** Memorization, Validation Loss, Train Loss, Model Capacity

## Memorization

**Core idea:** Memorization means reproducing training examples rather than generalizing.

**More Details:** A memorizing model may score well on training data but poorly on held-out data. For generation, it may regurgitate passages instead of producing new plausible text.

**Related:** Overfitting, Train Loss, Validation Loss, Generalization

## Generalization

**Core idea:** Generalization is performance on data or outputs not memorized from training.

**More Details:** A model that generalizes has learned reusable patterns, such as spelling, syntax, poetic structure, or style. Validation loss is used as a proxy for generalization.

**Related:** Validation Data, Overfitting, Memorization

## Regularization

**Core idea:** Regularization is anything that discourages overfitting.

**More Details:** The docs mention dropout, weight decay, smaller models, more data, and early stopping as ways to reduce overfitting. Regularization trades some fitting power for better generalization.

**Related:** Dropout, Weight Decay, Early Stopping, Overfitting

## Dropout

**Core idea:** Dropout randomly zeros activations during training.

**More Details:** This makes the model less reliant on any single internal feature and can reduce overfitting. The competition docs suggest adding dropout to attention and MLP blocks.

**Related:** Regularization, Overfitting, `model.train()`, `model.eval()`

## Early Stopping

**Core idea:** Early stopping means using the model checkpoint from the best validation loss instead of training to the end.

**More Details:** If validation loss rises after step 2000, the step 5000 checkpoint may be worse for generation. Early stopping keeps the model from continuing into overfitting.

**Related:** Validation Loss, Checkpoint, Overfitting

## Checkpoint

**Core idea:** A checkpoint is a saved snapshot of training state.

**More Details:** Workshop checkpoints include model weights, config, tokenizer mappings, and step number. They allow generation later without retraining and make competition submissions reproducible.

**Related:** `model_state_dict`, Config, `stoi`, `itos`, Reproducibility

## `model_state_dict`

**Core idea:** `model_state_dict` is PyTorch's saved mapping of model parameter names to tensors.

**More Details:** Loading it restores learned weights into a model with matching architecture. It is the core artifact needed to reproduce a trained model.

**Related:** Checkpoint, Parameters, PyTorch

## Config

**Core idea:** A config stores model hyperparameters.

**More Details:** `GPTConfig` includes `vocab_size`, `block_size`, `n_layer`, `n_head`, and `n_embd`. Saving config with a checkpoint ensures the model can be reconstructed before loading weights.

**Related:** GPTConfig, Hyperparameter, Checkpoint

## Hyperparameter

**Core idea:** A hyperparameter is a setting chosen before or during training, not learned directly.

**More Details:** Examples include learning rate, batch size, block size, number of layers, number of heads, embedding dimension, temperature, and top-k. Hyperparameters strongly affect quality, speed, and overfitting.

**Related:** Config, Model Size, Learning Rate, Generation Settings

## Model Size

**Core idea:** Model size describes how large the architecture is.

**More Details:** In the docs, model size is controlled mainly by `n_layer`, `n_head`, and `n_embd`. Larger models can produce better outputs with enough data but overfit faster on small data.

**Related:** Parameter Count, Model Capacity, `n_layer`, `n_head`, `n_embd`

## `n_layer`

**Core idea:** `n_layer` is the number of transformer blocks.

**More Details:** More layers let the model perform more stages of representation building. They also increase parameter count, compute, and overfitting risk.

**Related:** Transformer Block, Model Size, Parameter Count

## `n_head`

**Core idea:** `n_head` is the number of attention heads per transformer block.

**More Details:** More heads let attention track more relationships in parallel, but `n_embd` must divide evenly by `n_head`. Increasing heads usually goes along with increasing model width.

**Related:** Multi-Head Attention, Attention Head, Head Dimension

## `n_embd`

**Core idea:** `n_embd` is the embedding dimension and model width.

**More Details:** It determines the size of token vectors, hidden states, and many linear layer dimensions. Larger `n_embd` increases capacity and parameter count substantially.

**Related:** Embedding Dimension, Model Size, Head Dimension

## Data Size

**Core idea:** Data size is how much training text is available.

**More Details:** Larger datasets support larger models and longer training before overfitting. The docs emphasize scaling model size with data size, not simply making the model bigger.

**Related:** Dataset, Model Capacity, Overfitting, Chinchilla

## Parameter-to-Data Ratio

**Core idea:** Parameter-to-data ratio compares model size with dataset size.

**More Details:** The workshop highlights a rough 10:1 ratio: about 10M parameters and 1M characters. A high ratio makes memorization easy. More data or a smaller model improves the balance.

**Related:** Overfitting, Model Capacity, Data Size

## Autoregressive

**Core idea:** Autoregressive means generating or predicting one token at a time from previous tokens.

**More Details:** GPT is autoregressive because each next-token prediction depends only on prior context. Causal masking enforces this during training, and generation appends each sampled token before predicting the next one.

**Related:** Next-Token Prediction, Causal Masking, Generation

## Generation

**Core idea:** Generation is using a trained model to produce new text.

**More Details:** The model starts from a prompt, predicts logits for the next token, samples or chooses a token, appends it, and repeats. Generation quality depends on both the trained model and decoding settings.

**Related:** Autoregressive, Prompt, Temperature, Top-k Sampling

## Inference

**Core idea:** Inference is running the model to produce predictions without training.

**More Details:** Generation is a form of inference. During inference, gradients are not needed, and the model is usually in eval mode.

**Related:** Generation, `torch.no_grad()`, Evaluation

## Prompt

**Core idea:** A prompt is the starting text given to the model for generation.

**More Details:** The prompt becomes the initial token sequence. Different prompts can steer the same model toward different topics, structures, or styles. In the competition, the exact prompt is part of the submission command.

**Related:** Prompt Engineering, Generation, Tokenization

## Prompt Engineering

**Core idea:** Prompt engineering is choosing prompts to steer generated output.

**More Details:** For the poetry competition, a prompt might be a title, first line, or style cue. The model still produces the text, but the prompt influences the distribution of likely continuations.

**Related:** Prompt, Generation, Cherry-Pick

## `idx`

**Core idea:** `idx` is the tensor of current token IDs used during generation.

**More Details:** The generation loop repeatedly appends `next_token` to `idx`. It represents the full generated sequence so far, or at least the sequence being accumulated.

**Related:** Token ID, Generation, `idx_cond`, Concatenate

## `idx_cond`

**Core idea:** `idx_cond` is the context-limited slice of `idx` passed to the model.

**More Details:** If `idx` grows longer than `block_size`, the model only receives the last `block_size` tokens. This respects the model's maximum context window.

**Related:** `idx`, Block Size, Context Window, Generation

## Greedy Decoding

**Core idea:** Greedy decoding always chooses the most probable next token.

**More Details:** It is deterministic and simple, but often repetitive. Because it always takes the local best option, it can get trapped in boring or looping continuations.

**Related:** Argmax, Deterministic, Sampling, Temperature

## Argmax

**Core idea:** Argmax returns the index of the largest value.

**More Details:** In greedy decoding, argmax selects the token with the highest logit or probability. It removes randomness entirely.

**Related:** Greedy Decoding, Logits, Deterministic

## Deterministic

**Core idea:** Deterministic means the same input always gives the same output.

**More Details:** Greedy decoding is deterministic. Sampling is not deterministic unless a seed fixes the random number generator.

**Related:** Greedy Decoding, Seed, Reproducibility, Sampling

## Sampling

**Core idea:** Sampling chooses a token randomly according to probabilities.

**More Details:** Instead of always choosing the highest-probability token, sampling lets lower-probability tokens sometimes appear. This makes text more varied but can reduce coherence if the distribution is too flat.

**Related:** Probability Distribution, Multinomial, Temperature, Top-k Sampling

## Probability Distribution

**Core idea:** A probability distribution assigns probabilities that sum to one across outcomes.

**More Details:** For generation, the outcomes are possible next tokens. Softmax turns logits into this distribution, and sampling draws from it.

**Related:** Softmax, Logits, Sampling, Cross-Entropy

## Temperature

**Core idea:** Temperature controls how random generation is.

**More Details:** Dividing logits by a low temperature sharpens the distribution, making likely tokens more dominant. A high temperature flattens the distribution, giving rare tokens more chance. The docs suggest about 0.7–0.9 for coherent but varied text.

**Related:** Logits, Softmax, Sampling, Greedy Decoding

## Top-k Sampling

**Core idea:** Top-k sampling limits sampling to the `k` most likely tokens.

**More Details:** Tokens outside the top `k` have their logits set to negative infinity so softmax gives them zero probability. This prevents extremely unlikely choices while still allowing variety among plausible tokens.

**Related:** Sampling, Temperature, Negative Infinity, Probability Distribution

## Negative Infinity (`-inf`)

**Core idea:** `-inf` is used to make a token impossible after softmax.

**More Details:** In top-k filtering, logits below the cutoff are set to `-inf`. Since `exp(-inf)` is zero, those tokens receive zero probability.

**Related:** Top-k Sampling, Softmax, Logits

## Multinomial

**Core idea:** Multinomial sampling draws an index from a probability distribution.

**More Details:** The generation function uses `torch.multinomial` to choose the next token from softmax probabilities. This is what makes output vary from run to run.

**Related:** Sampling, Probability Distribution, Seed

## Seed

**Core idea:** A seed initializes the random number generator.

**More Details:** Setting the same seed makes random sampling reproducible. For the competition, the seed is included so judges can regenerate the exact submitted poem.

**Related:** Reproducibility, Sampling, Generation Command

## Reproducibility

**Core idea:** Reproducibility means someone else can get the same result.

**More Details:** For generation, reproducibility requires the same checkpoint, prompt, temperature, top-k, seed, and code. The competition submission asks for these exact ingredients.

**Related:** Seed, Checkpoint, Generation Command

## Generation Command

**Core idea:** A generation command is the exact terminal command used to produce output.

**More Details:** It includes the checkpoint path and generation settings such as prompt, temperature, top-k, and seed. In the competition, this command proves the poem came from the model.

**Related:** Reproducibility, Checkpoint, Prompt, Seed

## Cherry-Pick

**Core idea:** Cherry-picking means generating multiple outputs and choosing the best one.

**More Details:** The competition allows this. It does not change the model output manually; it only selects the strongest sample from several model-generated attempts.

**Related:** Sampling, Generation

## Checkpointing

**Core idea:** Checkpointing is periodically saving model state during training.

**More Details:** The workshop saves checkpoints every 1000 steps and a final checkpoint at the end. Frequent checkpoints let you return to better validation-loss regions instead of being stuck with the final model.

**Related:** Checkpoint, Early Stopping, Validation Loss

## Loss Log

**Core idea:** A loss log records loss values over time.

**More Details:** The workshop saves `loss_log.json` with training and validation losses. Plotting it shows whether the model is learning, plateauing, spiking, or overfitting.

**Related:** Loss Curve, Train Loss, Validation Loss

## Loss Curve

**Core idea:** A loss curve is a plot of loss over training steps.

**More Details:** It helps diagnose training behavior visually. A widening gap between train and validation loss indicates overfitting. Flat lines, spikes, or rising values suggest tuning or debugging is needed.

**Related:** Loss Log, Overfitting, Training Stability

## Google Colab

**Core idea:** Google Colab is a hosted notebook environment that can provide GPUs.

**More Details:** The docs include Colab instructions for attendees who are not training locally. Colab changes file paths and setup steps, but the model and training concepts are the same.

**Related:** Notebook, T4, Runtime

## Notebook

**Core idea:** A notebook is an interactive document for running code cells.

**More Details:** In Colab, attendees can write model, training, and generation code in notebook cells instead of local files. This can be easier for workshops but requires attention to paths and runtime setup.

**Related:** Google Colab, Runtime, Cell

## Runtime

**Core idea:** A runtime is the execution environment for code.

**More Details:** In Colab, changing the runtime type to GPU gives access to accelerated hardware. Locally, the runtime is your Python environment and available device.

**Related:** Google Colab, T4

## T4

**Core idea:** T4 is an NVIDIA GPU commonly available in Google Colab.

**More Details:** It can accelerate training compared with CPU. It uses CUDA through PyTorch.

**Related:** CUDA, Google Colab

## Dependency

**Core idea:** A dependency is an external package the code needs.

**More Details:** The docs mention installing packages like `torch`, `numpy`, `tqdm`, and `tiktoken` in Colab. Dependencies must be available before running the workshop code.

**Related:** PyTorch, `tqdm`, `tiktoken`, Colab

## PyTorch

**Core idea:** PyTorch is the deep learning library used in the workshop.

**More Details:** It provides tensors, neural network modules, autograd, optimizers, devices, model saving/loading, and functions like cross-entropy and scaled dot-product attention.

**Related:** Tensor, `nn.Module`, Autograd, Optimizer

## `nn.Module`

**Core idea:** `nn.Module` is PyTorch's base class for neural network components.

**More Details:** The workshop defines `GPT`, `CausalSelfAttention`, `MLP`, and `Block` as modules. Modules hold parameters and define forward passes.

**Related:** PyTorch, Model, Parameters, Forward Pass

## `nn.Embedding`

**Core idea:** `nn.Embedding` is PyTorch's lookup table module for embeddings.

**More Details:** Given token or position IDs, it returns learned vectors. The workshop uses it for both token embeddings and position embeddings.

**Related:** Embedding Table, Token Embedding, Position Embedding

## `nn.Linear`

**Core idea:** `nn.Linear` is PyTorch's linear layer module.

**More Details:** It applies a learned affine transformation to vectors. The workshop uses linear layers in attention, MLP, and the language-modeling head.

**Related:** Linear Layer, Projection, Weight, Bias

## `nn.LayerNorm`

**Core idea:** `nn.LayerNorm` is PyTorch's layer normalization module.

**More Details:** It normalizes each position's vector to stabilize transformer training. The workshop uses it before attention, before MLP, and after the final block stack.

**Related:** Layer Normalization, Pre-Norm, Training Stability

## Autograd

**Core idea:** Autograd is PyTorch's automatic differentiation system.

**More Details:** It records operations during the forward pass and computes gradients during `loss.backward()`. `torch.no_grad()` disables this recording when gradients are unnecessary.

**Related:** Backpropagation, Gradient, PyTorch, `torch.no_grad()`

## `tqdm`

**Core idea:** `tqdm` is a progress bar library.

**More Details:** The training loop uses it to show step progress, current loss, learning rate, validation messages, and generated samples during training.

**Related:** Training Loop, Progress Bar

## Monitoring

**Core idea:** Monitoring means watching training signals as the model learns.

**More Details:** Useful signals include train loss, validation loss, generated samples, loss curves, and checkpoint quality. Monitoring helps decide whether to stop early or tune settings.

**Related:** Loss Curve, Validation Loss, Sample Generation

## Sample Generation

**Core idea:** Sample generation means periodically generating text during training.

**More Details:** The docs generate text every 100 steps so attendees can watch progress from random characters to word-like and Shakespeare-like output. It gives qualitative feedback that loss alone cannot provide.

**Related:** Generation, Monitoring, Training Loop

## Baseline

**Core idea:** A baseline is a standard starting configuration for comparison.

**More Details:** The default 6L/6H/384D model is a baseline. Experiments with smaller or larger models, different learning rates, or longer context are meaningful because they can be compared against it.

**Related:** Experiment, Model Size, Hyperparameter

## Experiment

**Core idea:** An experiment changes one or more settings to learn their effect.

**More Details:** The docs suggest comparing model sizes, context lengths, and learning rates. Good experiments isolate variables so you know what caused the change in loss or generation quality.

**Related:** Baseline, Hyperparameter, Monitoring

## Trained From Scratch

**Core idea:** Trained from scratch means starting with randomly initialized weights, not pretrained weights.

**More Details:** The competition requires this so everyone learns and demonstrates the full training process. You may modify the code, but you may not load weights from another trained model.

**Related:** Pretrained Weights, Fine-Tuning

## Pretrained Weights

**Core idea:** Pretrained weights are parameters learned from previous training.

**More Details:** The competition forbids using them. The workshop references GPT-2 architecture ideas but does not permit loading GPT-2's learned parameters.

**Related:** Trained From Scratch, Fine-Tuning, Checkpoint

## Fine-Tuning

**Core idea:** Fine-tuning means continuing training from an already-trained model on new data.

**More Details:** Fine-tuning is common in practice, but the competition disallows it. Participants must train their own model from scratch.

**Related:** Pretrained Weights, Trained From Scratch

## Coherence

**Core idea:** Coherence means generated text makes sense as a whole.

**More Details:** In the poetry competition, coherence asks whether lines connect and form a plausible poem. Too much randomness can hurt coherence.

**Related:** Temperature, Top-k Sampling, Structure

## Structure

**Core idea:** Structure is the organization of generated text.

**More Details:** For poetry, structure includes line breaks, rhythm, stanzas, speaker labels, or rhyme-like patterns. Longer context and good poetry data can help the model learn structure.

**Related:** Context Window, Line Break, Dataset

## Poetry Dataset

**Core idea:** A poetry dataset is training text selected specifically for poetic output.

**More Details:** For the competition, sources might include Poetry Foundation, Project Gutenberg, Shakespeare, sonnets, or modern free verse. The goal is to align training data with the desired generated style.

[HuggingFace Poetry Dataset](https://huggingface.co/datasets/suayptalha/Poetry-Foundation-Poems)
[HuggingFace Gutenberg Poetry Dataset](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus)

**Related:** Data Curation, Dataset

## Poetry Foundation

**Core idea:** Poetry Foundation is a possible source of poetry data.

**More Details:** The docs list it as an example source with many poems across styles and eras. Any collected data should still be curated and cleaned.

[Poetry Foundation Site](https://www.poetryfoundation.org/)

**Related:** Poetry Dataset, Data Curation, Project Gutenberg

## Project Gutenberg

**Core idea:** Project Gutenberg is a source of public-domain texts, including poetry collections.

**More Details:** It can provide legally accessible poetry data, but downloaded books often contain headers, metadata, and non-poem text that should be removed.

[Project Gutenberg Site](https://www.gutenberg.org/)

**Related:** Poetry Dataset, Data Curation, Public Domain

## Public Domain

**Core idea:** Public domain works can be used without copyright restrictions.

**More Details:** Project Gutenberg primarily hosts public-domain texts. Public-domain data is useful for workshops because participants can share datasets and checkpoints more safely.

**Related:** Project Gutenberg, Dataset, Data Curation

## OpenWebText

**Core idea:** OpenWebText is a large web-text dataset.

**More Details:** The tokenization docs mention it as the kind of large dataset where BPE becomes appropriate. It is not used in the workshop's Shakespeare setup.

[Hugging Face Site](https://huggingface.co/datasets/Skylion007/openwebtext)

**Related:** BPE, Large Dataset, Tokenization

## TinyStories

**Core idea:** TinyStories is a dataset of simple stories used for training small language models.

**More Details:** The docs mention it as an example of much more data than Shakespeare. More data would allow a 10M parameter model to train longer without overfitting as quickly.

**Related:** Dataset, Data Size, Overfitting

## Chinchilla

**Core idea:** Chinchilla refers to research on balancing model size and training data.

**More Details:** The docs cite it for optimal scaling of data versus parameters. The workshop's practical lesson is similar: do not scale model size without also scaling data.

**Related:** Data Size, Parameter Count, Model Capacity

## Attention Is All You Need

**Core idea:** “Attention Is All You Need” is the original transformer paper.

**More Details:** It introduced the transformer architecture. The workshop uses a GPT-style descendant of that architecture, with modern choices such as pre-norm.

[Original Paper](https://arxiv.org/abs/1706.03762)

**Related:** Transformer, Attention, Post-Norm

## nanoGPT

**Core idea:** nanoGPT is a compact GPT training implementation by Andrej Karpathy.

**More Details:** The docs reference build-nanogpt as further reading. It is a useful next step after this workshop because it expands the same core ideas into a more complete implementation.

**Related:** GPT, Transformer, Further Reading

## nanochat

**Core idea:** nanochat is a full ChatGPT-style training pipeline referenced for further reading.

**More Details:** It goes beyond this workshop's single small GPT by covering more of the end-to-end process behind chat models.

**Related:** Further Reading, GPT, Training Pipeline

## Training Pipeline

**Core idea:** A training pipeline is the full workflow from data to trained model and generation.

**More Details:** In this workshop, the pipeline includes tokenization, model definition, training loop, checkpointing, generation, monitoring, and competition submission.

**Related:** Tokenization, Training Loop, Checkpoint, Generation
