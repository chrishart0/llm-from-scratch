# Technical Notes

This file is for details not already explained in `docs/`. It should stay biased toward package graph facts, operational caveats, and maintenance notes rather than re-teaching the workshop.

## Packages

Currently used by the baseline snippets:

- `torch>=2.8.0`
- `tqdm>=4.67.3`

Eagerly added for common follow-on experiments:

- `datasets>=4.5.0`
- `huggingface-hub>=1.8.0`
- `numpy>=2.0.2`
- `tiktoken>=0.12.0`

Lockfile resolution at the time of writing:

- `datasets`: `4.8.4`
- `huggingface-hub`: `1.9.0`
- `numpy`: `2.4.4`
- `tiktoken`: `0.12.0`
- `torch`: `2.11.0`
- `tqdm`: `4.67.3`

## Dependency Graph Notes

- `torch` does not declare `numpy` as a dependency in the current lockfile. `numpy` is present because this project declares it directly and because `datasets` pulls it in.
- `datasets` is the heaviest eager dependency. It brings in `pandas`, `pyarrow`, `multiprocess`, `dill`, `xxhash`, HTTP clients, and `tqdm`.
- `huggingface-hub` brings in `httpx`, `typer`, `pyyaml`, `fsspec`, and `hf-xet` on common CPU architectures.
- `tiktoken` brings in `regex` and `requests`.
- `torch` brings in `filelock`, `fsspec`, `jinja2`, `networkx`, `setuptools`, `sympy`, and `typing-extensions`; Linux resolves additional CUDA/NVIDIA packages.
- `tqdm` only adds `colorama` on Windows.

## Versioning

- The dependency specifiers in `pyproject.toml` are all lower bounds. They allow future major/minor releases unless the lockfile is honored.
- `uv.lock` is the real reproducibility boundary. `pyproject.toml` alone is intentionally loose.
- The lockfile includes wheels for multiple Python versions and platforms; it is larger than a single-machine install plan.
- Python is constrained to `>=3.12`, but the lockfile already contains artifacts for newer CPython versions too.
- If workshop reproducibility matters more than freshness, prefer exact pins or keep `uv.lock` committed and instruct users to install from it.

## Package Pruning

- A minimal local install can drop `datasets`, `huggingface-hub`, `numpy`, and `tiktoken` if the user only follows the baseline path.
- Dropping `datasets` removes most of the non-PyTorch dependency tree.
- Dropping `numpy` is technically possible for the baseline because all tensor creation in the current snippets can be done through `torch`.
- Keeping `tiktoken` is defensible if the environment is meant to support the tokenizer extension without another install step.
- Keeping `huggingface-hub` without `datasets` is only useful if code directly downloads hub artifacts.

## Serialization

- `torch.save` checkpoints are pickle-backed. Treat checkpoints as code-adjacent artifacts, not inert data files.
- `torch.load(..., weights_only=False)` can deserialize non-tensor Python objects. Only load checkpoints from trusted sources.
- If checkpoints need to be shareable across untrusted boundaries, prefer a tensor-only `state_dict` plus a separate JSON config/tokenizer metadata file.
- Dataclass configs stored inside checkpoints couple the file to the Python import path and class name used during save/load.

## Reproducibility

- `torch.manual_seed` is not a full cross-device determinism guarantee. CPU, CUDA, and MPS can still diverge because kernels and floating-point reduction order differ.
- Sampling reproducibility depends on more than seed: PyTorch version, backend, prompt normalization, tokenizer mapping, and exact checkpoint all matter.
- Sorting corpus characters makes the mapping deterministic for identical text bytes, but line-ending changes can alter the vocabulary.
- `open(..., "r")` uses the platform default encoding unless specified. For portable runs, prefer `encoding="utf-8"`.

## Data Hygiene

- Text files downloaded from different mirrors can differ in leading/trailing whitespace, license headers, Unicode normalization, or line endings.
- For dataset comparisons, record byte size and a content hash next to loss curves.
- If multiple corpora are concatenated, insert an explicit separator. Otherwise the model receives artificial transitions between unrelated documents.
- For competition submissions, save the exact preprocessing script or command, not just the final checkpoint.

## Device Edges

- MPS availability does not mean every operation has identical performance or numerics to CUDA.
- Moving batches to device inside the batch function hides transfer cost in the data path. That is fine here, but it makes profiling less obvious.
- Apple unified memory can make out-of-memory behavior look different from CUDA: the process may slow dramatically before failing.
- CPU fallback is useful for correctness checks, but performance expectations should be recorded separately per backend.

## Maintenance

- This file should not duplicate architecture, tokenization, loss-scale, or generation walkthroughs from `docs/`.
- If a note starts explaining how GPT works, it probably belongs in the tutorial docs instead.
- If a note explains why the environment behaves a certain way, why a dependency exists, or how artifacts should be handled, it belongs here.
