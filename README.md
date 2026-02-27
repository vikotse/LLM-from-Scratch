# LLM-from-Scratch

Build a Large Language Model from scratch, following the Stanford CS336 assignments.

## assignment1

**What you will implement**

1. Byte-pair encoding (BPE) tokenizer (§2)
2. Transformer language model (LM) (§3)
3. The cross-entropy loss function and the AdamW optimizer (§4)
4. The training loop, with support for serializing and loading model and optimizer state (§5)

**What you will run**
1. Train a BPE tokenizer on the TinyStories dataset.
2. Run your trained tokenizer on the dataset to convert it into a sequence of integer IDs.
3. Train a Transformer LM on the TinyStories dataset.
4. Generate samples and evaluate perplexity using the trained Transformer LM.
5. Train models on OpenWebText and submit your attained perplexities to a leaderboard.

### Directory structure

The `assignment1-basics` directory contains the following key subdirectories and files:

- `cs336_basics/`: Original starter code and materials from the CS336 assignment.
- `data/`: Downloaded datasets (TinyStories, OpenWebText samples, etc.).
- `model/`: Tokenizer vocab/merges files and related model artifacts.
- `runs/`: Training checkpoints and run artifacts.
- `script/`: Utility and training scripts (e.g., tokenizer training, binarization, LM training).
- `src/`: Main source code for the tokenizer, model, config, and training utilities.
- `tests/`: Unit tests and the adapter layer connecting your implementations to the autograder.
- `wandb/`: Weights & Biases run logs (if logging is enabled).
- `pyproject.toml` / `uv.lock`: Project configuration and dependency lockfile managed by `uv`.
- `make_submission.sh`: Helper script for packaging and submitting your assignment.

### Setup

#### Environment

Prepare the environment with `uv` as described in  
[assignment1 README – Environment](assignment1-basics/README.md#environment).

#### Data

Download the pretraining datasets as described in  
[assignment1 README – Download Data](assignment1-basics/README.md#download-data).

### Quick Start

#### Unit tests

Run unit tests for the components which have implemented:

1. `cd assignment1-basics`
2. Run unit tests (they call functions in  
   [assignment1-basics/tests/adapters.py](assignment1-basics/tests/adapters.py)):
   - Run all 48 tests: `uv run pytest`
   - Run a specific component test, e.g. `uv run pytest -k test_transformer_lm`

#### Training scripts

#### Run
1. Train a BPE tokenizer on the TinyStories dataset.

    `uv run python script/train_bpe_tokenizer.py`

2. Run your trained tokenizer on the dataset to convert it into a sequence of integer IDs.

    `uv run python script/tokenize_and_bin.py`

3. Train a Transformer LM on the TinyStories dataset.

    `uv run python script/train.py`
