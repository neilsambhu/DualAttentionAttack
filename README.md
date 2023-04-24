# DualAttentionAttack

## Setup

If you are setting up afresh, install the pre-requisites by running `install.sh`.

```bash
sh install.sh
```

## CLI APIs

### Prepare dataset

The followin command could be used to create the dataset from the original dataset for later training.

```bash
python main.py prepare \
  --batch-size 1 \
  --dataset ./dataset \
  --output ./dataset

python main.py prepare --help # for more information
```

### Training

To train the model on the prepared dataset, run the following command.

```bash
python main.py train \
  --dataset ./dataset
  --model-dst ./models

python main.py train --help # for more information
```
