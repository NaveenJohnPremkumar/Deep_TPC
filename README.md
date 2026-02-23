# Deep TPC: Temporal-Prior Conditioning for Time Series Forecasting

Official implementation of the paper **Deep TPC: Temporal-Prior Conditioning for Time Series Forecasting**.

## Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU training)

### Installation

```bash
pip install -r requirements_minimal.txt
```

For the full environment (including M4, preprocessing, etc.), use `requirements.txt`.

### Data

Place datasets in `./dataset/` (or set `--root_path`). Supported datasets include:
- **ETT** (ETTh1, ETTh2, ETTm1, ETTm2): [Download](https://github.com/zhouhaoyi/ETDataset)
- **Weather**, **Traffic**, **ECL**, **Solar**: See `data_provider/data_factory.py` for paths.

## Usage

### Long-term forecasting

**Training:**
```bash
cd src
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model GPT2WithMM \
  --data ETTh1 \
  --seq_len 672 --label_len 576 --token_len 96 \
  --test_seq_len 672 --test_label_len 576 --test_pred_len 96 \
  --batch_size 256 --learning_rate 0.0005 \
  --train_epochs 10 --use_amp --cosine --tmax 10 \
  --mix_embeds --drop_last \
  --mm_layers 0 2 4 6 8 10 --num_fusion_tokens 20 \
  --llm_ckp_dir gpt2
```

**Testing:**
```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model GPT2WithMM \
  --data ETTh1 \
  --llm_ckp_dir gpt2 \
  --test_dir <path_to_checkpoint_folder>
```

### Models

| Model | Description |
|-------|-------------|
| `GPT2WithMM` | GPT-2 with multimodal blocks; time patches as sequence, mark embeddings as MM context |
| `GPT2WithMM2` | GPT-2 with MM; prompt + time patches as sequence; mark or time as MM context |
| `GPT2WithMMWithPrompt` | GPT-2 with MM; prompt + time patches; uses text prompts |
| `AutoTimes_Gpt2` | Vanilla GPT-2 (no MM blocks) |
| `AutoTimes_Gpt2_concatanate` | Vanilla GPT-2 with concatenation-based mark fusion |

### Key arguments

- `--llm_ckp_dir`: Path to GPT-2 checkpoint (e.g. `gpt2` for auto-download, or local path)
- `--mm_layers`: Indices of GPT-2 layers replaced with MM blocks (e.g. `0 2 4 6 8 10`)
- `--num_fusion_tokens`: Number of learnable fusion tokens
- `--mix_embeds`: Fuse time-series and mark (timestamp) embeddings

## Citation

```bibtex
@article{deeptpc2025,
  title={Deep TPC: Temporal-Prior Conditioning for Time Series Forecasting},
  author={...},
  year={2025}
}
```

### SLURM

SLURM scripts are in `src/slurms/`. Before running:
1. Create `./logs/` for output/error files (or update `#SBATCH --output` and `--error` paths).
2. Set `#SBATCH --mail-user` to your email.
3. Adjust `#SBATCH --account` and `#SBATCH --partition` for your cluster.

## License

See LICENSE file.
