# Hitori-ppo

**Course context:** This repository is the course project for **IERG5350 Reinforcement Learning** . It implements **cold-start supervised fine-tuning**, **Maskable PPO**, a **U-Net-style** policy backbone, **variable-size** training/evaluation (padded observations), and **shaped** rewards for the RL stage.

**Main result (solve rate):** comparison among a CSP-style symbolic solver (wall-clock **≤ 1 s** per puzzle), **cold-start SFT** only, and the full **SFT + RL** pipeline. Values are per board side length \(n\).

| Size | CSP-style solver (≤ 1 s) | Cold-start SFT | SFT + RL |
|:---:|:---:|:---:|:---:|
| 4 | 1.000 | 0.740 | 0.770 |
| 5 | 1.000 | 0.750 | 0.800 |
| 6 | 1.000 | 0.685 | 0.805 |
| 7 | 1.000 | 0.595 | 0.670 |
| 8 | 1.000 | 0.565 | 0.645 |
| 9 | 1.000 | 0.515 | 0.585 |
| 10 | 0.995 | 0.455 | 0.510 |
| 11 | 0.950 | 0.340 | 0.400 |
| 12 | 0.900 | 0.315 | 0.410 |
| 13 | 0.780 | 0.245 | 0.430 |
| 14 | 0.610 | 0.275 | 0.450 |
| 15 | 0.465 | 0.195 | 0.395 |
| 16 | 0.265 | 0.145 | 0.350 |
| 17 | 0.170 | 0.110 | 0.260 |
| 18 | 0.095 | 0.125 | 0.395 |
| 19 | 0.055 | 0.105 | 0.310 |
| 20 | 0.020 | 0.085 | 0.285 |

**Evaluation protocol:** For each board size \(n\), we randomly sample **200 solvable** puzzles (same underlying generator as `hitori_env`) and report the **solve rate** (success fraction). The CSP column uses a **≤ 1 s** wall-clock limit per puzzle; SFT and SFT+RL columns follow the policy-eval settings used to produce those runs (Gym rollout until success/failure under the configured step limit).  


---

This repository trains **U-Net** (and other) policies on a [Gymnasium](https://gymnasium.farama.org/) **Hitori** board with [**MaskablePPO**](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) (`sb3-contrib`): legal moves are enforced with an **action mask**. The pipeline is:

1. Generate **supervision** (grid + expert shading) on random puzzles with the bundled **CSP solver**;
2. **`ppo.train_sft`**: behavioral cloning on those trajectories → **MaskablePPO `.zip`** compatible with `ppo.train`;
3. **`ppo.train`**: optional **`--pretrain-policy`** from SFT + **critic warmup**, then PPO under **sparse / shaped** rewards.

Random puzzles use the same `generate_random_hitori_game` as `hitori_env` `reset(seed=…)`; multi-size training pads each `n×n` board into a fixed **`max_size×max_size`** observation via **`PadToMaxBoardSizeWrapper`**.

---

## Environment setup

Use **Python 3.12** and a **CUDA build of PyTorch** (typical for SB3). From the repo root:

```bash
conda create -n hitori-gym python=3.12 -y
conda activate hitori-gym

python -m pip install --upgrade pip
# Install PyTorch for your CUDA version, e.g. from https://pytorch.org
# pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install -e .
```

Check GPU before long runs:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

See all flags: `python -m ppo.train --help`, `python -m ppo.train_sft --help`.

---

## End-to-end workflow and CLI (reference recipe)

Run the following from the **repository root** with `conda activate hitori-gym`. Adjust `--out-dir` / `--data-root` as needed.

### 1. Build SFT supervision data

`scripts/generate_solver_supervision_dataset.py`: same random puzzle distribution as training, multiprocess CSP in `hitori-solver/source`; only puzzles solved within wall-clock **`--time-limit`** are saved as `size_xx/item_*.npz`. **Resume-safe** (existing samples are not regenerated).

**Recommended: sides 4–20, 1000 saved puzzles per side, 128 workers, 60s:**

```bash
python scripts/generate_solver_supervision_dataset.py \
  --out-dir runs/data/solver_supervision_4_20 \
  --min-size 4 --max-size 20 \
  --per-size 1000 \
  --workers 128 \
  --time-limit 60 \
  --rng-base 100000
```

Outputs: `runs/data/solver_supervision_4_20/size_{NN}/item_*.npz`, plus `manifest.jsonl` and `dataset_state.json`.

### 2. Supervised fine-tuning: `ppo.train_sft`

Builds row-major expert trajectories from step **1** `--out-dir`, trains a U-Net Maskable policy; validation split is by **whole puzzles**, not shuffled steps. Writes **`sft_maskable_ppo_best.zip`** for `ppo.train --pretrain-policy`.

**Recommended (`max_n` / `max-size` 20 for downstream RL; up to 1000 puzzles per side):**

```bash
python -m ppo.train_sft \
  --data-root runs/data/solver_supervision_4_20 \
  --min-size 4 --max-size 20 \
  --max-n 20 \
  --per-size 1000 \
  --out-dir runs/sft/unet_bc_4_20 \
  --epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --num-attention-blocks 0 \
  --device cuda
```

Main artifacts: `runs/sft/unet_bc_4_20/sft_maskable_ppo_best.zip`, `sft_summary.json`, `sft_meta.json`.

### 3. RL training: `ppo.train`

**MaskablePPO** on mixed board sizes. **Reference “full” recipe**: **shaped** rewards, `--pretrain-policy` from step **2**, **100k critic warmup**, train/eval **4–20**, `max-size 20`, **4M** timesteps (same spirit as `cmd2.sh` / `final_exp` group 0).

```bash
CUDA_VISIBLE_DEVICES=0 python -m ppo.train \
  --model unet \
  --num-attention-blocks 0 \
  --learning-rate 1e-4 \
  --train-sizes 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --eval-sizes 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --max-size 20 \
  --vec-env subproc \
  --sampling-strategy round_robin \
  --n-envs 128 \
  --batch-size 256 \
  --n-steps 32 \
  --reward-mode shaped \
  --critic-warmup-timesteps 100000 \
  --pretrain-policy runs/sft/unet_bc_4_20/sft_maskable_ppo_best.zip \
  --total-timesteps 4000000 \
  --eval-freq 50000 \
  --eval-episodes 50 \
  --final-eval-episodes 200 \
  --eval-vec-envs 256 \
  --seed 44 \
  --device cuda \
  --out-dir runs/rl/unet_shaped_4_20_4m
```

Default eval suite seed: `eval_seed = seed + 10000` if `--eval-seed` is omitted. After training, each run subdirectory gets **`final_eval.txt`**, `learning_curve.csv`, checkpoints, etc.

**Evaluate an existing `.zip` only (same eval path as training):**

```bash
python -m ppo.eval_checkpoint \
  runs/sft/unet_bc_4_20/sft_maskable_ppo_best.zip \
  --max-size 20 --min-size 4 --eval-max-size 20 \
  --episodes 200 --eval-seed 10044 \
  --vec-envs 256 \
  --reward-mode shaped \
  --shaped-reward-norm none \
  --shaped-dense-mult 1.0 \
  --shaped-step-penalty -0.05 \
  --device cuda
```

---

## Repository layout (overview)

```text
hitori-ppo/
├── README.md                 # This file
├── requirements.txt          # gymnasium, sb3, torch, etc.
├── pyproject.toml            # Editable install: hitori_env + ppo
├── playground.py             # Local manual play (not the package entry)
├── hitori_env/               # Gymnasium: hitori_env/Hitori-v2
│   └── envs/                 # Rules, random generator, rendering
├── ppo/
│   ├── train.py              # MaskablePPO training entrypoint
│   ├── train_sft.py          # SFT / behavioral cloning
│   ├── eval_checkpoint.py    # Standalone .zip evaluation
│   ├── env.py                # Padding, masks, multi-size, eval slots
│   ├── evaluation.py         # evaluate_all_sizes, curves
│   └── policies.py           # U-Net / CNN / structured
├── hitori-solver/            # Vendored CSP (dataset script uses source/)
├── scripts/                  # Data gen, plots, batch experiments, …
├── runs/                     # Default logs and artifacts (often gitignored)
└── media/                    # Demo assets
```

---

## License

MIT — see [`LICENSE`](LICENSE) in the repo root. The environment and solver originate from upstream projects: [hitori-gym](https://github.com/Vibhu-Agarwal/hitori-gym), [hitori-solver](https://github.com/philiplugt/hitori-solver); refer to those repositories for their licenses and notices.
