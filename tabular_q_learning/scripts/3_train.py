#!/usr/bin/env python3
"""
3_train.py — CE-Only Training Loop

Trains the COCONUTTransformer from 2_model.py on the dataset produced by
1_generate_data.py using a single cross-entropy loss on action predictions:

    CE_loss = CrossEntropyLoss(action_logits, select_targets)  at <Select> positions

The model receives NO Q-value supervision. It must discover Q-value tracking
internally to predict optimal actions. The COCONUT continuous thought mechanism
(THINK → COT injection) provides the recurrent state needed for this.

Training details:
    - Optimizer  : AdamW(lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95))
    - Scheduler  : cosine decay with 100-step warmup, reset per curriculum stage
    - Mixed prec : torch.cuda.amp.autocast + GradScaler
    - Epochs     : 28  (--epochs flag)
    - Batch size : 64  (--batch_size flag)
    - Eval every : 500 steps

Curriculum (matching MWU Section 5.1 fine-grained approach):
    Stage 1:  epochs  1– 3  →  max_steps= 1
    Stage 2:  epochs  4– 6  →  max_steps= 2
    Stage 3:  epochs  7– 9  →  max_steps= 3
    Stage 4:  epochs 10–12  →  max_steps= 5
    Stage 5:  epochs 13–15  →  max_steps=10
    Stage 6:  epochs 16–18  →  max_steps=20
    Stage 7:  epochs 19–21  →  max_steps=30
    Stage 8:  epochs 22–28  →  max_steps=50  (7 epochs — harder sequences need more time)

Each stage starts a fresh cosine LR schedule with 100-step warmup.
10% replay from earlier stages to prevent catastrophic forgetting.

Ablation: pass --no_coconut to use model.forward() instead of model.forward_coconut().
This trains the same model without continuous thought feedback, as a baseline.
The use_coconut flag is saved in the checkpoint for automatic detection in eval.

Checkpoint format (saved to checkpoints/coconut_transformer.pt):
    {
        'model_state_dict': ...,
        'config':           config.to_dict(),
        'step':             global_step,
        'epoch':            epoch,
        'val_loss':         best_val_loss,
        'val_ce_loss':      ...,
        'val_acc':          ...,
        'use_coconut':      True/False,
    }
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

# Import model from 2_model.py — file starts with a digit so use importlib
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "coconut_model",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_model.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
COCONUTConfig      = _mod.COCONUTConfig
COCONUTTransformer = _mod.COCONUTTransformer


# ---------------------------------------------------------------------------
# Curriculum stage definitions: (last_epoch_inclusive, max_steps)
# ---------------------------------------------------------------------------

CURRICULUM_STAGES = [
    (4,  5),
    (8,  10),
    (12, 20),
    (18, 35),
    (28, 50)
]
STAGE_MAX_STEPS = [ms for _, ms in CURRICULUM_STAGES]


# ---------------------------------------------------------------------------
# Sequence truncation helper (for curriculum training)
# ---------------------------------------------------------------------------

def truncate_sequence(seq: Dict, max_steps: int) -> Dict:
    """Return a copy of seq truncated to at most max_steps rounds.

    input_ids has a 2-token prefix [TOK_NULL, TOK_START] followed by
    n_steps rounds of round_len tokens each. round_len is derived from
    the actual data so it works for any n_actions.
    """
    n = min(seq['n_steps'], max_steps)
    if n == seq['n_steps']:
        return seq   # nothing to truncate

    # Derive round_len from actual data (avoids hardcoding)
    full_n   = seq['n_steps']
    full_len = len(seq['input_ids'])
    round_len = (full_len - 2) // full_n   # 2-token prefix

    trunc = dict(seq)
    trunc['input_ids'] = seq['input_ids'][:2 + n * round_len]
    for field in ('reward_values', 'reward_positions',
                  'select_positions', 'select_targets',
                  'think_positions', 'cot_positions'):
        if field in seq:
            trunc[field] = seq[field][:n]
    trunc['n_steps'] = n
    return trunc


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCONUTDataset(Dataset):
    """Wraps the list of sequence dicts produced by 1_generate_data.py.

    Supports curriculum training via the max_steps attribute. Set
    train_ds.max_steps = k to limit sequences to k rounds per sample.
    With probability 0.1 (when stage_idx > 0) a random earlier-stage
    max_steps is sampled to prevent catastrophic forgetting.
    """

    def __init__(self, sequences: List[Dict]):
        self.sequences           = sequences
        self.max_steps           = None   # None = no truncation (use full sequence)
        self.stage_idx           = 0      # current curriculum stage index
        self.all_stage_max_steps = STAGE_MAX_STEPS

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        effective = self.max_steps
        if effective is not None and self.stage_idx > 0 and random.random() < 0.1:
            # 10% chance: sample from a previous stage to prevent forgetting
            effective = random.choice(self.all_stage_max_steps[:self.stage_idx])
        if effective is not None:
            seq = truncate_sequence(seq, effective)
        return seq


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length COCONUT sequences.

    Padding conventions
    -------------------
    input_ids        : padded with 0 (TOK_NULL)
    reward_positions : padded with -1  (valid = positions >= 0)
    reward_values    : padded with 0.0
    select_positions : padded with -1
    select_targets   : padded with 0   (ignored where positions == -1)
    think_positions  : padded with -1
    cot_positions    : padded with -1
    """
    max_seq   = max(len(s['input_ids']) for s in batch)
    max_steps = max(s['n_steps'] for s in batch)

    def pad_ids(ids: List[int], length: int) -> List[int]:
        return ids + [0] * (length - len(ids))

    def pad_ints(lst: List[int], length: int, pad: int = -1) -> List[int]:
        return lst + [pad] * (length - len(lst))

    def pad_floats(lst: List[float], length: int, pad: float = 0.0) -> List[float]:
        return lst + [pad] * (length - len(lst))

    input_ids_list        = []
    reward_values_list    = []
    reward_positions_list = []
    select_positions_list = []
    select_targets_list   = []
    think_positions_list  = []
    cot_positions_list    = []

    for s in batch:
        input_ids_list.append(pad_ids(s['input_ids'], max_seq))
        reward_values_list.append(pad_floats(s['reward_values'], max_steps))
        reward_positions_list.append(pad_ints(s['reward_positions'], max_steps))
        select_positions_list.append(pad_ints(s['select_positions'], max_steps))
        select_targets_list.append(pad_ints(s['select_targets'], max_steps, pad=0))
        think_positions_list.append(pad_ints(s.get('think_positions', []), max_steps))
        cot_positions_list.append(pad_ints(s.get('cot_positions', []), max_steps))

    return {
        'input_ids':        torch.tensor(input_ids_list,        dtype=torch.long),
        'reward_values':    torch.tensor(reward_values_list,    dtype=torch.float32),
        'reward_positions': torch.tensor(reward_positions_list, dtype=torch.long),
        'select_positions': torch.tensor(select_positions_list, dtype=torch.long),
        'select_targets':   torch.tensor(select_targets_list,   dtype=torch.long),
        'think_positions':  torch.tensor(think_positions_list,  dtype=torch.long),
        'cot_positions':    torch.tensor(cot_positions_list,    dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Scheduler: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup for `warmup_steps`, then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss computation — CE only
# ---------------------------------------------------------------------------

def compute_loss(
    action_logits:    torch.Tensor,   # [B, n_sel, n_actions]
    select_positions: torch.Tensor,   # [B, n_sel]  (-1 = pad)
    select_targets:   torch.Tensor,   # [B, n_sel]
) -> Tuple[torch.Tensor, float]:
    """Cross-entropy loss at SELECT positions + action accuracy.

    Returns (ce_loss, accuracy).
    """
    device = action_logits.device
    valid = (select_positions >= 0)           # [B, n_sel]
    n_valid = valid.sum().item()

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    logits  = action_logits[valid]            # [N, n_actions]
    targets = select_targets[valid]           # [N]

    loss = F.cross_entropy(logits, targets)
    acc  = (logits.argmax(dim=-1) == targets).float().mean().item()
    return loss, acc


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:       COCONUTTransformer,
    val_loader:  DataLoader,
    device:      torch.device,
    use_coconut: bool = True,
    max_batches: Optional[int] = 50,
) -> Tuple[float, float]:
    """Evaluate model on validation set.

    Returns (mean_ce, mean_acc) averaged over up to max_batches.
    """
    model.eval()
    ce_total  = 0.0
    acc_total = 0.0
    n_batches = 0

    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        if use_coconut:
            action_logits = model.forward_coconut(
                input_ids        = batch['input_ids'],
                reward_values    = batch['reward_values'],
                reward_positions = batch['reward_positions'],
                select_positions = batch['select_positions'],
                think_positions  = batch['think_positions'],
                cot_positions    = batch['cot_positions'],
            )
        else:
            action_logits = model(
                input_ids        = batch['input_ids'],
                reward_values    = batch['reward_values'],
                reward_positions = batch['reward_positions'],
                select_positions = batch['select_positions'],
            )

        ce, acc = compute_loss(
            action_logits,
            batch['select_positions'],
            batch['select_targets'],
        )
        ce_total  += ce.item()
        acc_total += acc
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0

    return ce_total / n_batches, acc_total / n_batches


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args) -> None:
    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    use_coconut = not args.no_coconut
    print(f"  COCONUT feedback: {'enabled' if use_coconut else 'DISABLED (ablation)'}")

    # ---- Load dataset ----
    data_path = args.data_path
    print(f"\nLoading dataset from {data_path} ...")
    checkpoint_data = torch.load(data_path, map_location='cpu', weights_only=False)
    train_seqs = checkpoint_data['train']
    val_seqs   = checkpoint_data['val']
    cfg_data   = checkpoint_data['config']
    print(f"  train: {len(train_seqs):,}  val: {len(val_seqs):,}")
    print(f"  n_states={cfg_data['n_states']}, n_actions={cfg_data['n_actions']}")

    # ---- Model config ----
    model_config = COCONUTConfig(
        n_states    = cfg_data['n_states'],
        n_actions   = cfg_data['n_actions'],
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
        d_model     = args.d_model,
        d_ff        = args.d_ff,
        dropout     = args.dropout,
        max_seq_len = args.max_seq_len,
    )

    # ---- DataLoaders ----
    train_ds = COCONUTDataset(train_seqs)
    val_ds   = COCONUTDataset(val_seqs)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == 'cuda'),
    )

    # ---- Model ----
    model = COCONUTTransformer(model_config).to(device)
    n_params = model.num_parameters()
    print(f"\nModel parameters: {n_params:,}")

    # ---- Optimizer ----
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.95),
    )

    # ---- AMP scaler (fp16/bf16 on A100) ----
    use_amp = (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_steps = len(train_loader) * args.epochs
    print(f"\nTraining for {args.epochs} epochs, {total_steps:,} steps total")
    print(f"  batch_size: {args.batch_size}  |  lr: {args.lr}  |  weight_decay: {args.weight_decay}")
    print(f"  AMP: {use_amp}")
    print(f"  Curriculum: {CURRICULUM_STAGES}")

    # ---- Weights & Biases ----
    if args.use_wandb:
        if not HAS_WANDB:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        run_tags = ["coconut-feedback", "curriculum", "ce-only"]
        if not use_coconut:
            run_tags.append("no-coconut-ablation")
        wandb.init(
            project = "coconut-qlearning",
            name    = args.run_name if args.run_name else None,
            tags    = run_tags,
            config  = {
                "architecture": "COCONUT CE-Only Transformer",
                "n_layers":     args.n_layers,
                "n_heads":      args.n_heads,
                "d_model":      args.d_model,
                "d_ff":         args.d_ff,
                "dropout":      args.dropout,
                "vocab_size":   model_config.vocab_size,
                "n_params":     n_params,
                "n_sequences":  cfg_data['n_sequences'],
                "n_states":     cfg_data['n_states'],
                "n_actions":    cfg_data['n_actions'],
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "lr":           args.lr,
                "weight_decay": args.weight_decay,
                "use_coconut":  use_coconut,
                "curriculum":   [{"last_epoch": e, "max_steps": s} for e, s in CURRICULUM_STAGES],
            }
        )

    # ---- Checkpointing ----
    ckpt_dir  = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'coconut_transformer.pt')

    best_val_loss = float('inf')
    global_step   = 0
    log_ce_acc    = 0.0
    log_acc_acc   = 0.0
    log_n         = 0
    t0            = time.time()
    current_stage = -1   # will be set on epoch 1

    # Training log: list of (step, train_ce, train_acc, val_ce, val_acc)
    training_log  = []

    # ---- Scheduler: initialized at first stage transition ----
    scheduler = None

    print()

    for epoch in range(1, args.epochs + 1):
        # ---- Curriculum stage transition ----
        new_stage = None
        for i, (last_ep, ms) in enumerate(CURRICULUM_STAGES):
            if epoch <= last_ep:
                new_stage = i
                break
        if new_stage is None:
            new_stage = len(CURRICULUM_STAGES) - 1

        if new_stage != current_stage:
            current_stage = new_stage
            stage_max_steps = CURRICULUM_STAGES[current_stage][1]
            train_ds.max_steps           = stage_max_steps
            train_ds.stage_idx           = current_stage
            train_ds.all_stage_max_steps = STAGE_MAX_STEPS
            print(f"[curriculum] Stage {current_stage + 1}/{len(CURRICULUM_STAGES)}: "
                  f"max_steps={stage_max_steps}")

            # Fresh cosine LR schedule with 100-step warmup for this stage
            stage_last_ep     = CURRICULUM_STAGES[current_stage][0]
            remaining_epochs  = stage_last_ep - epoch + 1
            stage_total_steps = len(train_loader) * remaining_epochs
            stage_warmup      = min(100, stage_total_steps // 10)
            scheduler = get_warmup_cosine_scheduler(optimizer, stage_warmup, stage_total_steps)

        model.train()

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_coconut:
                    action_logits = model.forward_coconut(
                        input_ids        = batch['input_ids'],
                        reward_values    = batch['reward_values'],
                        reward_positions = batch['reward_positions'],
                        select_positions = batch['select_positions'],
                        think_positions  = batch['think_positions'],
                        cot_positions    = batch['cot_positions'],
                    )
                else:
                    action_logits = model(
                        input_ids        = batch['input_ids'],
                        reward_values    = batch['reward_values'],
                        reward_positions = batch['reward_positions'],
                        select_positions = batch['select_positions'],
                    )

                loss, acc = compute_loss(
                    action_logits,
                    batch['select_positions'],
                    batch['select_targets'],
                )

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step  += 1
            log_ce_acc   += loss.item()
            log_acc_acc  += acc
            log_n        += 1

            # ---- Periodic logging + evaluation ----
            if global_step % args.eval_every == 0:
                avg_ce  = log_ce_acc  / log_n
                avg_acc = log_acc_acc / log_n

                val_ce, val_acc = evaluate(model, val_loader, device, use_coconut)
                model.train()

                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]

                print(
                    f"[ep {epoch:02d} | step {global_step:6d}]  "
                    f"train: CE={avg_ce:.4f} acc={avg_acc*100:.1f}%  |  "
                    f"val: CE={val_ce:.4f} acc={val_acc*100:.1f}%  |  "
                    f"lr={lr_now:.2e} t={elapsed:.0f}s"
                )

                # Accumulate training log for later plotting
                training_log.append((global_step, avg_ce, avg_acc, val_ce, val_acc))

                if args.use_wandb:
                    wandb.log({
                        "train/ce_loss":        avg_ce,
                        "train/accuracy":       avg_acc,
                        "val/ce_loss":          val_ce,
                        "val/accuracy":         val_acc,
                        "lr":                   lr_now,
                        "curriculum/stage":     current_stage + 1,
                        "curriculum/max_steps": CURRICULUM_STAGES[current_stage][1],
                        "epoch":                epoch,
                        "step":                 global_step,
                    })

                # Save best checkpoint by val CE loss
                if val_ce < best_val_loss:
                    best_val_loss = val_ce
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config':           model_config.to_dict(),
                        'step':             global_step,
                        'epoch':            epoch,
                        'val_loss':         best_val_loss,
                        'val_ce_loss':      val_ce,
                        'val_acc':          val_acc,
                        'use_coconut':      use_coconut,
                    }, ckpt_path)
                    print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

                log_ce_acc  = 0.0
                log_acc_acc = 0.0
                log_n       = 0

        # End-of-epoch evaluation
        val_ce, val_acc = evaluate(model, val_loader, device, use_coconut)
        model.train()
        print(
            f"\n[epoch {epoch:02d} END]  "
            f"val: CE={val_ce:.4f} acc={val_acc*100:.1f}%\n"
        )

        if val_ce < best_val_loss:
            best_val_loss = val_ce
            torch.save({
                'model_state_dict': model.state_dict(),
                'config':           model_config.to_dict(),
                'step':             global_step,
                'epoch':            epoch,
                'val_loss':         best_val_loss,
                'val_ce_loss':      val_ce,
                'val_acc':          val_acc,
                'use_coconut':      use_coconut,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

        if args.use_wandb:
            wandb.log({
                "epoch/val_ce_loss":      val_ce,
                "epoch/val_accuracy":     val_acc,
                "epoch/best_val_loss":    best_val_loss,
                "epoch/curriculum_stage": current_stage + 1,
                "epoch":                  epoch,
                "step":                   global_step,
            })

    # ---- Save training log for 4_evaluate.py to plot ----
    if training_log:
        steps_arr    = [r[0] for r in training_log]
        train_ce_arr = [r[1] for r in training_log]
        train_ac_arr = [r[2] for r in training_log]
        val_ce_arr   = [r[3] for r in training_log]
        val_ac_arr   = [r[4] for r in training_log]
        log_path = os.path.join(ckpt_dir, 'training_log.npz')
        import numpy as np
        np.savez(log_path,
                 steps=steps_arr,
                 train_ce=train_ce_arr,
                 train_acc=train_ac_arr,
                 val_ce=val_ce_arr,
                 val_acc=val_ac_arr)
        print(f"Training log saved to {log_path}")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val CE: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    if args.use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train COCONUT Q-Learning Transformer (CE only)')

    # Data
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'data', 'coconut_dataset.pt'))
    parser.add_argument('--checkpoint_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'checkpoints'))

    # Model hyperparameters
    parser.add_argument('--n_layers',    type=int,   default=4)
    parser.add_argument('--n_heads',     type=int,   default=8)
    parser.add_argument('--d_model',     type=int,   default=256)
    parser.add_argument('--d_ff',        type=int,   default=1024)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int,   default=1024)

    # Training hyperparameters
    parser.add_argument('--epochs',       type=int,   default=28)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--eval_every',   type=int,   default=500)
    parser.add_argument('--num_workers',  type=int,   default=4)

    # Ablation flag
    parser.add_argument('--no_coconut', action='store_true',
                        help='Use standard forward() instead of forward_coconut(). '
                             'Trains without COCONUT feedback — COT tokens use static '
                             'learned embeddings. Use for ablation comparison.')

    # W&B
    parser.add_argument('--use_wandb',  action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--run_name',   type=str, default=None,
                        help='W&B run name (default: auto-generated)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
