#!/usr/bin/env python3
"""
3_train.py — Dual-Loss Training Loop

Trains the COCONUTTransformer from 2_model.py on the dataset produced by
1_generate_data.py using two simultaneous loss functions:

    CE_loss  = CrossEntropyLoss(action_logits, select_targets)  at <Select> positions
    MSE_loss = MSELoss(q_value_preds, update_targets)            at <Update> positions
    Total    = CE_loss + MSE_loss

Training details:
    - Optimizer  : AdamW(lr=1e-4, weight_decay=1e-2)
    - Scheduler  : linear warmup (500 steps) then cosine decay to 0
    - Mixed prec : torch.cuda.amp.autocast + GradScaler (A100-friendly)
    - Epochs     : 20  (--epochs flag)
    - Batch size : 64  (--batch_size flag)
    - Eval every : 500 steps
    - Checkpoint : saves best model by total val loss

Checkpoint format (saved to checkpoints/coconut_transformer.pt):
    {
        'model_state_dict': ...,
        'config':           config.to_dict(),
        'step':             global_step,
        'epoch':            epoch,
        'val_loss':         best_val_loss,
        'val_ce_loss':      ...,
        'val_mse_loss':     ...,
    }
"""

import argparse
import math
import os
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
# Dataset
# ---------------------------------------------------------------------------

class COCONUTDataset(Dataset):
    """Wraps the list of sequence dicts produced by 1_generate_data.py."""

    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        return self.sequences[idx]


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
    update_positions : padded with -1
    update_targets   : padded with 0.0 (ignored where positions == -1)
    """
    # ---- Sequence length ----
    max_seq = max(len(s['input_ids']) for s in batch)

    # ---- Per-step count (n_steps varies) ----
    max_steps = max(s['n_steps'] for s in batch)

    def pad_ids(ids: List[int], length: int) -> List[int]:
        return ids + [0] * (length - len(ids))

    def pad_ints(lst: List[int], length: int, pad: int = -1) -> List[int]:
        return lst + [pad] * (length - len(lst))

    def pad_floats(lst: List[float], length: int, pad: float = 0.0) -> List[float]:
        return lst + [pad] * (length - len(lst))

    def pad_float_rows(lst: List[List[float]], length: int, n_actions: int) -> List[List[float]]:
        pad_row = [0.0] * n_actions
        return lst + [pad_row] * (length - len(lst))

    # Infer n_actions from update_targets
    n_actions = len(batch[0]['update_targets'][0])

    input_ids_list        = []
    reward_values_list    = []
    reward_positions_list = []
    select_positions_list = []
    select_targets_list   = []
    update_positions_list = []
    update_targets_list   = []

    for s in batch:
        n = s['n_steps']
        input_ids_list.append(pad_ids(s['input_ids'], max_seq))
        reward_values_list.append(pad_floats(s['reward_values'], max_steps))
        reward_positions_list.append(pad_ints(s['reward_positions'], max_steps))
        select_positions_list.append(pad_ints(s['select_positions'], max_steps))
        select_targets_list.append(pad_ints(s['select_targets'], max_steps, pad=0))
        update_positions_list.append(pad_ints(s['update_positions'], max_steps))
        update_targets_list.append(pad_float_rows(s['update_targets'], max_steps, n_actions))

    return {
        'input_ids':        torch.tensor(input_ids_list,        dtype=torch.long),
        'reward_values':    torch.tensor(reward_values_list,    dtype=torch.float32),
        'reward_positions': torch.tensor(reward_positions_list, dtype=torch.long),
        'select_positions': torch.tensor(select_positions_list, dtype=torch.long),
        'select_targets':   torch.tensor(select_targets_list,   dtype=torch.long),
        'update_positions': torch.tensor(update_positions_list, dtype=torch.long),
        'update_targets':   torch.tensor(update_targets_list,   dtype=torch.float32),
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
# Loss computation
# ---------------------------------------------------------------------------

def compute_losses(
    action_logits: torch.Tensor,   # [B, n_sel, n_actions]
    q_value_preds: torch.Tensor,   # [B, n_upd, n_actions]
    select_positions: torch.Tensor, # [B, n_sel]  (-1 = pad)
    select_targets: torch.Tensor,   # [B, n_sel]
    update_positions: torch.Tensor, # [B, n_upd]  (-1 = pad)
    update_targets: torch.Tensor,   # [B, n_upd, n_actions]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (ce_loss, mse_loss, total_loss).

    Padding mask: positions == -1 are excluded from both losses.
    Returns scalar zero tensors on the correct device when no valid tokens exist.
    """
    device = action_logits.device

    # ---- CE loss at <Select> positions ----
    sel_valid = select_positions >= 0                          # [B, n_sel]
    n_sel_valid = sel_valid.sum().item()
    if n_sel_valid > 0:
        al_valid = action_logits[sel_valid]                    # [N, n_actions]
        st_valid = select_targets[sel_valid]                   # [N]
        ce_loss = F.cross_entropy(al_valid, st_valid)
    else:
        ce_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # ---- MSE loss at <Update> positions ----
    upd_valid = update_positions >= 0                          # [B, n_upd]
    n_upd_valid = upd_valid.sum().item()
    if n_upd_valid > 0:
        qp_valid = q_value_preds[upd_valid]                    # [N, n_actions]
        qt_valid = update_targets[upd_valid]                   # [N, n_actions]
        mse_loss = F.mse_loss(qp_valid, qt_valid)
    else:
        mse_loss = torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = ce_loss + mse_loss
    return ce_loss, mse_loss, total_loss


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: COCONUTTransformer,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = 50,
) -> Tuple[float, float, float]:
    """Evaluate model on validation set.

    Returns (mean_ce, mean_mse, mean_total) averaged over up to max_batches.
    """
    model.eval()
    ce_total = 0.0
    mse_total = 0.0
    n_batches = 0

    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        action_logits, q_value_preds = model(
            input_ids        = batch['input_ids'],
            reward_values    = batch['reward_values'],
            reward_positions = batch['reward_positions'],
            select_positions = batch['select_positions'],
            update_positions = batch['update_positions'],
        )

        ce, mse, _ = compute_losses(
            action_logits, q_value_preds,
            batch['select_positions'], batch['select_targets'],
            batch['update_positions'], batch['update_targets'],
        )
        ce_total  += ce.item()
        mse_total += mse.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    mean_ce  = ce_total  / n_batches
    mean_mse = mse_total / n_batches
    return mean_ce, mean_mse, mean_ce + mean_mse


def train(args) -> None:
    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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

    # ---- Optimizer + Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.95),
    )
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = min(args.warmup_steps, total_steps // 10)
    scheduler     = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # ---- AMP scaler (fp16/bf16 on A100) ----
    use_amp = (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\nTraining for {args.epochs} epochs, {total_steps:,} steps total")
    print(f"  warmup: {warmup_steps} steps  |  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}  |  weight_decay: {args.weight_decay}")
    print(f"  AMP: {use_amp}")

    # ---- Weights & Biases ----
    if args.use_wandb:
        if not HAS_WANDB:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        wandb.init(
            project="coconut-qlearning",
            config={
                "architecture": "Continuous CoT Transformer",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_sequences": cfg_data['n_sequences'],
                "n_layers": args.n_layers,
                "d_model": args.d_model,
            }
        )

    # ---- Checkpointing ----
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'coconut_transformer.pt')

    best_val_loss  = float('inf')
    global_step    = 0
    log_ce_acc     = 0.0
    log_mse_acc    = 0.0
    log_n          = 0
    t0             = time.time()

    print()

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                action_logits, q_value_preds = model(
                    input_ids        = batch['input_ids'],
                    reward_values    = batch['reward_values'],
                    reward_positions = batch['reward_positions'],
                    select_positions = batch['select_positions'],
                    update_positions = batch['update_positions'],
                )
                ce_loss, mse_loss, total_loss = compute_losses(
                    action_logits, q_value_preds,
                    batch['select_positions'], batch['select_targets'],
                    batch['update_positions'], batch['update_targets'],
                )

            scaler.scale(total_loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            log_ce_acc  += ce_loss.item()
            log_mse_acc += mse_loss.item()
            log_n       += 1

            # ---- Periodic logging + evaluation ----
            if global_step % args.eval_every == 0:
                avg_ce  = log_ce_acc  / log_n
                avg_mse = log_mse_acc / log_n
                avg_tot = avg_ce + avg_mse

                val_ce, val_mse, val_tot = evaluate(model, val_loader, device)
                model.train()

                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]

                print(
                    f"[ep {epoch:02d} | step {global_step:6d}]  "
                    f"train: CE={avg_ce:.4f}  MSE={avg_mse:.4f}  Tot={avg_tot:.4f}  |  "
                    f"val: CE={val_ce:.4f}  MSE={val_mse:.4f}  Tot={val_tot:.4f}  |  "
                    f"lr={lr_now:.2e}  t={elapsed:.0f}s"
                )

                if args.use_wandb:
                    wandb.log({
                        "train/ce_loss":    avg_ce,
                        "train/mse_loss":   avg_mse,
                        "train/total_loss": avg_tot,
                        "val/ce_loss":      val_ce,
                        "val/mse_loss":     val_mse,
                        "val/total_loss":   val_tot,
                        "lr":               lr_now,
                        "epoch":            epoch,
                        "step":             global_step,
                    })

                # Save best checkpoint
                if val_tot < best_val_loss:
                    best_val_loss = val_tot
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config':          model_config.to_dict(),
                        'step':            global_step,
                        'epoch':           epoch,
                        'val_loss':        best_val_loss,
                        'val_ce_loss':     val_ce,
                        'val_mse_loss':    val_mse,
                    }, ckpt_path)
                    print(f"  -> Saved best checkpoint (val_loss={best_val_loss:.4f})")

                log_ce_acc  = 0.0
                log_mse_acc = 0.0
                log_n       = 0

        # End-of-epoch evaluation
        val_ce, val_mse, val_tot = evaluate(model, val_loader, device)
        model.train()
        print(
            f"\n[epoch {epoch:02d} END]  "
            f"val: CE={val_ce:.4f}  MSE={val_mse:.4f}  Tot={val_tot:.4f}\n"
        )

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            torch.save({
                'model_state_dict': model.state_dict(),
                'config':          model_config.to_dict(),
                'step':            global_step,
                'epoch':           epoch,
                'val_loss':        best_val_loss,
                'val_ce_loss':     val_ce,
                'val_mse_loss':    val_mse,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_loss={best_val_loss:.4f})")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    if args.use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train COCONUT Q-Learning Transformer')

    # Data
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'data', 'coconut_dataset.pt'))
    parser.add_argument('--checkpoint_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'checkpoints'))

    # Model hyperparameters
    parser.add_argument('--n_layers',    type=int,   default=4)
    parser.add_argument('--n_heads',     type=int,   default=4)
    parser.add_argument('--d_model',     type=int,   default=128)
    parser.add_argument('--d_ff',        type=int,   default=512)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int,   default=1024)

    # Training hyperparameters
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-2)
    parser.add_argument('--warmup_steps',  type=int,   default=500)
    parser.add_argument('--eval_every',    type=int,   default=500)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--use_wandb',     action='store_true',
                        help='Enable Weights & Biases logging')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
