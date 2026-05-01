#!/usr/bin/env python3
"""
3_train.py — BPTT Training with Recurrent Continuous Context

Trains the COCONUTTransformer from 2_model.py on the dataset produced by
1_generate_data.py using Backpropagation Through Time (BPTT).

No curriculum stages. The model uses continuous context from epoch 1.

BPTT loop per episode:
  1. Initialize context tokens c_1..c_{|A|} as learned parameters.
  2. For each transition t:
     a. Build discrete token sequence for step t.
     b. Feed context + tokens to model.forward_step.
     c. Accumulate CE loss from SELECT logits vs. a*.
     d. Extract UPDATE hidden state.
     e. Replace c_{a_t} with the UPDATE hidden state for step t+1.
  3. Backpropagate through the full episode (or truncated window).
"""

import argparse
import math
import os
import random
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

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "coconut_model",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_model.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
COCONUTConfig      = _mod.COCONUTConfig
COCONUTTransformer = _mod.COCONUTTransformer
build_vocab        = _mod.build_vocab


# ---------------------------------------------------------------------------
# Token sequence builder for a single transition
# ---------------------------------------------------------------------------

def build_step_tokens(
    transition: Dict,
    vocab: Dict,
    max_actions: int,
) -> Tuple[List[int], int, int, int]:
    """Build the discrete token sequence for one transition step.

    Returns (token_ids, reward_offset, select_offset, update_offset).

    Token layout:
      s_t, a_t, R, s_{t+1},
      [s_{t+1}, a_c, EVAL] * max_actions,
      SELECT, QCURR, QNEXT, UPDATE
    """
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_QCURR  = vocab['TOK_QCURR']
    TOK_QNEXT  = vocab['TOK_QNEXT']
    TOK_UPDATE = vocab['TOK_UPDATE']

    s      = transition['s']
    a      = transition['a']
    s_next = transition['s_next']

    ids = [TOK_S[s], TOK_A[a]]
    reward_offset = len(ids)
    ids.append(TOK_R)
    ids.append(TOK_S[s_next])

    for c in range(max_actions):
        ids.append(TOK_S[s_next])
        ids.append(TOK_A[c])
        ids.append(TOK_EVAL)

    select_offset = len(ids)
    ids.append(TOK_SELECT)
    ids.append(TOK_QCURR)
    ids.append(TOK_QNEXT)
    update_offset = len(ids)
    ids.append(TOK_UPDATE)

    return ids, reward_offset, select_offset, update_offset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EpisodeDataset(Dataset):
    def __init__(self, sequences: List[Dict], max_steps: Optional[int] = None):
        self.sequences = sequences
        self.max_steps = max_steps

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        if self.max_steps is not None:
            n = min(seq['n_steps'], self.max_steps)
            return {
                'transitions': seq['transitions'][:n],
                'n_steps':     n,
                'n_states':    seq['n_states'],
                'n_actions':   seq['n_actions'],
            }
        return seq


def collate_episodes(batch: List[Dict]) -> List[Dict]:
    return batch


# ---------------------------------------------------------------------------
# Scheduler: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Single-episode BPTT forward pass
# ---------------------------------------------------------------------------

def episode_forward(
    model: COCONUTTransformer,
    episode: Dict,
    vocab: Dict,
    max_actions: int,
    device: torch.device,
    truncate_bptt_window: int = 0,
) -> Tuple[torch.Tensor, float, int]:
    """Run BPTT through one episode. Returns (loss, accuracy, n_steps).

    truncate_bptt_window: if > 0, detach context from the graph every
    truncate_bptt_window steps to limit memory usage.
    """
    transitions = episode['transitions']
    n_steps = len(transitions)
    n_actions_actual = episode.get('n_actions', max_actions)

    context = model.get_init_context(1, device)  # [1, |A|, d]

    total_loss = torch.tensor(0.0, device=device)
    n_correct  = 0

    for t in range(n_steps):
        tr = transitions[t]
        token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, max_actions)

        token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
        reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)
        target = tr['a_star']

        if truncate_bptt_window > 0 and t > 0 and t % truncate_bptt_window == 0:
            context = context.detach()

        select_logits, update_hidden = model.forward_step(
            token_ids=token_ids,
            reward_value=reward_val,
            reward_offset=r_off,
            select_offset=s_off,
            update_offset=u_off,
            context=context,
        )

        if n_actions_actual < max_actions:
            select_logits[:, n_actions_actual:] = float('-inf')

        target_t = torch.tensor([target], dtype=torch.long, device=device)
        step_loss = F.cross_entropy(select_logits, target_t)
        total_loss = total_loss + step_loss

        pred = select_logits[0, :n_actions_actual].argmax().item()
        if pred == target:
            n_correct += 1

        a_t = tr['a']
        new_context = context.clone()
        new_context[0, a_t, :] = update_hidden[0]
        context = new_context

    avg_loss = total_loss / n_steps
    acc = n_correct / n_steps
    return avg_loss, acc, n_steps


# ---------------------------------------------------------------------------
# Batched BPTT forward pass (processes multiple episodes in parallel)
# ---------------------------------------------------------------------------

def batched_episode_forward(
    model: COCONUTTransformer,
    episodes: List[Dict],
    vocab: Dict,
    max_actions: int,
    device: torch.device,
    truncate_bptt_window: int = 0,
) -> Tuple[torch.Tensor, float, int]:
    """Run BPTT through a batch of episodes in parallel.

    All episodes in the batch are stepped through in lockstep. Episodes
    shorter than the longest are masked out after they end.
    """
    B = len(episodes)
    max_steps = max(ep['n_steps'] for ep in episodes)
    step_counts = [len(ep['transitions']) for ep in episodes]

    context = model.get_init_context(B, device)  # [B, |A|, d]

    total_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_valid   = 0

    for t in range(max_steps):
        active_mask = [i for i in range(B) if t < step_counts[i]]
        if not active_mask:
            break

        batch_tokens = []
        batch_rewards = []
        batch_targets = []
        batch_actions = []
        batch_n_actions = []

        sample_tr = episodes[active_mask[0]]['transitions'][t]
        token_list, r_off, s_off, u_off = build_step_tokens(
            sample_tr, vocab, max_actions
        )
        T_disc = len(token_list)

        all_token_lists = []
        for i in active_mask:
            tr = episodes[i]['transitions'][t]
            tl, _, _, _ = build_step_tokens(tr, vocab, max_actions)
            all_token_lists.append(tl)
            batch_rewards.append(tr['r'])
            batch_targets.append(tr['a_star'])
            batch_actions.append(tr['a'])
            batch_n_actions.append(episodes[i].get('n_actions', max_actions))

        token_ids = torch.tensor(all_token_lists, dtype=torch.long, device=device)
        reward_vals = torch.tensor(batch_rewards, dtype=torch.float32, device=device)

        active_ctx = context[active_mask]

        if truncate_bptt_window > 0 and t > 0 and t % truncate_bptt_window == 0:
            active_ctx = active_ctx.detach()
            context = context.detach()

        select_logits, update_hidden = model.forward_step(
            token_ids=token_ids,
            reward_value=reward_vals,
            reward_offset=r_off,
            select_offset=s_off,
            update_offset=u_off,
            context=active_ctx,
        )

        targets_t = torch.tensor(batch_targets, dtype=torch.long, device=device)

        n_act_t = torch.tensor(batch_n_actions, dtype=torch.long, device=device)
        act_range = torch.arange(max_actions, device=device).unsqueeze(0)
        phantom_mask = act_range >= n_act_t.unsqueeze(1)
        select_logits = select_logits.masked_fill(phantom_mask, float('-inf'))

        step_loss = F.cross_entropy(select_logits, targets_t)
        total_loss = total_loss + step_loss * len(active_mask)

        preds = select_logits.argmax(dim=-1)
        total_correct += (preds == targets_t).sum().item()
        total_valid += len(active_mask)

        new_context = context.clone()
        for j, i in enumerate(active_mask):
            a_t = batch_actions[j]
            new_context[i, a_t, :] = update_hidden[j]
        context = new_context

    avg_loss = total_loss / max(total_valid, 1)
    acc = total_correct / max(total_valid, 1)
    return avg_loss, acc, total_valid


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: COCONUTTransformer,
    val_loader: DataLoader,
    vocab: Dict,
    max_actions: int,
    device: torch.device,
    max_batches: Optional[int] = 50,
) -> Tuple[float, float]:
    model.eval()
    ce_total  = 0.0
    acc_total = 0.0
    n_batches = 0

    for episodes in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break

        loss, acc, _ = batched_episode_forward(
            model, episodes, vocab, max_actions, device,
        )
        ce_total  += loss.item()
        acc_total += acc
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0
    return ce_total / n_batches, acc_total / n_batches


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Load dataset ----
    data_path = args.data_path
    print(f"\nLoading dataset from {data_path} ...")
    checkpoint_data = torch.load(data_path, map_location='cpu', weights_only=False)
    train_seqs = checkpoint_data['train']
    val_seqs   = checkpoint_data['val']
    cfg_data   = checkpoint_data['config']
    print(f"  train: {len(train_seqs):,}  val: {len(val_seqs):,}")
    max_states  = cfg_data.get('max_states', 8)
    max_actions = cfg_data.get('max_actions', 4)
    print(f"  max_states={max_states}, max_actions={max_actions}")

    n_disc_per_step = 4 + 3 * max_actions + 3  # s,a,R,s' + evals + SELECT,QCURR,QNEXT,UPDATE
    max_seq_per_step = max_actions + n_disc_per_step

    # ---- Model config ----
    model_config = COCONUTConfig(
        max_states  = max_states,
        max_actions = max_actions,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
        d_model     = args.d_model,
        d_ff        = args.d_ff,
        dropout     = args.dropout,
        max_seq_len = max_seq_per_step + 16,
        use_ffns    = not args.no_ffns,
    )

    vocab = build_vocab(max_states, max_actions)

    # ---- Datasets ----
    train_ds = EpisodeDataset(train_seqs, max_steps=args.max_steps)
    val_ds   = EpisodeDataset(val_seqs, max_steps=args.max_steps)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_episodes,
        pin_memory  = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate_episodes,
        pin_memory  = False,
    )

    # ---- Model ----
    model = COCONUTTransformer(model_config).to(device)
    n_params = model.num_parameters()
    print(f"\nModel parameters: {n_params:,}")
    print(f"  vocab_size: {model_config.vocab_size}")
    print(f"  use_ffns: {model_config.use_ffns}")

    # ---- Optimizer ----
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.95),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(500, total_steps // 10)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # ---- AMP scaler ----
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nTraining for {args.epochs} epochs (BPTT, continuous context from epoch 1)")
    print(f"  batch_size: {args.batch_size}  |  lr: {args.lr}  |  weight_decay: {args.weight_decay}")
    print(f"  truncate_bptt: {args.truncate_bptt}")
    print(f"  max_steps: {args.max_steps}")
    print(f"  AMP: {use_amp}")

    # ---- Weights & Biases ----
    if args.use_wandb:
        if not HAS_WANDB:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        wandb.init(
            project = "coconut-qlearning",
            name    = args.run_name if args.run_name else None,
            tags    = ["recurrent-context", "bptt"],
            config  = {
                "architecture":    "Recurrent Context Transformer",
                "n_layers":        args.n_layers,
                "n_heads":         args.n_heads,
                "d_model":         args.d_model,
                "d_ff":            args.d_ff,
                "dropout":         args.dropout,
                "use_ffns":        model_config.use_ffns,
                "vocab_size":      model_config.vocab_size,
                "n_params":        n_params,
                "max_states":      max_states,
                "max_actions":     max_actions,
                "epochs":          args.epochs,
                "batch_size":      args.batch_size,
                "lr":              args.lr,
                "weight_decay":    args.weight_decay,
                "truncate_bptt":   args.truncate_bptt,
                "max_steps":       args.max_steps,
            }
        )

    # ---- Checkpointing ----
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.run_name:
        ckpt_path = os.path.join(ckpt_dir, f'coconut_transformer_{args.run_name}.pt')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'coconut_transformer.pt')

    best_val_loss  = float('inf')
    global_step    = 0
    log_loss       = 0.0
    log_acc        = 0.0
    log_n          = 0
    t0             = time.time()

    training_log = []

    print()

    for epoch in range(1, args.epochs + 1):
        model.train()

        for episodes in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, acc, n_valid = batched_episode_forward(
                    model, episodes, vocab, max_actions, device,
                    truncate_bptt_window=args.truncate_bptt,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            log_loss += loss.item()
            log_acc  += acc
            log_n    += 1

            if global_step % args.eval_every == 0:
                avg_loss = log_loss / log_n
                avg_acc  = log_acc / log_n

                val_ce, val_acc = evaluate(model, val_loader, vocab,
                                           max_actions, device)
                model.train()

                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]

                print(
                    f"[ep {epoch:02d} | step {global_step:6d}]  "
                    f"train: CE={avg_loss:.4f} acc={avg_acc*100:.1f}%  |  "
                    f"val: CE={val_ce:.4f} acc={val_acc*100:.1f}%  |  "
                    f"lr={lr_now:.2e} t={elapsed:.0f}s"
                )

                training_log.append((global_step, avg_loss, avg_acc, val_ce, val_acc))

                if args.use_wandb:
                    wandb.log({
                        "train/ce_loss":    avg_loss,
                        "train/accuracy":   avg_acc,
                        "val/ce_loss":      val_ce,
                        "val/accuracy":     val_acc,
                        "lr":               lr_now,
                        "epoch":            epoch,
                        "step":             global_step,
                    })

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
                    }, ckpt_path)
                    print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

                log_loss = 0.0
                log_acc  = 0.0
                log_n    = 0

        # End-of-epoch evaluation
        val_ce, val_acc = evaluate(model, val_loader, vocab, max_actions, device)
        model.train()
        print(f"\n[epoch {epoch:02d} END]  val: CE={val_ce:.4f} acc={val_acc*100:.1f}%\n")

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
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

        if args.use_wandb:
            wandb.log({
                "epoch/val_ce_loss":   val_ce,
                "epoch/val_accuracy":  val_acc,
                "epoch/best_val_loss": best_val_loss,
                "epoch":               epoch,
                "step":                global_step,
            })

    # ---- Save training log ----
    if training_log:
        import numpy as np
        steps_arr    = [r[0] for r in training_log]
        train_ce_arr = [r[1] for r in training_log]
        train_ac_arr = [r[2] for r in training_log]
        val_ce_arr   = [r[3] for r in training_log]
        val_ac_arr   = [r[4] for r in training_log]
        log_path = os.path.join(ckpt_dir, 'training_log.npz')
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
    parser = argparse.ArgumentParser(
        description='Train Recurrent Context Q-Learning Transformer (BPTT)')

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
    parser.add_argument('--no_ffns',     action='store_true',
                        help='Disable FFN layers (attention-only transformer)')

    # Training hyperparameters
    parser.add_argument('--epochs',       type=int,   default=25)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--eval_every',   type=int,   default=500)
    parser.add_argument('--num_workers',  type=int,   default=0)
    parser.add_argument('--max_steps',    type=int,   default=50,
                        help='Max transitions per episode during training')
    parser.add_argument('--truncate_bptt', type=int,  default=10,
                        help='Detach context graph every N steps (0 = full BPTT)')

    # W&B
    parser.add_argument('--use_wandb',  action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--run_name',   type=str, default=None,
                        help='W&B run name (default: auto-generated)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
