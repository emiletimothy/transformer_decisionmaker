#!/usr/bin/env python3
"""
3_train.py — Hao-Style Multi-Stage Curriculum Training

Trains the COCONUTTransformer from 2_model.py on the dataset produced by
1_generate_data.py using a Hao et al. (2025) multi-stage curriculum that
progressively replaces discrete explanation tokens with continuous thoughts.

Each round's single <COT> token is expanded into 3 explanation tokens:
    [<s_t>, <a_t>, <Q_bin>]

These are progressively replaced with continuous thoughts across 4 stages:
    Stage 0 (epochs  1-6):  fully discrete   — CE on <Select> + all 3 explain tokens
    Stage 1 (epochs  7-12): Q_bin→continuous  — CE on <Select> + s_t, a_t
    Stage 2 (epochs 13-18): a_t→continuous    — CE on <Select> + s_t only
    Stage 3 (epochs 19-24): fully continuous  — CE on <Select> only

Loss: CE_select + lambda_explain * CE_explain

Optimizer state is reset at each stage transition (following Hao et al.).
KV caching accelerates the COCONUT injection loop in Stages 1-3.

Ablation: pass --no_coconut to stay in Stage 0 forever (fully discrete).
"""

import argparse
import math
import os
import random
import sys
import time
from functools import partial
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
build_vocab        = _mod.build_vocab
discretize_q_value = _mod.discretize_q_value


# ---------------------------------------------------------------------------
# Hao-style curriculum stage definitions
# (last_epoch_inclusive, max_steps, n_continuous, description)
# ---------------------------------------------------------------------------

HAO_STAGES = [
    (6,  50, 0, "fully discrete: s_t a_t Q_bin"),
    (12, 50, 1, "Q_bin -> continuous"),
    (18, 50, 2, "a_t, Q_bin -> continuous"),
    (24, 50, 3, "fully continuous"),
]
STAGE_MAX_STEPS = [ms for _, ms, _, _ in HAO_STAGES]


# ---------------------------------------------------------------------------
# Sequence truncation helper (for curriculum training)
# ---------------------------------------------------------------------------

def truncate_sequence(seq: Dict, max_steps: int) -> Dict:
    """Return a copy of seq truncated to at most max_steps rounds."""
    n = min(seq['n_steps'], max_steps)
    if n == seq['n_steps']:
        return seq

    full_n   = seq['n_steps']
    full_len = len(seq['input_ids'])
    round_len = (full_len - 2) // full_n   # 2-token prefix

    trunc = dict(seq)
    trunc['input_ids'] = seq['input_ids'][:2 + n * round_len]
    for field in ('reward_values', 'reward_positions',
                  'select_positions', 'select_targets',
                  'think_positions', 'cot_positions',
                  'q_values_for_cot'):
        if field in seq:
            trunc[field] = seq[field][:n]
    trunc['n_steps'] = n
    return trunc


# ---------------------------------------------------------------------------
# Sequence expansion: replace single TOK_COT with 3 explanation tokens
# ---------------------------------------------------------------------------

def expand_sequence(
    seq: Dict,
    n_continuous: int,
    vocab: Dict,
    n_q_bins: int,
    q_bin_min: float,
    q_bin_max: float,
) -> Dict:
    """Expand each round's TOK_COT into [explain_0, explain_1, explain_2].

    Returns a new dict with expanded input_ids (15 tokens/round instead of 13)
    and new position arrays: explain_positions [n_steps, 3] and
    explain_targets [n_steps, 3].
    """
    n_steps  = seq['n_steps']
    old_ids  = seq['input_ids']
    qvc      = seq.get('q_values_for_cot', [])

    # Derive original round_len from actual data
    old_round_len = (len(old_ids) - 2) // n_steps  # 2-token prefix

    prefix = old_ids[:2]  # [TOK_NULL, TOK_START]

    new_ids              = list(prefix)
    new_reward_positions = []
    new_reward_values    = []
    new_select_positions = []
    new_select_targets   = []
    new_think_positions  = []
    new_explain_positions = []  # list of [pos0, pos1, pos2]
    new_explain_targets   = []  # list of [tok0, tok1, tok2]

    # Within original round layout (13 tokens for n_actions=2):
    #   offset 2  = TOK_R
    #   offset 10 = TOK_SELECT
    #   offset 11 = TOK_THINK
    #   offset 12 = TOK_COT (to be replaced)
    # These offsets are relative within the round.
    # We derive them from the stored position arrays for robustness.

    for r in range(n_steps):
        round_start = 2 + r * old_round_len
        round_tokens = old_ids[round_start:round_start + old_round_len]

        entry = qvc[r]
        st, at, q_new = entry['st'], entry['at'], entry['q_new']
        q_bin = discretize_q_value(q_new, n_q_bins, q_bin_min, q_bin_max)

        # Emit tokens 0..old_round_len-2 (everything BEFORE the last token = COT)
        for k in range(old_round_len - 1):
            pos = len(new_ids)
            # Track positions by matching original absolute positions
            orig_pos = round_start + k
            if orig_pos == seq['reward_positions'][r]:
                new_reward_positions.append(pos)
                new_reward_values.append(seq['reward_values'][r])
            if orig_pos == seq['select_positions'][r]:
                new_select_positions.append(pos)
                new_select_targets.append(seq['select_targets'][r])
            if orig_pos == seq['think_positions'][r]:
                new_think_positions.append(pos)
            new_ids.append(round_tokens[k])

        # Emit 3 explanation tokens in place of the single COT
        discrete_tokens = [
            vocab['TOK_S'][st],
            vocab['TOK_A'][at],
            vocab['TOK_QBIN'][q_bin],
        ]
        targets = list(discrete_tokens)

        explain_pos = []
        for j in range(3):
            pos = len(new_ids)
            explain_pos.append(pos)
            if j >= 3 - n_continuous:
                new_ids.append(vocab['TOK_COT'])  # placeholder for continuous
            else:
                new_ids.append(discrete_tokens[j])

        new_explain_positions.append(explain_pos)
        new_explain_targets.append(targets)

    return {
        'input_ids':         new_ids,
        'reward_values':     new_reward_values,
        'reward_positions':  new_reward_positions,
        'select_positions':  new_select_positions,
        'select_targets':    new_select_targets,
        'think_positions':   new_think_positions,
        'explain_positions': new_explain_positions,  # [n_steps][3]
        'explain_targets':   new_explain_targets,     # [n_steps][3]
        'n_steps':           n_steps,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCONUTDataset(Dataset):
    """Wraps the list of sequence dicts produced by 1_generate_data.py.

    Supports curriculum training via the max_steps attribute. With probability
    0.1 (when stage_idx > 0) a random earlier-stage max_steps is sampled to
    prevent catastrophic forgetting.
    """

    def __init__(self, sequences: List[Dict]):
        self.sequences           = sequences
        self.max_steps           = None
        self.stage_idx           = 0
        self.all_stage_max_steps = STAGE_MAX_STEPS

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        effective = self.max_steps
        if effective is not None and self.stage_idx > 0 and random.random() < 0.1:
            effective = random.choice(self.all_stage_max_steps[:self.stage_idx])
        if effective is not None:
            seq = truncate_sequence(seq, effective)
        return seq


# ---------------------------------------------------------------------------
# Collate function (Hao-style with explanation expansion)
# ---------------------------------------------------------------------------

def collate_fn_hao(
    batch: List[Dict],
    n_continuous: int,
    vocab: Dict,
    n_q_bins: int,
    q_bin_min: float,
    q_bin_max: float,
) -> Dict[str, torch.Tensor]:
    """Expand COT→3 explain tokens, then pad and batch."""

    # Expand each sequence
    expanded = [
        expand_sequence(s, n_continuous, vocab, n_q_bins, q_bin_min, q_bin_max)
        for s in batch
    ]

    max_seq   = max(len(s['input_ids']) for s in expanded)
    max_steps = max(s['n_steps'] for s in expanded)

    def pad_ids(ids, length):
        return ids + [0] * (length - len(ids))

    def pad_ints(lst, length, pad=-1):
        return lst + [pad] * (length - len(lst))

    def pad_floats(lst, length, pad=0.0):
        return lst + [pad] * (length - len(lst))

    input_ids_list        = []
    reward_values_list    = []
    reward_positions_list = []
    select_positions_list = []
    select_targets_list   = []
    think_positions_list  = []
    explain_positions_list = []  # [B, max_steps, 3]
    explain_targets_list   = []  # [B, max_steps, 3]

    for s in expanded:
        input_ids_list.append(pad_ids(s['input_ids'], max_seq))
        reward_values_list.append(pad_floats(s['reward_values'], max_steps))
        reward_positions_list.append(pad_ints(s['reward_positions'], max_steps))
        select_positions_list.append(pad_ints(s['select_positions'], max_steps))
        select_targets_list.append(pad_ints(s['select_targets'], max_steps, pad=0))
        think_positions_list.append(pad_ints(s['think_positions'], max_steps))

        # Pad explain_positions and explain_targets (each is [n_steps][3])
        ep = s['explain_positions']
        et = s['explain_targets']
        n_pad = max_steps - len(ep)
        ep_padded = ep + [[-1, -1, -1]] * n_pad
        et_padded = et + [[0, 0, 0]] * n_pad
        explain_positions_list.append(ep_padded)
        explain_targets_list.append(et_padded)

    return {
        'input_ids':         torch.tensor(input_ids_list,         dtype=torch.long),
        'reward_values':     torch.tensor(reward_values_list,     dtype=torch.float32),
        'reward_positions':  torch.tensor(reward_positions_list,  dtype=torch.long),
        'select_positions':  torch.tensor(select_positions_list,  dtype=torch.long),
        'select_targets':    torch.tensor(select_targets_list,    dtype=torch.long),
        'think_positions':   torch.tensor(think_positions_list,   dtype=torch.long),
        'explain_positions': torch.tensor(explain_positions_list, dtype=torch.long),
        'explain_targets':   torch.tensor(explain_targets_list,   dtype=torch.long),
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
# Loss computation — Hao-style dual CE
# ---------------------------------------------------------------------------

def compute_hao_loss(
    action_logits:     torch.Tensor,   # [B, n_sel, n_actions]
    select_positions:  torch.Tensor,   # [B, n_sel]  (-1 = pad)
    select_targets:    torch.Tensor,   # [B, n_sel]
    explain_logits:    torch.Tensor,   # [B, n_rounds, 3, vocab_size]
    explain_positions: torch.Tensor,   # [B, n_rounds, 3]
    explain_targets:   torch.Tensor,   # [B, n_rounds, 3]
    n_continuous:      int,
    lambda_explain:    float = 1.0,
) -> Tuple[torch.Tensor, float, float, float]:
    """Compute CE_select + lambda * CE_explain.

    Returns (total_loss, ce_select_val, ce_explain_val, select_accuracy).
    """
    device = action_logits.device
    n_discrete = 3 - n_continuous

    # ---- CE_select: action prediction at SELECT positions ----
    valid_sel = (select_positions >= 0)
    n_valid_sel = valid_sel.sum().item()

    if n_valid_sel == 0:
        ce_select = torch.tensor(0.0, device=device, requires_grad=True)
        acc = 0.0
    else:
        logits_sel  = action_logits[valid_sel]           # [N, n_actions]
        targets_sel = select_targets[valid_sel]          # [N]
        ce_select = F.cross_entropy(logits_sel, targets_sel)
        acc = (logits_sel.argmax(dim=-1) == targets_sel).float().mean().item()

    # ---- CE_explain: discrete explanation tokens ----
    if n_discrete > 0:
        # Only supervise the first n_discrete explain tokens per round
        ep_disc = explain_positions[:, :, :n_discrete]   # [B, n_rounds, n_discrete]
        et_disc = explain_targets[:, :, :n_discrete]     # [B, n_rounds, n_discrete]
        el_disc = explain_logits[:, :, :n_discrete, :]   # [B, n_rounds, n_discrete, V]

        valid_exp = (ep_disc >= 0)                       # [B, n_rounds, n_discrete]
        n_valid_exp = valid_exp.sum().item()

        if n_valid_exp == 0:
            ce_explain = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            logits_exp  = el_disc[valid_exp]             # [M, V]
            targets_exp = et_disc[valid_exp]             # [M]
            ce_explain = F.cross_entropy(logits_exp, targets_exp)
    else:
        ce_explain = torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = ce_select + lambda_explain * ce_explain
    return total_loss, ce_select.item(), ce_explain.item(), acc


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:        COCONUTTransformer,
    val_loader:   DataLoader,
    device:       torch.device,
    n_continuous:  int,
    max_batches:  Optional[int] = 50,
) -> Tuple[float, float]:
    """Evaluate model on validation set.

    Returns (mean_ce_select, mean_acc) averaged over up to max_batches.
    """
    model.eval()
    ce_total  = 0.0
    acc_total = 0.0
    n_batches = 0

    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        action_logits, _ = model.forward_hao(
            input_ids         = batch['input_ids'],
            reward_values     = batch['reward_values'],
            reward_positions  = batch['reward_positions'],
            select_positions  = batch['select_positions'],
            think_positions   = batch['think_positions'],
            explain_positions = batch['explain_positions'],
            n_continuous      = n_continuous,
        )

        # Eval metric: CE on select positions only (not explain)
        valid = (batch['select_positions'] >= 0)
        n_valid = valid.sum().item()
        if n_valid > 0:
            logits  = action_logits[valid]
            targets = batch['select_targets'][valid]
            ce  = F.cross_entropy(logits, targets).item()
            acc = (logits.argmax(-1) == targets).float().mean().item()
        else:
            ce, acc = 0.0, 0.0

        ce_total  += ce
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

    # ---- Vocab for sequence expansion ----
    vocab = build_vocab(cfg_data['n_states'], cfg_data['n_actions'],
                        model_config.n_q_bins)

    # ---- Datasets ----
    train_ds = COCONUTDataset(train_seqs)
    val_ds   = COCONUTDataset(val_seqs)

    # ---- Model ----
    model = COCONUTTransformer(model_config).to(device)
    n_params = model.num_parameters()
    print(f"\nModel parameters: {n_params:,}")
    print(f"  vocab_size: {model_config.vocab_size} (includes {model_config.n_q_bins} Q-bin tokens)")

    # ---- Optimizer ----
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.95),
    )

    # ---- AMP scaler ----
    use_amp = (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\nTraining for {args.epochs} epochs")
    print(f"  batch_size: {args.batch_size}  |  lr: {args.lr}  |  weight_decay: {args.weight_decay}")
    print(f"  AMP: {use_amp}")
    print(f"  Hao stages: {[(s[0], s[2], s[3]) for s in HAO_STAGES]}")

    # ---- Weights & Biases ----
    if args.use_wandb:
        if not HAS_WANDB:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        run_tags = ["hao-curriculum", "coconut-feedback", "ce-dual"]
        if not use_coconut:
            run_tags.append("no-coconut-ablation")
        wandb.init(
            project = "coconut-qlearning",
            name    = args.run_name if args.run_name else None,
            tags    = run_tags,
            config  = {
                "architecture": "COCONUT Hao-Curriculum Transformer",
                "n_layers":     args.n_layers,
                "n_heads":      args.n_heads,
                "d_model":      args.d_model,
                "d_ff":         args.d_ff,
                "dropout":      args.dropout,
                "vocab_size":   model_config.vocab_size,
                "n_q_bins":     model_config.n_q_bins,
                "n_params":     n_params,
                "n_sequences":  cfg_data.get('n_sequences', len(train_seqs) + len(val_seqs)),
                "n_states":     cfg_data['n_states'],
                "n_actions":    cfg_data['n_actions'],
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "lr":           args.lr,
                "weight_decay": args.weight_decay,
                "use_coconut":  use_coconut,
                "hao_stages":   [{"last_epoch": s[0], "max_steps": s[1],
                                  "n_continuous": s[2]} for s in HAO_STAGES],
            }
        )

    # ---- Checkpointing ----
    ckpt_dir  = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.run_name:
        ckpt_path = os.path.join(ckpt_dir, f'coconut_transformer_{args.run_name}.pt')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'coconut_transformer.pt')

    best_val_loss = float('inf')
    global_step   = 0
    log_ce_acc    = 0.0
    log_exp_acc   = 0.0
    log_acc_acc   = 0.0
    log_n         = 0
    t0            = time.time()
    current_stage = -1
    n_continuous  = 0

    training_log  = []
    scheduler     = None
    train_loader  = None
    val_loader    = None

    def make_loaders(n_cont):
        """Create train/val DataLoaders with the appropriate collate function."""
        cfn = partial(collate_fn_hao,
                      n_continuous=n_cont,
                      vocab=vocab,
                      n_q_bins=model_config.n_q_bins,
                      q_bin_min=model_config.q_bin_min,
                      q_bin_max=model_config.q_bin_max)
        tl = DataLoader(
            train_ds,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = args.num_workers,
            collate_fn  = cfn,
            pin_memory  = (device.type == 'cuda'),
        )
        vl = DataLoader(
            val_ds,
            batch_size  = args.batch_size * 2,
            shuffle     = False,
            num_workers = args.num_workers,
            collate_fn  = cfn,
            pin_memory  = (device.type == 'cuda'),
        )
        return tl, vl

    print()

    for epoch in range(1, args.epochs + 1):
        # ---- Determine Hao stage ----
        new_stage = None
        for i, (last_ep, ms, nc, desc) in enumerate(HAO_STAGES):
            if epoch <= last_ep:
                new_stage = i
                break
        if new_stage is None:
            new_stage = len(HAO_STAGES) - 1

        # For --no_coconut: override n_continuous to 0
        stage_n_continuous = HAO_STAGES[new_stage][2]
        if not use_coconut:
            stage_n_continuous = 0

        # ---- Stage transition ----
        if new_stage != current_stage or (current_stage == -1):
            current_stage = new_stage
            n_continuous  = stage_n_continuous
            stage_max_steps = HAO_STAGES[current_stage][1]

            train_ds.max_steps           = stage_max_steps
            train_ds.stage_idx           = current_stage
            train_ds.all_stage_max_steps = STAGE_MAX_STEPS

            desc = HAO_STAGES[current_stage][3]
            print(f"[hao] Stage {current_stage}/{len(HAO_STAGES)-1}: "
                  f"n_continuous={n_continuous}, max_steps={stage_max_steps} — {desc}")

            # Reset optimizer state (per Hao et al.)
            if current_stage > 0:
                optimizer.state.clear()
                print(f"  -> optimizer state reset")

            # Recreate DataLoaders with updated collate (different n_continuous)
            train_loader, val_loader = make_loaders(n_continuous)

            # Fresh cosine LR schedule
            stage_last_ep     = HAO_STAGES[current_stage][0]
            remaining_epochs  = stage_last_ep - epoch + 1
            stage_total_steps = len(train_loader) * remaining_epochs
            stage_warmup      = min(100, stage_total_steps // 10)
            scheduler = get_warmup_cosine_scheduler(optimizer, stage_warmup, stage_total_steps)

        model.train()

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                action_logits, explain_logits = model.forward_hao(
                    input_ids         = batch['input_ids'],
                    reward_values     = batch['reward_values'],
                    reward_positions  = batch['reward_positions'],
                    select_positions  = batch['select_positions'],
                    think_positions   = batch['think_positions'],
                    explain_positions = batch['explain_positions'],
                    n_continuous      = n_continuous,
                    truncate_bptt_window = 5,
                )

                loss, ce_sel, ce_exp, acc = compute_hao_loss(
                    action_logits,
                    batch['select_positions'],
                    batch['select_targets'],
                    explain_logits,
                    batch['explain_positions'],
                    batch['explain_targets'],
                    n_continuous   = n_continuous,
                    lambda_explain = 1.0,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            log_ce_acc  += ce_sel
            log_exp_acc += ce_exp
            log_acc_acc += acc
            log_n       += 1

            # ---- Periodic logging + evaluation ----
            if global_step % args.eval_every == 0:
                avg_ce  = log_ce_acc  / log_n
                avg_exp = log_exp_acc / log_n
                avg_acc = log_acc_acc / log_n

                val_ce, val_acc = evaluate(model, val_loader, device, n_continuous)
                model.train()

                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]

                print(
                    f"[ep {epoch:02d} | step {global_step:6d}]  "
                    f"train: CE_sel={avg_ce:.4f} CE_exp={avg_exp:.4f} acc={avg_acc*100:.1f}%  |  "
                    f"val: CE={val_ce:.4f} acc={val_acc*100:.1f}%  |  "
                    f"lr={lr_now:.2e} t={elapsed:.0f}s"
                )

                training_log.append((global_step, avg_ce, avg_acc, val_ce, val_acc))

                if args.use_wandb:
                    wandb.log({
                        "train/ce_select":      avg_ce,
                        "train/ce_explain":     avg_exp,
                        "train/accuracy":       avg_acc,
                        "val/ce_loss":          val_ce,
                        "val/accuracy":         val_acc,
                        "lr":                   lr_now,
                        "curriculum/stage":     current_stage,
                        "curriculum/n_continuous": n_continuous,
                        "curriculum/max_steps": HAO_STAGES[current_stage][1],
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
                        'n_continuous':     n_continuous,
                        'hao_stage':        current_stage,
                    }, ckpt_path)
                    print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

                log_ce_acc  = 0.0
                log_exp_acc = 0.0
                log_acc_acc = 0.0
                log_n       = 0

        # End-of-epoch evaluation
        val_ce, val_acc = evaluate(model, val_loader, device, n_continuous)
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
                'n_continuous':     n_continuous,
                'hao_stage':        current_stage,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

        if args.use_wandb:
            wandb.log({
                "epoch/val_ce_loss":      val_ce,
                "epoch/val_accuracy":     val_acc,
                "epoch/best_val_loss":    best_val_loss,
                "epoch/curriculum_stage": current_stage,
                "epoch":                  epoch,
                "step":                   global_step,
            })

    # ---- Save training log ----
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
    parser = argparse.ArgumentParser(
        description='Train COCONUT Q-Learning Transformer (Hao-style curriculum)')

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
    parser.add_argument('--epochs',       type=int,   default=24)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--eval_every',   type=int,   default=500)
    parser.add_argument('--num_workers',  type=int,   default=4)

    # Ablation flag
    parser.add_argument('--no_coconut', action='store_true',
                        help='Stay in Stage 0 (fully discrete) forever. '
                             'No continuous thoughts — pure discrete baseline.')

    # W&B
    parser.add_argument('--use_wandb',  action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--run_name',   type=str, default=None,
                        help='W&B run name (default: auto-generated)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
