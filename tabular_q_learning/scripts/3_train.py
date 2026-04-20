#!/usr/bin/env python3
"""
3_train.py — Hao-Style Multi-Stage Curriculum Training (Two-Phase Per-Round Layout)

Trains the COCONUTTransformer from 2_model.py on the dataset produced by
1_generate_data.py using a 5-stage curriculum that progressively increases the
fraction of ROUNDS using the continuous thought block (1 TOK_COT token) vs. the
discrete thought block (3 tokens: s_t, a_t, Q_bin).

Per-round token layout:
  Phase 1 (11 tokens, |A|=2): s_t, a_t, R, s_next, [s_next, a_c, EVAL]*|A|, SELECT
  Phase 2 discrete (7 tokens): ANEXT, QCURR, QNEXT, UPDATE, s_t, a_t, Q_bin
  Phase 2 continuous (5 tokens): ANEXT, QCURR, QNEXT, UPDATE, COT

Stage schedule (curriculum over rounds, not tokens):
    Stage 0 (epochs  1-5):  0% continuous rounds   — all discrete
    Stage 1 (epochs  6-10): 25% continuous rounds
    Stage 2 (epochs 11-15): 50% continuous rounds
    Stage 3 (epochs 16-20): 75% continuous rounds
    Stage 4 (epochs 21-25): 100% continuous rounds  — all continuous

Loss: CE_select (always) + lambda_thought * CE_thought (discrete rounds only)

Optimizer state is reset at each stage transition (following Hao et al.).
KV caching accelerates the COCONUT injection loop in Stages 1-4.
"""

import argparse
import math
import os
import random
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
# (last_epoch_inclusive, max_steps, frac_continuous, description)
# ---------------------------------------------------------------------------

HAO_STAGES = [
    (5,  50, 0.00, "Stage 0: 0% continuous rounds (all discrete)"),
    (10, 50, 0.25, "Stage 1: 25% continuous rounds"),
    (15, 50, 0.50, "Stage 2: 50% continuous rounds"),
    (20, 50, 0.75, "Stage 3: 75% continuous rounds"),
    (25, 50, 1.00, "Stage 4: 100% continuous rounds (all continuous)"),
]
STAGE_MAX_STEPS = [ms for _, ms, _, _ in HAO_STAGES]


# ---------------------------------------------------------------------------
# Sequence truncation helper
# ---------------------------------------------------------------------------

def truncate_sequence(seq: Dict, max_steps: int) -> Dict:
    """Return a copy of seq truncated to at most max_steps rounds."""
    n = min(seq['n_steps'], max_steps)
    if n == seq['n_steps']:
        return seq

    full_n   = seq['n_steps']
    full_len = len(seq['input_ids'])
    round_len = (full_len - 2) // full_n  # 2-token prefix

    trunc = dict(seq)
    trunc['input_ids'] = seq['input_ids'][:2 + n * round_len]
    for field in ('reward_values', 'reward_positions',
                  'select_positions', 'select_targets',
                  'a_next_positions', 'q_curr_positions', 'q_next_positions',
                  'update_positions', 'thought_positions',
                  'q_values_for_cot'):
        if field in seq:
            trunc[field] = seq[field][:n]
    trunc['n_steps'] = n
    return trunc


# ---------------------------------------------------------------------------
# Mixed-sequence builder: replaces expand_sequence
# ---------------------------------------------------------------------------

def build_mixed_sequence(
    seq: Dict,
    continuous_round_mask: List[bool],
    vocab: Dict,
    n_q_bins: int,
    q_bin_min: float,
    q_bin_max: float,
) -> Dict:
    """Build the actual token sequence mixing discrete and continuous rounds.

    Takes a stored discrete sequence (18 tokens/round for |A|=2) and a per-round
    mask. For continuous rounds, replaces the 3-token thought block with 1 TOK_COT.
    For discrete rounds, patches the Q_bin placeholder with the real bin value.

    All position arrays (select, update, thought) are recomputed against the new
    absolute token offsets in the mixed sequence.

    Parameters
    ----------
    seq                   : stored dataset dict (discrete form, from 1_generate_data.py)
    continuous_round_mask : [n_rounds] bool — True = emit 1 COT, False = emit 3-token block
    vocab                 : vocabulary dict from build_vocab
    n_q_bins, q_bin_min, q_bin_max : Q-value discretization params

    Returns
    -------
    dict with mixed token sequence and recomputed position arrays:
        input_ids, reward_values, reward_positions, select_positions,
        select_targets, update_positions, thought_positions ([n][3], -1 for invalid),
        thought_targets ([n][3]), continuous_round_mask (as passed), n_steps
    """
    n_steps   = seq['n_steps']
    old_ids   = seq['input_ids']
    qvc       = seq.get('q_values_for_cot', [])

    old_round_len = (len(old_ids) - 2) // n_steps  # 18 for n_actions=2

    TOK_COT    = vocab['TOK_COT']
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_QBIN   = vocab['TOK_QBIN']

    new_ids              = list(old_ids[:2])  # [TOK_NULL, TOK_START]
    new_reward_values    = []
    new_reward_positions = []
    new_select_positions = []
    new_select_targets   = []
    new_update_positions = []
    new_thought_positions = []   # [n_steps][3]
    new_thought_targets  = []   # [n_steps][3]

    for r in range(n_steps):
        round_start  = 2 + r * old_round_len
        round_tokens = old_ids[round_start:round_start + old_round_len]

        entry = qvc[r]
        st, at, q_new = entry['st'], entry['at'], entry['q_new']
        q_bin = discretize_q_value(q_new, n_q_bins, q_bin_min, q_bin_max)
        q_bin_tok = TOK_QBIN[q_bin]

        # Phase 1 + Phase 2 scaffold: tokens 0..old_round_len-4 (first 14 tokens for |A|=2)
        # The last 3 tokens of the stored round are the thought block placeholder.
        # Layout offsets within the round (for |A|=2, old_round_len=18):
        #   0..10  : Phase 1 (s_t, a_t, R, s_next, evals, SELECT)
        #   11..14 : Phase 2 scaffold (ANEXT, QCURR, QNEXT, UPDATE)
        #   15..17 : thought block placeholder (s_t token, a_t token, TOK_NULL placeholder)
        scaffold_end = old_round_len - 3  # index of first thought token (15 for |A|=2)

        # Emit Phase 1 + scaffold tokens, fixing position tracking
        for k in range(scaffold_end):
            abs_orig = round_start + k
            pos_new  = len(new_ids)

            # reward position
            if abs_orig in seq.get('reward_positions', []):
                new_reward_positions.append(pos_new)
                r_idx = seq['reward_positions'].index(abs_orig)
                new_reward_values.append(seq['reward_values'][r_idx])

            # select position
            if len(seq['select_positions']) > r and abs_orig == seq['select_positions'][r]:
                new_select_positions.append(pos_new)
                new_select_targets.append(seq['select_targets'][r])

            # update position
            if len(seq['update_positions']) > r and abs_orig == seq['update_positions'][r]:
                new_update_positions.append(pos_new)

            new_ids.append(round_tokens[k])

        # Emit thought block
        if continuous_round_mask[r]:
            # Continuous: 1 TOK_COT token
            cot_pos = len(new_ids)
            new_ids.append(TOK_COT)
            new_thought_positions.append([cot_pos, -1, -1])
            new_thought_targets.append([0, 0, 0])
        else:
            # Discrete: 3 tokens [s_t, a_t, Q_bin], with real Q_bin substituted
            p0 = len(new_ids); new_ids.append(TOK_S[st])
            p1 = len(new_ids); new_ids.append(TOK_A[at])
            p2 = len(new_ids); new_ids.append(q_bin_tok)
            new_thought_positions.append([p0, p1, p2])
            new_thought_targets.append([TOK_S[st], TOK_A[at], q_bin_tok])

    return {
        'input_ids':            new_ids,
        'reward_values':        new_reward_values,
        'reward_positions':     new_reward_positions,
        'select_positions':     new_select_positions,
        'select_targets':       new_select_targets,
        'update_positions':     new_update_positions,
        'thought_positions':    new_thought_positions,
        'thought_targets':      new_thought_targets,
        'continuous_round_mask': list(continuous_round_mask),
        'n_steps':              n_steps,
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
# Collate function
# ---------------------------------------------------------------------------

def collate_fn_hao(
    batch: List[Dict],
    stage_idx: int,
    vocab: Dict,
    n_q_bins: int,
    q_bin_min: float,
    q_bin_max: float,
) -> Dict[str, torch.Tensor]:
    """Build mixed discrete/continuous sequences, then pad and batch."""
    frac = HAO_STAGES[stage_idx][2]

    expanded = []
    for s in batch:
        n_rounds = s['n_steps']
        n_cont   = int(math.floor(frac * n_rounds))
        # First n_cont rounds are continuous, rest are discrete
        crm = [True] * n_cont + [False] * (n_rounds - n_cont)
        expanded.append(build_mixed_sequence(s, crm, vocab, n_q_bins, q_bin_min, q_bin_max))

    max_seq   = max(len(s['input_ids']) for s in expanded)
    max_steps = max(s['n_steps'] for s in expanded)

    def pad_ids(ids, length):
        return ids + [0] * (length - len(ids))

    def pad_ints(lst, length, pad=-1):
        return lst + [pad] * (length - len(lst))

    def pad_floats(lst, length, pad=0.0):
        return lst + [pad] * (length - len(lst))

    input_ids_list         = []
    reward_values_list     = []
    reward_positions_list  = []
    select_positions_list  = []
    select_targets_list    = []
    update_positions_list  = []
    thought_positions_list = []  # [B, max_steps, 3]
    thought_targets_list   = []  # [B, max_steps, 3]
    cont_mask_list         = []  # [B, max_steps]

    for s in expanded:
        input_ids_list.append(pad_ids(s['input_ids'], max_seq))
        reward_values_list.append(pad_floats(s['reward_values'], max_steps))
        reward_positions_list.append(pad_ints(s['reward_positions'], max_steps))
        select_positions_list.append(pad_ints(s['select_positions'], max_steps))
        select_targets_list.append(pad_ints(s['select_targets'], max_steps, pad=0))
        update_positions_list.append(pad_ints(s['update_positions'], max_steps))

        tp = s['thought_positions']
        tt = s['thought_targets']
        cm = s['continuous_round_mask']
        n_pad = max_steps - len(tp)
        tp_padded = tp + [[-1, -1, -1]] * n_pad
        tt_padded = tt + [[0, 0, 0]] * n_pad
        cm_padded = cm + [False] * n_pad
        thought_positions_list.append(tp_padded)
        thought_targets_list.append(tt_padded)
        cont_mask_list.append(cm_padded)

    return {
        'input_ids':            torch.tensor(input_ids_list,         dtype=torch.long),
        'reward_values':        torch.tensor(reward_values_list,     dtype=torch.float32),
        'reward_positions':     torch.tensor(reward_positions_list,  dtype=torch.long),
        'select_positions':     torch.tensor(select_positions_list,  dtype=torch.long),
        'select_targets':       torch.tensor(select_targets_list,    dtype=torch.long),
        'update_positions':     torch.tensor(update_positions_list,  dtype=torch.long),
        'thought_positions':    torch.tensor(thought_positions_list, dtype=torch.long),
        'thought_targets':      torch.tensor(thought_targets_list,   dtype=torch.long),
        'continuous_round_mask': torch.tensor(cont_mask_list,        dtype=torch.bool),
    }


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
# Loss computation
# ---------------------------------------------------------------------------

def compute_hao_loss(
    action_logits:         torch.Tensor,   # [B, n_rounds, n_actions]
    select_positions:      torch.Tensor,   # [B, n_rounds]  (-1 = pad)
    select_targets:        torch.Tensor,   # [B, n_rounds]
    thought_logits:        torch.Tensor,   # [B, n_rounds, 3, vocab_size]
    thought_positions:     torch.Tensor,   # [B, n_rounds, 3]
    thought_targets:       torch.Tensor,   # [B, n_rounds, 3]
    continuous_round_mask: torch.Tensor,   # [B, n_rounds] bool
    lambda_thought:        float = 1.0,
) -> Tuple[torch.Tensor, float, float, float]:
    """Compute CE_select + lambda * CE_thought.

    CE_select: always active at every SELECT position.
    CE_thought: active only for discrete rounds (continuous_round_mask == False)
                at all 3 thought positions that are valid (>= 0).

    Returns (total_loss, ce_select_val, ce_thought_val, select_accuracy).
    """
    device = action_logits.device

    # ---- CE_select ----
    valid_sel = (select_positions >= 0)
    n_valid_sel = valid_sel.sum().item()

    if n_valid_sel == 0:
        ce_select = torch.tensor(0.0, device=device, requires_grad=True)
        acc = 0.0
    else:
        logits_sel  = action_logits[valid_sel]
        targets_sel = select_targets[valid_sel]
        ce_select = F.cross_entropy(logits_sel, targets_sel)
        acc = (logits_sel.argmax(dim=-1) == targets_sel).float().mean().item()

    # ---- CE_thought: discrete rounds only ----
    # valid = position >= 0 AND round is discrete (not continuous)
    disc_mask = ~continuous_round_mask.unsqueeze(-1)           # [B, n_rounds, 1] broadcast
    valid_thought = (thought_positions >= 0) & disc_mask       # [B, n_rounds, 3]
    n_valid_thought = valid_thought.sum().item()

    if n_valid_thought == 0:
        ce_thought = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        logits_th  = thought_logits[valid_thought]             # [M, vocab_size]
        targets_th = thought_targets[valid_thought]            # [M]
        ce_thought = F.cross_entropy(logits_th, targets_th)

    total_loss = ce_select + lambda_thought * ce_thought
    return total_loss, ce_select.item(), ce_thought.item(), acc


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:       COCONUTTransformer,
    val_loader:  DataLoader,
    device:      torch.device,
    max_batches: Optional[int] = 50,
) -> Tuple[float, float]:
    """Evaluate on validation set. Returns (mean_ce_select, mean_acc)."""
    model.eval()
    ce_total  = 0.0
    acc_total = 0.0
    n_batches = 0

    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        action_logits, _ = model.forward_hao(
            input_ids             = batch['input_ids'],
            reward_values         = batch['reward_values'],
            reward_positions      = batch['reward_positions'],
            select_positions      = batch['select_positions'],
            update_positions      = batch['update_positions'],
            thought_positions     = batch['thought_positions'],
            continuous_round_mask = batch['continuous_round_mask'],
        )

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
        run_tags = ["hao-curriculum", "coconut-feedback", "two-phase-layout"]
        if not use_coconut:
            run_tags.append("no-coconut-ablation")
        wandb.init(
            project = "coconut-qlearning",
            name    = args.run_name if args.run_name else None,
            tags    = run_tags,
            config  = {
                "architecture":  "COCONUT Two-Phase Transformer",
                "n_layers":      args.n_layers,
                "n_heads":       args.n_heads,
                "d_model":       args.d_model,
                "d_ff":          args.d_ff,
                "dropout":       args.dropout,
                "vocab_size":    model_config.vocab_size,
                "n_q_bins":      model_config.n_q_bins,
                "n_params":      n_params,
                "n_sequences":   cfg_data.get('n_sequences', len(train_seqs) + len(val_seqs)),
                "n_states":      cfg_data['n_states'],
                "n_actions":     cfg_data['n_actions'],
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "lr":            args.lr,
                "weight_decay":  args.weight_decay,
                "use_coconut":   use_coconut,
                "hao_stages":    [{"last_epoch": s[0], "max_steps": s[1],
                                   "frac_continuous": s[2]} for s in HAO_STAGES],
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
    log_ce_sel     = 0.0
    log_ce_th      = 0.0
    log_acc        = 0.0
    log_n          = 0
    t0             = time.time()
    current_stage  = -1
    frac_continuous = 0.0

    training_log = []
    scheduler    = None
    train_loader = None
    val_loader   = None

    def make_loaders(s_idx: int):
        cfn = partial(collate_fn_hao,
                      stage_idx  = s_idx,
                      vocab      = vocab,
                      n_q_bins   = model_config.n_q_bins,
                      q_bin_min  = model_config.q_bin_min,
                      q_bin_max  = model_config.q_bin_max)
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
        for i, (last_ep, ms, fc, desc) in enumerate(HAO_STAGES):
            if epoch <= last_ep:
                new_stage = i
                break
        if new_stage is None:
            new_stage = len(HAO_STAGES) - 1

        # For --no_coconut: stay at stage 0 (all discrete) forever
        if not use_coconut:
            new_stage = 0

        # ---- Stage transition ----
        if new_stage != current_stage:
            current_stage   = new_stage
            frac_continuous = HAO_STAGES[current_stage][2]
            stage_max_steps = HAO_STAGES[current_stage][1]

            train_ds.max_steps           = stage_max_steps
            train_ds.stage_idx           = current_stage
            train_ds.all_stage_max_steps = STAGE_MAX_STEPS

            desc = HAO_STAGES[current_stage][3]
            print(f"[hao] Stage {current_stage}/{len(HAO_STAGES)-1}: "
                  f"frac_continuous={frac_continuous:.0%}, max_steps={stage_max_steps} — {desc}")

            # Reset optimizer state (per Hao et al.)
            if current_stage > 0:
                optimizer.state.clear()
                print(f"  -> optimizer state reset")

            train_loader, val_loader = make_loaders(current_stage)

            # Fresh cosine LR schedule for this stage
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
                action_logits, thought_logits = model.forward_hao(
                    input_ids             = batch['input_ids'],
                    reward_values         = batch['reward_values'],
                    reward_positions      = batch['reward_positions'],
                    select_positions      = batch['select_positions'],
                    update_positions      = batch['update_positions'],
                    thought_positions     = batch['thought_positions'],
                    continuous_round_mask = batch['continuous_round_mask'],
                    truncate_bptt_window  = 5,
                )

                loss, ce_sel, ce_th, acc = compute_hao_loss(
                    action_logits,
                    batch['select_positions'],
                    batch['select_targets'],
                    thought_logits,
                    batch['thought_positions'],
                    batch['thought_targets'],
                    batch['continuous_round_mask'],
                    lambda_thought = 1.0,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            log_ce_sel  += ce_sel
            log_ce_th   += ce_th
            log_acc     += acc
            log_n       += 1

            # ---- Periodic logging + evaluation ----
            if global_step % args.eval_every == 0:
                avg_ce_sel = log_ce_sel / log_n
                avg_ce_th  = log_ce_th  / log_n
                avg_acc    = log_acc    / log_n

                val_ce, val_acc = evaluate(model, val_loader, device)
                model.train()

                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]

                print(
                    f"[ep {epoch:02d} | step {global_step:6d}]  "
                    f"train: CE_sel={avg_ce_sel:.4f} CE_th={avg_ce_th:.4f} acc={avg_acc*100:.1f}%  |  "
                    f"val: CE={val_ce:.4f} acc={val_acc*100:.1f}%  |  "
                    f"lr={lr_now:.2e} t={elapsed:.0f}s"
                )

                training_log.append((global_step, avg_ce_sel, avg_acc, val_ce, val_acc))

                if args.use_wandb:
                    wandb.log({
                        "train/ce_select":          avg_ce_sel,
                        "train/ce_thought":         avg_ce_th,
                        "train/accuracy":           avg_acc,
                        "val/ce_loss":              val_ce,
                        "val/accuracy":             val_acc,
                        "lr":                       lr_now,
                        "curriculum/stage_idx":     current_stage,
                        "curriculum/frac_continuous": frac_continuous,
                        "curriculum/max_steps":     HAO_STAGES[current_stage][1],
                        "epoch":                    epoch,
                        "step":                     global_step,
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
                        'use_coconut':      use_coconut,
                        'stage_idx':        current_stage,
                        'frac_continuous':  frac_continuous,
                    }, ckpt_path)
                    print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

                log_ce_sel = 0.0
                log_ce_th  = 0.0
                log_acc    = 0.0
                log_n      = 0

        # End-of-epoch evaluation
        val_ce, val_acc = evaluate(model, val_loader, device)
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
                'stage_idx':        current_stage,
                'frac_continuous':  frac_continuous,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_ce={best_val_loss:.4f})")

        if args.use_wandb:
            wandb.log({
                "epoch/val_ce_loss":        val_ce,
                "epoch/val_accuracy":       val_acc,
                "epoch/best_val_loss":      best_val_loss,
                "epoch/curriculum_stage":   current_stage,
                "epoch":                    epoch,
                "step":                     global_step,
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
        description='Train COCONUT Q-Learning Transformer (two-phase per-round curriculum)')

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
    parser.add_argument('--epochs',       type=int,   default=25)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--eval_every',   type=int,   default=500)
    parser.add_argument('--num_workers',  type=int,   default=4)

    # Ablation flag
    parser.add_argument('--no_coconut', action='store_true',
                        help='Stay in Stage 0 (fully discrete) forever. Baseline ablation.')

    # W&B
    parser.add_argument('--use_wandb',  action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--run_name',   type=str, default=None,
                        help='W&B run name (default: auto-generated)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
