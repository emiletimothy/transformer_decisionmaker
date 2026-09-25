#!/usr/bin/env python3
"""
eval_clean_steps.py — Which rule does the model actually follow?

Plain agreement with the teacher's label a* = argmax Q_{gamma=0.9}(s', .) cannot
separate Q-learning from simpler rules, because on most steps they give the same
label (see data_shortcuts.py). This script runs a checkpoint
teacher-forced exactly as in training (3_train.batched_episode_forward), keeps
its greedy choice at every step, and replays alternative labels from the same
history:

  q_g0.9       the teacher itself (bootstrapped Q-learning, data's alpha)
  q_g0.5       Q-learning with gamma = 0.5
  q_g0         myopic incremental reward average (gamma = 0)
  most_tried   reward-blind: action tried most often at s'
  last_astar   copy the most recent a* token revealed for s' (the model sees
               the teacher's a* in its input, so this is a real shortcut)

For every rule X it reports, on the steps where X and the teacher both have a
unique argmax and DISAGREE, how often the model picks the teacher's action vs
X's action. Those steps are the only evidence that separates the two.

  python3 eval_clean_steps.py \
      --checkpoints ../checkpoints/a.pt ../checkpoints/b.pt \
      --data ../data/bootstrap_dataset.pt ../data/coconut_dataset.pt \
      --out ../figures/comparison/disagreement.csv
"""
import argparse
import csv
import importlib.util
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from collections import defaultdict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load('coconut_model', '2_model.py')
T = _load('coconut_train', '3_train.py')

GAMMAS = (0.9, 0.5, 0.0)
RULES = ['q_g0.5', 'q_g0', 'most_tried', 'last_astar']


def _argmax_unique(v):
    """Return (argmax, is_unique)."""
    best = v.max()
    idx = np.flatnonzero(v == best)
    return int(idx[0]), len(idx) == 1


def replay_rules(seq, alpha):
    """Per-step labels (and uniqueness flags) for every rule, from the history."""
    nS, nA = seq['n_states'], seq['n_actions']
    Qs = {g: np.zeros((nS, nA), dtype=np.float32) for g in GAMMAS}
    tried = np.zeros((nS, nA))
    last_astar = np.full(nS, -1)
    rows = []
    for tr in seq['transitions']:
        s, a, r, s2 = tr['s'], tr['a'], tr['r'], tr['s_next']
        row = {}
        # last_astar is what was revealed BEFORE this step's label.
        row['last_astar'] = (int(last_astar[s2]), last_astar[s2] >= 0)
        for g, Q in Qs.items():
            mq = float(np.max(Q[s2]))
            Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r + g * mq)
            row[f'q_g{g:g}'] = _argmax_unique(Q[s2])
        tried[s, a] += 1
        row['most_tried'] = _argmax_unique(tried[s2])
        row['teacher'] = tr['a_star']
        rows.append(row)
        last_astar[s2] = tr['a_star']
    return rows


@torch.no_grad()
def model_choices(model, episodes, vocab, max_actions, device):
    """Greedy SELECT choice per step, teacher-forced as in 3_train."""
    B = len(episodes)
    n_act = episodes[0]['n_actions']
    steps = [len(ep['transitions']) for ep in episodes]
    context = model.get_init_context(B, n_act, device)
    preds = [[] for _ in range(B)]
    for t in range(max(steps)):
        active = [i for i in range(B) if t < steps[i]]
        toks, rews, acts = [], [], []
        for i in active:
            tr = episodes[i]['transitions'][t]
            tl, r_off, s_off, u_off = T.build_step_tokens(tr, vocab, n_act)
            toks.append(tl)
            rews.append(tr['r'])
            acts.append(tr['a'])
        logits, upd = model.forward_step(
            token_ids=torch.tensor(toks, dtype=torch.long, device=device),
            reward_value=torch.tensor(rews, dtype=torch.float32, device=device),
            reward_offset=r_off, select_offset=s_off, update_offset=u_off,
            context=context[active],
        )
        logits[:, n_act:] = float('-inf')
        for j, i in enumerate(active):
            preds[i].append(int(logits[j].argmax()))
        write = model.contextualize(upd)
        context = context.clone()
        for j, i in enumerate(active):
            context[i, acts[j], :] = write[j]
    return preds


def evaluate(ckpt_path, data_path, split, max_seqs, batch_size, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = M.COCONUTConfig.from_dict(ckpt['config'])
    model = M.COCONUTTransformer(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    vocab = M.build_vocab(config.max_states, config.max_actions)

    d = torch.load(data_path, map_location='cpu', weights_only=False)
    alpha = float(d['config'].get('alpha', 0.1))
    seqs = d[split][:max_seqs]

    by_na = defaultdict(list)
    for sq in seqs:
        by_na[sq['n_actions']].append(sq)

    # stats[group][key] -> list of bools
    stats = defaultdict(lambda: defaultdict(list))
    for na, group in by_na.items():
        for b in range(0, len(group), batch_size):
            eps = group[b:b + batch_size]
            preds = model_choices(model, eps, vocab, config.max_actions, device)
            for sq, pr in zip(eps, preds):
                rows = replay_rules(sq, alpha)
                q = sq['q_snapshots']
                for t, (row, p) in enumerate(zip(rows, pr)):
                    s2 = sq['transitions'][t]['s_next']
                    qt = np.asarray(q[t])[s2]
                    if (qt == qt.max()).sum() > 1:
                        continue  # teacher label is a random tie-break
                    teach = row['teacher']
                    for grp in ('ALL', f"task={sq.get('task', 'orig')}"):
                        st = stats[grp]
                        st['acc_teacher'].append(p == teach)
                        for g in GAMMAS:
                            lab, uniq = row[f'q_g{g:g}']
                            if uniq:
                                st[f'acc_q_g{g:g}'].append(p == lab)
                        for rule in RULES:
                            lab, uniq = row[rule]
                            if not uniq or lab == teach:
                                continue
                            st[f'{rule}:follow_teacher'].append(p == teach)
                            st[f'{rule}:follow_rule'].append(p == lab)
                            st[f'{rule}:chance'].append(1.0 / na)
                            # 'clean': copying the last revealed a* for s'
                            # would NOT give the teacher's label, so following
                            # the teacher here cannot be explained by copying.
                            la, la_ok = row['last_astar']
                            if rule != 'last_astar' and not (la_ok and la == teach):
                                st[f'{rule}:clean:follow_teacher'].append(p == teach)
                                st[f'{rule}:clean:follow_rule'].append(p == lab)
    return stats, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', default=[str(paths.checkpoint(m)) for m in paths.MODELS])
    ap.add_argument('--data', nargs='+', default=[f'{paths.DATASET}:val'])
    ap.add_argument('--split', default='val',
                    help="default split; override per file as path.pt:train")
    ap.add_argument('--max_seqs', type=int, default=5000)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--out', default=str(paths.FIGURES / 'comparison' / 'clean_steps' / 'clean_steps.csv'),
                    help='CSV output path')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    out_rows = []
    for ck in args.checkpoints:
        for spec in args.data:
            dp, _, split = spec.partition(':')
            split = split or args.split
            stats, alpha = evaluate(ck, dp, split, args.max_seqs,
                                    args.batch_size, device)
            name, dname = os.path.basename(ck), f'{os.path.basename(dp)}:{split}'
            print(f'\n=== {name}  on  {dname} (alpha={alpha})', flush=True)
            for grp in sorted(stats, key=lambda k: (k != 'ALL', k)):
                st = stats[grp]
                acc = '  '.join(
                    f"{k[4:]}={np.mean(st[k]):.3f}" for k in
                    ['acc_teacher'] + [f'acc_q_g{g:g}' for g in GAMMAS])
                print(f'  {grp:14s} n={len(st["acc_teacher"]):7d}  agreement: {acc}')
                for rule in RULES:
                    ft = st[f'{rule}:follow_teacher']
                    if not ft:
                        continue
                    fr = st[f'{rule}:follow_rule']
                    ch = np.mean(st[f'{rule}:chance'])
                    cft = st[f'{rule}:clean:follow_teacher']
                    cfr = st[f'{rule}:clean:follow_rule']
                    clean = (f'  | clean n={len(cft):5d} teacher {np.mean(cft):.3f} '
                             f'{rule} {np.mean(cfr):.3f}') if cft else ''
                    print(f'      teacher vs {rule:11s} n={len(ft):6d}  '
                          f'follows teacher {np.mean(ft):.3f}  follows {rule} {np.mean(fr):.3f}  '
                          f'(chance {ch:.3f}){clean}')
                    out_rows.append({
                        'checkpoint': name, 'data': dname, 'alpha': alpha,
                        'group': grp, 'rule': rule, 'n': len(ft),
                        'follow_teacher': np.mean(ft),
                        'follow_teacher_sem': np.std(ft) / np.sqrt(len(ft)),
                        'follow_rule': np.mean(fr),
                        'follow_rule_sem': np.std(fr) / np.sqrt(len(fr)),
                        'chance': ch,
                        'clean_n': len(cft),
                        'clean_follow_teacher': np.mean(cft) if cft else np.nan,
                        'clean_follow_rule': np.mean(cfr) if cfr else np.nan,
                        'acc_teacher': np.mean(st['acc_teacher']),
                        **{f'acc_q_g{g:g}': np.mean(st[f'acc_q_g{g:g}']) for g in GAMMAS},
                    })
    if args.out and out_rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
