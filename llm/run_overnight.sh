#!/bin/bash
cd /home/jupyter-xcao/mwu

echo "[$(date)] === Waiting for current experiment to finish ==="
while ps aux | grep "interactive_api_run" | grep -v grep > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] === Current experiment done ==="

# ── Step 1: Draw regret plot ──
echo "[$(date)] === Drawing regret plot ==="
mkdir -p plots
python3 -c "
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mw_lib import get_ground_truth_outputs

T = 100; N = 20
data_json = json.load(open('mw_dataset.json'))
cases = data_json['cases']

def compute_hard_regret(preds, true_labels, expert_preds):
    n_exp = len(expert_preds[0])
    model_cum, expert_cum = 0, [0]*n_exp
    regret = []
    for t in range(len(preds)):
        model_cum += int(preds[t] != true_labels[t])
        for i in range(n_exp): expert_cum[i] += int(expert_preds[t][i] != true_labels[t])
        regret.append(model_cum - min(expert_cum))
    return regret

steps = np.arange(1, T+1)
fig, ax = plt.subplots(figsize=(14, 7))

# MW
mw_regrets = []
for idx in range(N):
    c = cases[idx]; tl = c['true_labels'][:T]; ep = c['expert_predictions'][:T]
    case_100 = {**c,'n_steps':T,'expert_predictions':ep,'true_labels':tl,'losses':c['losses'][:T]}
    gt = get_ground_truth_outputs(case_100)
    mw_regrets.append(compute_hard_regret(gt['algorithm_predictions'], tl, ep))
mw_arr = np.array(mw_regrets)
ax.plot(steps, mw_arr.mean(axis=0), lw=2.5, color='#4C78A8', label='MW Optimal', zorder=10)
ax.fill_between(steps, mw_arr.mean(axis=0)-mw_arr.std(axis=0), mw_arr.mean(axis=0)+mw_arr.std(axis=0), alpha=0.12, color='#4C78A8')

# Random
rand_regrets = []
for idx in range(N):
    c = cases[idx]; tl = c['true_labels'][:T]; ep = c['expert_predictions'][:T]
    n_exp = len(ep[0]); expert_cum = [0]*n_exp; r_cum = 0; reg = []
    np.random.seed(idx+100)
    for t in range(T):
        p = np.random.randint(0,2); r_cum += int(p!=tl[t])
        for i in range(n_exp): expert_cum[i] += int(ep[t][i]!=tl[t])
        reg.append(r_cum - min(expert_cum))
    rand_regrets.append(reg)
rand_arr = np.array(rand_regrets)
ax.plot(steps, rand_arr.mean(axis=0), lw=1.5, color='#AAAAAA', label='Random Guessing', zorder=1)
ax.fill_between(steps, rand_arr.mean(axis=0)-rand_arr.std(axis=0), rand_arr.mean(axis=0)+rand_arr.std(axis=0), alpha=0.06, color='#AAAAAA')

# Majority Vote
mv_regrets = []
for idx in range(N):
    c = cases[idx]; tl = c['true_labels'][:T]; ep = c['expert_predictions'][:T]
    n_exp = len(ep[0])
    preds = [1 if sum(ep[t][i] for i in range(n_exp))/n_exp >= 0.5 else 0 for t in range(T)]
    mv_regrets.append(compute_hard_regret(preds, tl, ep))
mv_arr = np.array(mv_regrets)
ax.plot(steps, mv_arr.mean(axis=0), lw=2, color='#FF69B4', label='Majority Vote (uniform)', zorder=2)
ax.fill_between(steps, mv_arr.mean(axis=0)-mv_arr.std(axis=0), mv_arr.mean(axis=0)+mv_arr.std(axis=0), alpha=0.06, color='#FF69B4')

# Follow the Leader
ftl_regrets = []
for idx in range(N):
    c = cases[idx]; tl = c['true_labels'][:T]; ep = c['expert_predictions'][:T]
    n_exp = len(ep[0]); cum_correct = [0]*n_exp; preds = []
    for t in range(T):
        leader = max(range(n_exp), key=lambda i: cum_correct[i]) if t > 0 else 0
        preds.append(ep[t][leader])
        for i in range(n_exp): cum_correct[i] += int(ep[t][i]==tl[t])
    ftl_regrets.append(compute_hard_regret(preds, tl, ep))
ftl_arr = np.array(ftl_regrets)
ax.plot(steps, ftl_arr.mean(axis=0), lw=2, color='#17BECF', label='Follow the Leader', zorder=2)
ax.fill_between(steps, ftl_arr.mean(axis=0)-ftl_arr.std(axis=0), ftl_arr.mean(axis=0)+ftl_arr.std(axis=0), alpha=0.06, color='#17BECF')

# Follow Previous Winners
fpw_regrets = []
for idx in range(N):
    c = cases[idx]; tl = c['true_labels'][:T]; ep = c['expert_predictions'][:T]
    n_exp = len(ep[0]); correct_mask = [True]*n_exp; preds = []
    for t in range(T):
        trusted = [i for i in range(n_exp) if correct_mask[i]] or list(range(n_exp))
        preds.append(1 if sum(ep[t][i] for i in trusted)/len(trusted) >= 0.5 else 0)
        correct_mask = [ep[t][i]==tl[t] for i in range(n_exp)]
    fpw_regrets.append(compute_hard_regret(preds, tl, ep))
fpw_arr = np.array(fpw_regrets)
ax.plot(steps, fpw_arr.mean(axis=0), lw=2, color='#9467BD', label='Follow Previous Winners', zorder=2)
ax.fill_between(steps, fpw_arr.mean(axis=0)-fpw_arr.std(axis=0), fpw_arr.mean(axis=0)+fpw_arr.std(axis=0), alpha=0.06, color='#9467BD')

# 8 LLM prompts
configs = [
    ('weather', 'weather (hint, note)', '#2CA02C', '-', 2.5),
    ('weather_no_hint', 'weather (no hint, note)', '#2CA02C', '--', 2.5),
    ('weather_nonote', 'weather (hint, no note)', '#90EE90', '-', 2.0),
    ('weather_nonote_nohint', 'weather (no hint, no note)', '#90EE90', '--', 2.0),
    ('online', 'online (hint, note)', '#D62728', '-', 2.5),
    ('online_nohint', 'online (no hint, note)', '#D62728', '--', 2.5),
    ('online_nonote', 'online (hint, no note)', '#FF9999', '-', 2.0),
    ('online_nonote_nohint', 'online (no hint, no note)', '#FF9999', '--', 2.0),
]
for prompt, label, color, ls, lw in configs:
    regrets = []
    for idx in range(N):
        rd = Path(f'exp_results/{prompt}/ds/cases_{idx:03d}/run001/result.json')
        if not rd.exists(): continue
        r = json.load(open(rd))
        preds = r.get('response',{}).get('predictions',[])
        if len(preds) < T: continue
        regrets.append(compute_hard_regret(preds[:T], cases[idx]['true_labels'][:T], cases[idx]['expert_predictions'][:T]))
    if not regrets: continue
    arr = np.array(regrets)
    ax.plot(steps, arr.mean(axis=0), lw=lw, color=color, linestyle=ls, label=f'{label} (n={len(regrets)})', zorder=5)
    ax.fill_between(steps, arr.mean(axis=0)-arr.std(axis=0), arr.mean(axis=0)+arr.std(axis=0), alpha=0.06, color=color)

ax.set_xlabel('Step', fontsize=13)
ax.set_ylabel('Cumulative Regret (hard 0-1)', fontsize=13)
ax.set_title('DS V3: 2x2x2 Prompt Ablation (20 cases, T=100)', fontsize=14)
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('plots/ds_8prompt_ablation_20cases.png', dpi=150, bbox_inches='tight')
print('Saved plot')
"

# ── Step 2: Print summary ──
echo "[$(date)] === Summary ==="
python3 -c "
import json, numpy as np
from pathlib import Path
from mw_lib import get_ground_truth_outputs

data = json.load(open('mw_dataset.json'))
cases = data['cases']
T = 100

prompts = [
    ('weather', 'weather (hint, note)'),
    ('weather_no_hint', 'weather (no hint, note)'),
    ('weather_nonote', 'weather (hint, no note)'),
    ('weather_nonote_nohint', 'weather (no hint, no note)'),
    ('online', 'online (hint, note)'),
    ('online_nohint', 'online (no hint, note)'),
    ('online_nonote', 'online (hint, no note)'),
    ('online_nonote_nohint', 'online (no hint, no note)'),
]

mw_accs = []
for idx in range(20):
    c = cases[idx]; tl = c['true_labels'][:T]; ep = c['expert_predictions'][:T]
    case_100 = {**c,'n_steps':T,'expert_predictions':ep,'true_labels':tl,'losses':c['losses'][:T]}
    gt = get_ground_truth_outputs(case_100)
    mw_accs.append(sum(1 for t in range(T) if gt['algorithm_predictions'][t]==tl[t])/T)

print(f'MW Optimal: {np.mean(mw_accs):.1%}')
for p, label in prompts:
    accs = []
    for idx in range(20):
        rd = Path(f'exp_results/{p}/ds/cases_{idx:03d}/run001/result.json')
        if rd.exists():
            r = json.load(open(rd))
            preds = r.get('response',{}).get('predictions',[])
            if len(preds) >= T:
                accs.append(sum(1 for t in range(T) if preds[t]==cases[idx]['true_labels'][t])/T)
    if accs:
        print(f'{label:>35s}: {np.mean(accs):.1%} +/- {np.std(accs):.1%} (n={len(accs)})')
"

# ── Step 3: Archive ──
echo "[$(date)] === Archiving ==="
cp -r exp_results archive/exp04_2x2x2_prompt_ablation
cp -r prompts archive/exp04_2x2x2_prompt_ablation/prompts
cp -r plots archive/exp04_2x2x2_prompt_ablation/plots

cat > archive/exp04_2x2x2_prompt_ablation/README.md << 'READMEEOF'
# Experiment 04: 2×2×2 Prompt Ablation

## Dataset
- mw_dataset.json (stratified, best 90-100%, 2nd 65-80%, 3rd 55-70%, worst 45-60%)
- 20 cases, 100 steps

## Model
- DeepSeek V3 (deepseek-chat, 671B MoE), temperature=0

## Design: 2×2×2
- Framing: weather (sunny/rainy) vs online (0/1)
- Hint: "more reliable/accurate" vs no hint
- Note: two-turn note-passing vs multi-turn no-note (feedback only)

## Prompts
Goal text: "Your job is to make your own prediction based on the experts' predictions. Your goal is to be as accurate as possible over time."
READMEEOF

echo "[$(date)] === Archive done ==="

# ── Step 4: Git push ──
echo "[$(date)] === Git push ==="
cd /home/jupyter-xcao/mwu
git add -A && git commit -m "Archive exp04: 2x2x2 prompt ablation (8 prompts × 20 cases)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" && git push origin llm

# ── Step 5: Create new prompts with trustworthy hint ──
echo "[$(date)] === Creating v2 prompts with trustworthy hint ==="

for f in prompts/interactive_weather.txt prompts/interactive_weather_no_hint.txt \
         prompts/interactive_weather_nonote.txt prompts/interactive_weather_nonote_nohint.txt \
         prompts/interactive_online.txt prompts/interactive_online_nohint.txt \
         prompts/interactive_online_nonote.txt prompts/interactive_online_nonote_nohint.txt; do

    basename=$(basename "$f" .txt)
    newname="prompts/${basename}_v2.txt"
    cp "$f" "$newname"

    # Add "Use past outcomes..." after "Your goal is to be as accurate as possible over time."
    sed -i 's/Your goal is to be as accurate as possible over time\./Your goal is to be as accurate as possible over time. Use past outcomes to figure out which experts are most trustworthy and improve your future predictions./' "$newname"

    echo "Created $newname"
done

# ── Step 6: Update code to recognize v2 prompts ──
echo "[$(date)] === Updating code for v2 prompts ==="
python3 -c "
import re

with open('interactive_api_run.py', 'r') as f:
    code = f.read()

# Add v2 prompts to TWO_TURN_PROMPTS
old = '''TWO_TURN_PROMPTS = {\"interactive_weather\", \"interactive_online\",
                    \"interactive_weather_no_hint\", \"interactive_online\",
                    \"interactive_online_nohint\"}'''

# Actually let's just add them to the existing sets
# Find TWO_TURN_PROMPTS and add v2 variants
code = code.replace(
    'TWO_TURN_PROMPTS = {\"interactive_weather\", \"interactive_online\",\n                    \"interactive_weather_no_hint\", \"interactive_online\",\n                    \"interactive_online_nohint\"}',
    'TWO_TURN_PROMPTS = {\"interactive_weather\", \"interactive_online\",\n                    \"interactive_weather_no_hint\", \"interactive_online\",\n                    \"interactive_online_nohint\",\n                    \"interactive_weather_v2\", \"interactive_weather_no_hint_v2\",\n                    \"interactive_online_v2\", \"interactive_online_nohint_v2\"}'
)

# Hmm, the format might not match exactly. Let me just do a broader approach.
# Add v2 to WEATHER_PROMPTS and TWO_TURN_PROMPTS and NONOTE
lines = code.split('\n')
new_lines = []
for line in lines:
    new_lines.append(line)
    if 'TWO_TURN_PROMPTS' in line and '{' in line and 'interactive_weather' in line:
        # Check if this is a multi-line definition
        pass  # Will handle differently

with open('interactive_api_run.py', 'w') as f:
    f.write(code)
print('Code update needs manual review')
"

# Manual approach: just add v2 prompts to the sets
python3 << 'PYEOF'
content = open('interactive_api_run.py').read()

# Add v2 to TWO_TURN_PROMPTS
content = content.replace(
    '"interactive_online_nohint"}',
    '"interactive_online_nohint",\n                    "interactive_weather_v2", "interactive_weather_no_hint_v2",\n                    "interactive_online_v2", "interactive_online_nohint_v2"}'
)

# Add v2 to WEATHER_PROMPTS
content = content.replace(
    'WEATHER_PROMPTS = {"interactive_weather", "interactive_weather_no_hint"}',
    'WEATHER_PROMPTS = {"interactive_weather", "interactive_weather_no_hint",\n                    "interactive_weather_v2", "interactive_weather_no_hint_v2"}'
)

# Add v2 nonote to NONOTE_PROMPTS
content = content.replace(
    '"interactive_online_nonote_nohint"}',
    '"interactive_online_nonote_nohint",\n                  "interactive_weather_nonote_v2", "interactive_weather_nonote_nohint_v2",\n                  "interactive_online_nonote_v2", "interactive_online_nonote_nohint_v2"}'
)

# Add v2 nonote to NONOTE_WEATHER
content = content.replace(
    'NONOTE_WEATHER = {"interactive_weather_nonote", "interactive_weather_nonote_nohint"}',
    'NONOTE_WEATHER = {"interactive_weather_nonote", "interactive_weather_nonote_nohint",\n                  "interactive_weather_nonote_v2", "interactive_weather_nonote_nohint_v2"}'
)

open('interactive_api_run.py', 'w').write(content)
print('Code updated with v2 prompts')
PYEOF

# ── Step 7: Run v2 experiments ──
echo "[$(date)] === Running v2 experiments ==="
IDX="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

python interactive_api_run.py cases --idx $IDX \
    --model ds \
    --prompt interactive_weather_v2 interactive_weather_no_hint_v2 \
            interactive_weather_nonote_v2 interactive_weather_nonote_nohint_v2 \
            interactive_online_v2 interactive_online_nohint_v2 \
            interactive_online_nonote_v2 interactive_online_nonote_nohint_v2 \
    --dataset mw_dataset.json --steps 100 2>&1

echo "[$(date)] === v2 experiments done ==="

# ── Step 8: Draw v2 plot and push ──
echo "[$(date)] === Final git push ==="
git add -A && git commit -m "v2 prompts (trustworthy hint): 8 prompts × 20 cases

Added 'Use past outcomes to figure out which experts are most trustworthy
and improve your future predictions.' to all 8 prompts.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" && git push origin llm

echo "[$(date)] === ALL DONE ==="
