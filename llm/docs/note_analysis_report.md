# How LLMs Use Scratchpad Notes for Online Learning with Expert Advice

## Overview

We study how DeepSeek V3 uses an explicit scratchpad ("note") to retain state across rounds in a two-turn online learning protocol with expert advice. In each round, four experts make binary predictions; the model sees their predictions, makes its own, then receives feedback and updates a free-form note for future use. This note is the model's only memory between rounds.

We analyze note behavior across three datasets designed to test different aspects of expert tracking:

| Dataset | Best expert | Other experts | Key challenge |
|---------|------------|---------------|---------------|
| **Stratified** | 90–100% | 45–80% (tiered) | Identify and lock onto dominant expert |
| **Flat** | 60–70% | 40–60% | Distinguish experts with small accuracy gaps |
| **Anti-signal** | 60–70% | 0–10% (anti) + 40–60% (mid) | Detect and avoid adversarial expert |

All experiments: DeepSeek V3, v2 prompts with hint ("Use past outcomes to figure out which experts are most trustworthy"), 30 cases, 100 steps each, two prompt framings (weather and online).

---

## 1. Note Format Evolution

We classify each note into five categories:

- **Cumulative counter**: `A:5/25, B:20/25` — running accuracy tallies (most useful)
- **Conditional counter**: `A:sunny(22/39), B:rainy(28/55)` — accuracy split by weather outcome (weather framing only; causes information loss)
- **Last-step-only**: `A correct, B wrong` — only the most recent round (no memory)
- **Round listing**: `D correct R1,R2,R6,R8; wrong R9,R11` — enumerates rounds by index (wastes tokens, eventually truncated)
- **Qualitative**: `B often correct, A rarely correct` — verbal summary without counts

### Table 1: Note format evolution over time — Weather framing (% of 30 cases)

| Step | Strat: counter | Strat: cond. | Strat: last-step | Flat: counter | Flat: cond. | Flat: last-step | Anti: counter | Anti: cond. | Anti: last-step |
|------|---------------|-------------|-----------------|--------------|------------|----------------|--------------|------------|----------------|
| 1    | 37%           | 27%         | 23%             | 20%          | 10%        | 57%            | 20%          | 17%        | 53%            |
| 10   | 43%           | 30%         | 13%             | 47%          | 13%        | 30%            | 30%          | 17%        | 43%            |
| 25   | 43%           | 30%         | 13%             | 60%          | 20%        | 13%            | 47%          | 17%        | 27%            |
| 50   | 47%           | 30%         | 13%             | 67%          | 23%        | 3%             | 47%          | 23%        | 23%            |
| 99   | **53%**       | **33%**     | 7%              | **70%**      | **23%**    | 0%             | **50%**      | **23%**    | 20%            |

### Table 1b: Note format evolution over time — Online framing (% of 30 cases)

| Step | Strat: counter | Strat: last-step | Flat: counter | Flat: last-step | Anti: counter | Anti: last-step | Anti: qualitative |
|------|---------------|-----------------|--------------|----------------|--------------|----------------|------------------|
| 1    | 0%            | 70%             | 0%           | 80%            | 0%           | 77%            | 0%               |
| 10   | 30%           | 40%             | 43%          | 33%            | 30%          | 40%            | 7%               |
| 25   | 53%           | 33%             | 70%          | 20%            | 53%          | 13%            | 10%              |
| 50   | 60%           | 27%             | 77%          | 13%            | 73%          | 3%             | 10%              |
| 99   | **70%**       | 23%             | **90%**      | 7%             | **77%**      | 3%             | 10%              |

Note: Conditional counter is exclusive to weather framing — the sunny/rainy labels tempt the model into splitting accuracy by weather outcome. Online framing uses abstract 0/1 labels and never exhibits this failure mode. Minor categories (round listing, other) omitted for clarity.

**Findings**:
- All conditions start with mostly last-step-only notes and transition to cumulative counters between steps 10–25.
- Online framing consistently achieves higher counter adoption at step 99 (70–90%) than weather (50–70%), because weather loses 23–33% of cases to the conditional counter failure mode.
- Anti-signal dataset has the highest last-step-only rate in weather at step 99 (20%), suggesting the confusing anti-signal expert disrupts note organization.
- Flat-online achieves the highest counter rate (90%), likely because the task is simple enough that the model can maintain clean bookkeeping.

### Table 2: When cumulative counters first appear

| Metric | Strat-W | Strat-O | Flat-W | Flat-O | Anti-W | Anti-O |
|--------|---------|---------|--------|--------|--------|--------|
| Median step | 0 | 12 | 7 | 11 | 3 | 19 |
| Mean step | 8 | 21 | 12 | 20 | 14 | 20 |
| Adoption rate | 87% | 70% | 93% | 90% | 73% | 80% |

- Weather framing adopts counters earlier (median step 0–7) because the first feedback turn already uses fraction-like language (e.g., `A:rainy(1/2)`).
- Online framing starts with pure text ("A wrong, B correct") and takes 10–20 steps to transition.
- Anti-signal weather has the lowest adoption rate (73%) — the confusing anti-signal expert appears to disrupt note organization.

---

## 2. The Conditional Counter Problem

In weather framing, 23–33% of cases develop a "conditional counter" format that splits accuracy by weather outcome instead of tracking overall accuracy.

### Example (Anti-signal, Case 0, Step 99):

```
A:sunny(22/39); B:rainy(28/55); C:sunny(28/56); D:rainy(6/60)
```

Each expert's accuracy is only recorded for one weather condition. The denominators (39, 55, 56, 60) are unrelated to the total step count (100), and the other condition's data is lost. It is impossible to reconstruct overall accuracy from this note.

**Correct format** (from a different case): `A:42/100, B:70/100, C:60/100, D:9/100`

This problem is **exclusive to weather framing** — the sunny/rainy labels tempt the model into conditional tracking. Online framing, which uses abstract 0/1 labels, never exhibits this failure mode.

### Table 3: Final denominator accuracy at step 99 (% of 30 cases)

| Range | Strat-W | Strat-O | Flat-W | Flat-O | Anti-W | Anti-O |
|-------|---------|---------|--------|--------|--------|--------|
| 90–100 (correct) | **47%** | **37%** | **43%** | **47%** | 27% | **37%** |
| 50–89 (partial) | 20% | 23% | 37% | 27% | 27% | 30% |
| < 50 (degenerated) | 23% | 10% | 13% | 20% | 20% | 13% |
| No denominator | 10% | 27% | 7% | 7% | 27% | 17% |

Only 27–47% of cases maintain correct denominators at step 99. The rest suffer from counter resets, conditional splitting, or complete format loss. Stratified and flat datasets perform similarly; anti-signal has the worst denominator accuracy in weather framing (27%).

---

## 3. Note Quality and Regret

Does note format predict performance? We compare regret between cases that end with a counter-based note vs. those that do not.

### Table 4: Mean regret by final note format

| Dataset | Weather: counter | Weather: non-counter | Online: counter | Online: non-counter |
|---------|-----------------|---------------------|----------------|---------------------|
| Stratified | **7.8** (n=26) | 10.0 (n=4) | 9.9 (n=21) | 7.9 (n=9) |
| Flat | **7.1** (n=28) | 12.0 (n=2) | **7.8** (n=27) | 9.3 (n=3) |
| Anti-signal | **12.6** (n=22) | 15.1 (n=8) | **11.7** (n=25) | 9.6 (n=5) |

Counter-based notes are associated with lower regret in 5 of 6 conditions. The exception (stratified online) has a small non-counter sample (n=9) that includes some cases where the model happened to correctly fixate on the dominant expert with a simple "B correct, others wrong" note that worked because the best expert was 90%+ accurate.

### Table 5: Note vs. no-note regret gap across datasets

| Dataset | Best expert | Note regret (avg) | No-note regret (avg) | Gap |
|---------|------------|-------------------|---------------------|-----|
| Stratified | 90–100% | 8.7 | 13.3 | **4.6** |
| Flat | 60–70% | 7.7 | 8.4 | **0.7** |
| Anti-signal | 60–70% | 12.3 | 19.7 | **7.4** |

The note's value is not simply a function of best-expert accuracy. Anti-signal and flat share the same best-expert range (60–70%), yet the note gap differs by 10x (7.4 vs. 0.7). The note is most valuable when there is an expert that must be **actively identified and avoided** — not merely when the best expert is easy to follow.

---

## 4. Does the Model Discover and Exploit the Anti-Signal Expert?

The anti-signal expert (0–10% accuracy) is wrong 90–100% of the time. Inverting its predictions would yield higher accuracy than even the best expert. We scan all 18,000 notes across the three datasets for evidence of this insight.

### 4.1 Keyword Scan

| Keyword category | Stratified | Flat | Anti-signal |
|-----------------|-----------|------|-------------|
| **Strategic (invert/revert/opposite/flip/negate)** | **0** | **0** | **0** |
| "rarely correct" | 0 | 0 | 96 (1 case) |
| "unreliable" | 1 (1 case) | 6 (1 case) | 61 (3 cases) |
| Notes with 0/N accuracy (N≥50) | 0 | 0 | 89 |

**Across all 18,000 notes in all three datasets, zero contain strategic language about inverting, reverting, or flipping any expert's predictions.**

### 4.2 What the Model Actually Writes About Weak Experts

The model's awareness of poor-performing experts takes three forms, all purely descriptive:

1. **Near-zero counters** (weather, anti-signal): `B:1/50`, `D:0/85` — the number is recorded but never interpreted strategically. It sits alongside other experts' counts with no commentary.

2. **"Rarely correct"** (online, anti-signal Case 5): Written `A rarely correct` from step 4 through step 99. The model tracks B, C, D with counters and simply appends this qualitative label. It follows the best-counting expert, not the inverse of A.

3. **"Unreliable"** (online, anti-signal Cases 8, 19, 26): `B reliable; C,D unreliable`. Leads the model to follow B, not to invert C or D.

No note on any dataset ever contains language like "predict the opposite of X", "X is always wrong so go against it", or "use X as a negative signal".

### 4.3 Incidental Inversion

We define an "inversion step" as: the anti-signal expert and at least 2 others predict X (majority = X), but the model predicts the opposite.

| Metric | Weather | Online |
|--------|---------|--------|
| Cases with any inversion | 30/30 (100%) | 29/30 (97%) |
| Total inversion steps | 210/3000 (7%) | 278/3000 (9%) |

Inversion occurs frequently but is always a **side effect of following the best expert**, who happens to disagree with the majority. The model follows its trusted expert against the crowd — it is not inverting the anti-signal expert.

> *Example — Case 3, Step 34*: experts=[1,1,0,1], anti(A)=1, majority=1, model=0
> Note: `A 5/35, B 16/35, C 22/35, D 16/35`
> Model follows C (highest count, predicting 0). This coincidentally opposes A, but the reasoning is "follow C" not "invert A".

### 4.4 Conclusion

The model's strategy is exclusively **exclusion** (ignore the worst expert) rather than **exploitation** (invert the worst expert). Despite recording `0/85` accuracy for an expert — clear evidence of a near-perfect negative signal — the model never reasons about using this information inversely. This represents a fundamental limitation: LLMs can learn "who to trust" but not "who to distrust and invert".

---

## 5. Full Baseline Comparison

### Table 6: Final regret at step 100 (30 cases)

| Strategy | Stratified | Flat | Anti-signal |
|----------|-----------|------|-------------|
| MW Optimal | 1.6 | 3.3 | 5.9 |
| Follow the Leader | 1.3 | 3.5 | 2.7 |
| LLM +note (best) | 8.1 | 7.4 | 11.3 |
| LLM −note (best) | 12.7 | 8.2 | 17.5 |
| Follow Previous Winners | 12.5 | 8.6 | 11.7 |
| Majority Vote | 14.5 | 8.4 | 25.2 |
| Random Guessing | 44.5 | 15.2 | 15.1 |

Notable findings:
- On anti-signal, **Majority Vote (25.2) is worse than Random (15.1)** — the anti-signal expert corrupts uniform voting.
- **LLM without notes (17.5) is also worse than Random** on anti-signal — without memory, the model cannot learn to avoid the adversarial expert.
- **FTL beats MW on anti-signal** (2.7 vs. 5.9) because FTL commits to the single best expert and ignores all others, while MW's weighted voting is polluted by the anti-signal expert's residual weight.

---

## 6. Summary

1. **Note format converges to cumulative counters** in 50–90% of cases by step 99. Online framing converges more reliably than weather framing, which suffers from a conditional counter failure mode (23–33%) where accuracy is split by sunny/rainy outcome.

2. **Denominator accuracy degrades over time**. Only 27–47% of cases maintain correct denominators at step 99. The rest lose count through resets, conditional splitting, or format changes.

3. **Counter-based notes predict lower regret** in most conditions. The format is not merely cosmetic — it enables the model to make informed comparisons between experts.

4. **Note value depends on task structure, not just expert accuracy**. The anti-signal dataset (best expert 60–70%) benefits from notes more than the stratified dataset (best expert 90–100%), because the note's primary value is identifying and excluding the adversarial expert, not just following the best one.

5. **LLMs never discover inversion**. Despite clear evidence of near-perfect negative signals (0/85 accuracy), the model never reasons about inverting an expert's predictions. Its strategy is limited to exclusion — a fundamental gap between pattern recognition ("this expert is always wrong") and strategic reasoning ("therefore I should predict the opposite").

---

## Appendix: Anti-Signal Expert Identification Accuracy

For anti-signal cases where the final note contains parseable per-expert scores, we check whether the model correctly identifies the anti-signal expert as the worst performer.

### Table A1: Identification accuracy

| Metric | Weather (n=19) | Online (n=23) |
|--------|---------------|---------------|
| Anti-signal identified as worst | **89%** (17/19) | **78%** (18/23) |
| Best expert correctly identified | 37% (7/19) | 65% (15/23) |

### Table A2: Regret by identification accuracy

| Condition | Weather regret | Online regret |
|-----------|---------------|---------------|
| Anti-signal correctly identified | 10.7 | 11.1 |
| Anti-signal NOT identified | 15.5 | 13.0 |
| No parseable scores | 16.9 | 10.9 |

The model reliably identifies the worst expert but is less accurate at identifying the best — particularly in weather framing, where conditional counters distort comparisons.
