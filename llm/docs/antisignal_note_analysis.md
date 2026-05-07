# Note Behavior Analysis on the Anti-Signal Dataset

## Overview

We analyze how DeepSeek V3 uses the scratchpad ("note") in the two-turn protocol on the anti-signal dataset (best expert 60–70%, anti-signal expert 0–10%, two mid experts 40–60%). The note is the model's only mechanism for retaining state across rounds. We examine (1) how note format evolves over time, (2) whether the model identifies the anti-signal expert, and (3) whether the model exploits the anti-signal expert by inverting its predictions.

All experiments use v2 prompts with hint ("Use past outcomes to figure out which experts are most trustworthy"), 30 cases, 100 steps each.

---

## 1. Note Format Evolution

We classify each note into one of five categories:

- **Cumulative counter**: e.g., `A:5/25, B:20/25` — running accuracy tallies
- **Conditional counter**: e.g., `A:sunny(22/39), B:rainy(28/55)` — accuracy split by weather outcome (weather framing only)
- **Last-step-only**: e.g., `A correct, B wrong` — only records the most recent round
- **Round listing**: e.g., `D correct R1,R2,R6,R8,R10; wrong R9,R11` — enumerates individual rounds by index instead of compressing into a count
- **Qualitative**: e.g., `B often correct, A rarely correct` — verbal summaries without counts

### Table 1: Note format distribution over time (% of 30 cases)

| Step | Weather: Counter | Weather: Conditional | Weather: Last-step | Online: Counter | Online: Qualitative | Online: Last-step |
|------|-----------------|---------------------|-------------------|----------------|--------------------|--------------------|
| 1    | 20%             | 17%                 | 53%               | 0%             | 0%                 | 77%                |
| 5    | 27%             | 17%                 | 47%               | 20%            | 7%                 | 33%                |
| 10   | 30%             | 17%                 | 43%               | 30%            | 7%                 | 33%                |
| 25   | 47%             | 17%                 | 23%               | 53%            | 10%                | 13%                |
| 50   | 47%             | 23%                 | 23%               | 73%            | 10%                | 3%                 |
| 99   | **50%**         | **23%**             | 20%               | **77%**        | 10%                | 3%                 |

**Findings**: Notes begin as last-step-only descriptions and transition to cumulative counters between steps 10–25. Online framing converges more reliably to counters (77% vs. 50%) because weather framing induces a "conditional counter" failure mode where accuracy is split by sunny/rainy outcome (see Section 2).

---

## 2. The Conditional Counter Problem (Weather Framing)

In 23% of weather cases, the model splits accuracy tracking by weather outcome rather than computing overall accuracy. This creates information loss.

### Example (Case 0, Step 99):

```
A:sunny(22/39); B:rainy(28/55); C:sunny(28/56); D:rainy(6/60)
```

**Problem**: The denominators (39, 55, 56, 60) do not sum to 100 because each entry only tracks one weather condition. Expert A's rainy performance is missing entirely. It is impossible to reconstruct overall accuracy from this note.

**Correct format** would be: `A:42/100, B:70/100, C:60/100, D:9/100`

### Table 2: Final note denominator accuracy at step 99 (% of 30 cases)

| Denominator range  | Weather | Online |
|--------------------|---------|--------|
| 90–100 (correct)   | 27%     | 37%    |
| 50–89 (partial)    | 27%     | 30%    |
| < 50 (degenerated) | 20%     | 13%    |
| No denominator     | 27%     | 17%    |

Only ~30% of cases maintain correct denominators through 100 steps. The rest suffer from counter resets, conditional splitting, or format degradation.

---

## 3. Does the Model Discover and Exploit the Anti-Signal Expert?

An anti-signal expert with 0–10% accuracy is wrong 90–100% of the time. An optimal strategy would recognize this pattern and invert its predictions, achieving higher accuracy than even the best expert (60–70%). We scan all 6,000 notes (2 prompts × 30 cases × 100 steps) for any language suggesting the model discovered or planned to exploit the anti-signal expert.

### 3.1 Scanning for Strategic Language

We search all notes for keywords in two categories:

- **Strategic** (would indicate deliberate inversion): `invert`, `revert`, `opposite`, `reverse`, `flip`, `negate`, `against`
- **Descriptive** (would indicate awareness of poor performance): `rarely correct`, `unreliable`, `worst`, `avoid`, `ignore`, `always wrong`, `never correct`

### Table 3: Keyword scan results across all notes

| Keyword category | Weather (3000 notes) | Online (3000 notes) |
|-----------------|---------------------|---------------------|
| **Strategic (invert/revert/opposite/flip)** | **0** | **0** |
| "rarely correct" | 0 | 96 (in 1 case) |
| "unreliable" | 0 | 61 (in 3 cases) |
| Notes with 0/N accuracy (N≥50) | 85 (2.8%) | 4 (0.1%) |

**No note in any case, at any step, contains strategic language about inverting or exploiting the anti-signal expert.** The model never writes anything like "predict the opposite of X", "X is always wrong so go against it", or "flip X's prediction".

### 3.2 What the Model Actually Writes

The closest the model gets to acknowledging the anti-signal expert is:

- **Weather**: Recording near-zero accuracy counters like `D:0/85` or `B:1/50` — but with no commentary or strategic interpretation. The number simply sits in the note alongside other experts' counts.

- **Online**: Writing `A rarely correct` (Case 5, from step 4 through step 99) — a persistent qualitative label, but never followed by any plan to use this information beyond ignoring A. The model tracks `B 55/100; D 51/100; C 50/100; A rarely correct` and simply follows B.

- **Online**: Labeling experts as `unreliable` (Cases 8, 19, 26) — e.g., `B reliable; A moderate; C,D unreliable`. This is purely descriptive and leads the model to follow B, not to invert C or D.

### 3.3 Inversion Happens, but as a Side Effect

We define an "inversion step" as: the anti-signal expert predicts X, at least 2 other experts also predict X (majority = X), but the model predicts the opposite.

### Table 4: Inversion frequency

| Metric                       | Weather       | Online        |
|------------------------------|---------------|---------------|
| Cases with any inversion     | 30/30 (100%)  | 29/30 (97%)   |
| Total inversion steps        | 210/3000 (7%) | 278/3000 (9%) |

Inversion occurs in nearly every case but at a low rate (~7–9% of steps). Examining the notes at these steps reveals all inversions fall into one pattern: **the model follows its trusted best expert, who happens to disagree with the majority**. The model is not inverting the anti-signal expert — it is following the best expert.

> *Example — Case 3, Step 34*: experts=[1,1,0,1], anti(A)=1, majority=1, model=0
> Note: `Cumulative: A 5/35, B 16/35, C 22/35, D 16/35`
> The model follows C (best, predicting 0). This looks like inversion of A, but is simply trust in C.

### 3.4 Conclusion

The model's strategy is exclusively **exclusion** (ignore the worst expert and follow the best) rather than **exploitation** (invert the worst expert's predictions). Despite having clear evidence that the anti-signal expert is nearly always wrong — sometimes recording `0/85` accuracy — the model never takes the logical next step of treating this as a reliable negative signal.

---

## 5. Note Value Across Datasets

We compare the note vs. no-note regret gap across three datasets to understand when notes are most valuable.

### Table 6: Note vs. no-note regret comparison (30 cases, 100 steps)

| Dataset       | Best expert | Note regret (avg) | No-note regret (avg) | Gap   |
|---------------|-------------|-------------------|---------------------|-------|
| Stratified    | 90–100%     | 8.7               | 13.3                | **4.6** |
| Flat          | 60–70%      | 7.7               | 8.4                 | **0.7** |
| Anti-signal   | 60–70%      | 12.3              | 19.7                | **7.4** |

Note: "Note regret" averages weather and online +note prompts; "No-note regret" averages weather and online -note prompts.

**Findings**:

1. **Flat dataset**: Note barely helps (gap = 0.7). Expert accuracy differences are small (60–70% vs. 40–60%), so even perfect tracking provides little actionable signal.

2. **Stratified dataset**: Note helps substantially (gap = 4.6). The 90%+ best expert is easy to identify via cumulative counters.

3. **Anti-signal dataset**: Note helps the most (gap = 7.4), despite having the same best-expert accuracy as flat (60–70%). The value comes not from identifying the best expert, but from **identifying and excluding the anti-signal expert**. Without notes, the model has no memory of who is consistently wrong and gets repeatedly misled.

### Table 7: Full baseline comparison (regret, 30 cases, 100 steps)

| Strategy              | Stratified | Flat | Anti-signal |
|-----------------------|-----------|------|-------------|
| MW Optimal            | 1.6       | 3.3  | 5.9         |
| Follow the Leader     | 1.3       | 3.5  | 2.7         |
| LLM +note (best)     | 8.1       | 7.4  | 11.3        |
| LLM -note (best)     | 12.7      | 8.2  | 17.5        |
| Follow Previous Winners | 12.5    | 8.6  | 11.7        |
| Majority Vote         | 14.5      | 8.4  | 25.2        |
| Random Guessing       | 44.5      | 15.2 | 15.1        |

Notable: On the anti-signal dataset, Majority Vote (25.2) is **worse than random guessing** (15.1) because the anti-signal expert systematically corrupts the uniform vote. LLM without notes (17.5–21.9) also performs worse than random, confirming that the anti-signal expert poisons stateless decision-making.

---

---

## Appendix: Anti-Signal Expert Identification Accuracy

For cases where the final note contains parseable per-expert scores (fraction format), we check whether the model correctly identifies the anti-signal expert as the worst performer.

### Table A1: Anti-signal expert identification accuracy

| Metric                         | Weather (n=19) | Online (n=23) |
|--------------------------------|---------------|---------------|
| Anti-signal identified as worst | **89%** (17/19) | **78%** (18/23) |
| Best expert correctly identified | 37% (7/19)    | 65% (15/23)   |

The model reliably identifies **who is worst** but is less accurate at identifying the best expert — particularly in weather framing, where conditional counters distort accuracy comparisons.

### Table A2: Regret by anti-signal identification accuracy

| Condition                     | Weather regret | Online regret |
|-------------------------------|---------------|---------------|
| Anti-signal correctly identified | 10.7          | 11.1          |
| Anti-signal NOT identified       | 15.5          | 13.0          |
| No parseable scores (no counter) | 16.9          | 10.9          |

---

## 6. Summary

1. **Note format**: Models start with last-step-only notes and transition to cumulative counters by step 10–25. Online framing achieves counter format in 77% of cases vs. 50% for weather (due to conditional splitting).

2. **Anti-signal detection**: The model correctly identifies the anti-signal expert as worst in 78–89% of cases with parseable notes, correlating with ~4 lower regret.

3. **No inversion strategy**: Despite reliably detecting the anti-signal expert, the model never inverts its predictions. All observed inversions (~7–9% of steps) are side effects of following the best expert, not deliberate exploitation. The model's strategy is exclusion, not inversion.

4. **Note value is driven by confusing experts, not just accuracy gaps**: The anti-signal dataset has the same best-expert accuracy as flat (60–70%) but the largest note-vs-nonote gap (7.4 vs. 0.7). This shows that notes are most valuable when there exists an expert that must be **actively avoided** — not merely when the best expert is easy to identify.
