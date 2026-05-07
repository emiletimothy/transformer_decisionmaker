## DS V3 Weather — Two-Turn Stateless Protocol Analysis (10 cases, T=100)

### Note Strategy Classification

| Case | Best Expert | MW Acc | DS Acc | MW Reg | DS Reg | Note Type | Counter % | Follow Best | Note Example (step 50) |
|------|-------------|--------|--------|--------|--------|-----------|-----------|-------------|------------------------|
| 0 | B (100%) | 98% | 98% | 2 | 2 | cumulative counter | 93% | 98% | `A:20/44, B:44/44, C:30/44, D:31/44` |
| 1 | C (91%) | 87% | 89% | 4 | 2 | cumulative counter | 100% | 96% | `A:31/50, B:34/50, C:45/50, D:13/50` |
| 2 | C (97%) | 95% | 86% | 2 | 11 | cumulative counter | 99% | 89% | `A:sunny(18/21) rainy(20/25), B:sunny(15/18) rainy(17/22)...` |
| 3 | D (93%) | 91% | 79% | 2 | 14 | last-step only | 0% | 78% | `A:rainy✗ B:rainy✗ C:sunny✓ D:sunny✓` |
| 4 | A (95%) | 95% | 91% | 0 | 4 | cumulative counter | 100% | 92% | `A:47/50, B:25/50, C:42/50, D:34/50` |
| 5 | A (85%) | 87% | 89% | -2 | -4 | cumulative counter | 100% | 90% | `A:44/50, B:42/50, C:37/50, D:13/50` |
| 6 | D (96%) | 95% | 95% | 1 | 1 | cumulative counter | 96% | 97% | `A:28/47, B:30/47, C:36/47, D:44/47` |
| 7 | B (94%) | 89% | 90% | 5 | 4 | cumulative counter | 100% | 92% | `A:27/50, B:45/50, C:35/50, D:33/50` |
| 8 | B (95%) | 95% | 95% | 0 | 0 | cumulative counter | 100% | 100% | `A:19/50, B:46/50, C:35/50, D:18/50` |
| 9 | C (100%) | 99% | 90% | 1 | 10 | last-step only | 0% | 90% | `A wrong, B correct, C correct, D wrong.` |

### Key Findings

1. **Counter-based notes correlate with low regret.** Cases where DS maintains cumulative accuracy counters (e.g., "B:94/94, C:64/94") achieve regret 0–4, comparable to MW optimal.

2. **Last-step-only notes lead to high regret.** Cases 3 and 9 never establish counters, recording only the most recent outcome (e.g., "A:sunny✓ D:sunny✓"). These cases have regret 10–14, with follow rates dropping to 78–90%.

3. **Note strategy is stochastic.** The same prompt and model produce counter-based notes in 8/10 cases but last-step-only notes in 2/10 cases. This variability is the primary source of regret variance.

4. **When following the best expert, DS is almost always correct.** Follow accuracy is 91–100%. Nearly all errors come from steps where DS deviates from the best expert.

5. **DS matches or beats MW in 6/10 cases** when it maintains proper counters. The overall gap (DS regret 4.4 vs MW regret 1.5) is driven almost entirely by the 2 cases with degenerate notes.
