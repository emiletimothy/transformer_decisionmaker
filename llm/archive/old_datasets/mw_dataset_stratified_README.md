# Dataset: mw_dataset_stratified.json

## Overview
- **30 cases**, 200 steps each, 4 experts
- **η** = 0.117741 (standard MW rate: sqrt(2·ln(4)/200))
- **Seed**: 42

## Expert Accuracy Tiers (stratified sampling)
Each case has one expert per tier, randomly shuffled across expert indices.

| Tier | Target Accuracy | Role |
|------|----------------|------|
| Best | 90% - 100% | Clear leader, should be identified |
| Second | 65% - 80% | Decent, distracting |
| Third | 55% - 70% | Slightly above random |
| Worst | 45% - 60% | Near random |

## Generation
- True labels: uniform random binary
- Expert predictions: Bernoulli(acc) per step — correct with probability `acc`, wrong otherwise
- Expert-to-tier assignment: randomly shuffled per case (best expert is not always expert 0)

## Baseline Performance
- **Best expert**: mean 95%, range 88-100%
- **MW algorithm**: mean 94%, range 88.5-99.5%

## Per-case Summary
| Case | Expert 0 | Expert 1 | Expert 2 | Expert 3 | Best | MW |
|------|----------|----------|----------|----------|------|-----|
| 0 | 44% | **100%** | 66% | 78% | 100% | 98% |
| 1 | 60% | 76% | **90%** | 53% | 90% | 90% |
| 2 | 74% | 63% | **97%** | 54% | 97% | 98% |
| 3 | 76% | 50% | 74% | **94%** | 94% | 94% |
| 4 | **96%** | 56% | 77% | 62% | 96% | 96% |
| 5 | 80% | **88%** | 68% | 55% | 88% | 88% |
| 6 | 55% | 55% | 81% | **95%** | 95% | 95% |
| 7 | 56% | **96%** | 74% | 75% | 96% | 96% |
| 8 | 44% | **94%** | 78% | 60% | 94% | 94% |
| 9 | 69% | 72% | **100%** | 52% | 100% | 99% |
| 10 | 52% | 58% | 73% | **98%** | 98% | 98% |
| 11 | 74% | 71% | **95%** | 49% | 95% | 95% |
| 12 | 62% | 68% | **88%** | 57% | 88% | 88% |
| 13 | 70% | **93%** | 46% | 74% | 93% | 93% |
| 14 | 69% | **97%** | 74% | 50% | 97% | 97% |
| 15 | 56% | 79% | 64% | **98%** | 98% | 98% |
| 16 | **90%** | 66% | 60% | 76% | 90% | 90% |
| 17 | 80% | 56% | **91%** | 58% | 91% | 91% |
| 18 | **100%** | 73% | 60% | 63% | 100% | 100% |
| 19 | 80% | **94%** | 61% | 62% | 94% | 94% |
| 20 | 62% | **98%** | 72% | 50% | 98% | 98% |
| 21 | 38% | 73% | 68% | **97%** | 97% | 97% |
| 22 | 73% | 62% | 46% | **98%** | 98% | 98% |
| 23 | 48% | **93%** | 73% | 66% | 93% | 93% |
| 24 | 82% | **98%** | 61% | 53% | 98% | 98% |
| 25 | **92%** | 58% | 72% | 45% | 92% | 92% |
| 26 | 59% | **93%** | 54% | 76% | 93% | 93% |
| 27 | 60% | 48% | 64% | **92%** | 92% | 92% |
| 28 | **94%** | 54% | 72% | 51% | 94% | 94% |
| 29 | **94%** | 70% | 70% | 42% | 94% | 94% |
