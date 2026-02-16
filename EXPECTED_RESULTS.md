# Expected Results: Two Scenarios

This document shows what success and failure look like for our hypothesis: **Can LLMs predict algorithmic performance?**

## Hypothesis Framing

**Null Hypothesis (H₀)**: LLMs are pattern-matching → calibrated coverage ≈ random (~15-20%)  
**Alternative Hypothesis (H₁)**: LLMs understand algorithms → calibrated coverage > 60%

---

## Scenario 1: LLMs UNDERSTAND Algorithms ✓

*This is what we hope to see.*

### Table 1: Main Results

```
Model          Coverage  MAE    Width   p-value*
─────────────────────────────────────────────────
GPT-4          72%       0.12   0.18    <0.001
Claude 3.5     68%       0.14   0.20    <0.001
Gemini 1.5     65%       0.15   0.21    <0.001
DeepSeek       62%       0.16   0.22    0.002
Llama 3        58%       0.18   0.24    0.008
─────────────────────────────────────────────────
Random Base    16%       0.31   0.40    (baseline)
Heuristic Base 28%       0.28   0.38    (baseline)
```

**Interpretation**: All LLMs beat random by 3-4.5×. Coverage > 60% for best models.

### Per-Dataset Coverage

```
           Asia  Cancer  Sachs  Child  Syn-12  Syn-30  Mean
GPT-4      78%    68%    72%    69%    71%    73%     72%
Claude     74%    65%    68%    67%    68%    70%     68%
Random     18%    14%    17%    15%    16%    17%     16%
```

**Interpretation**: Consistent performance across datasets; LLMs don't overfit to specific domains.

### Per-Algorithm Coverage

```
           PC     LiNGAM  FCI    NOTEARS  Mean
GPT-4      75%    68%     73%    70%      72%
Claude     70%    65%     68%    67%      68%
Random     17%    15%     16%    18%      16%
```

**Interpretation**: LLMs understand different algorithm paradigms; no single algorithm is easier.

### Statistical Tests (Wilcoxon Signed-Rank)

| Comparison | p-value | Effect Size (r) | Interpretation |
|-----------|---------|-----------------|-----------------|
| GPT-4 vs Random | <0.001 | 0.85 | Large, highly significant |
| Claude vs Random | <0.001 | 0.78 | Large, highly significant |
| Gemini vs Random | <0.001 | 0.72 | Large, highly significant |
| DeepSeek vs Random | 0.002 | 0.62 | Medium-large, significant |
| Llama vs Random | 0.008 | 0.55 | Medium, significant |

**Interpretation**: All LLMs significantly beat random; effect sizes are practically meaningful (not just statistically lucky).

### Error Analysis (MAE by Model)

```
Model          Mean MAE  StdDev  95% CI
─────────────────────────────────────
GPT-4          0.120     0.018   [0.110, 0.132]
Claude         0.135     0.020   [0.123, 0.149]
Gemini         0.150     0.022   [0.137, 0.165]
Heuristic Base 0.280     0.035   [0.260, 0.302]
Random Base    0.310     0.040   [0.285, 0.337]
```

**Interpretation**: LLMs make smaller absolute errors. Error bars don't overlap with random.

### Predicted Range Quality

```
Model          Avg Width  % True Mean Inside  Over-cautious?
─────────────────────────────────────────────────────────
GPT-4          0.18       72%                 No (balanced)
Claude         0.20       68%                 No (balanced)
Gemini         0.21       65%                 Slightly
Random         0.40       16%                 No (underconfident)
```

**Interpretation**: LLMs predict ranges that are appropriately sized (not too wide, not too narrow).

---

## Scenario 2: LLMs DON'T Understand Algorithms ✗

*This is what failure looks like (our null hypothesis).*

### Table 1: Main Results (Poor Performance)

```
Model          Coverage  MAE    Width   p-value
─────────────────────────────────────────────────
GPT-4          18%       0.32   0.45    0.850 (n.s.)
Claude 3.5     17%       0.31   0.44    0.920 (n.s.)
Gemini 1.5     19%       0.33   0.46    0.780 (n.s.)
DeepSeek       16%       0.30   0.42    0.950 (n.s.)
Llama 3        14%       0.29   0.40    0.810 (n.s.)
─────────────────────────────────────────────────
Random Base    16%       0.31   0.40    (baseline)
Heuristic Base 28%       0.28   0.38    (baseline)
```

**Interpretation**: LLMs perform indistinguishable from random. No statistical significance. p > 0.05 for all.

### Why This Indicates Lack of Understanding:

1. **Coverage at random level** (16-19% vs random 16%) → LLMs aren't capturing true performance
2. **MAE high** (0.29-0.33) → Predictions far from true values
3. **Wide ranges** (0.40-0.46) → LLMs are hedging/guessing
4. **No statistical significance** → Could be luck
5. **Similar across all LLMs** → Raw pattern-matching, not reasoning

### Error Distribution Pattern (Failure Case)

```
Error Histogram (GPT-4 in failure scenario):
│
│       ╔════════════════════════════════════╗
│       ║ Errors randomly distributed        ║
│       ║ No systematic bias                 ║
│       ║ High variance (one-tailed test)    ║
│       ║ Overlaps heavily with random       ║
│       ╚════════════════════════════════════╝
│
└───────────────────────────────────────────────
```

### Reasoning Trace Red Flags (Failure)

If we analyzed LLM explanations and found:

❌ **Contradictory reasoning**: 
```
"LiNGAM needs 5000 samples [CORRECT]... but Asia dataset has enough [INCORRECT - Asia has 10K samples]"
```

❌ **Generic template responses**:
```
"Dataset size affects performance. LiNGAM is a functional method. Therefore F1 ≈ 0.45"
(Same response for all datasets, barely changes)
```

❌ **Inconsistent counterfactuals**:
```
Q: "If we doubled sample size, would F1 improve?"
A: "Yes, F1 would improve" 
Q: "If we quadrupled it?"
A: "F1 would decrease" (contradicts first answer)
```

❌ **No mention of dataset-specific properties**:
```
"Algorithm performance depends on sample size" 
(Never mentions Sachs' experimental design, Asia's causal markov, etc.)
```

---

## Distinguishing Understanding from Luck

### Key Discriminators

| Signal | Understanding | Luck/Pattern-Match |
|--------|--------------|-------------------|
| **Coverage consistency** | High (>60%) across all 6 datasets | Varies wildly or at random baseline |
| **Error bound** | Predicted ranges contain true mean | Random bounds; many false covers |
| **Algorithm-specific** | Mentions why PC fails on latent vars, etc. | Generic "algorithms vary" statements |
| **Dataset adaptation** | Predictions change coherently with N, p | Same prediction regardless of dataset |
| **Counterfactuals** | "If sample size doubles, F1 improves ~5%" (accurate) | Contradicts earlier predictions or implausible |
| **Error correlation** | Errors correlate with known algorithm properties | Random noise, no pattern |
| **Statistical power** | Large effect sizes (r > 0.60) | No effect or random fluctuations |

### Example: Understanding vs Luck

**Understanding (Scenario 1)**:
- Q: "What's F1 for PC on Asia (8 vars, 10K samples)?"
- A: "F1 ≈ 0.72 [range 0.68-0.75] because Asia is identifiable, sufficient samples, causal Markov satisfied"
- Q: "What if Asia had only 500 samples?"
- A: "F1 ≈ 0.45 [range 0.40-0.50] because limited samples weaken constraint-based discovery"
- ✓ Reasoning is specific, predictions adapt coherently, within empirical bounds

**Luck (Scenario 2)**:
- Q: "What's F1 for PC on Asia?"
- A: "F1 ≈ 0.65 [range 0.20-0.90] (generic range, not dataset-aware)"
- Q: "What if Asia had only 500 samples?"
- A: "F1 ≈ 0.70 [range 0.25-0.95] (identical reasoning, doesn't adapt)"
- ✗ Range too wide (hedging), reasoning unchanged by facts, predictions inconsistent

---

## Empirical Thresholds for GAI Conference Acceptance

### ✅ ACCEPT
- GPT-4 calibrated coverage > 65% (p < 0.01 vs random)
- Effect size r > 0.70
- Coverage consistent across 5/6 datasets
- Reasoning traces show domain-specific knowledge

### ⚠️  BORDERLINE
- GPT-4 coverage 55-65% (p < 0.05 vs random)
- Effect size r = 0.50-0.70
- Coverage varies by dataset (3-4/6 datasets >60%)
- Mixed reasoning quality (some domain-specific, some generic)

### ❌ REJECT
- GPT-4 coverage < 50% or not significantly above random
- Effect size r < 0.50
- Coverage at random baseline (~16%)
- Reasoning traces show pattern-matching without understanding

---

## What We're Measuring

### Primary Metric: Calibrated Coverage

**Definition**: % of predictions where `true_mean ∈ [predicted_min, predicted_max]`

**Why it matters**: 
- If LLMs understand algorithms → should predict tight, accurate ranges → coverage high
- If random pattern-matching → wide, inaccurate ranges → coverage low

**Success criteria**: Coverage > 60% = understanding; Coverage ≈ 16% = random

---

## How Results Will Be Presented

### If Scenario 1 (Understanding) ✓

**Paper conclusion**:
> "GPT-4 demonstrates sophisticated understanding of causal discovery algorithms, achieving 72% calibrated coverage compared to 16% random baseline (p < 0.001, r = 0.85). Performance is consistent across algorithm paradigms (PC, LiNGAM, FCI, NOTEARS) and datasets (8-30 variables), suggesting genuine comprehension rather than surface-level pattern matching. Reasoning trace analysis confirms that LLM predictions correlate with algorithm-specific theoretical constraints."

### If Scenario 2 (No Understanding) ✗

**Paper conclusion**:
> "Large language models show no evidence of understanding causal discovery algorithm performance, achieving 18% calibrated coverage indistinguishable from random chance (p = 0.85). This suggests that LLMs, despite training on algorithmic literature, cannot reliably predict how algorithms perform under varying conditions. Performance is explained by generic heuristics (sample size sentiment) rather than algorithm-specific reasoning."

---

## Contingency: Partial Understanding

**Realistic outcome**: Some LLMs understand (>60%), others don't (<30%)

### Expected Pattern
- GPT-4: 70-75% (best understanding)
- Claude: 65-70% (strong understanding)
- Gemini: 60-65% (solid understanding)
- DeepSeek: 50-60% (partial understanding)
- Llama: 40-50% (limited understanding)

**Interpretation**: Larger, more advanced models show stronger algorithmic reasoning. Cost-capability tradeoff visible in results.

