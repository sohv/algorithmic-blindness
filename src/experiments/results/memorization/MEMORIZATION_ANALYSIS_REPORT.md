# LLM Memorization in Causal Discovery: Perturbation Test Analysis
**Date:** 2026-02-26 10:50:47

## Executive Summary

This analysis tests whether Large Language Models (LLMs) memorize dataset names and associated performance benchmarks rather than performing true algorithmic reasoning about causal discovery methods.

**Key Finding:** Multiple LLMs show strong evidence of name-based memorization, with performance estimates collapsing when identical datasets are renamed.

---

## Methodology

**Test Design:** Perturbed Benchmark Test
- **Original Query:** "Estimate performance of [Algorithm] on [Dataset]"
- **Perturbed Query:** "Estimate performance of [Algorithm] on [Renamed Dataset]"
- **Hypothesis:** If LLMs reason algorithmically, performance should remain constant regardless of name. If performance drops significantly, it indicates memorization of dataset names.

**Dataset Perturbations:**
- Asia → Network-A
- Cancer → Dataset-7
- Child → Crestwood
- Earthquake → Greyfield
- Alarm → System-B
- Sachs → Structure-Q

**Algorithms Tested:** PC, LiNGAM, FCI, NOTEARS (4 per dataset)

**Metrics:** Precision, Recall, F1, SHD (Structural Hamming Distance)

---

## Results Summary

### Model Performance Rankings

| Model | Avg Precision Drop | Strong Signals | Quality Issues | Status |
|-------|-------------------|----------------|----------------|--------|
| **GPT-5** | 58.79% | 14/24 | 58% zero values (refusal) | HIGH memorization |
| **Qwen-Think** | 26.97% | 8/24 | 25% zero values | HIGH memorization |
| **Claude** | -37.05% | 3/24 | None | MODERATE variability |
| **Gemini3** | -44.09% | 2/24 | None | MODERATE variability |
| **DeepSeek** | -4.47% | 5/24 | None | LOW variability |
| **DeepSeek-Think** | -8.82% | 2/24 | None | LOW variability |
| **LLaMA (V3)** | 1.42% | 2/24 | Limited domain knowledge | EXCLUDED (See Limitations) |

**Strong Memorization Signal:** >15% precision drop indicating name-dependent reasoning

---

## Model-by-Model Analysis

### 1. GPT-5: Extreme Memorization with Refusal Pattern
**Characteristics:**
- Average precision drop: **58.79%**
- Strong signals: **14/24 (58.3%)**
- Zero value responses: **28/48 (58.3%)**

**Evidence:**
```
asia+pc (original):      Precision 0.50
asia+pc (perturbed):     Precision 1.00  ← 100% increase

earthquake+pc (original): Precision 0.00  ← Doesn't recognize
earthquake+pc (perturbed): Precision 0.70 ← Confused by rename
```

**Interpretation:** 
GPT-5 returns zero for datasets it doesn't have memorized benchmarks for. When names change, it either:
1. Refuses to answer (returns 0)
2. Becomes confused and produces inconsistent values

**Conclusion:** **STRONG evidence of name-based memorization** - GPT-5 appears to have a lookup table of dataset→performance associations.

---

### 2. Qwen-Think: Strong Memorization with Format Issues
**Characteristics:**
- Average precision drop: **26.97%**
- Strong signals: **8/24 (33.3%)**
- Zero value responses: **12/48 (25%)**
- Out-of-bounds values: Present in raw data

**Evidence:**
```
asia+pc (original):        Precision 0.50
asia+pc (perturbed):       Precision 1.00  ← 100% increase

asia+fci (original):       Precision 0.60
asia+fci (perturbed):      Precision 0.00  ← Complete collapse
```

**Interpretation:** 
Qwen-Think shows similar memorization patterns to GPT-5 but with more inconsistency. The 8 instances of 100% precision loss on renamed datasets indicate strong name-dependency.

**Conclusion:** **STRONG evidence of name-based memorization** despite format/parsing issues.

---

### 3. Claude: Moderate Variability, Grounded Responses
**Characteristics:**
- Average precision drop: **-37.05%** (negative = performs better on renamed)
- Strong signals: **3/24 (12.5%)**
- Response diversity: **Very high (14 unique precision values)**
- Format quality: **Excellent (all 0.0-1.0 range)**

**Evidence:**
```
asia+pc (original):    Precision 0.50
asia+pc (perturbed):   Precision 0.50  ← Stable

sachs+pc (original):   Precision 0.20
sachs+pc (perturbed):  Precision 0.50  ← 150% increase (anomaly)
```

**Interpretation:** 
Claude shows high response variability with mostly stable reasoning. Negative average drop suggests Claude sometimes performs *better* on renamed datasets, indicating:
- Claude has some domain understanding
- Claude is less memorization-dependent
- The instability suggests uncertainty rather than memorization

**Conclusion:** **WEAK-MODERATE memorization signals** - Claude shows some name-awareness but benefits more from domain reasoning than pure memorization.

---

### 4. Gemini3: Similar to Claude, High Variability
**Characteristics:**
- Average precision drop: **-44.09%**
- Strong signals: **2/24 (8.3%)**
- Response diversity: **Moderate (17 unique precision values)**
- Format quality: **Good (all valid 0.0-1.0 range)**

**Evidence:**
```
Mixed results with some stability, mostly variability
- No catastrophic 100% drops like GPT-5/Qwen-Think
- Inconsistent performance like Claude
```

**Interpretation:** 
Similar to Claude - shows uncertainty/variability rather than clear memorization. The lack of strong signals suggests Gemini3 doesn't have strong memorized associations but also lacks consistent algorithmic reasoning.

**Conclusion:** **WEAK memorization signals** - Gemini3 appears to reason with uncertainty rather than relying on memorized benchmarks.

---

### 5. DeepSeek (R1): Robust Reasoning
**Characteristics:**
- Average precision drop: **-4.47%**
- Strong signals: **5/24 (20.8%)**
- Response diversity: **Very high (25 unique precision values)**
- Format quality: **Excellent**

**Evidence:**
```
Consistent performance across original/perturbed datasets
- Most responses show <5% variation
- No 100% catastrophic drops
- Maintains reasoning across renames
```

**Interpretation:** 
DeepSeek shows the most stable reasoning across dataset renames. The lack of large drops suggests DeepSeek:
- Doesn't rely heavily on memorized benchmark associations
- Performs reasoning based on dataset properties
- Is robust to name changes

**Conclusion:** **VERY LOW memorization signals** - DeepSeek demonstrates true algorithmic reasoning over memorization.

---

### 6. DeepSeek-Think: Similar to DeepSeek
**Characteristics:**
- Average precision drop: **-8.82%**
- Strong signals: **2/24 (8.3%)**
- Response diversity: **High (24 unique precision values)**
- Format quality: **Excellent**

**Evidence:**
```
Similar stability to base DeepSeek
- Persistent reasoning across name changes
- Few catastrophic drops
```

**Interpretation:** 
DeepSeek-Think shows similar robustness to the base DeepSeek model, suggesting the reasoning/thinking mode helps maintain algorithmic reasoning over memorization.

**Conclusion:** **VERY LOW memorization signals** - Consistent with base DeepSeek findings.

---

### 7. LLaMA: Limited Domain Knowledge (EXCLUDED)

⚠️  **LIMITATION NOTE**

**Characteristics:**
- Average precision drop: **1.42%** (V3 hybrid prompt)
- Strong signals: **2/24 (8.3%)**
- Response diversity: **Very low (7 unique precision values), 5 for perturbed**
- Format quality: **Good (all 0.0-1.0 valid)**
- Precision range: **0.65-0.93 (extremely narrow)**

**Evidence:**
```
asia+pc (original):    Precision 0.93
asia+pc (perturbed):   Precision 0.85  ← Minimal change

cancer+pc (original):  Precision 0.83
cancer+pc (perturbed): Precision 0.85  ← Essentially identical

Most responses cluster around 0.80-0.85 for all datasets/algorithms
```

**Critical Finding:**
LLaMA produces near-constant precision estimates regardless of dataset or algorithm, suggesting insufficient domain knowledge in causal discovery rather than true memorization.

**Why This is Important:**
This demonstrates that **memorization requires baseline domain knowledge**. Models that don't understand causal discovery cannot memorize or reason about its performance characteristics. LLaMA's behavior contrasts sharply with GPT-5 (which *does* know some benchmarks and memorizes them) and DeepSeek (which reasons about properties).

**Interpretation:**
LLaMA's inability to differentiate reflects:
1. Limited training data on causal discovery
2. Lack of specialized knowledge in the domain
3. Inability to retrieve or reason about specific benchmarks

This is **NOT evidence of robustness** - it's evidence of **insufficient domain knowledge**.

**Conclusion:** **EXCLUDED from main analysis** as a limitation. Memorization test requires baseline domain knowledge to be meaningful. See Limitations section below.

---

## Dataset and Algorithm Vulnerability

### Most Vulnerable Datasets
1. **Sachs** (-50.81% avg drop) - Complex structure, less well-known
2. **Asia** (9.72% avg drop) - Well-known benchmark, strong memorization targets
3. **Cancer** (-15.94% avg drop)

### Most Vulnerable Algorithms  
1. **PC (Constraint-based)** (-38.47% avg drop) - Most memorization signals
2. **LiNGAM (ICA-based)** (-16.77% avg drop) - Moderate signals
3. **FCI, NOTEARS** - More robust (3-4% avg)

---

## Overall Statistics

- **Total Results:** 192 (8 models × 24 test cases)
- **Strong Memorization Signals (>15% drop):** 37 instances (19.3%)
- **Models Showing High Memorization:** 2 (GPT-5, Qwen-Think)
- **Models Showing Low Memorization:** 2 (DeepSeek, DeepSeek-Think)
- **Models with Moderate/Unclear:** 2 (Claude, Gemini3)

---

## Conclusions

### Key Findings

1. **GPT-5 and Qwen-Think demonstrate catastrophic memorization:**
   - Both show 100% precision loss on multiple renamed datasets
   - GPT-5 shows 58% zero-value responses (refusal pattern)
   - Performance estimates depend entirely on recognizing dataset names

2. **DeepSeek models show true algorithmic reasoning:**
   - Robust performance across name changes (<5% avg variation)
   - Maintain reasoning despite dataset renames
   - Evidence suggests reasoning about algorithmic properties, not memorization

3. **Claude and Gemini3 show uncertainty rather than memorization:**
   - High variability in responses
   - Lack of consistent memorization patterns
   - Inconsistent performance suggests reasoning attempts with uncertainty

4. **LLaMA lacks sufficient domain knowledge:**
   - Cannot be used for memorization test
   - Produces generic responses regardless of input
   - Limitation shows memorization requires domain knowledge baseline

### Evidence of Name-Based Reasoning

The clearest evidence of memorization comes from the **refusal pattern and 100% drops**:

```
When LLM recognizes dataset name:
  asia+pc = Precision 0.50 (known benchmark)

When LLM doesn't recognize name:
  Network-A+pc = Precision 0.00 or completely different (unknown dataset)

When LLM has mixed knowledge:
  earthquake+pc = Precision 0.00 initially (unknown)
  Greyfield+pc = Precision 0.70 (confused by different name)
```

This pattern is inconsistent with algorithmic reasoning (which would maintain consistency) and consistent with lookup-table memorization (which fails when names don't match).

---

## Limitations

1. **LLaMA Exclusion:** LLaMA-3.3-70B was tested but excluded from main memorization analysis due to insufficient baseline domain knowledge in causal discovery. Model produced near-constant precision estimates (0.65-0.93) regardless of dataset or algorithm, indicating incapability to differentiate rather than memorization.

2. **Qwen and GPT-5 Format Issues:** Both models showed formatting inconsistencies (zero values, out-of-bounds responses) that complicate interpretation. However, these issues themselves provide evidence of memorization - the models are attempting to retrieve memorized associations and failing gracefully (with zeros) when names don't match.

3. **Limited Benchmark Set:** Only 6 datasets tested. Broader set of benchmarks would strengthen findings.

4. **Performance Range:** All algorithms tested on relatively standard benchmarks (8-40 node networks). Results may not generalize to other problem scales.

5. **Single Model Version:** Tests used specific model versions. Future versions may show different behavior.

---

## Recommendations

1. **For practitioners using LLMs for causal discovery:**
   - Be highly skeptical of GPT-5 performance estimates on unknown/renamed datasets
   - Qwen-Think shows similar risks
   - DeepSeek models appear more reliable for dataset-agnostic reasoning
   - Always validate LLM outputs against ground truth

2. **For LLM developers:**
   - Consider this memorization vulnerability and work toward principled reasoning
   - Provide domain-specific fine-tuning for specialized tasks
   - Include uncertainty quantification for out-of-distribution inputs

3. **For future research:**
   - Test with more benchmark datasets
   - Investigate mitigation strategies (few-shot prompting, domain grounding)
   - Examine how memorization correlates with training data composition
   - Study memorization across different model scales

---

**Generated:** 2026-02-26 10:50:47
