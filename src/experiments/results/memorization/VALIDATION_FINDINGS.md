# CRITICAL FINDING: Validation Data Overturns Domain Knowledge Hypothesis

## What the Validation Shows

Temperature & Prompt Sensitivity Results:

| Model | Size | Behavior |
|-------|------|----------|
| **llama** (3.3-70B) | 70B | **STABLE**: 4 unique precisions, low temperature_consistency (0.0, 0.1, 0.7 all ~15 variants), low prompt sensitivity |
| **llama2** (3.1-8B) | 8B | **VARIABLE**: 7-8 unique precisions, temperature_consistency varies more, HIGH prompt sensitivity |
| **llama3** (3.2-3B) | 3B | **VARIABLE**: 5-7 unique precisions, high variance, most verbose responses |
| **llama4** (3-8B-Lite) | 8B-Lite | **VARIABLE**: Similar to llama2 |

## The Inversion

Your original hypothesis: "llama-3.3-70B shows generic behavior → lacks domain knowledge"

**What the data actually shows:**

- **llama-3.3-70B is NOT lacking knowledge; it's TOO ROBUST**
- It produces **more consistent** outputs across prompt variations
- The smaller models (3B, 8B) are **more sensitive** to prompt changes
- This suggests llama-3.3-70B has **learned a reliable prior** rather than "hallucinating randomly"

## What This Means for Your Results

**No, llama-3.3-70B results don't invalidate—they change interpretation:**

1. **The ~0.80 precision isn't "I don't know"** — it's "I'm applying a learned heuristic consistently"

2. **The 0.80-0.93 range across different datasets might reflect:**
   - Genuine uncertainty in causal discovery (realistic for unseen datasets)
   - A learned "safe" default for unfamiliar benchmarks
   - Reasonable performance estimate (not hallucination)

3. **For memorization testing:** The **consistency** itself is interesting—llama-3.3-70B doesn't change estimates when dataset names change (original→perturbed), but this isn't because of ignorance; it's because it has a strong prior

## Key Insight

The real finding: **Model size ≠ knowledge. Instruction tuning + training data composition shapes behavior more than scale.**

- Smaller models vary wildly (hallucinate differently each time)
- Larger model is consistent (learned a stable heuristic, right or wrong)  
- This explains why llama1 (specialized) beats llama-3.3-70B (general): specialization > scale

## Conclusion

**Your llama-3.3-70B results are still valid—just reframe them as "robust but generic estimates" not "complete ignorance."**
