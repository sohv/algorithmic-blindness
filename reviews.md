Large Language Models are Algorithmically Blind
Download PDF
Sohan Venkatesh, Ashish Mahendran Kurapath, Tejas Melkote 
25 Feb 2026 (modified: 31 May 2026)
Submitted to UAI 2026
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
BibTeX
CC BY 4.0
Keywords: Large Language Models, Causal Discovery, Uncertainty Estimation, Algorithmic Reasoning
TL;DR: Frontier large language models fail to produce calibrated performance predictions for causal discovery algorithms, exhibiting systematic coverage errors even when provided structured prompts.
Abstract:
Large language models (LLMs) demonstrate remarkable breadth of knowledge, yet their ability to reason about computational processes remains poorly understood. Closing this gap matters for practitioners who rely on LLMs to guide algorithm selection and deployment. We address this limitation using causal discovery as a testbed and evaluate eight frontier LLMs against ground truth derived from large-scale algorithm executions and find systematic, near-total failure. Models produce ranges far wider than true confidence intervals yet still fail to contain the true algorithmic mean in the majority of instances; most perform worse than random guessing and the marginal above-random performance of the best model is most consistent with benchmark memorization rather than principled reasoning. We term this failure algorithmic blindness and argue it reflects a fundamental gap between declarative knowledge about algorithms and calibrated procedural prediction.

Submission Number: 448
Discussion
Filter by reply type...

Filter by author...




13 / 13 replies shown
Add:
Paper Decision
Decisionby Program Chairs31 May 2026 at 02:33 (modified: 01 Jun 2026 at 17:49)Program Chairs, Senior Area Chairs, Area Chairs, AuthorsRevisions
Decision: Reject
Meta Review of Submission448 by Area Chair 4ksg
Meta Reviewby Area Chair 4ksg19 May 2026 at 19:39 (modified: 01 Jun 2026 at 17:50)Senior Area Chairs, Area Chairs, Authors, Program ChairsRevisions
Metareview:
This paper evaluates the capability of frontier Large Language Models (LLMs) to perform calibrated interval predictions of algorithm performance, utilizing the domain of causal discovery as a concrete evaluation testbed. Across a massive 5,200 bootstrap-based algorithmic runs, the paper demonstrates a systemic failure mode: the mean calibrated interval coverage across models is a mere 15.9%, with seven out of eight frontier LLMs performing significantly below a trivial uniform-random baseline. Reviewers praised the paper's rigorous and extensive empirical setup, noting that focusing on quantitative uncertainty calibration intervals rather than qualitative judgments or point-estimate accuracy offers an original perspective on model predictive limits.

However, the paper faced widespread pushback regarding its overgeneral claims and sweeping title ("Large Language Models are Algorithmically Blind"), with reviewers arguing that causal discovery is a highly narrow and unusual domain, and model failure within it provides very weak evidence for broad, domain-agnostic algorithmic blindness. Reviewers also highlighted that the metadata provided in the prompts (e.g., sample size and data type) was too sparse and restrictive to quantitatively estimate precise metrics, and questioned the strength of the authors' conclusions attributing above-chance results to benchmark memorization rather than distribution-shift uncertainty awareness. In the rebuttal, the authors defended the task's real-world motivation by citing existing automated selection frameworks and detailed concrete methodological expansions—such as launching an anonymized benchmark experiment, introducing non-Gaussian synthetic datasets, and planning a human expert baseline study in future.

The authors promised crucial corrections in a future revision—such as testing anonymized prompt descriptions, providing a human expert baseline, and adding non-Gaussian synthetic datasets. However, because they did not provide these empirical results during the active review period, major concerns regarding over-claimed memorization probes, the narrow evaluation scope, and insufficient baseline controls remained fundamentally unresolved.

Confidence: 5: The area chair is absolutely certain
Official Review of Submission448 by Reviewer o67S
Official Reviewby Reviewer o67S12 Apr 2026 at 16:36 (modified: 01 Jun 2026 at 18:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer o67S, AuthorsRevisions
Q1 summary:
The authors evaluate the ability of frontier models to predict the performance of algorithms. Specifically, they systematically evaluate the performance of eight frontier models on predicting the accuracy of the causal structure learned by four different algorithms for causal discovery.

Q2 novelty: 2: Fair: The paper contributes some new ideas.
Q2 correctness: 3: Good: The paper appears to be technically sound, but I have not carefully checked the details.
Q2 evidence based support: 2: Fair: the main claims are somewhat supported by evidence but the experimental evaluation may be weak, important baselines may be missing, proofs lack rigor, or assumptions are not sufficiently motivated.
Q2 reproducibility: 3: Good: key resources are available and key details are sufficiently well-described for competent researchers to confidently reproduce the main results.
Q2 writing clarity: 3: Good: The paper is well organized but the presentation could be improved.
Q3 strengths:
The experimental methodology appears fairly extensive and thoughtful. The authors evaluated 13 datasets (both real and synthetic), four different algorithms for causal discovery, four different peformance measures, and eight different frontier models. The authors also compare the frontier model results to two baseline systems, and they attempt to systematically evaluate whether the frontier models are memorizing results of engaging in some form of reasoning.

The results are quite clear and fairly consistent. Frontier models are show to perform very poorly at predicting quantitative measures of the accuracy of causal model structure inferred by causal discovery algorithms.

The paper reads well.

Q4 weaknesses:
The paper makes several overgeneral claims. While the Discussion (Section 5) concludes in a fairly narrow way that “LLMs should not be used as zero-shot performance predictors for causal discovery algorithm selection,” other parts of the paper make far grander claims. The title proclaims an extremely general conclusion (“Large Language Models are Algorithmically Blind”). The Introduction states that: “We ask whether frontier LLMs can provide calibrated predictions of algorithm performance.” Section 4.2 claims that “These findings confirm that LLMs provide no systematic reasoning advantage for algorithm performance prediction.”

Causal discovery is a very narrow and highly unusual domain of discourse. That is, merely because LLMs do a poor job making predictions about causal discovery algorithms is only very weak evidence for the (very general) claims in the paper.

Most human reasoners (indeed, most researchers in causal inference) would not be able to perform this task very effectively. This is particularly true given the information provided in typical queries (Figure 7), which is merely dataset name, number of variables, number of samples, data type, and algorithm name. For example, there is no information about the adherence of the data set to key assumptions of many of the algorithms (e.g., faithfulness, causal sufficiency, modularity, positivity, etc.). There is also no information about how key aspects of how each algorithm is implemented (e.g., whether PC uses conditional independence tests that assume linear models or instead uses some more modern CI test). As plenty of papers in causal inference have show, these factors will have strong effects on the measures being estimated by the frontier models.

The authors define “algorithmic blindness” as “the inability of a model to form calibrated probabilistic beliefs about algorithm performance from problem structure and algorithmic description alone...” This term seems likely to be misunderstood by most readers, particularly since it is used in the title. My initial interpretation of the term was that it referred to the inability of LMs to correctly reason about how an algorithm worked at a procedural level. A more immediately interpretable term would be “performance prediction” or “performance estimation”.

Q5 comments:
Overall, then, a key question is whether it is reasonable to expect an LLM to accurately predict the performance of a given algorithm for causal discovery, given the information provided in the prompts. This is particularly true for algorithms with as complex a performance relationship as algorithms for causal discovery. It is unclear (to this reviewer, at least) whether any reasonable intelligence (human or artificial) could be expected to make accurate predictions in this domain. Clearly, it might be possible to learn some degree of dependence between some high-level characteristics of a data set and the characteristics of the resulting model, but that dependence would be expected to be fairly weak, at best.

The paper would be greatly improved by a discussion, fairly early in the paper, on the ways in which it might be possible to accurately predict how a given causal discovery algorithm will perform given the input data used in the experiments. Furthermore, the authors should also discuss why their results are surprising and novel. That is, why would any substantial set of researchers or practitioners expect that frontier models would perform well at the tasks that they examine?

Q6 rating: 3: Reject: For instance, a paper with technical flaws, limited novelty, weak experimental evaluation, inadequate reproducibility, incompletely addressed ethical considerations.
Q7 justification:
The experimental evidence, while competently done, does produce results that are surprising or novel. That is, most researchers would already have expected that frontier models would not be able to accurately predict the effectiveness of algorithms for causal discovery (or, if they could to some degree, that would be based on memorization).

Q8 confidence: 4: Quite confident. I tried to check the important points carefully. It is unlikely, though conceivable, that I missed some aspects that could otherwise have impacted my evaluation. I am familiar with the research topic and most of the related work.
Q9 confirmation: I have read the UAI reviewing instructions and certify that I comply with them.
Rebuttal by Authors
Rebuttalby Authors (Ashish Mahendran Kurapath, Sohan Venkatesh, Tejas Melkote)01 May 2026 at 21:40 (modified: 01 May 2026 at 21:51)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We appreciate the detailed feedback and address each concern directly.

On overclaiming: We accept this criticism and will scope the title and abstract to explicitly frame causal discovery as a testbed rather than a universal claim. The term "algorithmic blindness" will be defined more precisely and earlier in the paper.
On novelty and whether results are surprising: The reviewer suggests most researchers would expect this failure. We respectfully disagree. A growing body of recent work explicitly assumes or demonstrates LLMs have useful quantitative reasoning about algorithm behavior in exactly this setting. Causal-Copilot (Wang et al. (2025)) deploys LLMs to automate algorithm selection for causal discovery, explicitly relying on LLM knowledge of algorithm assumptions and dataset characteristics to recommend methods. Tornede et al. (2023) argue LLMs hold promise for AutoML including algorithm selection and configuration. Yang et al. (2023) show LLM-guided optimization outperforms random baselines in structured search. Our result is a direct empirical challenge to these assumptions. If the failure were expected and obvious, these systems would not have been built.
On task difficulty for humans: This is precisely our point. We are not claiming the task is easy. We are claiming LLMs specifically fail despite demonstrating declarative knowledge of these algorithms. The failure mode is the gap between what LLMs know and what they can predict, not the absolute difficulty of the task. A random baseline at 36.5% demonstrates the task is solvable at above-chance levels. LLMs fall dramatically below this.
On missing algorithm implementation details: Our prompts deliberately reflect realistic practitioner usage, the same information a practitioner would have when deciding which algorithm to run. This is a feature, not a limitation. The question is whether LLMs can serve as zero-shot algorithm selectors under realistic conditions, not idealized ones.
Official Review of Submission448 by Reviewer psTX
Official Reviewby Reviewer psTX12 Apr 2026 at 05:14 (modified: 01 Jun 2026 at 18:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer psTX, AuthorsRevisions
Q1 summary:
This paper investigates whether LLMs can accurately predict the performance of algorithms on specific problem instances, focusing on causal discovery as the evaluation domain. The objective is to determine if LLMs can generate calibrated uncertainty estimates for algorithm performance from problem structure alone. The paper evaluated eight frontier LLMs across 13 datasets, four causal discovery algorithms, and four performance metrics, using multiple prompt formulations.

Q2 novelty: 3: Good: The paper makes non-trivial advances over the current state-of-the-art.
Q2 correctness: 2: Fair: The paper has minor, easily fixable, technical flaws that do not impact the validity of the main results.
Q2 evidence based support: 2: Fair: the main claims are somewhat supported by evidence but the experimental evaluation may be weak, important baselines may be missing, proofs lack rigor, or assumptions are not sufficiently motivated.
Q2 reproducibility: 3: Good: key resources are available and key details are sufficiently well-described for competent researchers to confidently reproduce the main results.
Q2 writing clarity: 3: Good: The paper is well organized but the presentation could be improved.
Q3 strengths:
Evaluating uncertainty calibration rather than point-estimate accuracy provides a novel test angle of the models' predictive limits.
The experimental design controls for multiple confounders. The inclusion of both real-world benchmark datasets and synthetic datasets allows for a dissociation between memorization and algorithmic reasoning.
The study is comprehensive, evaluating 8 distinct frontier models, using 3 different prompt formulations to ensure findings are robust to phrasing (detailed in Appendix B), and relying on rigorous ground truth computation (100 runs per condition).
Q4 weaknesses:
The core premise of the paper, which is using LLMs to predict the numerical performance metrics of an algorithm on a specific dataset, lacks strong real-world motivation. The paper justifies that this as a way to reduce the need for "costly empirical evaluation," but in practice, a practitioner would likely just run the algorithm on the dataset (or use an LLM to write the code to do so) to obtain the exact performance metrics, rather than relying on an LLM's zero-shot, highly uncertain estimation. The task appears contrived to test LLM limitations rather than addressing a practical use case for algorithm selection.
The paper implies that LLMs lack algorithmic reasoning, which relies heavily on their failure to provide calibrated confidence intervals. However, producing calibrated interval predictions is a difficult task that often requires specific training or conformal prediction techniques, even for specialized models. Failure in this task might stem from an inability to calibrate output ranges rather than a complete lack of algorithmic understanding.
The paper used argument "under memorization, LLMs should produce tighter ranges for benchmark datasets whose statistics they have retrieved and wider ranges for novel synthetic data.", which I do not fully understand. The narrower-wider width phenomenon might also indicate that LLMs are uncertainty aware and, since that synthetic dataset is out of its training distribution, they will produce wider range of outputs. My point here is that there could be multiple explanations for the phenomenon and it does not necessarily imply memorization in LLMs.
Q5 comments:
Please see my comments above.

Q6 rating: 4: Borderline reject: Technically solid paper where reasons to reject outweigh reasons to accept. Please use sparingly.
Q7 justification:
While the scale of the evaluation is impressive, the core task is of questionable utility, and the conclusions regarding "algorithmic blindness" looks over-claimed and insufficiently decoupled from the inherent difficulty of zero-shot interval calibration and the lack of convincing evidence in the memorization probes.

Q8 confidence: 3: Somewhat confident, but there's a chance I missed some aspects. I did not carefully check some of the details, e.g. novelty, proof of a theorem, experimental design, or statistical validity of conclusions. I am somewhat familiar with the topic but may not know all related work.
Q9 confirmation: I have read the UAI reviewing instructions and certify that I comply with them.
Rebuttal by Authors
Rebuttalby Authors (Ashish Mahendran Kurapath, Sohan Venkatesh, Tejas Melkote)01 May 2026 at 21:50Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Rebuttal:
We appreciate the constructive feedback and address each concern directly.

On real-world motivation: The motivation extends beyond cost reduction. Algorithm selection is a genuine problem. Practitioners routinely choose between PC, FCI, LiNGAM and NOTEARS without running all four. If LLMs could provide calibrated guidance this would be directly useful. Our result shows they cannot, which is important for practitioners considering LLM-assisted algorithm selection.
On calibration difficulty: We agree calibrated interval prediction is hard. However our random baseline achieves 36.5% coverage by construction, demonstrating the task is not inherently impossible. Seven of eight frontier LLMs fall significantly below this trivial baseline. The failure is not that the task is hard. It is that LLMs perform worse than random guessing, which is a much stronger negative result.
On the uncertainty-awareness alternative explanation: This is a fair point and we take it seriously. However three converging signals jointly support memorization over uncertainty awareness. First, range width compression is not uniform. It is strongest for Claude (0.26x) which has the most training exposure to causal discovery literature and weakest for Gemini 3 and Qwen-Think. Pure uncertainty awareness would predict uniform compression across models. Second, cross-model agreement collapses 2.6x on synthetic data. If models were genuinely uncertainty-aware they would still agree on their uncertain estimates. Instead they diverge wildly, indicating independent guessing. Third, the algorithm x metric dissociation, specifically NOTEARS achieving 41.3% SHD coverage versus near-zero for PC/FCI, is inexplicable under uncertainty awareness but directly predicted by memorization of NOTEARS benchmark papers. We will clarify this multi-signal argument more explicitly in the revision.
Official Review of Submission448 by Reviewer j7Mn
Official Reviewby Reviewer j7Mn11 Apr 2026 at 21:19 (modified: 01 Jun 2026 at 18:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer j7Mn, AuthorsRevisions
Q1 summary:
This paper evaluates whether LLMs can make calibrated interval predictions of algorithm performance on causal discovery as a concrete, measurable testbed. The authors evaluate 8 LLMs across 13 datasets, 4 algorithms, 4 metrics, and 3 prompt formulations. They find systematic failure with seven of eight models performing below a random-interval baseline. They present multiple behavioral probes that they argue are most consistent with memorization rather than principled algorithmic reasoning.

Q2 novelty: 3: Good: The paper makes non-trivial advances over the current state-of-the-art.
Q2 correctness: 2: Fair: The paper has minor, easily fixable, technical flaws that do not impact the validity of the main results.
Q2 evidence based support: 2: Fair: the main claims are somewhat supported by evidence but the experimental evaluation may be weak, important baselines may be missing, proofs lack rigor, or assumptions are not sufficiently motivated.
Q2 reproducibility: 3: Good: key resources are available and key details are sufficiently well-described for competent researchers to confidently reproduce the main results.
Q2 writing clarity: 3: Good: The paper is well organized but the presentation could be improved.
Q3 strengths:
This paper offers a rigorous negative result on the usefulness of zero-shot LLMs for calibrated algorithm performance prediction, an increasingly relevant claim as LLMs are proposed for model/algorithm selection tasks.
The empirical setup is fairly broad. It spans 8 models, 52 dataset-algorithm conditions, 3 prompt formulations, and 5,200 total algorithm runs for the empirical reference values. I also appreciate that the authors did not rely on a single prompt template and instead aggregate across three formulations, which makes the main result more robust to prompt wording artifacts.
Q4 weaknesses:
(1) The paper focuses on zero-shot quantitative prediction of algorithm performance for causal discovery algorithms. But jumping from that to “the failure mode is not domain specific” claim is not justified by the experiments. The authors only test one domain, one task family, four algorithms, and one style of prediction target. The leap from this to the claim in the title ( and the abstract and the introduction) is not sufficiently supported by evidence in my opinion.
(2) The evaluation target is generally reasonable since the empirical mean would generally be expected to lie within a predicted range of typical outcomes. However, the prompts do not clearly specify whether the desired interval is intended to capture run-level variability, central mass, or uncertainty around the mean.
(3) The paper argues that the limited above-random performance is “most consistent with memorization rather than principled reasoning.” I do not think the evidence presented is sufficient to support that claim strongly. The memorization argument relies on indirect signals: benchmark vs. synthetic gaps, range-width compression, and cross-model agreement patterns. These are suggestive, but they do not rule out alternative explanations. I think the claim needs to be softened.
Q5 comments:
I have a few questions for the authors:

(1) The random baseline is described to sample predicted ranges uniformly at random. Could you please specify how the lower and upper bounds are sampled and whether interval widths are biased wide or narrow?
(2) Have the authors tested anonymized benchmark descriptions, where dataset names are removed, to better separate memorization from structure-based reasoning?
(3) Why is containment of the empirical mean the primary scoring rule, rather than containment of the empirical 95% interval or use of a proper interval scoring rule that takes into account both accuracy and sharpness?
(4) Are LLMs systematically biased in one direction (e.g., optimistic), such that even wide ranges are shifted away from ground truth? A bias analysis (not just width analysis) should clarify this.
Q6 rating: 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject. Please use sparingly.
Q7 justification:
The paper investigates an important question and reports an interesting negative result. However, the current version overstates some of the conclusion and leaves several important methodological details insufficiently clear (please refer to my questions above). I would be happy to improve my score if the authors narrow the claims and provide more clarity regarding my concerns.

Q8 confidence: 3: Somewhat confident, but there's a chance I missed some aspects. I did not carefully check some of the details, e.g. novelty, proof of a theorem, experimental design, or statistical validity of conclusions. I am somewhat familiar with the topic but may not know all related work.
Q9 confirmation: I have read the UAI reviewing instructions and certify that I comply with them.
Rebuttal by Authors
Rebuttalby Authors (Ashish Mahendran Kurapath, Sohan Venkatesh, Tejas Melkote)01 May 2026 at 21:48Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Rebuttal:
We appreciate the detailed and constructive feedback and address each point directly.

On domain generality: We accept this criticism. We will remove the claim "the failure mode is not domain specific" and replace it with a carefully scoped claim. We note that causal discovery was chosen precisely because it is a best-case scenario for LLMs. Algorithms are well-documented, benchmarks are prominent in training corpora, and theoretical properties are clearly defined. Failure here is therefore more surprising and more informative than failure in an obscure domain.
On prompt clarity: We will add explicit clarification to all three prompt formulations that the interval should capture the range of typical outcomes across repeated runs.
On softening memorization claims: We will soften "most consistent with memorization" to "suggestive of memorization" and present the three signals as converging indirect evidence rather than proof.
We now answer the reviewer's specific questions directly.
a. Q1 on random baseline construction: The random baseline samples the lower bound uniformly from [0, metric_max] and the upper bound uniformly from [lower_bound, metric_max], producing unbiased interval widths. We will add this specification explicitly to Section 3.4.
b. Q2 on anonymized benchmark experiment: We are actively running this experiment, removing dataset names from prompts and rerunning all eight models. Results will be included in the camera-ready version if accepted.
c. Q3 on containment of empirical mean: We chose the empirical mean as the target because it is the most stable estimator from 100 bootstrap runs and directly answers the practitioner question of whether typical performance falls in the predicted range. We will add a supplementary analysis using containment of the empirical 95% CI as an alternative target.
d. Q4 on bias direction: We are running a systematic bias direction analysis on existing data. Preliminary results confirm LLMs are systematically optimistic, with predicted ranges shifted toward higher precision and recall values relative to ground truth. Full results will be included in the revision.
Official Review of Submission448 by Reviewer KsDD
Official Reviewby Reviewer KsDD10 Apr 2026 at 08:33 (modified: 01 Jun 2026 at 18:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer KsDD, AuthorsRevisions
Q1 summary:
This paper tests whether frontier LLMs can give calibrated interval predictions for causal discovery algorithm performance. Across 8 models, 13 datasets, 4 algorithms, 4 metrics, and bootstrap-based runs, mean calibrated coverage is only , with  models below a random baseline. The paper argues the lone above-random result is more consistent with benchmark memorization than principled reasoning.

Q2 novelty: 2: Fair: The paper contributes some new ideas.
Q2 correctness: 2: Fair: The paper has minor, easily fixable, technical flaws that do not impact the validity of the main results.
Q2 evidence based support: 2: Fair: the main claims are somewhat supported by evidence but the experimental evaluation may be weak, important baselines may be missing, proofs lack rigor, or assumptions are not sufficiently motivated.
Q2 reproducibility: 2: Fair: key resources are unavailable but key details are sufficiently well-described for an expert to confidently reproduce the main results.
Q2 writing clarity: 4: Excellent: The paper is well-organized and clearly written.
Q3 strengths:
Decision-relevant problem formulation through interval coverage rather than qualitative judgments.
Broad and systematic empirical evaluation across models, datasets, algorithms, metrics, and prompts.
Strong presentation quality with useful appendix diagnostics.
Q4 weaknesses:
Memorization interpretation is confounded by synthetic-data design that favors NOTEARS and disadvantages LiNGAM assumptions.
No human expert baseline, which weakens the specificity of the "algorithmic blindness" claim.
Statistical/reproducibility details remain incomplete for high-confidence interpretation.
Q5 comments:
This is a valuable negative-result paper. The calibration framing is meaningful, and the broad experimental sweep makes the low-coverage result worth taking seriously.

My main reservation is interpretation strength. I am convinced by the robust finding that calibration is poor. I am less convinced that the current controls isolate memorization as the dominant mechanism, because the synthetic setup interacts unevenly with algorithm assumptions.

I have 2 questions for authors:

Can you add synthetic settings with non-Gaussian noise and report whether the LiNGAM degradation pattern persists?
Can you provide a human-expert interval-prediction baseline on a representative subset to separate LLM-specific failure from task difficulty?
Q6 rating: 4: Borderline reject: Technically solid paper where reasons to reject outweigh reasons to accept. Please use sparingly.
Q7 justification:
I score it 4/10 because the negative result on poor calibration is interesting, the empirical sweep is broad, and the paper is very clear. But the central memorization interpretation is confounded by synthetic data that favor NOTEARS over LiNGAM, related-work positioning is incomplete, there is no human expert baseline, and the statistical / baseline details are not strong enough for acceptance.

Q8 confidence: 3: Somewhat confident, but there's a chance I missed some aspects. I did not carefully check some of the details, e.g. novelty, proof of a theorem, experimental design, or statistical validity of conclusions. I am somewhat familiar with the topic but may not know all related work.
Q9 confirmation: I have read the UAI reviewing instructions and certify that I comply with them.
Q10 ethics:
No ethical concerns.

Rebuttal by Authors
Rebuttalby Authors (Ashish Mahendran Kurapath, Sohan Venkatesh, Tejas Melkote)01 May 2026 at 21:41 (modified: 01 May 2026 at 21:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We appreciate the thorough and constructive review and address each concern directly.

On memorization confound with synthetic data: The reviewer is correct that linear Gaussian synthetic data favors NOTEARS and disadvantages LiNGAM. However we argue this strengthens rather than weakens the memorization argument. LiNGAM's linear non-Gaussianity assumption is no harder to reason about on synthetic graphs than on benchmarks. If LLMs understood the assumption they would predict poor LiNGAM performance on Gaussian data regardless of whether it is synthetic or benchmark. The key observation is that LiNGAM achieves 27.1% benchmark coverage but collapses to 3.9% on synthetic data generated under identical Gaussian conditions. The collapse tracks training data presence, not a change in algorithmic properties or data characteristics. We will add non-Gaussian synthetic datasets in the revision to directly test whether LiNGAM degradation persists under assumption-satisfying conditions, which will further isolate the memorization mechanism.
On human expert baseline: We acknowledge this is a genuine gap. We note however that the primary finding does not depend on it. Even if human experts also perform near random on this task, that would not explain why seven of eight frontier LLMs perform significantly worse than a trivial random interval baseline. The gap between LLMs and random guessing is the core finding. A human baseline addresses task difficulty but not the specific LLM failure relative to random. We will add a small expert study on a representative subset in the revision and discuss this distinction explicitly.
On related work positioning: We will strengthen the related work section to include recent work on LLMs for causal discovery algorithm selection, specifically Causal-Copilot (Wang et al. (2025)), which deploys LLMs for automated algorithm recommendation in causal discovery, and the survey by Zhu et al. (2024) on LLMs for causal discovery, both of which assume or demonstrate LLM competence in qualitative algorithm reasoning. Our result provides the first quantitative calibration test of these assumptions.
On statistical and reproducibility details: We will add full specification of the random baseline construction, bootstrap resampling procedure, and confidence intervals around all coverage estimates in the revision.
Official Review of Submission448 by Reviewer u9TJ
Official Reviewby Reviewer u9TJ25 Mar 2026 at 11:10 (modified: 01 Jun 2026 at 18:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer u9TJ, AuthorsRevisions
Q1 summary:
This paper study the problem whether frontier LLMs could predict the performance of algorithms by using the causal discovery task as a testbed. The authors term the systematic failure of LLMs as “algorithmic blindness”, where it can be observed that frontier LLMs perform no better than random guess. These findings indicate the gap between declarative knowledge and calibrated prediction in LLMs.

Q2 novelty: 3: Good: The paper makes non-trivial advances over the current state-of-the-art.
Q2 correctness: 3: Good: The paper appears to be technically sound, but I have not carefully checked the details.
Q2 evidence based support: 2: Fair: the main claims are somewhat supported by evidence but the experimental evaluation may be weak, important baselines may be missing, proofs lack rigor, or assumptions are not sufficiently motivated.
Q2 reproducibility: 3: Good: key resources are available and key details are sufficiently well-described for competent researchers to confidently reproduce the main results.
Q2 writing clarity: 3: Good: The paper is well organized but the presentation could be improved.
Q3 strengths:
The setting of this paper is novel: causal discovery algorithms typically have distinct theoretical assumptions, making it theoretically possible to estimate algorithm performance on the base of dataset metadata only.
The experimental results are surprising yet insightful. No models exhibit a clear advantage over the random baseline, proving that LLMs are currently incapable of estimating algorithm performances. Meanwhile, LLMs generally show lower coverage score but higher prediction range width, indicating that the correct coverage of existing benchmarks may come from memorization rather than computation.
Q4 weaknesses:
The main weakness of this paper is that the setting may be too harsh for LLMs. As demonstrated in Figure 7, LLMs only have access to the metadata of datasets (name, num of samples & variables, datatype). These information may be sufficient to qualitatively determine whether an algorithm could function normally (for example the number of samples should be larger than variables), but is insufficient to quantitively estimate the final metrics. For example, when feeding data with Gaussian noise to LiNGAM (which requires non-Gaussian noise), it will crash, but whether the “crash” will lead to a F1 of 0.2 or 0.1 is unknown.
Causal discovery is a very special task, where existing algorithms have clear theoretical assumptions. For most practical tasks, the limitations of algorithms is unclear, making the conclusions restricted in the causal discovery task.
Q5 comments:
The experimental settings should be explained more clearly. What are the possible values of <data_type> and <complexity> in the prompts? And why Formulation 2 does not have a <data_type> field, while Formulation 1 and 3 do not have <complexity> fields? The authors should provide an example of formulated prompts.
As stated above, the metadata may be insufficient to quantitively estimate the final metrics. More information like Kurtosis (which evaluates how "Gaussian" the data is) may be required.
Frontier models could now function as agents. Will it be possible for LLMs to guide algorithm selection by executing the algorithm on a small dataset in a sandbox?
Q6 rating: 6: Weak Accept: Technically solid paper, with no major concerns with respect to provided evidence, resources, reproducibility, and ethical considerations.
Q7 justification:
While I feel that the current setting is “unfair” for LLMs where too little information is provided, the motivation and findings of this paper is novel and interesting, which I weigh more heavily.

Q8 confidence: 3: Somewhat confident, but there's a chance I missed some aspects. I did not carefully check some of the details, e.g. novelty, proof of a theorem, experimental design, or statistical validity of conclusions. I am somewhat familiar with the topic but may not know all related work.
Q9 confirmation: I have read the UAI reviewing instructions and certify that I comply with them.
Rebuttal by Authors
Rebuttalby Authors (Ashish Mahendran Kurapath, Sohan Venkatesh, Tejas Melkote)01 May 2026 at 21:42 (modified: 01 May 2026 at 21:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We appreciate the positive assessment and address the remaining concerns directly.

On prompt metadata sufficiency: We agree that dataset metadata alone may be insufficient for precise quantitative prediction. This is central to our argument. LLMs are being asked to do what practitioners implicitly expect them to do when recommending algorithms and they fail. The point is not that the task is easy but that LLMs claim to know these algorithms yet cannot translate that knowledge into calibrated predictions even under realistic information constraints.
On kurtosis and additional metadata: Adding kurtosis is an interesting direction for future work. We note however that even with kurtosis the LiNGAM synthetic collapse is not explained by data statistics alone. The collapse tracks training data presence across all models, not just on datasets where Gaussianity violations are present.
On prompt field clarification: We will provide fully instantiated example prompts for all three formulations with concrete values filled in and clarify the data_type and complexity fields explicitly.
On LLM agents running algorithms in sandbox: This is an excellent direction for future work and we will add it to Section 7. It is however orthogonal to our research question. We are evaluating zero-shot calibrated prediction, not agentic execution.
Official Comment by Reviewer u9TJ
Official Commentby Reviewer u9TJ02 May 2026 at 16:01Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thank you for your response. I will maintain my score.

Algorithmic Blindness in Large Language Models: A Calibration Study of Performance Prediction
Download PDF
Sohan Venkatesh, Ashish Mahendran Kurapath, Tejas Melkote 
24 Jun 2026 (modified: 29 Jun 2026)
COLM 2026 Workshop SciFM Submission
Sci-FM, Reviewers, Authors
Revisions
CC BY 4.0
Keywords: calibration, large language models, algorithm selection, memorization
TL;DR: Frontier LLMs show poor calibration when predicting algorithm performance, with the one above-random model's gain pointing to benchmark memorization rather than reasoning.
Abstract:
Large language models (LLMs) demonstrate remarkable breadth of knowledge, yet their ability to reason about computational processes remains poorly understood. Closing this gap matters for practitioners who rely on LLMs to guide algorithm selection and deployment. We address this limitation using causal discovery as a testbed and evaluate eight frontier LLMs against ground truth derived from algorithm executions. We find systematic, near-total failure across models. The predicted ranges are far wider than true confidence intervals yet still fail to contain the true algorithmic mean in most cases. Most models perform worse than random guessing. The best model's marginal improvement points to benchmark memorization rather than principled reasoning. We term this failure algorithmic blindness and argue it reflects a fundamental gap between declarative knowledge about algorithms and calibrated procedural prediction.

Email Sharing: We authorize the sharing of all author emails with Program Chairs.
Data Release: We authorize the release of our submission and author names to the public in the event of acceptance.
Submission Number: 52
Filter by reply type...

Filter by author...




2 / 2 replies shown
Add:
Paper Decision
Decisionby Program Chairs25 Jul 2026 at 02:40 (modified: 25 Jul 2026 at 14:00)Program Chairs, Reviewers, AuthorsRevisions
Decision: Accept (Poster)
Add:

Official Comment
Review
Official Reviewby Reviewer u68Q19 Jul 2026 at 17:25Program Chairs, Reviewers, Authors
Summary:
This paper studies whether LLMs can predict the performance of causal discovery algorithms. Across eight LLMs and thirteen datasets, the authors find poor calibration and argue that this reflects a gap between declarative algorithmic knowledge and procedural performance prediction.

Review:
This paper studies an important question through a systematic empirical evaluation. Please see the detailed strengths and weaknesses below.

Strengths:
The paper studies an important and practically relevant question: whether LLMs can provide reliable quantitative guidance for algorithm selection.
The evaluation is comprehensive within causal discovery, and presents a consistent negative result across these settings.
Weaknesses:
If I understand correctly, the models receive only coarse dataset information rather than the actual data or detailed algorithmic procedures. This may make the task too under-specified for LLMs to answer reliably. It is also unclear to me whether the baselines have access to exactly the same information.
The experiments cover only causal discovery. Evaluating additional algorithmic domains would better support the broader negative claim.
The causal discovery setup could be explained more carefully for readers unfamiliar with the area.
Questions:
Do the LLM inputs exclude the actual dataset and detailed algorithmic procedure, containing only the dataset name, number of nodes, number of samples, data type or complexity, and algorithm name? If so, why is this information sufficient for predicting algorithm performance?
Would the conclusions change if the model were given access to the actual dataset and allowed to write and execute code?
Rating: 4: Weak Accept
Confidence: 3: Moderately confident

