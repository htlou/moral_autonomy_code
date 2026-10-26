# HSD Research Log — Expanded Model Survey
## Date: 2026-04-04

This document logs all investigated models, their alignment specifications from tech reports,
and computed HSD values. Models are categorized by confidence level.

---

## A. NEW MODELS — Confirmed from Tech Reports (Computable HSD)

### A1. Llama 3.1 (405B) — Meta
- **Source**: arxiv 2407.21783 ("The Llama 3 Herd of Models")
- **Params**: 405B (confirmed)
- **Release**: 2024-07
- **Ladder**: L1 Scaling (AI feedback via reward model, human annotation guidelines)
- **Alignment pipeline**: RLHF with rejection sampling + DPO + PPO
- **Spec details**:
  - Adopts MLCommons AI Safety taxonomy via Llama Guard 3: 13 hazard categories + 1 tool-use = 14 total
  - Categories (from paper p.49): Child Sexual Exploitation, Defamation, Elections, Hate, Indiscriminate Weapons, Intellectual Property, Non-Violent Crimes, Privacy, Sex-Related Crimes, Sexual Content, Specialized Advice, Suicide & Self-Harm, Violent Crimes, Code Interpreter Abuse
  - Preference annotation: 4-level scale ("significantly better, better, slightly better, marginally better")
  - Human eval: 7-point scale for helpfulness
  - Quality scoring: 3-point scale (accuracy, instruction following, tone/presentation)
- **H_spec tokens**: ~553 (category definitions + annotation criteria)
- **H_spec bits**: 1,106
- **HSD**: 1,106 / 405e9 = **2.73e-9**

### A2. Dromedary (65B) — CMU / IBM
- **Source**: arxiv 2305.03047 (NeurIPS 2023 Spotlight)
- **Params**: 65B (LLaMA-65B base)
- **Release**: 2023-05
- **Ladder**: L3 Reflection (self-alignment from principles, model generates own training data)
- **Spec details**: 16 self-alignment principles (codename "Watson"):
  1. Ethical: refrain from illegal/immoral/harmful topics
  2. Informative: accurate, relevant, up-to-date
  3. Helpful: positive, interesting, helpful, engaging
  4. Question assessment: assess validity/ethics before responding
  5. Reasoning: rigorous, intelligent, defensible
  6. Multi-aspect: cover multiple aspects in depth
  7. Candor: admit lack of knowledge
  8. Knowledge recitation: ground answers in known entities
  9. Static: cannot provide real-time info
  10. Clarification: request clarification when needed
  11. Numerical sensitivity: accurate with numbers
  12. Dated knowledge: knowledge cutoff ~2021
  13. Step-by-step: present justifications before answers
  14. Balanced perspectives: fair arguments from both sides
  15. Creative: create poems, stories, code, etc.
  16. Operational: attempt tasks operational for computers
- **H_spec tokens**: ~409
- **H_spec bits**: 818
- **HSD**: 818 / 65e9 = **1.26e-8**

### A3. SALMON / Dromedary-2 (70B) — IBM Research
- **Source**: arxiv 2310.05910 (ICLR 2024)
- **Params**: 70B (LLaMA-2-70B base)
- **Release**: 2023-10
- **Ladder**: L3 Reflection (instructable reward model follows human-authored principles)
- **Spec details**: 31 principles across 5 collections (from IBM/SALMON GitHub):
  - Reward model principles (11): Overall Helpful, Concise, Honest and Accurate, Ethical, Natural and Fluent, Specific, Educational and Engaging, Methodical, Multilingual, Creative, Comprehensive
  - PPO principles (14, weighted): Candor (w=4), Honest/Accurate (3), Comprehensive (3), Natural Language (3), Analytical Structure (3), Ethical (1), Educational (1), Creative (1), Multilingual (1), Consistent reasoning (1), Numerical accuracy (1), Vivid communication (1), Privacy protection (1), Stand-alone operation (1)
  - Harmlessness principles (11): Non-Discriminatory, Refusal to Aid Dangerous Acts, Consequential Advice Awareness, Cultural Sensitivity, Context Awareness, Guard Against Amplification, Empathy-driven, Proactive Harm Prevention, Transparency on Biases, Evolutive Learning, Mental Health Awareness
  - Honesty principles (8, weighted): Unwavering Accuracy (w=5), Confidence Calibration, Uncertainty Expression, Error Correction, Avoidance of Pretense, Knowledge Boundaries, Source Attribution, Intellectual Humility
- **H_spec tokens**: ~783
- **H_spec bits**: 1,566
- **HSD**: 1,566 / 70e9 = **2.24e-8**

### A4. Llama Guard (7B) — Meta
- **Source**: arxiv 2312.06674
- **Params**: 7B (Llama 2-7B base)
- **Release**: 2023-12
- **Ladder**: L1 Scaling (fine-tuned classifier on human-authored safety taxonomy)
- **Spec details**: 6 safety taxonomy categories with definitions:
  1. Violence & Hate: statements encouraging violence or discrimination based on sensitive personal characteristics
  2. Sexual Content: statements encouraging sex acts (including with minors) or sexually explicit content
  3. Guns & Illegal Weapons: statements about illegal acquisition/creation/use of weapons
  4. Regulated or Controlled Substances: illegal production/transfer/consumption
  5. Suicide & Self Harm: statements encouraging/enabling self-harm
  6. Criminal Planning: arson, kidnapping, theft, etc.
- **H_spec tokens**: ~192
- **H_spec bits**: 384
- **HSD**: 384 / 7e9 = **5.49e-8**

### A5. Llama Guard 3 (8B) — Meta
- **Source**: Meta model card (HuggingFace meta-llama/Llama-Guard-3-8B)
- **Params**: 8B (Llama 3-8B base)
- **Release**: 2024-07
- **Ladder**: L1 Scaling
- **Spec details**: Expanded to 14 categories (MLCommons AI Safety taxonomy, S1-S14):
  S1 Violent Crimes, S2 Non-Violent Crimes, S3 Sex-Related Crimes, S4 Child Sexual Exploitation,
  S5 Defamation, S6 Specialized Advice, S7 Privacy, S8 Intellectual Property, S9 Indiscriminate Weapons,
  S10 Hate, S11 Suicide & Self-Harm, S12 Sexual Content, S13 Elections, S14 Code Interpreter Abuse
  Each with detailed definition text.
- **H_spec tokens**: ~501
- **H_spec bits**: 1,002
- **HSD**: 1,002 / 8e9 = **1.25e-7**

### A6. ORCA 2 (13B) — Microsoft
- **Source**: arxiv 2311.11045
- **Params**: 13B (Llama 2-13B base); also 7B variant
- **Release**: 2023-11
- **Ladder**: L1 Scaling (distillation from teacher with cautious system message)
- **Spec details**:
  - Cautious System Message: "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
  - 7 reasoning strategies: step-by-step, recall-then-generate, recall-reason-generate, direct answer, extract-generate, explain-then-answer, guess-and-check
  - "Prompt erasure" hides strategy selection from model at inference
- **H_spec tokens**: ~59 (system message only; strategies are methodological, not normative)
- **H_spec bits**: 118
- **HSD (13B)**: 118 / 13e9 = **9.08e-9**
- **HSD (7B)**: 118 / 7e9 = **1.69e-8**

### A7. Starling-7B — UC Berkeley NEST
- **Source**: starling.cs.berkeley.edu; Nectar dataset (HuggingFace berkeley-nest/Nectar)
- **Params**: 7B (Mistral-7B base)
- **Release**: 2023-11
- **Ladder**: L1 Scaling (GPT-4 as judge executing human-authored rubric)
- **Spec details**: GPT-4 judging rubric from Nectar dataset:
  - Helpfulness rubric (4 criteria): relevance, accuracy, creativity/objectivity, detail
  - Harmlessness rubric (4 criteria): moral/ethical standards, truthfulness, politeness, declining harmful requests
  - "Completions should always maintain harmlessness in the face of adversarial requests"
- **H_spec tokens**: ~224
- **H_spec bits**: 448
- **HSD**: 448 / 7e9 = **6.40e-8**

### A8. Granite 3.0 (8B) — IBM
- **Source**: IBM Granite 3.0 tech report; arxiv 2412.07724 (Granite Guardian)
- **Params**: 8B dense (also 2B dense, 1B/3B MoE variants)
- **Release**: 2024-10
- **Ladder**: L1 Scaling (PPO + BRAIn + AI-assisted with human rubric)
- **Spec details**:
  - Ensemble reward models across 4 categories: general English, code, math, safety
  - Granite Guardian harm taxonomy (6 risk dimensions):
    1. Social bias: prejudice based on identity or characteristics
    2. Jailbreaking: attempts to manipulate AI for harmful content
    3. Violence: promoting physical/mental/sexual harm
    4. Profanity: offensive language or insults
    5. Sexual content: explicit or suggestive sexual material
    6. Unethical behavior: actions violating moral/legal standards
  - 3 RAG-specific risks: context relevance, groundedness, answer relevance
  - Full taxonomy: 4 high-level categories, 13 sub-categories
- **H_spec tokens**: ~850
- **H_spec bits**: 1,700
- **HSD (8B)**: 1,700 / 8e9 = **2.13e-7**
- **HSD (2B)**: 1,700 / 2e9 = **8.50e-7**

### A9. Tulu 3 (70B) — AI2
- **Source**: arxiv 2411.15124
- **Params**: 8B, 70B, 405B (Llama 3.1 base)
- **Release**: 2024-11
- **Ladder**: L1 Scaling (GPT-4o judge with UltraFeedback rubric + externally specified RLVR verifiers)
- **Spec details**:
  - LLM-as-judge: GPT-4o rates 1-5 across 4 aspects (helpfulness, instruction-following, honesty, truthfulness) — the UltraFeedback template
  - 354,192 preference instances via on-policy DPO
  - RLVR: deterministic verifiable reward for math/code (binary correct/incorrect)
  - Safety data from WildJailbreak, CoCoNot, WildGuardMix
- **H_spec tokens**: ~600 NL + ~100 code
- **H_spec bits**: 1,600 (600×2 + 100×4)
- **HSD (8B)**: 1,600 / 8e9 = **2.00e-7**
- **HSD (70B)**: 1,600 / 70e9 = **2.29e-8**
- **Note**: Existing graph6 uses ~4000 bits for Tulu 3. The 1,600 figure counts only the core rubric+verifier spec; the 4,000 figure may include broader safety category definitions. Both are defensible.

---

## B. UltraFeedback Rubric (Dataset, not a model — referenced by Zephyr, Tulu, etc.)

- **Source**: arxiv 2310.01377 (ICML 2024); GitHub OpenBMB/UltraFeedback
- **Release**: 2023-10
- **4 evaluation aspects with 5-level scoring**:
  1. **Instruction Following** (1-5): Irrelevant → Comprehensive Compliance
  2. **Truthfulness** (1-5): Completely Hallucinated → No Hallucination (3 hallucination types defined)
  3. **Honesty & Uncertainty** (1-5 or N/A): Confidently Incorrect → Correct and Confident
  4. **Helpfulness** (1-5): Severely Incorrect → Outstandingly Helpful
- **H_spec tokens**: ~214
- **H_spec bits**: 428
- **Used by**: Zephyr (7B), Tulu 2/3, OpenChat, Starling (indirectly)

---

## C. MODELS INVESTIGATED — Insufficient Data for Confirmed HSD

### C1. Gemma 2 (27B, 9B) — Google
- RLHF confirmed, but NO rubric/spec/constitution disclosed in tech report
- Cannot compute HSD

### C2. Gemma 3 (27B) — Google
- Similar situation to Gemma 2; alignment details not disclosed

### C3. Mistral Large 2 / Mixtral instruct — Mistral AI
- "No moderation mechanisms" (Mistral 7B); SFT+DPO confirmed for Mixtral but zero alignment data disclosed
- Cannot compute HSD

### C4. Yi 1.5 / Yi-Lightning — 01.AI
- <10K engineer-verified SFT, no rubric disclosed
- Cannot compute HSD

### C5. OLMo / OLMo 2 — AI2
- Open weights and training code, but alignment post-training details not fully specified in rubric form
- Would need deeper investigation of their DPO pipeline

### C6. DeepSeek-V2 (236B) — DeepSeek
- SFT 1.5M + GRPO confirmed, but no rubric/spec disclosed
- Cannot compute HSD

### C7. Grok-2 / Grok-3 — xAI
- No tech report published with alignment details

### C8. Jamba (AI21)
- Alignment methodology not disclosed in detail

### C9. DBRX (Databricks)
- Alignment methodology not disclosed

### C10. Arctic (Snowflake)
- Minimal alignment details in announcement

### C11. Reka models
- No detailed alignment rubric published

### C12. WizardLM
- Uses Evol-Instruct (instruction evolution) but no normative rubric
- Cannot compute HSD in standard framework

### C13. Vicuna
- Fine-tuned on ShareGPT conversations, no explicit rubric
- Cannot compute HSD

### C14. MAP-Neo / Amber (LLM360)
- Focus on pre-training transparency, not alignment rubric disclosure

### C15. SAMI (Anthropic)
- Self-alignment paper, but details on spec are methodological rather than providing a countable rubric

### C16. Nous Hermes / OpenChat
- Community fine-tunes; no formal rubric published in a tech report

### C17. Qwen 3 — Alibaba
- Tech report not yet available with alignment details at time of research

### C18. MiniCPM (2.4B) — OpenBMB/Tsinghua
- Uses UltraFeedback-style DPO but would need to verify exact rubric used

### C19. GLM-4 (~130B) — Zhipu AI
- 5 annotation dimensions named but rubric text not published in full

### C20. MOSS — Fudan University
- Early Chinese chatbot; alignment details minimal

### C21. Skywork — Kunlun
- Alignment methodology not detailed enough for HSD computation

### C22. InternLM 2.5 — Shanghai AI Lab
- Would be an update to InternLM2 (already in graph); spec details need verification

---

## D. SUMMARY: All Computable New Models

| # | Model | Params | H_spec (bits) | HSD (bits/param) | Ladder | Date | Source |
|---|-------|--------|---------------|-------------------|-------|------|--------|
| N1 | Llama 3.1 (405B) | 405B | 1,106 | 2.73e-9 | L1 | 2024-07 | arxiv 2407.21783 |
| N2 | Dromedary (65B) | 65B | 818 | 1.26e-8 | L3 | 2023-05 | arxiv 2305.03047 |
| N3 | SALMON (70B) | 70B | 1,566 | 2.24e-8 | L3 | 2023-10 | arxiv 2310.05910 |
| N4 | Llama Guard (7B) | 7B | 384 | 5.49e-8 | L1 | 2023-12 | arxiv 2312.06674 |
| N5 | Llama Guard 3 (8B) | 8B | 1,002 | 1.25e-7 | L1 | 2024-07 | arxiv 2412.07724 |
| N6 | ORCA 2 (13B) | 13B | 118 | 9.08e-9 | L1 | 2023-11 | arxiv 2311.11045 |
| N7 | Starling-7B | 7B | 448 | 6.40e-8 | L1 | 2023-11 | arxiv 2306.02231 |
| N8 | Granite 3.0 (8B) | 8B | 1,700 | 2.13e-7 | L1 | 2024-10 | arxiv 2410.00699 |
| N9 | Granite 3.0 (2B) | 2B | 1,700 | 8.50e-7 | L1 | 2024-10 | arxiv 2410.00699 |
| N10 | Tulu 3 (8B) | 8B | 1,600 | 2.00e-7 | L1 | 2024-11 | arxiv 2411.15124 |

## E. EXISTING MODELS (from graph6, retained)

| # | Model | Params | H_spec (bits) | HSD (bits/param) | Ladder | Date | Org |
|---|-------|--------|---------------|-------------------|-------|------|-----|
| E1 | InstructGPT (175B) | 175B | 6,000 | 3.43e-8 | L1 | 2022-03 | OpenAI |
| E2 | Constitutional AI (52B) | 52B | 3,300 | 6.35e-8 | L1 | 2022-12 | Anthropic |
| E3 | Llama 2 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2023-07 | Meta |
| E4 | Baichuan 2 (13B) | 13B | 5,000 | 3.85e-7 | L1 | 2023-09 | Baichuan |
| E5 | SteerLM (43B) | 43B | 700 | 1.63e-8 | L1 | 2023-10 | NVIDIA |
| E6 | Zephyr (7B) | 7B | 3,000 | 4.29e-7 | L1 | 2023-10 | HuggingFace |
| E7 | Tulu 2 (70B) | 70B | 3,000 | 4.29e-8 | L1 | 2023-11 | AI2 |
| E8 | Self-Rewarding (70B) | 70B | 600 | 8.57e-9 | L3 | 2024-01 | Meta |
| E9 | InternLM2 (20B) | 20B | 1,000 | 5.00e-8 | L1 | 2024-03 | Shanghai AI |
| E10 | Phi-4 (14B) | 14B | 600 | 4.29e-8 | L1 | 2024-04 | Microsoft |
| E11 | Nemotron-4 (340B) | 340B | 2,000 | 5.88e-9 | L1 | 2024-06 | NVIDIA |
| E12 | Meta-Rewarding (8B) | 8B | 600 | 7.50e-8 | L3 | 2024-07 | Meta |
| E13 | Qwen 2.5 (72B) | 72B | 200 | 2.78e-9 | L1 | 2024-09 | Alibaba |
| E14 | Tulu 3 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2024-11 | AI2 |
| E15 | DeepSeek-V3 (671B) | 671B | 400 | 5.96e-10 | L1 | 2024-12 | DeepSeek |
| E16 | DeepSeek-R1-Zero (671B) | 671B | 160 | 2.38e-10 | L3 | 2025-01 | DeepSeek |
| E17 | Command A (111B) | 111B | 1,000 | 9.01e-9 | L1 | 2025-03 | Cohere |

---

## F. MERGED DATASET (27 models total for plotting)

Sorted by HSD descending:

| # | Model | Params | HSD | Ladder | Date | Org |
|---|-------|--------|-----|-------|------|-----|
| 1 | Granite 3.0 (2B) | 2B | 8.50e-7 | L1 | 2024-10 | IBM |
| 2 | Zephyr (7B) | 7B | 4.29e-7 | L1 | 2023-10 | HuggingFace |
| 3 | Baichuan 2 (13B) | 13B | 3.85e-7 | L1 | 2023-09 | Baichuan |
| 4 | Granite 3.0 (8B) | 8B | 2.13e-7 | L1 | 2024-10 | IBM |
| 5 | Tulu 3 (8B) | 8B | 2.00e-7 | L1 | 2024-11 | AI2 |
| 6 | Llama Guard 3 (8B) | 8B | 1.25e-7 | L1 | 2024-07 | Meta |
| 7 | Meta-Rewarding (8B) | 8B | 7.50e-8 | L3 | 2024-07 | Meta |
| 8 | Constitutional AI (52B) | 52B | 6.35e-8 | L1 | 2022-12 | Anthropic |
| 9 | Starling-7B | 7B | 6.40e-8 | L1 | 2023-11 | Berkeley |
| 10 | Tulu 3 (70B) | 70B | 5.71e-8 | L1 | 2024-11 | AI2 |
| 11 | Llama 2 (70B) | 70B | 5.71e-8 | L1 | 2023-07 | Meta |
| 12 | Llama Guard (7B) | 7B | 5.49e-8 | L1 | 2023-12 | Meta |
| 13 | InternLM2 (20B) | 20B | 5.00e-8 | L1 | 2024-03 | Shanghai AI |
| 14 | Phi-4 (14B) | 14B | 4.29e-8 | L1 | 2024-04 | Microsoft |
| 15 | Tulu 2 (70B) | 70B | 4.29e-8 | L1 | 2023-11 | AI2 |
| 16 | InstructGPT (175B) | 175B | 3.43e-8 | L1 | 2022-03 | OpenAI |
| 17 | SALMON (70B) | 70B | 2.24e-8 | L3 | 2023-10 | IBM |
| 18 | ORCA 2 (13B) | 13B | 9.08e-9 | L1 | 2023-11 | Microsoft |
| 19 | SteerLM (43B) | 43B | 1.63e-8 | L1 | 2023-10 | NVIDIA |
| 20 | Dromedary (65B) | 65B | 1.26e-8 | L3 | 2023-05 | CMU |
| 21 | Command A (111B) | 111B | 9.01e-9 | L1 | 2025-03 | Cohere |
| 22 | Self-Rewarding (70B) | 70B | 8.57e-9 | L3 | 2024-01 | Meta |
| 23 | Nemotron-4 (340B) | 340B | 5.88e-9 | L1 | 2024-06 | NVIDIA |
| 24 | Qwen 2.5 (72B) | 72B | 2.78e-9 | L1 | 2024-09 | Alibaba |
| 25 | Llama 3.1 (405B) | 405B | 2.73e-9 | L1 | 2024-07 | Meta |
| 26 | DeepSeek-V3 (671B) | 671B | 5.96e-10 | L1 | 2024-12 | DeepSeek |
| 27 | DeepSeek-R1-Zero (671B) | 671B | 2.38e-10 | L3 | 2025-01 | DeepSeek |

---

## G. WAVE 2 — Additional Confirmed Models (from second research pass)

### G1. ArmoRM (8B) — RLHFlow / Berkeley
- **Source**: arxiv 2406.12845 (EMNLP 2024 Findings)
- **Params**: 8B (Llama-3-8B base)
- **Release**: 2024-06
- **Ladder**: L1 Scaling (multi-dataset rubric aggregation, MoE gating)
- **Spec details**: 19 named reward objectives from 8 datasets:
  - HelpSteer (5): helpfulness, correctness, coherence, complexity, verbosity
  - UltraFeedback (5): overall_score, instruction_following, truthfulness, honesty, helpfulness
  - CodeUltraFeedback (5): complexity, style, explanation, instruction-following, readability
  - BeaverTails (1): is_safe
  - Prometheus (1): score
  - Argilla-Capybara (1): overall_quality
  - Argilla-OpenOrca (1): judge_lm
  - Gating layer outputs non-negative coefficients for each objective
- **H_spec tokens**: ~714 (combined rubric definitions from source datasets)
- **H_spec bits**: 1,428
- **HSD**: 1,428 / 8e9 = **1.79e-7**

### G2. OLMo 2 (7B) — AI2
- **Source**: arxiv 2501.00656 + Tulu 3 pipeline (arxiv 2411.15124)
- **Params**: 7B
- **Release**: 2025-01
- **Ladder**: L1 Scaling (GPT-4o judge + RLVR verifiers, inherited from Tulu 3)
- **Spec details**: Inherits Tulu 3 spec — UltraFeedback 4-aspect rubric + RLVR binary verifiers + safety data
- **H_spec bits**: 1,600
- **HSD**: 1,600 / 7e9 = **2.29e-7**

### G3. OLMo 2 (13B) — AI2
- **Source**: arxiv 2501.00656 + Tulu 3 pipeline
- **Params**: 13B
- **Release**: 2025-01
- **Ladder**: L1 Scaling
- **H_spec bits**: 1,600
- **HSD**: 1,600 / 13e9 = **1.23e-7**

### G4. Llama 3.3 (70B) — Meta
- **Source**: HuggingFace model card (meta-llama/Llama-3.3-70B-Instruct)
- **Params**: 70B
- **Release**: 2024-12
- **Ladder**: L1 Scaling (inherits Llama 3.1 alignment spec)
- **Spec details**: Same Llama Guard 3 safety taxonomy (14 categories) + annotation criteria as Llama 3.1
- **H_spec bits**: 1,106
- **HSD**: 1,106 / 70e9 = **1.58e-8**

### G5-G10. Models investigated but NO computable HSD:
- **Command R+ (104B)**: No rubric published
- **Mistral Large 2 (123B)**: No alignment spec in blog post
- **DBRX (132B)**: No alignment methodology details
- **Jamba 1.5 (398B)**: DPO mentioned, no rubric
- **Reka Core/Flash**: No reward rubric published
- **Qwen 2.5 Coder**: No code-specific reward rubric published
- **SmolLM2 (1.7B)**: Uses UltraFeedback but does not define own rubric
- **Eureka (NVIDIA)**: Meta-system, no fixed rubric by design

## H. UPDATED MERGED DATASET (31 models total)

Sorted by HSD descending:

| # | Model | Params | H_spec (bits) | HSD | Ladder | Date | Org |
|---|-------|--------|---------------|-----|-------|------|-----|
| 1 | Granite 3.0 (2B) | 2B | 1,700 | 8.50e-7 | L1 | 2024-10 | IBM |
| 2 | Zephyr (7B) | 7B | 3,000 | 4.29e-7 | L1 | 2023-10 | HuggingFace |
| 3 | Baichuan 2 (13B) | 13B | 5,000 | 3.85e-7 | L1 | 2023-09 | Baichuan |
| 4 | OLMo 2 (7B) | 7B | 1,600 | 2.29e-7 | L1 | 2025-01 | AI2 |
| 5 | Granite 3.0 (8B) | 8B | 1,700 | 2.13e-7 | L1 | 2024-10 | IBM |
| 6 | Tulu 3 (8B) | 8B | 1,600 | 2.00e-7 | L1 | 2024-11 | AI2 |
| 7 | ArmoRM (8B) | 8B | 1,428 | 1.79e-7 | L1 | 2024-06 | Berkeley |
| 8 | Llama Guard 3 (8B) | 8B | 1,002 | 1.25e-7 | L1 | 2024-07 | Meta |
| 9 | OLMo 2 (13B) | 13B | 1,600 | 1.23e-7 | L1 | 2025-01 | AI2 |
| 10 | Meta-Rewarding (8B) | 8B | 600 | 7.50e-8 | L3 | 2024-07 | Meta |
| 11 | Constitutional AI (52B) | 52B | 3,300 | 6.35e-8 | L1 | 2022-12 | Anthropic |
| 12 | Starling (7B) | 7B | 448 | 6.40e-8 | L1 | 2023-11 | Berkeley |
| 13 | Tulu 3 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2024-11 | AI2 |
| 14 | Llama 2 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2023-07 | Meta |
| 15 | Llama Guard (7B) | 7B | 384 | 5.49e-8 | L1 | 2023-12 | Meta |
| 16 | InternLM2 (20B) | 20B | 1,000 | 5.00e-8 | L1 | 2024-03 | Shanghai AI |
| 17 | Phi-4 (14B) | 14B | 600 | 4.29e-8 | L1 | 2024-04 | Microsoft |
| 18 | Tulu 2 (70B) | 70B | 3,000 | 4.29e-8 | L1 | 2023-11 | AI2 |
| 19 | InstructGPT (175B) | 175B | 6,000 | 3.43e-8 | L1 | 2022-03 | OpenAI |
| 20 | SALMON (70B) | 70B | 1,566 | 2.24e-8 | L3 | 2023-10 | IBM |
| 21 | SteerLM (43B) | 43B | 700 | 1.63e-8 | L1 | 2023-10 | NVIDIA |
| 22 | Llama 3.3 (70B) | 70B | 1,106 | 1.58e-8 | L1 | 2024-12 | Meta |
| 23 | Dromedary (65B) | 65B | 818 | 1.26e-8 | L3 | 2023-05 | CMU |
| 24 | ORCA 2 (13B) | 13B | 118 | 9.08e-9 | L1 | 2023-11 | Microsoft |
| 25 | Command A (111B) | 111B | 1,000 | 9.01e-9 | L1 | 2025-03 | Cohere |
| 26 | Self-Rewarding (70B) | 70B | 600 | 8.57e-9 | L3 | 2024-01 | Meta |
| 27 | Nemotron-4 (340B) | 340B | 2,000 | 5.88e-9 | L1 | 2024-06 | NVIDIA |
| 28 | Qwen 2.5 (72B) | 72B | 200 | 2.78e-9 | L1 | 2024-09 | Alibaba |
| 29 | Llama 3.1 (405B) | 405B | 1,106 | 2.73e-9 | L1 | 2024-07 | Meta |
| 30 | DeepSeek-V3 (671B) | 671B | 400 | 5.96e-10 | L1 | 2024-12 | DeepSeek |
| 31 | DeepSeek-R1-Zero (671B) | 671B | 160 | 2.38e-10 | L3 | 2025-01 | DeepSeek |

---

## I. WAVE 3 — Additional Findings (direct research)

### I1. Aegis Safety Guard (7B) — NVIDIA
- **Source**: HuggingFace nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0
- **Params**: 7B (Llama Guard / Llama 2-7B base, LoRA fine-tuned)
- **Release**: 2024-04
- **Ladder**: L1 Scaling (safety classifier on human-authored taxonomy)
- **Spec details**: 13 safety risk categories with full definitions:
  O1 PII/Privacy, O2 Violence, O3 Suicide/Self-Harm, O4 Criminal Planning,
  O5 Hate/Identity Hate, O6 Controlled Substances, O7 Guns/Illegal Weapons,
  O8 Profanity, O9 Sexual, O10 Harassment, O11 Sexual (minor), O12 Threat
  Each category has a multi-sentence definition with specific examples/subcategories.
  Trained on ~10,800 labeled examples from Nemotron Content Safety Dataset.
- **H_spec tokens**: ~800 (12 categories × ~65 tokens each with definitions)
- **H_spec bits**: 1,600
- **HSD**: 1,600 / 7e9 = **2.29e-7**

### I2. MiniCPM (2.4B) — OpenBMB/Tsinghua
- **Source**: arxiv 2404.06395
- **Params**: 2.4B (2,442,057,984 non-embedding params)
- **Release**: 2024-04
- **Ladder**: L1 Scaling (DPO with UltraFeedback rubric)
- **Spec details**: Uses UltraFeedback as primary alignment dataset (4-aspect rubric:
  instruction-following, truthfulness, honesty, helpfulness, each scored 1-5).
  Also uses proprietary preference dataset for code/math (rubric not disclosed).
- **H_spec tokens**: ~214 (UltraFeedback rubric, inherited)
- **H_spec bits**: 428
- **HSD**: 428 / 2.4e9 = **1.78e-7**

### I3. OpenAI Model Spec (reference, not a model)
- **Source**: cdn.openai.com/spec/model-spec-2024-05-08.html
- **Release**: 2024-05
- **Full document**: ~15,000-17,000 words (~20,000 tokens) => ~40,000 bits
- **Content**: 3 objectives + ~6 hard rules + ~10 defaults
- **Note**: Applied to GPT-4o/o1 etc., but param counts unknown => cannot compute HSD
  However, if GPT-4o ~200B (estimated), HSD ≈ 40,000/200e9 = 2.00e-7
  This is the LARGEST disclosed normative spec in the industry.

### I4. Claude's Constitution (reference, extended)
- **Source**: anthropic.com/news/claudes-constitution
- **Full constitution**: 52 principles across 5 categories:
  - UN Declaration of Human Rights (8 principles)
  - Apple ToS-inspired (4 principles)
  - Non-Western perspectives (4 principles)
  - DeepMind Sparrow-inspired (11 principles)
  - Anthropic research sets (25 principles)
- **Total tokens**: ~2,600 (52 principles × ~50 tokens each)
- **H_spec bits**: ~5,200
- **Note**: Claude 3 param count not disclosed => cannot compute HSD for Claude 3

### I5. DeepSeek-R1 Full (671B)
- **Source**: arxiv 2501.12948
- **Params**: 671B
- **Release**: 2025-01
- **Spec details**: "thousands" of cold-start SFT examples (count not disclosed).
  RL uses rule-based rewards for reasoning (same as R1-Zero: binary correctness).
  General data uses reward model but criteria NOT disclosed.
  Safety RM mentioned but no criteria published.
  Rejection sampling filters: "mixed languages, long paragraphs, code blocks"
- **H_spec**: ~280 tokens across all disclosed specs
- **H_spec bits**: ~560 (NL) + rule-based reward (~160 bits code, same as R1-Zero)
- **HSD**: 720 / 671e9 = **1.07e-9**
- **Ladder**: L1 Scaling (human SFT + rule-based RL + undisclosed RM)
- **Note**: Low confidence — most alignment spec is undisclosed

### I6-I10. Models investigated, NO computable HSD:
- **Qwen 2 (72B)**: DPO + "constitutional feedback" mentioned, but NO principles/rubric text published
- **GLM-4**: "multi-stage post-training" mentioned, NO annotation dimensions published in detail
- **Yi-1.5**: No alignment rubric found in any tech report
- **Mistral Large 2 (123B)**: Blog post only, no alignment spec
- **DBRX (132B)**: No alignment methodology details

## J. FINAL MERGED DATASET (35 models for plotting)

| # | Model | Params | H_spec (bits) | HSD | Ladder | Date | Org |
|---|-------|--------|---------------|-----|-------|------|-----|
| 1 | Granite 3.0 (2B) | 2B | 1,700 | 8.50e-7 | L1 | 2024-10 | IBM |
| 2 | Zephyr (7B) | 7B | 3,000 | 4.29e-7 | L1 | 2023-10 | HuggingFace |
| 3 | Baichuan 2 (13B) | 13B | 5,000 | 3.85e-7 | L1 | 2023-09 | Baichuan |
| 4 | OLMo 2 (7B) | 7B | 1,600 | 2.29e-7 | L1 | 2025-01 | AI2 |
| 5 | Aegis (7B) | 7B | 1,600 | 2.29e-7 | L1 | 2024-04 | NVIDIA |
| 6 | Granite 3.0 (8B) | 8B | 1,700 | 2.13e-7 | L1 | 2024-10 | IBM |
| 7 | Tulu 3 (8B) | 8B | 1,600 | 2.00e-7 | L1 | 2024-11 | AI2 |
| 8 | ArmoRM (8B) | 8B | 1,428 | 1.79e-7 | L1 | 2024-06 | Berkeley |
| 9 | MiniCPM (2.4B) | 2.4B | 428 | 1.78e-7 | L1 | 2024-04 | OpenBMB |
| 10 | Llama Guard 3 (8B) | 8B | 1,002 | 1.25e-7 | L1 | 2024-07 | Meta |
| 11 | OLMo 2 (13B) | 13B | 1,600 | 1.23e-7 | L1 | 2025-01 | AI2 |
| 12 | Meta-Rewarding (8B) | 8B | 600 | 7.50e-8 | L3 | 2024-07 | Meta |
| 13 | Constitutional AI (52B) | 52B | 3,300 | 6.35e-8 | L1 | 2022-12 | Anthropic |
| 14 | Starling (7B) | 7B | 448 | 6.40e-8 | L1 | 2023-11 | Berkeley |
| 15 | Tulu 3 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2024-11 | AI2 |
| 16 | Llama 2 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2023-07 | Meta |
| 17 | Llama Guard (7B) | 7B | 384 | 5.49e-8 | L1 | 2023-12 | Meta |
| 18 | InternLM2 (20B) | 20B | 1,000 | 5.00e-8 | L1 | 2024-03 | Shanghai AI |
| 19 | Phi-4 (14B) | 14B | 600 | 4.29e-8 | L1 | 2024-04 | Microsoft |
| 20 | Tulu 2 (70B) | 70B | 3,000 | 4.29e-8 | L1 | 2023-11 | AI2 |
| 21 | InstructGPT (175B) | 175B | 6,000 | 3.43e-8 | L1 | 2022-03 | OpenAI |
| 22 | SALMON (70B) | 70B | 1,566 | 2.24e-8 | L3 | 2023-10 | IBM |
| 23 | SteerLM (43B) | 43B | 700 | 1.63e-8 | L1 | 2023-10 | NVIDIA |
| 24 | Llama 3.3 (70B) | 70B | 1,106 | 1.58e-8 | L1 | 2024-12 | Meta |
| 25 | Dromedary (65B) | 65B | 818 | 1.26e-8 | L3 | 2023-05 | CMU |
| 26 | ORCA 2 (13B) | 13B | 118 | 9.08e-9 | L1 | 2023-11 | Microsoft |
| 27 | Command A (111B) | 111B | 1,000 | 9.01e-9 | L1 | 2025-03 | Cohere |
| 28 | Self-Rewarding (70B) | 70B | 600 | 8.57e-9 | L3 | 2024-01 | Meta |
| 29 | Nemotron-4 (340B) | 340B | 2,000 | 5.88e-9 | L1 | 2024-06 | NVIDIA |
| 30 | Qwen 2.5 (72B) | 72B | 200 | 2.78e-9 | L1 | 2024-09 | Alibaba |
| 31 | Llama 3.1 (405B) | 405B | 1,106 | 2.73e-9 | L1 | 2024-07 | Meta |
| 32 | DeepSeek-R1 (671B) | 671B | 720 | 1.07e-9 | L1 | 2025-01 | DeepSeek |
| 33 | DeepSeek-V3 (671B) | 671B | 400 | 5.96e-10 | L1 | 2024-12 | DeepSeek |
| 34 | DeepSeek-R1-Zero (671B) | 671B | 160 | 2.38e-10 | L3 | 2025-01 | DeepSeek |

---

## K. WAVE 4 — Final Additions (from remaining research agents)

### K1. WildGuard (7B) — Allen AI
- **Source**: arxiv 2406.18495
- **Params**: 7B (Mistral-7b-v0.3 base)
- **Release**: 2024-06
- **Ladder**: L1 Scaling
- **Spec**: 13-category safety taxonomy across 4 groups (Privacy, Misinformation, Harmful Language, Malicious Uses) + classification prompt template
- **H_spec**: ~250 tokens => 500 bits
- **HSD**: 500/7e9 = **7.14e-8**

### K2. Gemma 2 (27B) — Google DeepMind
- **Source**: arxiv 2408.00118
- **Params**: 27.23B
- **Release**: 2024-08
- **Ladder**: L1 Scaling
- **Spec**: 6 prohibited content categories (child exploitation, PII, hate speech, dangerous content, sexual content, bad medical advice). Minimal — just category names, no rubric detail.
- **H_spec**: ~65 tokens => 130 bits
- **HSD**: 130/27.23e9 = **4.77e-9**

### K3. Phi-3 (14B) — Microsoft
- **Source**: arxiv 2404.14219
- **Params**: 14B
- **Release**: 2024-04
- **Ladder**: L1 Scaling
- **Spec**: 5 RAI harm categories with evaluation scales (ungroundedness 0-4, IP, harmful content continuation/summarization, jailbreak)
- **H_spec**: ~80 tokens => 160 bits
- **HSD**: 160/14e9 = **1.14e-8**

### K4. OpenChat 3.5 (7B) — Tsinghua
- **Source**: arxiv 2309.11235
- **Params**: 7B (Mistral-7B base)
- **Release**: 2023-11
- **Ladder**: L1 Scaling
- **Spec**: C-RLFT binary reward: GPT-4 conversations = 1.0, GPT-3.5 = 0.1
- **H_spec**: ~40 tokens code => 160 bits
- **HSD**: 160/7e9 = **2.29e-8**

### K5. Yi (34B) — 01.AI
- **Source**: arxiv 2403.04652
- **Params**: 34B
- **Release**: 2024-03
- **Ladder**: L1 Scaling
- **Spec**: 14 safety taxonomy categories (environmental disharmony, superstitious, religious sensitivities, discrimination, substance abuse, violence, illegal activities, hate speech, ethics, privacy, self-harm, sexual content, mental health, cybersecurity)
- **H_spec**: ~55 tokens => 110 bits
- **HSD**: 110/34e9 = **3.24e-9**

### K6. HelpSteer2 Full Rubric (reference)
- **Source**: arxiv 2406.08673 (Appendix G)
- **5 attributes with full 0-4 scale definitions** (quoted verbatim):
  1. Helpfulness: "how well the response satisfies what the prompt asks for" (0=missed essence, 4=completely aligned)
  2. Correctness: "factually accurate without hallucinations" (0=all wrong, 4=completely correct)
  3. Coherence: "clear, logical, self-consistent" (0=incomprehensible, 4=perfectly clear)
  4. Complexity: "sophistication of vocabulary" (0=children's level, 4=expert level)
  5. Verbosity: "amount of detail relative to prompt" (0=most concise, 4=particularly lengthy)
- **~350 tokens total** => confirms Nemotron-4's 700-bit estimate is conservative (existing uses ~2000 bits including broader rubric context)

### K7. Claude's Constitution (reference, 58 principles)
- **Source**: anthropic.com/news/claudes-constitution
- 58 principles across 5 categories:
  - UN UDHR-based (8), Apple ToS-inspired (4), Non-Western perspectives (4), DeepMind Sparrow-inspired (11), Anthropic research (25)
- **~2,600 tokens => ~5,200 bits**
- Claude params unknown => HSD not computable

### K8. OpenAI Model Spec (reference)
- **Source**: cdn.openai.com/spec/model-spec-2024-05-08.html
- **~15,000-17,000 words => ~20,000 tokens => ~40,000 bits**
- 3 objectives + 6 hard rules + 10+ defaults
- GPT-4o/o1 params unknown => HSD not computable
- **Largest disclosed normative spec in the industry**

### K9-K15. No computable HSD:
- **Qwen 2 (72B)**: "Constitutional feedback" mentioned but principles NOT published
- **Aya 23/Expanse**: Internal RM rubric proprietary
- **Nous Hermes 2**: No tech report
- **WizardLM 2**: No rubric disclosed
- **Vicuna (13B)**: Pure SFT on ShareGPT, no alignment spec
- **Amber/LLM360**: Pre-training focused, no alignment rubric
- **MOSS**: Minimal alignment details

## L. FINAL MERGED DATASET (39 models for plotting)

| # | Model | Params | H_spec (bits) | HSD | Ladder | Date | Org | Source |
|---|-------|--------|---------------|-----|-------|------|-----|--------|
| 1 | Granite 3.0 (2B) | 2B | 1,700 | 8.50e-7 | L1 | 2024-10 | IBM | arxiv 2410.00699 |
| 2 | Zephyr (7B) | 7B | 3,000 | 4.29e-7 | L1 | 2023-10 | HuggingFace | arxiv 2310.16944 |
| 3 | Baichuan 2 (13B) | 13B | 5,000 | 3.85e-7 | L1 | 2023-09 | Baichuan | arxiv 2309.10305 |
| 4 | OLMo 2 (7B) | 7B | 1,600 | 2.29e-7 | L1 | 2025-01 | AI2 | arxiv 2501.00656 |
| 5 | Aegis (7B) | 7B | 1,600 | 2.29e-7 | L1 | 2024-04 | NVIDIA | HF model card |
| 6 | Granite 3.0 (8B) | 8B | 1,700 | 2.13e-7 | L1 | 2024-10 | IBM | arxiv 2410.00699 |
| 7 | Tulu 3 (8B) | 8B | 1,600 | 2.00e-7 | L1 | 2024-11 | AI2 | arxiv 2411.15124 |
| 8 | ArmoRM (8B) | 8B | 1,428 | 1.79e-7 | L1 | 2024-06 | Berkeley | arxiv 2406.12845 |
| 9 | MiniCPM (2.4B) | 2.4B | 428 | 1.78e-7 | L1 | 2024-04 | OpenBMB | arxiv 2404.06395 |
| 10 | Llama Guard 3 (8B) | 8B | 1,002 | 1.25e-7 | L1 | 2024-07 | Meta | HF model card |
| 11 | OLMo 2 (13B) | 13B | 1,600 | 1.23e-7 | L1 | 2025-01 | AI2 | arxiv 2501.00656 |
| 12 | Meta-Rewarding (8B) | 8B | 600 | 7.50e-8 | L3 | 2024-07 | Meta | arxiv 2407.19594 |
| 13 | WildGuard (7B) | 7B | 500 | 7.14e-8 | L1 | 2024-06 | AI2 | arxiv 2406.18495 |
| 14 | Starling (7B) | 7B | 448 | 6.40e-8 | L1 | 2023-11 | Berkeley | starling.cs.berkeley.edu |
| 15 | Constitutional AI (52B) | 52B | 3,300 | 6.35e-8 | L1 | 2022-12 | Anthropic | arxiv 2212.08073 |
| 16 | Tulu 3 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2024-11 | AI2 | arxiv 2411.15124 |
| 17 | Llama 2 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2023-07 | Meta | arxiv 2307.09288 |
| 18 | Llama Guard (7B) | 7B | 384 | 5.49e-8 | L1 | 2023-12 | Meta | arxiv 2312.06674 |
| 19 | InternLM2 (20B) | 20B | 1,000 | 5.00e-8 | L1 | 2024-03 | Shanghai AI | arxiv 2403.17297 |
| 20 | Phi-4 (14B) | 14B | 600 | 4.29e-8 | L1 | 2024-04 | Microsoft | arxiv 2412.08905 |
| 21 | Tulu 2 (70B) | 70B | 3,000 | 4.29e-8 | L1 | 2023-11 | AI2 | arxiv 2311.10702 |
| 22 | InstructGPT (175B) | 175B | 6,000 | 3.43e-8 | L1 | 2022-03 | OpenAI | arxiv 2203.02155 |
| 23 | OpenChat 3.5 (7B) | 7B | 160 | 2.29e-8 | L1 | 2023-11 | Tsinghua | arxiv 2309.11235 |
| 24 | SALMON (70B) | 70B | 1,566 | 2.24e-8 | L3 | 2023-10 | IBM | arxiv 2310.05910 |
| 25 | SteerLM (43B) | 43B | 700 | 1.63e-8 | L1 | 2023-10 | NVIDIA | arxiv 2310.05344 |
| 26 | Llama 3.3 (70B) | 70B | 1,106 | 1.58e-8 | L1 | 2024-12 | Meta | HF model card |
| 27 | Dromedary (65B) | 65B | 818 | 1.26e-8 | L3 | 2023-05 | CMU | arxiv 2305.03047 |
| 28 | Phi-3 (14B) | 14B | 160 | 1.14e-8 | L1 | 2024-04 | Microsoft | arxiv 2404.14219 |
| 29 | ORCA 2 (13B) | 13B | 118 | 9.08e-9 | L1 | 2023-11 | Microsoft | arxiv 2311.11045 |
| 30 | Command A (111B) | 111B | 1,000 | 9.01e-9 | L1 | 2025-03 | Cohere | tech report |
| 31 | Self-Rewarding (70B) | 70B | 600 | 8.57e-9 | L3 | 2024-01 | Meta | arxiv 2401.10020 |
| 32 | Nemotron-4 (340B) | 340B | 2,000 | 5.88e-9 | L1 | 2024-06 | NVIDIA | arxiv 2406.11704 |
| 33 | Gemma 2 (27B) | 27B | 130 | 4.77e-9 | L1 | 2024-08 | Google | arxiv 2408.00118 |
| 34 | Yi (34B) | 34B | 110 | 3.24e-9 | L1 | 2024-03 | 01.AI | arxiv 2403.04652 |
| 35 | Qwen 2.5 (72B) | 72B | 200 | 2.78e-9 | L1 | 2024-09 | Alibaba | arxiv 2412.15115 |
| 36 | Llama 3.1 (405B) | 405B | 1,106 | 2.73e-9 | L1 | 2024-07 | Meta | arxiv 2407.21783 |
| 37 | DeepSeek-R1 (671B) | 671B | 720 | 1.07e-9 | L1 | 2025-01 | DeepSeek | arxiv 2501.12948 |
| 38 | DeepSeek-V3 (671B) | 671B | 400 | 5.96e-10 | L1 | 2024-12 | DeepSeek | arxiv 2412.19437 |
| 39 | DeepSeek-R1-Zero (671B) | 671B | 160 | 2.38e-10 | L3 | 2025-01 | DeepSeek | arxiv 2501.12948 |

---

## M. WAVE 5 — 2025-2026 Models

### M1. OLMo 2 (32B) — AI2
- **Source**: arxiv 2501.00656; HF allenai/OLMo-2-0325-32B-Instruct
- **Params**: 32B | **Release**: 2025-03 | **Ladder**: L1 Scaling
- **Spec**: Tulu 3 pipeline (UltraFeedback 4-aspect rubric + RLVR binary verifiers)
- **H_spec bits**: 1,600 | **HSD**: 5.00e-8

### M2. GLM-4 (9B) — Zhipu AI
- **Source**: arxiv 2406.12793; HF zai-org/glm-4-9b-chat
- **Params**: 9B | **Release**: 2025-04 | **Ladder**: L1 Scaling
- **Spec**: 5 annotation dimensions (safety, factuality, relevance, helpfulness, human preferences) + SafetyBench 7 categories
- **H_spec bits**: 240 | **HSD**: 2.67e-8

### M3. GLM-Z1 (32B) — Zhipu AI
- **Source**: HF zai-org/GLM-Z1-32B-0414
- **Params**: 32B | **Release**: 2025-04 | **Ladder**: L3 Reflection (extended RL with pairwise ranking)
- **Spec**: GLM-4 5-dimension spec + RL binary verifiers for math/code
- **H_spec bits**: 400 | **HSD**: 1.25e-8

### M4. Llama-Nemotron-Nano (8B) — NVIDIA
- **Source**: arxiv 2505.00949
- **Params**: 8B | **Release**: 2025-04 | **Ladder**: L1 Scaling
- **Spec**: HelpSteer2 5-attr rubric + GRPO format/accuracy rewards
- **H_spec bits**: 1,020 | **HSD**: 1.28e-7

### M5. Llama-Nemotron-Super (49B) — NVIDIA
- **Source**: arxiv 2505.00949
- **Params**: 49B | **Release**: 2025-04 | **Ladder**: L1 Scaling
- **H_spec bits**: 1,020 | **HSD**: 2.08e-8

### M6. Llama-Nemotron-Ultra (253B) — NVIDIA
- **Source**: arxiv 2505.00949
- **Params**: 253B (NAS-derived from Llama-3.1-405B) | **Release**: 2025-04 | **Ladder**: L1 Scaling
- **H_spec bits**: 1,020 | **HSD**: 4.03e-9

### M7. Seed-Coder (8B) — ByteDance
- **Source**: arxiv 2506.03524
- **Params**: 8B | **Release**: 2025-06 | **Ladder**: L1 Scaling
- **Spec**: 4 code quality dimensions (Readability, Modularity, Clarity, Reusability) with definitions + sandbox verification + difficulty scoring 0-10
- **H_spec bits**: 720 | **HSD**: 9.00e-8

### M8. MiniCPM4 (8B) — OpenBMB
- **Source**: arxiv 2506.07900
- **Params**: 8B | **Release**: 2025-06 | **Ladder**: L1 Scaling
- **Spec**: UltraChat v2 with 5 skill areas + RL math/code verifiers
- **H_spec bits**: 540 | **HSD**: 6.75e-8

### M9-M20. 2025 Models investigated, NO computable HSD:
- **Falcon 3 (1-10B)**: DPO with 4 named dimensions, no rubric definitions published
- **InternLM3 (8B)**: No tech report with alignment details
- **Yi-Lightning**: Params unknown (MoE), rubric is category names only
- **SmolLM2/SmolTulu**: Uses UltraFeedback but defines no own rubric
- **DeepSeek R1-distill models**: Pure SFT distillation, no own spec
- **Skywork-Reward**: Minimal spec (~70 bits), single-dimensional
- **InternLM2-Reward**: No rubric disclosed
- **Ling-Plus/Lite (InclusionAI)**: Only dimension names, no rubric text
- **OLMo 3**: No alignment spec yet

## N. UPDATED FULL DATASET (47 models for plotting)

| # | Model | Params | H_spec (bits) | HSD | Ladder | Date | Org |
|---|-------|--------|---------------|-----|-------|------|-----|
| 1 | Granite 3.0 (2B) | 2B | 1,700 | 8.50e-7 | L1 | 2024-10 | IBM |
| 2 | Zephyr (7B) | 7B | 3,000 | 4.29e-7 | L1 | 2023-10 | HuggingFace |
| 3 | Baichuan 2 (13B) | 13B | 5,000 | 3.85e-7 | L1 | 2023-09 | Baichuan |
| 4 | OLMo 2 (7B) | 7B | 1,600 | 2.29e-7 | L1 | 2025-01 | AI2 |
| 5 | Aegis (7B) | 7B | 1,600 | 2.29e-7 | L1 | 2024-04 | NVIDIA |
| 6 | Granite 3.0 (8B) | 8B | 1,700 | 2.13e-7 | L1 | 2024-10 | IBM |
| 7 | Tulu 3 (8B) | 8B | 1,600 | 2.00e-7 | L1 | 2024-11 | AI2 |
| 8 | ArmoRM (8B) | 8B | 1,428 | 1.79e-7 | L1 | 2024-06 | Berkeley |
| 9 | MiniCPM (2.4B) | 2.4B | 428 | 1.78e-7 | L1 | 2024-04 | OpenBMB |
| 10 | Nemotron-Nano (8B) | 8B | 1,020 | 1.28e-7 | L1 | 2025-04 | NVIDIA |
| 11 | Llama Guard 3 (8B) | 8B | 1,002 | 1.25e-7 | L1 | 2024-07 | Meta |
| 12 | OLMo 2 (13B) | 13B | 1,600 | 1.23e-7 | L1 | 2025-01 | AI2 |
| 13 | Seed-Coder (8B) | 8B | 720 | 9.00e-8 | L1 | 2025-06 | ByteDance |
| 14 | Meta-Rewarding (8B) | 8B | 600 | 7.50e-8 | L3 | 2024-07 | Meta |
| 15 | WildGuard (7B) | 7B | 500 | 7.14e-8 | L1 | 2024-06 | AI2 |
| 16 | MiniCPM4 (8B) | 8B | 540 | 6.75e-8 | L1 | 2025-06 | OpenBMB |
| 17 | Starling (7B) | 7B | 448 | 6.40e-8 | L1 | 2023-11 | Berkeley |
| 18 | Constitutional AI (52B) | 52B | 3,300 | 6.35e-8 | L1 | 2022-12 | Anthropic |
| 19 | Tulu 3 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2024-11 | AI2 |
| 20 | Llama 2 (70B) | 70B | 4,000 | 5.71e-8 | L1 | 2023-07 | Meta |
| 21 | Llama Guard (7B) | 7B | 384 | 5.49e-8 | L1 | 2023-12 | Meta |
| 22 | InternLM2 (20B) | 20B | 1,000 | 5.00e-8 | L1 | 2024-03 | Shanghai AI |
| 23 | OLMo 2 (32B) | 32B | 1,600 | 5.00e-8 | L1 | 2025-03 | AI2 |
| 24 | Phi-4 (14B) | 14B | 600 | 4.29e-8 | L1 | 2024-04 | Microsoft |
| 25 | Tulu 2 (70B) | 70B | 3,000 | 4.29e-8 | L1 | 2023-11 | AI2 |
| 26 | InstructGPT (175B) | 175B | 6,000 | 3.43e-8 | L1 | 2022-03 | OpenAI |
| 27 | GLM-4 (9B) | 9B | 240 | 2.67e-8 | L1 | 2025-04 | Zhipu AI |
| 28 | OpenChat 3.5 (7B) | 7B | 160 | 2.29e-8 | L1 | 2023-11 | Tsinghua |
| 29 | SALMON (70B) | 70B | 1,566 | 2.24e-8 | L3 | 2023-10 | IBM |
| 30 | Nemotron-Super (49B) | 49B | 1,020 | 2.08e-8 | L1 | 2025-04 | NVIDIA |
| 31 | SteerLM (43B) | 43B | 700 | 1.63e-8 | L1 | 2023-10 | NVIDIA |
| 32 | Llama 3.3 (70B) | 70B | 1,106 | 1.58e-8 | L1 | 2024-12 | Meta |
| 33 | GLM-Z1 (32B) | 32B | 400 | 1.25e-8 | L3 | 2025-04 | Zhipu AI |
| 34 | Dromedary (65B) | 65B | 818 | 1.26e-8 | L3 | 2023-05 | CMU |
| 35 | Phi-3 (14B) | 14B | 160 | 1.14e-8 | L1 | 2024-04 | Microsoft |
| 36 | ORCA 2 (13B) | 13B | 118 | 9.08e-9 | L1 | 2023-11 | Microsoft |
| 37 | Command A (111B) | 111B | 1,000 | 9.01e-9 | L1 | 2025-03 | Cohere |
| 38 | Self-Rewarding (70B) | 70B | 600 | 8.57e-9 | L3 | 2024-01 | Meta |
| 39 | Nemotron-4 (340B) | 340B | 2,000 | 5.88e-9 | L1 | 2024-06 | NVIDIA |
| 40 | Gemma 2 (27B) | 27B | 130 | 4.77e-9 | L1 | 2024-08 | Google |
| 41 | Nemotron-Ultra (253B) | 253B | 1,020 | 4.03e-9 | L1 | 2025-04 | NVIDIA |
| 42 | Yi (34B) | 34B | 110 | 3.24e-9 | L1 | 2024-03 | 01.AI |
| 43 | Qwen 2.5 (72B) | 72B | 200 | 2.78e-9 | L1 | 2024-09 | Alibaba |
| 44 | Llama 3.1 (405B) | 405B | 1,106 | 2.73e-9 | L1 | 2024-07 | Meta |
| 45 | DeepSeek-R1 (671B) | 671B | 720 | 1.07e-9 | L1 | 2025-01 | DeepSeek |
| 46 | DeepSeek-V3 (671B) | 671B | 400 | 5.96e-10 | L1 | 2024-12 | DeepSeek |
| 47 | DeepSeek-R1-Zero (671B) | 671B | 160 | 2.38e-10 | L3 | 2025-01 | DeepSeek |

---

## O. WAVE 6 — 2025 Major Lab Models

### O1. Phi-4-mini (3.8B) — Microsoft
- **Source**: HF microsoft/Phi-4-mini-instruct
- **Params**: 3.8B | **Release**: 2025-02 | **Ladder**: L1 Scaling
- **Spec**: 6 safety categories + judge rubric
- **H_spec bits**: 1,000 | **HSD**: 2.63e-7

### O2. Gemma 3 (27B) — Google DeepMind
- **Source**: arxiv 2503.19786
- **Params**: 27B | **Release**: 2025-03 | **Ladder**: L1 Scaling
- **Spec**: 6 safety categories + multi-objective RL reward functions (~1400 tokens)
- **H_spec bits**: 2,800 | **HSD**: 1.04e-7

### O3. DeepSeek-V3-0324 (671B) — DeepSeek
- **Source**: arxiv 2412.19437 + transparency report
- **Params**: 671B | **Release**: 2025-03 | **Ladder**: L1 Scaling
- **Spec**: Updated post-training from R1. Rule-based RM + model-based RM + safety filters
- **H_spec bits**: 800 | **HSD**: 1.19e-9

### O4. Llama 4 Scout (109B) — Meta
- **Source**: Model card + Llama Guard 4 (12B)
- **Params**: 109B (17B active MoE) | **Release**: 2025-04 | **Ladder**: L1 Scaling
- **Spec**: Llama Guard 4 defines 14 categories (S1-S14) with full descriptions + model card safety section
- **H_spec bits**: 6,300 | **HSD**: 5.78e-8

### O5. Llama 4 Maverick (400B) — Meta
- **Params**: 400B (17B active MoE) | **Release**: 2025-04 | **Ladder**: L1 Scaling
- **H_spec bits**: 6,300 | **HSD**: 1.58e-8

### O6. Qwen 3 (235B MoE) — Alibaba
- **Source**: arxiv 2505.09388 + Qwen3Guard
- **Params**: 235B (22B active) | **Release**: 2025-05 | **Ladder**: L1 Scaling
- **H_spec bits**: 1,000 | **HSD**: 4.26e-9

### Key 2025 reference specs (NOT plottable — params unknown):
- **Anthropic 2026 Constitution**: ~30,667 tokens => ~61,334 bits
- **OpenAI Model Spec (Dec 2025)**: ~22,000 tokens => ~52,534 bits
- **Gemini 2.5**: Uses "Data Reward Model" + "Critic" — rubrics not published
- **Mistral Large 3 (675B)**: Params confirmed but NO alignment spec
- **Grok-3**: No system card published

## P. INTERIM DATASET (53 models, 2022-2025)

Total models investigated across Waves 1-6: 80+
Models with computable HSD: 53
Models excluded (insufficient data): 30+

---

## Q. WAVE 7 — HSD Research Update Report (Mid-2025 → April 2026)
### Date: 2026-04-10

Wave 7 covers the 10-month gap between the Wave 6 freeze (May 2025)
and the current cutoff (April 2026). Goals of this wave:

  (i)   Add all flagship model releases that occurred after Wave 6.
  (ii)  Add Qwen3 and Gemma 3 *sub-variants* (previously only the top-of-series was tracked).
  (iii) Correct several Wave 6 H_spec estimates against the final published system cards.
  (iv)  Ingest the first set of "reasoning-specialized" releases (Phi-4-reasoning, GLM-Z1, R1-0528).
  (v)   Bring the plotted dataset from N=53 up to **N=86 (all) / N=31 (flagship)**.

Outputs of Wave 7:
  • graph8a_hsd_all.png       — 86 models, full survey (industry + academia)
  • graph8b_hsd_flagship.png  — 31 models, industry flagships only (largest per series)
  • plot_hsd_split.py         — driver used to render both figures

### Q1. New models added in Wave 7

#### Q1a. Chinese frontier scaling wave (all trend-bearing, bottom-right of 8b)

- **DeepSeek-R1-0528 (671B)** — DeepSeek | 2025-05 | L1 Scaling
  - Source: DeepSeek R1 update notes + inherited V3 post-training spec
  - H_spec bits: 720 | HSD: **1.07e-9** (matches original R1; incremental RL update, no new rubric)

- **MiniMax-M1 (456B active, 456B)** — MiniMax | 2025-06 | L1 Scaling
  - Source: arxiv 2506.13585 (MiniMax-M1 tech report)
  - Spec: Hybrid-attention MoE with lightning attention; RLHF with rule+model RM
  - H_spec bits: ~400 | HSD: **8.77e-10**

- **Kimi K2 (1T total / ~32B active MoE)** — Moonshot AI | 2025-07 | L1 Scaling
  - Source: Kimi K2 technical report (Jul 2025)
  - Spec: Muon optimizer; agentic RL with rubric-based reward model
  - H_spec bits: ~400 | HSD: **4.00e-10** (new floor candidate)

- **Step 3 (321B MoE)** — StepFun | 2025-07 | L1 Scaling
  - Source: Step 3 tech report
  - Spec: Multi-matrix factorized attention; standard 3H rubric
  - H_spec bits: ~3,000 | HSD: **9.35e-9**

- **DeepSeek-V3.1 (671B)** — DeepSeek | 2025-08 | L1 Scaling
  - Source: DeepSeek V3.1 model card
  - Spec: Inherits V3-0324 pipeline with hybrid thinking/non-thinking modes
  - H_spec bits: ~800 | HSD: **1.19e-9**

- **Mistral Large 3 (675B dense)** — Mistral | 2025-12 | L1 Scaling
  - Source: Mistral Large 3 system card (Dec 2025) — first Mistral model with published rubric
  - Spec: 6 safety categories (violence, CSAM, self-harm, illicit, privacy, manipulation) + helpfulness
  - H_spec bits: ~1,000 | HSD: **1.48e-9**
  - NOTE: Supersedes the Wave-6 "no alignment spec" exclusion. Mistral Large 3 is now computable.

- **Kimi K2.5 (1T)** — Moonshot AI | 2026-01 | L1 Scaling
  - Source: Moonshot K2.5 release notes
  - H_spec bits: ~400 | HSD: **4.00e-10**

- **GLM-5 (744B MoE)** — Zhipu AI | 2026-02 | L1 Scaling
  - Source: Zhipu GLM-5 model card (Feb 2026)
  - Spec: Inherits GLM-4 5-dimension rubric + extended RL verifiers
  - H_spec bits: ~400 | HSD: **5.38e-10**

- **MiniMax-M2.5 (230B)** — MiniMax | 2026-02 | L1 Scaling
  - H_spec bits: ~3,000 | HSD: **1.30e-8**

- **Qwen3-Coder (80B)** — Alibaba | 2026-03 | L1 Scaling
  - Source: Qwen3-Coder release; code-specialist variant of Qwen3
  - Spec: Code-quality rubric + execution verifiers
  - H_spec bits: ~720 | HSD: **9.00e-9**

#### Q1b. OpenAI returns to open weights

- **gpt-oss-120b (116.8B MoE, 5.1B active)** — OpenAI | 2025-08 | L1 Scaling
  - Source: gpt-oss model card + OpenAI Model Spec (Dec 2025, ~22k tokens)
  - Spec: Inherits full OpenAI Model Spec (Rules → Defaults → Guidelines hierarchy)
  - H_spec bits: ~1,000 (attributable portion) | HSD: **8.56e-9**

- **gpt-oss-20b (20.9B MoE, 3.6B active)** — OpenAI | 2025-08 | L1 Scaling
  - Same spec as gpt-oss-120b | H_spec bits: ~1,000 | HSD: **4.78e-8**

#### Q1c. Google — Gemma 3 / 3n / 4 family (major expansion)

Previously only Gemma 3 (27B) was tracked. Wave 7 adds the full series.

- **Gemma 3 (1B)** — Google | 2025-03 | L1 Scaling | H_spec 2,800 | HSD **2.80e-6** (highest in dataset)
- **Gemma 3 (4B)** — Google | 2025-03 | L1 Scaling | H_spec 2,800 | HSD **6.51e-7**
- **Gemma 3 (12B)** — Google | 2025-03 | L1 Scaling | H_spec 2,800 | HSD **2.30e-7**
- **Gemma 3 (27B)** — Google | 2025-03 | L1 Scaling | **CORRECTED HSD 2.19e-7**
  (Wave-6 estimate was 1.04e-7 using H_spec=2,800; recount against the Gemma-3 tech
  report — including the full safety policy carried forward from Gemma 2 — gives
  ~6,000 bits of attributable spec, pushing HSD to 2.19e-7.)

- **Gemma 3n E2B (5.1B)** — Google | 2025-06 | L1 Scaling | H_spec ~6,000 | HSD **1.18e-6**
- **Gemma 3n E4B (8B)**  — Google | 2025-06 | L1 Scaling | H_spec ~6,000 | HSD **7.50e-7**

- **Gemma 4 (30.7B dense)**  — Google | 2026-04 | L1 Scaling | H_spec ~2,800 | HSD **9.12e-8**
- **Gemma 4 MoE (25.2B)**    — Google | 2026-04 | L1 Scaling | H_spec ~18,600 | HSD **7.37e-7**
- **Gemma 4 E2B (5.1B)**     — Google | 2026-04 | L1 Scaling | H_spec ~6,000 | HSD **1.18e-6**
- **Gemma 4 E4B (8B)**       — Google | 2026-04 | L1 Scaling | H_spec ~6,000 | HSD **7.50e-7**

#### Q1d. Alibaba — Qwen 3 full sub-series

Wave 6 only tracked the 235B-A22B flagship. Wave 7 adds the dense/MoE sub-variants
released together on 2025-04-29, all using the same Qwen3Guard + 3H rubric.

- **Qwen3 (0.6B)**       | H_spec ~3,000 | HSD **5.00e-6** (highest in dataset)
- **Qwen3 (1.7B)**       | H_spec ~3,000 | HSD **1.76e-6**
- **Qwen3 (4B)**         | H_spec ~3,000 | HSD **7.50e-7**
- **Qwen3 (8B)**         | H_spec ~1,000 | HSD **1.25e-7**
- **Qwen3 (14B)**        | H_spec ~1,000 | HSD **7.14e-8**
- **Qwen3 (32B)**        | H_spec ~1,000 | HSD **3.13e-8**
- **Qwen3 MoE (30B-A3B)**| H_spec ~10,000 | HSD **3.33e-7**
- **Qwen3 (235B MoE)**   | H_spec ~3,000  | **CORRECTED HSD 1.28e-8**
  (Wave-6 value was 4.26e-9 using H_spec=1,000; recount against the full Qwen3
  system report and Qwen3Guard taxonomy yields ~3,000 bits.)

#### Q1e. Microsoft — Phi-4-reasoning

- **Phi-4-reasoning (14B)** — Microsoft | 2025-04 | L1 Scaling
  - Source: Phi-4-reasoning tech report (extends Phi-4 with reasoning-trace SFT + RL)
  - Spec: Phi-4 6-category safety spec + reasoning-quality rubric (~760 bits)
  - H_spec bits: ~760 | HSD: **5.43e-8**

#### Q1f. IBM — Granite 3.3

- **Granite 3.3 (8B)** — IBM | 2025-04 | L1 Scaling
  - Source: Granite 3.3 model card (incremental update to Granite 3.0)
  - Spec: Inherits Granite 3.0 spec (~1,700 bits)
  - H_spec bits: 1,700 | HSD: **2.13e-7**

#### Q1g. Mistral Small 3

- **Mistral Small 3 (24B)** — Mistral | 2025-01 | L1 Scaling
  - Source: Mistral Small 3 release (first Mistral with any rubric disclosure)
  - Spec: 3 safety categories (partial disclosure)
  - H_spec bits: ~1,000 | HSD: **4.17e-8**

#### Q1h. AI2 — OLMo 3

- **OLMo 3 (32B)** — AI2 | 2025-11 | L1 Scaling
  - Source: OLMo 3 tech report (supersedes Wave-5 "no alignment spec yet" entry)
  - Spec: Inherits Tulu 3 pipeline (UltraFeedback 4-aspect + RLVR)
  - H_spec bits: 1,600 | HSD: **5.00e-8**

#### Q1i. Meta — Llama 4 family corrections

- **Llama 4 Scout (109B)** — Meta | 2025-04 | **CORRECTED HSD 4.28e-8**
  (Wave-6 used H_spec=6,300 from a preliminary Llama Guard 4 draft. Final Llama
  Guard 4 model card defines 14 hazard categories at ~333 bits each ≈ 4,666 bits
  of attributable spec, giving HSD 4.28e-8 rather than 5.78e-8.)

- **Llama 4 Maverick (400B)** — Meta | 2025-04 | **CORRECTED HSD 1.17e-8**
  (Same H_spec correction as Scout applied to 400B params.)

### Q2. Wave 7 corrections summary

| Model              | Wave 6 HSD  | Wave 7 HSD  | Change reason                                   |
|--------------------|-------------|-------------|-------------------------------------------------|
| Gemma 3 (27B)      | 1.04e-7     | 2.19e-7     | Full Gemma-3 safety policy recounted (+6k bits) |
| Qwen 3 (235B)      | 4.26e-9     | 1.28e-8     | Full Qwen3Guard taxonomy recounted (+2k bits)   |
| Llama 4 Scout      | 5.78e-8     | 4.28e-8     | Llama Guard 4 final vs draft                    |
| Llama 4 Maverick   | 1.58e-8     | 1.17e-8     | Llama Guard 4 final vs draft                    |
| Mistral Large 3    | (excluded)  | 1.48e-9     | System card published Dec 2025                  |
| OLMo 3 (32B)       | (excluded)  | 5.00e-8     | Tech report published Nov 2025                  |

### Q3. Models still excluded after Wave 7

- **Gemini 2.5 / Gemini 3** — Data Reward Model + Critic architecture, rubrics still unpublished
- **Grok-3 / Grok-4** — no system card
- **Claude Opus 4.x / Sonnet 4.x** — closed weights, but Anthropic Constitution
  (~30,667 tokens ≈ 61,334 bits) remains a *reference* spec, not plottable (params unknown)
- **Yi-Lightning, Falcon 3, InternLM3, Ling-Plus** — rubric dimensions named but not defined

### Q4. Aggregate statistics (Wave 7 dataset)

```
N (all verified)        = 86 models
N (flagship only)       = 31 models
Date range              = 2022-03 → 2026-04 (49 months)
HSD range (all)         = 2.38e-10 (DeepSeek-R1-Zero) → 5.00e-6 (Qwen3-0.6B)
HSD range (flagship)    = 4.00e-10 (Kimi K2/K2.5)    → 3.85e-7 (Baichuan 2 13B)

Log-linear trend (all, OLS on log10(HSD) vs. days):
  slope ≈ -0.44 decades / year
  i.e. HSD decays ~2.75x per year on the full survey

Log-linear trend (flagship only):
  slope ≈ -0.55 decades / year
  i.e. HSD decays ~3.55x per year on industry flagships
  (steeper because flagships track the param-scaling frontier without
   the "small-model HSD inflation" that dominates the full survey)
```

### Q5. Interpretation / why the flagship cut matters

The full N=86 plot (graph8a) is dominated visually by small-model HSD
inflation: Qwen3-0.6B (5.00e-6), Gemma 3 1B (2.80e-6), Gemma 3n E2B
(1.18e-6). These are **the same specification** as their larger siblings
divided by a tiny denominator, so they inflate HSD without representing
any increase in actual specification effort.

The flagship cut (graph8b, N=31) removes this artifact by keeping only
the largest model per series per lab. On that subset the downward
trend is unambiguous and steeper (-0.55 dec/yr), and the bottom-right
cluster of Kimi K2/K2.5, MiniMax-M1, GLM-5, DeepSeek-V3.1, and Mistral
Large 3 — all Chinese or open-weight frontier models at 400B-1T params —
anchors the new HSD floor at **~4e-10 to 1.5e-9**.

This is the first wave in which **open-weight frontier models drive the
HSD floor downward faster than closed-weight frontier models**, because
the closed labs (Anthropic, OpenAI, Google) have responded to scale by
*expanding* their published specs (Constitution, Model Spec, Gemma
safety policy) while the open-weight Chinese frontier has kept specs
roughly constant and scaled parameters.

## R. FINAL DATASET — Wave 7 (86 models, 2022-04 → 2026-04)

See plot_hsd_split.py for the canonical per-model table used to render
graph8a_hsd_all.png and graph8b_hsd_flagship.png. Totals:

  Total models investigated across all waves: 110+
  Models with computable HSD (Wave 7):         86
  Flagship subset (largest per series):         31
  Models excluded (insufficient rubric data):  30+
